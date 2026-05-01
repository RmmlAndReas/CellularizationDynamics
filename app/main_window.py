from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QFileDialog,
    QSizePolicy,
    QComboBox,
    QGroupBox,
)

from .services.analyze_worker import AnalyzeWorker, GenerateFigureWorker
from .services.config_io import load_or_create_config
from .services.metadata_reader import read_imagej_params
from .services.io import atomic_write_tiff
from .services.output_layout import work_dir_for, ensure_work_tree
from .services.pipeline_adapter import (
    save_compat_roi_placeholder,
    compute_cytoplasm_size_over_time,
    update_apical_height_in_config,
)
from scripts.annotation_source import (
    build_apical_alignment_v2,
    raw_clicks_to_time_depth,
    write_annotation_tsv,
)
from .services.sample_state import AcquisitionParams, SampleState
from .services.session_restore import has_restorable_session, restore_interactive_session
from .widgets.defaults_panel import DefaultsPanel
from .widgets.drop_list import DropListWidget
from .widgets.threshold_panel import ThresholdPanel
from .widgets.front_panel import FrontPanel
from .widgets.result_panel import ResultPanel
from .widgets.movie_preview_dialog import MoviePreviewDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cellularization Desktop")
        self.resize(1280, 800)

        self.samples: dict[str, SampleState] = {}
        self.current_path: str | None = None
        self._analyze_worker = None
        self._figure_worker = None
        self._movie_preview: MoviePreviewDialog | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self.threshold_panel = ThresholdPanel()
        self.front_panel = FrontPanel()
        self.result_panel = ResultPanel()
        self.result_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.defaults_panel = DefaultsPanel()
        threshold_controls = self.threshold_panel.detach_footer_widget()

        # Top area: large two-image workspace on the left, controls+preview on the right.
        top_layout = QHBoxLayout()
        left_images = QWidget()
        left_images_layout = QHBoxLayout(left_images)
        left_images_layout.setContentsMargins(0, 0, 0, 0)
        left_images_layout.setSpacing(8)
        # Fill full workspace height so both plot columns share vertical space.
        self.threshold_panel.setMinimumSize(320, 200)
        self.front_panel.setMinimumSize(320, 200)
        plot_policy = QSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.threshold_panel.setSizePolicy(plot_policy)
        self.front_panel.setSizePolicy(plot_policy)
        left_images_layout.addWidget(self.threshold_panel, 1)
        left_images_layout.addWidget(self.front_panel, 1)
        top_layout.addWidget(left_images, 3)

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        apical_tip = (
            "Longest run: apical from the longest cytoplasm segment per column.\n"
            "Island: click masked islands on the left kymograph; column-wise vertical center = apex."
        )
        apical_box = QGroupBox("Apical detection")
        apical_inner = QVBoxLayout(apical_box)
        apical_inner.setContentsMargins(6, 8, 6, 6)
        apical_inner.setSpacing(4)
        apical_row = QHBoxLayout()
        apical_row.setSpacing(6)
        self.apical_mode_label = QLabel("Method:")
        self.apical_mode_label.setToolTip(apical_tip)
        self.apical_mode_combo = QComboBox()
        self.apical_mode_combo.setToolTip(apical_tip)
        self.apical_mode_combo.addItem("Island (click to select)")
        self.apical_mode_combo.addItem("Longest cytoplasm run")
        self.apical_mode_combo.setCurrentIndex(0)
        self.apical_mode_combo.currentIndexChanged.connect(self._on_apical_mode_changed)
        apical_row.addWidget(self.apical_mode_label, 0)
        apical_row.addWidget(self.apical_mode_combo, 1)
        apical_inner.addLayout(apical_row)
        apical_inner.addWidget(threshold_controls)
        self.defaults_panel.insert_widget_after_movie(apical_box)

        right_layout.addWidget(self.defaults_panel, 0)
        right_layout.addWidget(self.result_panel, 1)
        right_column.setMinimumWidth(330)
        top_layout.addWidget(right_column, 1)
        root.addLayout(top_layout, 1)

        controls = QHBoxLayout()
        self.open_files_btn = QPushButton("Open Files")
        self.analyze_btn = QPushButton("Analyze")
        self.save_btn = QPushButton("Save")
        self.generate_btn = QPushButton("Generate Outputs")
        self.generate_btn.setToolTip(
            "Run spline fitting, export geometry CSV, and regenerate result figures."
        )
        self.preview_movie_btn = QPushButton("Preview movie…")
        self.preview_movie_btn.setToolTip(
            "Open the delta movie (MP4 with front markers) in a separate window."
        )
        self.status_label = QLabel("Idle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.save_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.preview_movie_btn.setEnabled(False)

        controls.addWidget(self.open_files_btn)
        controls.addWidget(self.analyze_btn)
        controls.addWidget(self.save_btn)
        controls.addWidget(self.generate_btn)
        controls.addWidget(self.preview_movie_btn)
        controls.addWidget(self.status_label, 1)
        root.addLayout(controls)

        root.addWidget(QLabel("Selected files"))
        self.drop_list = DropListWidget(self.defaults_panel.values)
        self.drop_list.setMinimumHeight(140)
        root.addWidget(self.drop_list, 0)

        self.drop_list.file_selected.connect(self._on_file_selected)
        self.drop_list.params_changed.connect(self._on_params_changed)
        self.drop_list.file_removed.connect(self._on_file_removed)
        self.defaults_panel.changed.connect(self._on_defaults_changed)

        self.open_files_btn.clicked.connect(self.open_files_dialog)
        self.analyze_btn.clicked.connect(self.analyze_current)
        self.save_btn.clicked.connect(self.save_current)
        self.generate_btn.clicked.connect(self.generate_figure)
        self.preview_movie_btn.clicked.connect(self._open_movie_preview)

        self.threshold_panel.threshold_changed.connect(self._on_threshold_changed)
        self.threshold_panel.brightness_changed.connect(self._on_brightness_changed)
        self.threshold_panel.island_clicked.connect(self._on_island_clicked)
        self.front_panel.raw_point_added.connect(self._on_raw_point_added)
        self.front_panel.raw_point_moved.connect(self._on_raw_point_moved)
        self.front_panel.undo_requested.connect(self._on_undo)
        self.front_panel.clear_requested.connect(self._on_clear)

    def _current_state(self) -> SampleState | None:
        if not self.current_path:
            return None
        return self.samples.get(self.current_path)

    def _on_file_selected(self, path: str):
        self.current_path = path
        # Refresh main settings directly from the clicked movie metadata.
        meta = read_imagej_params(path)
        if meta:
            self.defaults_panel.apply_metadata(meta)
        params = self.drop_list.params_for(path)
        if path not in self.samples:
            raw = Path(path)
            state = SampleState(raw, work_dir_for(path), AcquisitionParams(**params))
            state.state_changed.connect(self._refresh_panels)
            self.samples[path] = state
        else:
            self.samples[path].acq_params = AcquisitionParams(**params)
        self._sync_apical_mode_combo(self.samples[path].use_island_mode)
        self._refresh_panels()
        self._refresh_output_preview()
        st = self.samples[path]
        if st.kymograph is None:
            self.save_btn.setEnabled(False)
            self.generate_btn.setEnabled(False)
        QTimer.singleShot(0, self._try_auto_analyze_current)

    def _try_auto_analyze_current(self) -> None:
        state = self._current_state()
        if state is None or state.kymograph is not None:
            return
        if self._analyze_worker is not None and self._analyze_worker.isRunning():
            return
        if not has_restorable_session(state.work_dir):
            return
        self.analyze_current()

    def open_files_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select timelapse TIFF files",
            str(Path.home()),
            "TIFF files (*.tif *.tiff);;All files (*)",
        )
        if not paths:
            return

        for path in paths:
            self.drop_list.add_movie(path)

        # Auto-select first picked file if nothing is selected yet.
        if self.drop_list.currentItem() is None and self.drop_list.count() > 0:
            self.drop_list.setCurrentRow(0)

    def _on_params_changed(self, path: str, params: dict):
        if path in self.samples:
            self.samples[path].acq_params = AcquisitionParams(**params)

    def _on_defaults_changed(self, _params: dict):
        # Keep non-overridden rows in sync with top-level defaults so Analyze
        # immediately reflects changed defaults without re-adding files.
        for path in self.drop_list.all_paths():
            if self.drop_list.has_override(path):
                continue
            if path in self.samples:
                self.samples[path].acq_params = AcquisitionParams(**self.drop_list.params_for(path))
        if self.current_path and self.current_path in self.samples and not self.drop_list.has_override(self.current_path):
            self._refresh_panels()

    def _on_file_removed(self, path: str):
        self.samples.pop(path, None)
        if self.current_path == path:
            self.current_path = None
            self.save_btn.setEnabled(False)
            self.generate_btn.setEnabled(False)
            self.preview_movie_btn.setEnabled(False)
            self.status_label.setText("Idle")

    def _refresh_panels(self):
        state = self._current_state()
        if state is None or state.kymograph is None:
            return

        self.threshold_panel.set_data(
            state.kymograph,
            state.threshold,
            state.mask,
            state.acq_params.movie_time_interval_sec,
            state.use_island_mode,
        )
        self.threshold_panel.set_selected_island_mask(state.selected_island_mask())

        crop_top = max(0, state.ref_row - int(round(3.0 / max(state.acq_params.px2micron, 1e-9))))
        points_display = state.display_points(
            state.ref_row,
            state.acq_params.px2micron,
            state.acq_params.movie_time_interval_sec,
        )
        self.front_panel.set_data(
            state.straight_kymo,
            points_display,
            state.shifts,
            crop_top,
            state.ref_row,
            state.acq_params.px2micron,
            state.acq_params.movie_time_interval_sec,
        )
        self.front_panel.set_brightness(self.threshold_panel.brightness_factor())

        self.save_btn.setEnabled(True)

    def _on_threshold_changed(self, threshold: float):
        state = self._current_state()
        if state is None:
            return
        state.set_threshold(threshold)
        if state.use_island_mode:
            self.status_label.setText("Island mode: threshold changed, reselect islands")
        else:
            self.status_label.setText("Unsaved changes")
        self.generate_btn.setEnabled(False)

    def _on_brightness_changed(self, brightness: float):
        self.front_panel.set_brightness(brightness)

    def _sync_apical_mode_combo(self, use_island: bool) -> None:
        self.apical_mode_combo.blockSignals(True)
        self.apical_mode_combo.setCurrentIndex(0 if use_island else 1)
        self.apical_mode_combo.blockSignals(False)

    def _on_apical_mode_changed(self, _index: int) -> None:
        state = self._current_state()
        if state is None:
            return
        enabled = self.apical_mode_combo.currentIndex() == 0
        state.set_use_island_mode(enabled)
        self.generate_btn.setEnabled(False)
        if enabled:
            self.status_label.setText("Island mode: click threshold islands on left kymograph")
        else:
            self.status_label.setText("Unsaved changes")

    def _on_island_clicked(self, x: float, y: float):
        state = self._current_state()
        if state is None:
            return
        selected = state.select_island_at(x, y)
        if selected:
            self.status_label.setText("Unsaved changes")
            self.generate_btn.setEnabled(False)
        else:
            self.status_label.setText("Island mode: selection reset (click islands to select)")
            self.generate_btn.setEnabled(False)

    def _on_raw_point_added(self, x: float, y: float):
        state = self._current_state()
        if state is None:
            return
        state.add_front_point_raw(x, y)
        self.status_label.setText("Unsaved changes")
        self.generate_btn.setEnabled(False)

    def _on_raw_point_moved(self, index: int, x: float, y: float):
        state = self._current_state()
        if state is None:
            return
        state.update_front_point_raw(index, x, y)
        self.status_label.setText("Unsaved changes")
        self.generate_btn.setEnabled(False)

    def _on_undo(self):
        state = self._current_state()
        if state is not None:
            state.undo_front_point()

    def _on_clear(self):
        state = self._current_state()
        if state is not None:
            state.clear_front_points()

    def analyze_current(self):
        state = self._current_state()
        if state is None:
            QMessageBox.warning(self, "No input", "Drop and select a TIFF first.")
            return
        if self.current_path:
            state.acq_params = AcquisitionParams(**self.drop_list.params_for(self.current_path))
        # New Analyze run starts a fresh interactive session for this sample.
        state.front_points_raw = []
        state.dirty = False

        ensure_work_tree(state.work_dir)
        load_or_create_config(state.work_dir, asdict(state.acq_params))

        self._analyze_worker = AnalyzeWorker(
            state.raw_movie_path,
            state.work_dir,
            self.current_path or "",
        )
        self._analyze_worker.progress.connect(self.status_label.setText)
        self._analyze_worker.done.connect(self._on_analyze_done)
        self._analyze_worker.failed.connect(self._on_worker_failed)
        self._analyze_worker.start()
        self.status_label.setText("Analyze started...")

    def _on_analyze_done(self, kymo):
        worker = self._analyze_worker
        path = worker.sample_path if worker else self.current_path
        state = self.samples.get(path) if path else None
        if state is None:
            return
        state.set_kymograph(kymo)
        restored, note = restore_interactive_session(state)
        if self.current_path == path:
            if restored:
                self._sync_apical_mode_combo(state.use_island_mode)
                self.status_label.setText(note)
                self.generate_btn.setEnabled(True)
            elif note:
                self._sync_apical_mode_combo(state.use_island_mode)
                self.status_label.setText(note)
                self.generate_btn.setEnabled(False)
            else:
                self.status_label.setText("Analyze complete")
                self.generate_btn.setEnabled(False)
            self.save_btn.setEnabled(True)
            self._refresh_panels()

    def _on_worker_failed(self, message: str):
        QMessageBox.critical(self, "Worker error", message)
        self.status_label.setText("Error")

    def save_current(self):
        state = self._current_state()
        if state is None or state.kymograph is None or state.mask is None:
            QMessageBox.warning(self, "Nothing to save", "Analyze first.")
            return

        if len(state.front_points_raw) < 2:
            QMessageBox.warning(self, "Need points", "Add at least two front points before saving.")
            return

        track = state.work_dir / "track"
        ensure_work_tree(state.work_dir)

        mask_u8 = state.mask.astype(np.uint8) * 255
        atomic_write_tiff(track / "YolkMask.tif", mask_u8)

        points_raw = np.asarray(state.front_points_raw, dtype=float)
        time_interval_min = state.acq_params.movie_time_interval_sec / 60.0
        time_min, depth_px = raw_clicks_to_time_depth(
            points_raw,
            state.shifts,
            state.kymograph.shape[1] if state.kymograph is not None else 0,
            time_interval_min,
        )
        alignment = build_apical_alignment_v2(
            mode="island" if state.use_island_mode else "longest_run",
            island_labels=sorted(state.selected_island_labels)
            if state.use_island_mode
            else [],
            threshold=float(state.threshold),
            kymograph_shape=state.kymograph.shape,
            movie_time_interval_sec=float(state.acq_params.movie_time_interval_sec),
            time_min=time_min,
            depth_px=depth_px,
        )
        with (track / "apical_alignment.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(alignment, f, default_flow_style=False, sort_keys=False)

        write_annotation_tsv(str(track / "VerticalKymoCelluSelection.tsv"), time_min, depth_px)

        save_compat_roi_placeholder(str(track))

        _, _, _, apical_px, _, _, _ = compute_cytoplasm_size_over_time(
            str(state.work_dir),
            {"px2micron": state.acq_params.px2micron},
            min_run_length_px=5,
        )
        update_apical_height_in_config(str(state.work_dir), apical_px, state.acq_params.px2micron)

        state.dirty = False
        self.generate_btn.setEnabled(True)
        self.status_label.setText("Saved")

    def generate_figure(self):
        state = self._current_state()
        if state is None:
            return

        smoothing = float(state.acq_params.smoothing)
        degree = int(state.acq_params.degree)
        time_interval_min = state.acq_params.movie_time_interval_sec / 60.0

        self._figure_worker = GenerateFigureWorker(state.work_dir, smoothing, degree, time_interval_min)
        self._figure_worker.progress.connect(self.status_label.setText)
        self._figure_worker.done.connect(self._on_figure_done)
        self._figure_worker.failed.connect(self._on_worker_failed)
        self._figure_worker.start()

    def _on_figure_done(self):
        self.status_label.setText("Outputs ready")
        self._refresh_output_preview()

    def _update_movie_preview_enabled(self):
        state = self._current_state()
        if state is None:
            self.preview_movie_btn.setEnabled(False)
            return
        mp4 = state.work_dir / "results" / "Cellularization_trimmed_delta.mp4"
        self.preview_movie_btn.setEnabled(mp4.is_file())

    def _open_movie_preview(self):
        state = self._current_state()
        if state is None:
            return
        mp4 = state.work_dir / "results" / "Cellularization_trimmed_delta.mp4"
        if not mp4.is_file():
            QMessageBox.information(
                self,
                "No movie",
                "Run Generate Outputs first to create the delta movie (MP4).",
            )
            return
        if self._movie_preview is None:
            self._movie_preview = MoviePreviewDialog(self)
        self._movie_preview.set_source(mp4)
        self._movie_preview.show()
        self._movie_preview.raise_()
        self._movie_preview.activateWindow()

    def _refresh_output_preview(self):
        state = self._current_state()
        if state is None:
            return
        png_path = state.work_dir / "results" / "Cellularization.png"
        self.result_panel.show_png(png_path)
        self._update_movie_preview_enabled()


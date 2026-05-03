from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tifffile
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellularization_dynamics.core.generate_outputs import (
    EXPORT_MP4_CRF,
    EXPORT_MP4_FPS,
    draw_timestamp_bottom_left_rgb,
    format_experiment_hms,
    furrow_relative_stamp_seconds,
    load_apical_line_series_for_movie,
    load_config,
    load_front_furrow_stamp_time_bounds_minutes,
    load_spline,
    mark_delta_on_trimmed_movie,
)
from cellularization_dynamics.core.work_state import get_movie_path


def _format_ms(ms: int) -> str:
    if ms < 0:
        ms = 0
    s = ms // 1000
    m, s = divmod(s, 60)
    return f"{m:d}:{s:02d}"


def _draw_dashed_horizontal_u8(
    rgb: np.ndarray,
    y: float,
    color: tuple[int, int, int],
    *,
    dash_px: int = 14,
    gap_px: int = 8,
    thickness_px: int = 2,
) -> None:
    """Draw a full-width dashed horizontal line on uint8 RGB [H,W,3] in place."""
    h, w, _c = rgb.shape
    y0 = int(np.clip(int(round(y)), 0, h - 1))
    r, g, b = color
    x = 0
    draw_segment = True
    while x < w:
        seg = dash_px if draw_segment else gap_px
        x_end = min(w, x + seg)
        if draw_segment:
            for xi in range(x, x_end):
                for dy in range(-(thickness_px // 2), thickness_px - (thickness_px // 2)):
                    yy = y0 + dy
                    if 0 <= yy < h:
                        rgb[yy, xi, 0] = r
                        rgb[yy, xi, 1] = g
                        rgb[yy, xi, 2] = b
        x = x_end
        draw_segment = not draw_segment


class _SaveMovieWorker(QThread):
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        work_dir: Path,
        *,
        kymograph_brightness: float,
        show_apical_line: bool,
        apical_line_rgb: tuple[int, int, int],
        mp4_fps: float,
        mp4_crf: int,
        show_timestamp: bool,
        timestamp_rgb: tuple[int, int, int],
        timestamp_size_px: int,
    ):
        super().__init__()
        self._work_dir = work_dir
        self._kymograph_brightness = kymograph_brightness
        self._show_apical_line = show_apical_line
        self._apical_line_rgb = apical_line_rgb
        self._mp4_fps = float(mp4_fps)
        self._mp4_crf = int(mp4_crf)
        self._show_timestamp = bool(show_timestamp)
        self._timestamp_rgb = timestamp_rgb
        self._timestamp_size_px = int(timestamp_size_px)

    def run(self) -> None:
        try:
            wd = str(self._work_dir)
            cfg = load_config(wd)
            t_min, f_px = load_spline(wd)
            movie_path = get_movie_path(wd)
            movie = tifffile.imread(movie_path)
            if movie.ndim == 2:
                movie = movie[np.newaxis, :, :]
            mark_delta_on_trimmed_movie(
                wd,
                cfg,
                t_min,
                f_px,
                movie=movie,
                kymograph_brightness=self._kymograph_brightness,
                show_apical_line=self._show_apical_line,
                apical_line_rgb=self._apical_line_rgb,
                mp4_fps=self._mp4_fps,
                mp4_crf=self._mp4_crf,
                show_timestamp=self._show_timestamp,
                timestamp_rgb=self._timestamp_rgb,
                timestamp_size_px=self._timestamp_size_px,
            )
            self.finished_ok.emit()
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


class MoviePreviewDialog(QDialog):
    """
    Modeless preview for Cellularization_front_markers.mp4 (play, pause, seek).

    Uses OpenCV + QLabel instead of QMediaPlayer: conda-forge ``pyqt6`` does not
    include ``PyQt6.QtMultimedia`` bindings.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Front markers preview")
        self.setMinimumSize(640, 520)

        self._slider_drag = False
        self._cap: cv2.VideoCapture | None = None
        self._frame_count = 1
        self._fps = 10.0
        self._current_frame = 0
        self._playing = False
        self._last_rgb_raw: np.ndarray | None = None

        self._work_dir: Path | None = None
        self._geom_time_ap: np.ndarray | None = None
        self._geom_apical_v: np.ndarray | None = None
        self._movie_time_interval_sec = 10.0
        self._cfg_kymograph_brightness = 1.0
        self._apical_rgb: tuple[int, int, int] = (255, 255, 0)
        self._stamp_rgb: tuple[int, int, int] = (255, 255, 255)
        self._furrow_stamp_t0_min: float | None = None
        self._furrow_stamp_t1_min: float | None = None
        self._save_worker: _SaveMovieWorker | None = None

        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(320, 240)
        self._video_label.setStyleSheet("background-color: #1a1a1a;")

        self._play_btn = QPushButton("Play")
        self._play_btn.clicked.connect(self._toggle_play)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(10)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._slider.sliderMoved.connect(self._on_slider_moved)

        self._time_label = QLabel("0:00 / 0:00")

        self._brightness_label = QLabel("Brightness")
        self._brightness_label.setToolTip(
            "Scales displayed intensity. Use Save MP4 to bake this (with config baseline) into a new file."
        )
        self._brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self._brightness_slider.setMinimum(20)
        self._brightness_slider.setMaximum(300)
        self._brightness_slider.setValue(100)
        self._brightness_slider.setToolTip(
            "100 = match encoded MP4 decode; higher brightens. Save applies ×(slider/100) on top of saved kymograph brightness."
        )
        self._brightness_slider.valueChanged.connect(self._on_preview_brightness_changed)

        brightness_row = QHBoxLayout()
        brightness_row.addWidget(self._brightness_label, 0)
        brightness_row.addWidget(self._brightness_slider, 1)

        self._save_fps_label = QLabel("Save FPS")
        self._save_fps_label.setToolTip(
            "Playback speed in the preview and frame rate written into the MP4 on Save. "
            "Opening a file sets this from the file; change it here to match what you want before saving."
        )
        self._save_fps_spin = QDoubleSpinBox()
        self._save_fps_spin.setRange(0.25, 120.0)
        self._save_fps_spin.setDecimals(3)
        self._save_fps_spin.setSingleStep(1.0)
        self._save_fps_spin.setValue(float(EXPORT_MP4_FPS))
        self._save_fps_spin.valueChanged.connect(self._on_save_fps_spin_changed)
        self._fps_exp_rate_label = QLabel("")
        self._fps_exp_rate_label.setToolTip(
            "Experiment minutes advanced per second of playback: Save FPS × movie_time_interval_sec / 60. "
            "Match this value across movies to compare the same effective timeline speed."
        )
        save_fps_row = QHBoxLayout()
        save_fps_row.addWidget(self._save_fps_label, 0)
        save_fps_row.addWidget(self._save_fps_spin, 0)
        save_fps_row.addStretch(1)
        save_fps_row.addWidget(self._fps_exp_rate_label, 0)

        self._save_crf_label = QLabel("CRF")
        self._save_crf_label.setToolTip(
            "libx264 quality for Save MP4: lower = better quality, larger files "
            "(typical range ~18–28; 0 ≈ lossless, 51 worst)."
        )
        self._save_crf_spin = QSpinBox()
        self._save_crf_spin.setRange(0, 51)
        self._save_crf_spin.setValue(int(EXPORT_MP4_CRF))
        save_crf_row = QHBoxLayout()
        save_crf_row.addWidget(self._save_crf_label, 0)
        save_crf_row.addWidget(self._save_crf_spin, 1)

        self._stamp_cb = QCheckBox("Timestamp")
        self._stamp_cb.setToolTip(
            "Furrow-relative time (HH:MM:SS) lower-left when output.csv is available: "
            "00:00:00 until the front is first detected, then elapsed time, then frozen "
            "after the last detected front. Otherwise, elapsed time from frame 0 × "
            "movie_time_interval_sec."
        )
        self._stamp_cb.setEnabled(False)
        self._stamp_cb.stateChanged.connect(lambda _s: self._update_display_from_raw())
        self._stamp_size_label = QLabel("Size")
        self._stamp_size_spin = QSpinBox()
        self._stamp_size_spin.setRange(2, 96)
        self._stamp_size_spin.setValue(8)
        self._stamp_size_spin.setEnabled(False)
        self._stamp_size_spin.valueChanged.connect(lambda _v: self._update_display_from_raw())
        self._stamp_color_btn = QPushButton()
        self._stamp_color_btn.setFixedSize(28, 22)
        self._stamp_color_btn.setToolTip("Timestamp text color (preview and Save MP4)")
        self._stamp_color_btn.setEnabled(False)
        self._stamp_color_btn.clicked.connect(self._pick_stamp_color)
        stamp_row = QHBoxLayout()
        stamp_row.addWidget(self._stamp_cb, 0)
        stamp_row.addWidget(self._stamp_size_label, 0)
        stamp_row.addWidget(self._stamp_size_spin, 0)
        stamp_row.addWidget(self._stamp_color_btn, 0)
        stamp_row.addStretch(1)

        self._apical_cb = QCheckBox("Show apical")
        self._apical_cb.setToolTip(
            "Draw apical border (from output.csv) as a dashed line — same sense as the yellow apical cue on the kymograph."
        )
        self._apical_cb.setEnabled(False)
        self._apical_cb.stateChanged.connect(lambda _s: self._update_display_from_raw())

        self._apical_color_btn = QPushButton()
        self._apical_color_btn.setFixedSize(28, 22)
        self._apical_color_btn.setToolTip("Apical line color (preview and Save MP4)")
        self._apical_color_btn.clicked.connect(self._pick_apical_color)
        self._apical_color_btn.setEnabled(False)

        self._save_btn = QPushButton("Save MP4…")
        self._save_btn.setToolTip(
            "Overwrite results/Cellularization_front_markers.mp4 using current brightness, "
            "Save FPS, CRF, and apical-line options (re-encodes from the acquisition TIFF)."
        )
        self._save_btn.clicked.connect(self._on_save_mp4)
        self._save_btn.setEnabled(False)

        apical_row = QHBoxLayout()
        apical_row.addWidget(self._apical_cb, 0)
        apical_row.addWidget(self._apical_color_btn, 0)
        apical_row.addStretch(1)
        apical_row.addWidget(self._save_btn, 0)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #c00;")
        self._error_label.setWordWrap(True)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)

        controls = QHBoxLayout()
        controls.addWidget(self._play_btn)
        controls.addWidget(self._slider, 1)
        controls.addWidget(self._time_label)

        root = QVBoxLayout(self)
        root.addWidget(self._video_label, 1)
        root.addLayout(controls)
        root.addLayout(brightness_row)
        root.addLayout(save_fps_row)
        root.addLayout(save_crf_row)
        root.addLayout(stamp_row)
        root.addLayout(apical_row)
        root.addWidget(self._error_label)

        self._sync_apical_color_button()
        self._sync_stamp_color_button()
        self._update_fps_experiment_rate_hint()

    def _on_save_fps_spin_changed(self, value: float) -> None:
        """Keep preview play speed and mm:ss labels in sync with Save FPS."""
        v = float(value)
        if v <= 0:
            return
        self._fps = v
        self._timer.setInterval(max(1, int(round(1000.0 / self._fps))))
        if self._playing:
            self._timer.stop()
            self._timer.start()
        self._update_time_labels()
        self._update_fps_experiment_rate_hint()

    def _update_fps_experiment_rate_hint(self) -> None:
        """Experiment minutes per playback second: FPS × movie_time_interval_sec / 60."""
        fps = float(self._save_fps_spin.value())
        dt = float(self._movie_time_interval_sec)
        if fps > 0 and dt > 0:
            mps = fps * dt / 60.0
            self._fps_exp_rate_label.setText(f"({mps:.4f} min/s)")
        else:
            self._fps_exp_rate_label.setText("")

    def _sync_apical_color_button(self) -> None:
        r, g, b = self._apical_rgb
        self._apical_color_btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #444;"
        )

    def _pick_apical_color(self) -> None:
        c = QColorDialog.getColor(
            QColor(*self._apical_rgb),
            self,
            "Apical line color",
        )
        if c.isValid():
            self._apical_rgb = (c.red(), c.green(), c.blue())
            self._sync_apical_color_button()
            self._update_display_from_raw()

    def _sync_stamp_color_button(self) -> None:
        r, g, b = self._stamp_rgb
        self._stamp_color_btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #444;"
        )

    def _pick_stamp_color(self) -> None:
        c = QColorDialog.getColor(
            QColor(*self._stamp_rgb),
            self,
            "Timestamp color",
        )
        if c.isValid():
            self._stamp_rgb = (c.red(), c.green(), c.blue())
            self._sync_stamp_color_button()
            self._update_display_from_raw()

    def set_source(
        self,
        mp4_path: Path,
        work_dir: Path | None = None,
        *,
        preserve_timestamp: bool = False,
    ) -> None:
        want_stamp = preserve_timestamp and self._stamp_cb.isChecked()
        self._error_label.setText("")
        self._release_cap()
        self._last_rgb_raw = None
        self._brightness_slider.blockSignals(True)
        self._brightness_slider.setValue(100)
        self._brightness_slider.blockSignals(False)

        self._work_dir = work_dir.resolve() if work_dir is not None else None
        self._geom_time_ap = None
        self._geom_apical_v = None
        self._furrow_stamp_t0_min = None
        self._furrow_stamp_t1_min = None
        self._cfg_kymograph_brightness = 1.0
        can_save = self._work_dir is not None
        self._save_btn.setEnabled(can_save)
        self._save_fps_spin.setEnabled(can_save)
        self._save_crf_spin.setEnabled(can_save)
        self._stamp_cb.setEnabled(False)
        if not preserve_timestamp:
            self._stamp_cb.setChecked(False)
        self._stamp_size_spin.setEnabled(False)
        self._stamp_color_btn.setEnabled(False)

        if self._work_dir is not None:
            try:
                cfg = load_config(str(self._work_dir))
                self._cfg_kymograph_brightness = float(cfg.get("kymograph_brightness", 1.0))
                self._movie_time_interval_sec = float(cfg.get("movie_time_interval_sec", 10.0))
            except Exception:
                self._cfg_kymograph_brightness = 1.0
                self._movie_time_interval_sec = 10.0
            furrow_bounds = load_front_furrow_stamp_time_bounds_minutes(self._work_dir)
            if furrow_bounds is not None:
                self._furrow_stamp_t0_min, self._furrow_stamp_t1_min = furrow_bounds
            ap_series = load_apical_line_series_for_movie(self._work_dir)
            if ap_series is not None:
                self._geom_time_ap, self._geom_apical_v, self._movie_time_interval_sec = ap_series
                self._apical_cb.setEnabled(True)
                self._apical_color_btn.setEnabled(True)
            else:
                self._apical_cb.setEnabled(False)
                self._apical_cb.setChecked(False)
                self._apical_color_btn.setEnabled(False)
        else:
            self._save_btn.setEnabled(False)
            self._save_fps_spin.setEnabled(False)
            self._save_crf_spin.setEnabled(False)
            self._stamp_cb.setEnabled(False)
            if not preserve_timestamp:
                self._stamp_cb.setChecked(False)
            self._stamp_size_spin.setEnabled(False)
            self._stamp_color_btn.setEnabled(False)
            self._apical_cb.setEnabled(False)
            self._apical_cb.setChecked(False)
            self._apical_color_btn.setEnabled(False)

        path = str(mp4_path.resolve())
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._error_label.setText(f"Could not open video: {path}")
            self._save_fps_spin.blockSignals(True)
            self._save_fps_spin.setValue(float(EXPORT_MP4_FPS))
            self._save_fps_spin.blockSignals(False)
            self._save_crf_spin.blockSignals(True)
            self._save_crf_spin.setValue(int(EXPORT_MP4_CRF))
            self._save_crf_spin.blockSignals(False)
            self._update_fps_experiment_rate_hint()
            return

        self._cap = cap
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_count = max(1, n)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        self._fps = fps if fps > 1e-3 else 10.0
        self._save_fps_spin.blockSignals(True)
        self._save_fps_spin.setValue(self._fps)
        self._save_fps_spin.blockSignals(False)
        self._timer.setInterval(max(1, int(round(1000.0 / self._fps))))
        self._stamp_cb.setEnabled(True)
        self._stamp_size_spin.setEnabled(True)
        self._stamp_color_btn.setEnabled(True)
        self._update_fps_experiment_rate_hint()

        self._slider.setMaximum(max(0, self._frame_count - 1))
        self._current_frame = 0
        self._playing = False
        self._play_btn.setText("Play")
        self._set_frame(0)
        if want_stamp:
            self._stamp_cb.setChecked(True)

    def _release_cap(self) -> None:
        self._timer.stop()
        self._playing = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _toggle_play(self) -> None:
        if self._cap is None:
            return
        if self._playing:
            self._playing = False
            self._timer.stop()
            self._play_btn.setText("Play")
        else:
            if self._current_frame >= self._frame_count - 1:
                self._set_frame(0)
            self._playing = True
            self._play_btn.setText("Pause")
            self._timer.start()

    def _on_timer_tick(self) -> None:
        if not self._playing or self._cap is None:
            return
        next_f = self._current_frame + 1
        if next_f >= self._frame_count:
            self._playing = False
            self._timer.stop()
            self._play_btn.setText("Play")
            return
        self._set_frame(next_f)

    def _set_frame(self, idx: int) -> None:
        if self._cap is None:
            return
        idx = int(np.clip(idx, 0, max(0, self._frame_count - 1)))
        self._current_frame = idx
        if not self._slider_drag:
            self._slider.blockSignals(True)
            self._slider.setValue(idx)
            self._slider.blockSignals(False)
        self._render_current_frame()
        self._update_time_labels()

    def _render_current_frame(self) -> None:
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._last_rgb_raw = np.ascontiguousarray(rgb)
        self._update_display_from_raw()

    def _on_preview_brightness_changed(self, _value: int) -> None:
        self._update_display_from_raw()

    def _update_display_from_raw(self) -> None:
        if self._last_rgb_raw is None:
            return
        base = self._last_rgb_raw
        if (
            self._apical_cb.isChecked()
            and self._geom_time_ap is not None
            and self._geom_apical_v is not None
        ):
            base = self._last_rgb_raw.copy()
            time_min = self._current_frame * self._movie_time_interval_sec / 60.0
            ap_y = float(np.interp(time_min, self._geom_time_ap, self._geom_apical_v))
            if np.isfinite(ap_y):
                _draw_dashed_horizontal_u8(base, ap_y, self._apical_rgb)
        factor = float(self._brightness_slider.value()) / 100.0
        adj = np.clip(base.astype(np.float32) * factor, 0.0, 255.0).astype(np.uint8)
        if self._stamp_cb.isChecked():
            disp = np.ascontiguousarray(adj)
            mdt = float(self._movie_time_interval_sec)
            if self._furrow_stamp_t0_min is not None and self._furrow_stamp_t1_min is not None:
                t_sec = furrow_relative_stamp_seconds(
                    self._current_frame,
                    mdt,
                    self._furrow_stamp_t0_min,
                    self._furrow_stamp_t1_min,
                )
            else:
                t_sec = int(round(self._current_frame * mdt))
            stamp = format_experiment_hms(t_sec)
            draw_timestamp_bottom_left_rgb(
                disp,
                stamp,
                color_rgb=self._stamp_rgb,
                size_px=int(self._stamp_size_spin.value()),
            )
        else:
            disp = np.ascontiguousarray(adj)
        self._apply_pixmap_from_rgb(disp)

    def _effective_encode_brightness(self) -> float:
        preview_factor = float(self._brightness_slider.value()) / 100.0
        return max(0.2, min(3.0, self._cfg_kymograph_brightness * preview_factor))

    def _on_save_mp4(self) -> None:
        if self._work_dir is None:
            return
        if self._save_worker is not None and self._save_worker.isRunning():
            return
        save_fps = float(self._save_fps_spin.value())
        if save_fps <= 0:
            QMessageBox.warning(self, "Save FPS", "Save FPS must be positive.")
            return
        self._save_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._save_worker = _SaveMovieWorker(
            self._work_dir,
            kymograph_brightness=self._effective_encode_brightness(),
            show_apical_line=self._apical_cb.isChecked(),
            apical_line_rgb=self._apical_rgb,
            mp4_fps=save_fps,
            mp4_crf=int(self._save_crf_spin.value()),
            show_timestamp=self._stamp_cb.isChecked(),
            timestamp_rgb=self._stamp_rgb,
            timestamp_size_px=int(self._stamp_size_spin.value()),
        )
        self._save_worker.finished_ok.connect(self._on_save_finished)
        self._save_worker.failed.connect(self._on_save_failed)
        self._save_worker.start()

    def _on_save_finished(self) -> None:
        QApplication.restoreOverrideCursor()
        if self._save_worker is not None:
            try:
                self._save_worker.finished_ok.disconnect()
                self._save_worker.failed.disconnect()
            except TypeError:
                pass
            self._save_worker.deleteLater()
            self._save_worker = None
        self._save_btn.setEnabled(self._work_dir is not None)
        QMessageBox.information(
            self,
            "Saved",
            "Updated results/Cellularization_front_markers.mp4 with the current settings.",
        )
        if self._work_dir is not None:
            mp4 = self._work_dir / "results" / "Cellularization_front_markers.mp4"
            self.set_source(mp4, self._work_dir, preserve_timestamp=True)

    def _on_save_failed(self, msg: str) -> None:
        QApplication.restoreOverrideCursor()
        if self._save_worker is not None:
            try:
                self._save_worker.finished_ok.disconnect()
                self._save_worker.failed.disconnect()
            except TypeError:
                pass
            self._save_worker.deleteLater()
            self._save_worker = None
        self._save_btn.setEnabled(self._work_dir is not None)
        QMessageBox.warning(self, "Save failed", msg)

    def _apply_pixmap_from_rgb(self, rgb: np.ndarray) -> None:
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())
        scaled = pix.scaled(
            self._video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_label.setPixmap(scaled)

    def _update_time_labels(self) -> None:
        if self._fps <= 0:
            return
        pos_ms = int(self._current_frame / self._fps * 1000)
        dur_ms = int(max(0, self._frame_count - 1) / self._fps * 1000)
        self._time_label.setText(f"{_format_ms(pos_ms)} / {_format_ms(dur_ms)}")

    def _on_slider_pressed(self) -> None:
        self._slider_drag = True

    def _on_slider_released(self) -> None:
        self._slider_drag = False
        self._set_frame(self._slider.value())

    def _on_slider_moved(self, value: int) -> None:
        if not self._slider_drag:
            return
        self._set_frame(value)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_rgb_raw is not None:
            self._update_display_from_raw()

    def closeEvent(self, event: QCloseEvent):
        if self._save_worker is not None and self._save_worker.isRunning():
            self._save_worker.wait(3000)
        self._release_cap()
        self._last_rgb_raw = None
        super().closeEvent(event)

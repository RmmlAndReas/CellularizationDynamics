from __future__ import annotations

from pathlib import Path
from ..services.metadata_reader import read_imagej_params

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QDialog,
    QFormLayout,
    QDialogButtonBox,
    QDoubleSpinBox,
    QSpinBox,
)


class ParamDialog(QDialog):
    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Per-movie acquisition params")
        layout = QFormLayout(self)

        self.px = QDoubleSpinBox()
        self.px.setRange(0.0001, 1000)
        self.px.setDecimals(6)
        self.px.setValue(float(params["px2micron"]))
        self.px.setToolTip(
            "Microns per pixel conversion. Used to convert pixel depths to microns."
        )

        self.dt = QDoubleSpinBox()
        self.dt.setRange(0.001, 100000)
        self.dt.setDecimals(3)
        self.dt.setValue(float(params["movie_time_interval_sec"]))
        self.dt.setToolTip(
            "Original movie frame interval in seconds (before keep_every downsampling)."
        )

        self.keep = QSpinBox()
        self.keep.setRange(1, 1000)
        self.keep.setValue(int(params["keep_every"]))
        self.keep.setToolTip(
            "Keep every Nth frame when building the trimmed movie and kymograph."
        )

        self.smoothing = QDoubleSpinBox()
        self.smoothing.setRange(0.0, 1e6)
        self.smoothing.setDecimals(4)
        self.smoothing.setValue(float(params.get("smoothing", 0.0)))
        self.smoothing.setToolTip(
            "Spline smoothing factor used during Generate Figure. "
            "0 = pass exactly through points; higher values give smoother fits."
        )

        self.degree = QSpinBox()
        self.degree.setRange(1, 5)
        self.degree.setValue(int(params.get("degree", 1)))
        self.degree.setToolTip(
            "Spline polynomial degree (1..5). Higher degree allows more curvature."
        )

        layout.addRow("px2micron", self.px)
        layout.addRow("movie_time_interval_sec", self.dt)
        layout.addRow("keep_every", self.keep)
        layout.addRow("smoothing", self.smoothing)
        layout.addRow("degree", self.degree)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict:
        return {
            "px2micron": float(self.px.value()),
            "movie_time_interval_sec": float(self.dt.value()),
            "keep_every": int(self.keep.value()),
            "smoothing": float(self.smoothing.value()),
            "degree": int(self.degree.value()),
        }


class DropListWidget(QListWidget):
    file_selected = pyqtSignal(str)
    params_changed = pyqtSignal(str, dict)
    file_removed = pyqtSignal(str)

    def __init__(self, default_params_provider):
        super().__init__()
        self.default_params_provider = default_params_provider
        self.setAcceptDrops(True)
        self._entries: dict[str, dict] = {}
        self.currentItemChanged.connect(self._emit_current)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.suffix.lower() in {".tif", ".tiff"}:
                self.add_movie(str(p))
        event.acceptProposedAction()

    def add_movie(self, path: str):
        if path in self._entries:
            return

        self._entries[path] = {"override_params": None}

        item = QListWidgetItem(self)
        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(4, 2, 4, 2)
        row_layout.addWidget(QLabel(Path(path).name))
        del_btn = QPushButton("Trash")
        del_btn.setToolTip("Remove file from list")
        del_btn.clicked.connect(lambda: self.remove_movie(path))
        row_layout.addWidget(del_btn)

        item.setSizeHint(row_widget.sizeHint())
        self.addItem(item)
        self.setItemWidget(item, row_widget)
        item.setData(256, path)

    def _edit_params(self, path: str):
        # Refresh from TIFF metadata on each click, then let user adjust.
        params = self.params_for(path)
        meta = read_imagej_params(path)
        if meta:
            params.update(meta)

        dialog = ParamDialog(params, self)
        if dialog.exec():
            self._entries[path]["override_params"] = dialog.values()
            self.params_changed.emit(path, dict(self._entries[path]["override_params"]))

    def params_for(self, path: str) -> dict:
        override = self._entries[path].get("override_params")
        if override is not None:
            return dict(override)
        # Use live defaults for non-overridden items, so changing top-level
        # defaults updates Analyze behavior for already-dropped files.
        return dict(self.default_params_provider())

    def has_override(self, path: str) -> bool:
        return self._entries.get(path, {}).get("override_params") is not None

    def all_paths(self) -> list[str]:
        return list(self._entries.keys())

    def remove_movie(self, path: str):
        row = None
        for i in range(self.count()):
            item = self.item(i)
            if item.data(256) == path:
                row = i
                break
        if row is None:
            return

        self.takeItem(row)
        self._entries.pop(path, None)
        self.file_removed.emit(path)

    def _emit_current(self, current, _previous):
        if current is not None:
            path = current.data(256)
            self.file_selected.emit(path)

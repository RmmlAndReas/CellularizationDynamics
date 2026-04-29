from __future__ import annotations

from pathlib import Path
import yaml

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QFormLayout, QDoubleSpinBox, QSpinBox


class DefaultsPanel(QWidget):
    changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.storage_path = Path(__file__).resolve().parents[1] / ".last_defaults.yaml"

        layout = QFormLayout(self)
        self.px2micron = QDoubleSpinBox()
        self.px2micron.setRange(0.0001, 1000.0)
        self.px2micron.setDecimals(6)
        self.px2micron.setValue(0.2738095)
        self.px2micron.setToolTip(
            "Microns per pixel conversion. Used to convert pixel depths to microns."
        )

        self.movie_dt = QDoubleSpinBox()
        self.movie_dt.setRange(0.001, 100000.0)
        self.movie_dt.setDecimals(3)
        self.movie_dt.setValue(10.0)
        self.movie_dt.setToolTip(
            "Original movie frame interval in seconds (before keep_every downsampling)."
        )

        self.keep_every = QSpinBox()
        self.keep_every.setRange(1, 1000)
        self.keep_every.setValue(3)
        self.keep_every.setToolTip(
            "Keep every Nth frame when building the trimmed movie and kymograph."
        )

        self.smoothing = QDoubleSpinBox()
        self.smoothing.setRange(0.0, 1e6)
        self.smoothing.setDecimals(4)
        self.smoothing.setValue(0.0)
        self.smoothing.setToolTip(
            "Spline smoothing factor used during Generate Figure. "
            "0 = pass exactly through points; higher values give smoother fits."
        )

        self.degree = QSpinBox()
        self.degree.setRange(1, 5)
        self.degree.setValue(1)
        self.degree.setToolTip(
            "Spline polynomial degree (1..5). Higher degree allows more curvature."
        )

        layout.addRow("px2micron", self.px2micron)
        layout.addRow("movie_time_interval_sec", self.movie_dt)
        layout.addRow("keep_every", self.keep_every)
        layout.addRow("smoothing", self.smoothing)
        layout.addRow("degree", self.degree)

        self._load_persisted()

        for w in (
            self.px2micron,
            self.movie_dt,
            self.keep_every,
            self.smoothing,
            self.degree,
        ):
            w.valueChanged.connect(self._on_change)

    def values(self) -> dict:
        return {
            "px2micron": float(self.px2micron.value()),
            "movie_time_interval_sec": float(self.movie_dt.value()),
            "keep_every": int(self.keep_every.value()),
            "smoothing": float(self.smoothing.value()),
            "degree": int(self.degree.value()),
        }

    def apply_metadata(self, metadata: dict):
        """Apply known TIFF metadata fields into the main settings panel."""
        self.px2micron.blockSignals(True)
        self.movie_dt.blockSignals(True)
        try:
            if "px2micron" in metadata:
                self.px2micron.setValue(float(metadata["px2micron"]))
            if "movie_time_interval_sec" in metadata:
                self.movie_dt.setValue(float(metadata["movie_time_interval_sec"]))
        finally:
            self.px2micron.blockSignals(False)
            self.movie_dt.blockSignals(False)
        self._on_change(None)

    def _on_change(self, _value):
        payload = self.values()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self.storage_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)
        self.changed.emit(payload)

    def _load_persisted(self):
        if not self.storage_path.exists():
            return
        with self.storage_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "px2micron" in data:
            self.px2micron.setValue(float(data["px2micron"]))
        if "movie_time_interval_sec" in data:
            self.movie_dt.setValue(float(data["movie_time_interval_sec"]))
        if "keep_every" in data:
            self.keep_every.setValue(int(data["keep_every"]))
        if "smoothing" in data:
            self.smoothing.setValue(float(data["smoothing"]))
        if "degree" in data:
            self.degree.setValue(int(data["degree"]))

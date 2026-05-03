from __future__ import annotations

from pathlib import Path
import yaml

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QFormLayout,
    QVBoxLayout,
    QGroupBox,
    QDoubleSpinBox,
    QSpinBox,
    QSizePolicy,
)


class DefaultsPanel(QWidget):
    changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.storage_path = Path(__file__).resolve().parents[1] / ".last_defaults.yaml"
        # Do not grow past content height; extra space stays with the preview below.
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Maximum,
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self.movie_box = QGroupBox("Movie parameters")
        movie_form = QFormLayout(self.movie_box)
        movie_form.setSpacing(4)
        movie_form.setContentsMargins(6, 8, 6, 6)
        movie_form.setVerticalSpacing(4)
        self.line_box = QGroupBox("Line postprocessing parameters")
        line_form = QFormLayout(self.line_box)
        line_form.setSpacing(4)
        line_form.setContentsMargins(6, 8, 6, 6)
        line_form.setVerticalSpacing(4)

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
            "Time between consecutive movie frames in seconds (kymograph time axis)."
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

        movie_form.addRow("px2micron", self.px2micron)
        movie_form.addRow("Movie time interval (s)", self.movie_dt)
        line_form.addRow("smoothing", self.smoothing)
        line_form.addRow("degree", self.degree)

        root.addWidget(self.movie_box)
        root.addWidget(self.line_box)

        self._load_persisted()

        for w in (
            self.px2micron,
            self.movie_dt,
            self.smoothing,
            self.degree,
        ):
            w.valueChanged.connect(self._on_change)

    def insert_widget_after_movie(self, widget: QWidget) -> None:
        """Insert a widget between Movie parameters and Line postprocessing."""
        lay = self.layout()
        if not isinstance(lay, QVBoxLayout):
            return
        idx = lay.indexOf(self.line_box)
        if idx < 0:
            lay.addWidget(widget)
            return
        lay.insertWidget(idx, widget)

    def values(self) -> dict:
        return {
            "px2micron": float(self.px2micron.value()),
            "movie_time_interval_sec": float(self.movie_dt.value()),
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
        if "smoothing" in data:
            self.smoothing.setValue(float(data["smoothing"]))
        if "degree" in data:
            self.degree.setValue(int(data["degree"]))

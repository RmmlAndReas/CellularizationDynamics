from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import ListedColormap

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QLabel,
    QSizePolicy,
)
from PyQt6.QtCore import Qt

from .figure_style import apply_window_background_to_figure


class ThresholdPanel(QWidget):
    threshold_changed = pyqtSignal(float)
    brightness_changed = pyqtSignal(float)
    island_clicked = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.layout_main = QVBoxLayout(self)
        self.layout_main.setContentsMargins(0, 0, 0, 0)
        self.layout_main.setSpacing(4)

        self.fig = Figure(figsize=(5, 4), layout="constrained")
        apply_window_background_to_figure(self, self.fig)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.ax = self.fig.add_subplot(111)
        self.layout_main.addWidget(self.canvas, 1)

        self.footer = QWidget(self)
        footer_layout = QVBoxLayout(self.footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(4)

        self.threshold_label = QLabel("threshold")
        self.threshold_label.setToolTip(
            "Intensity threshold for apical/cytoplasm mask. "
            "Changing this moves the detected apical border."
        )
        self.threshold_label.setMinimumWidth(72)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setToolTip(
            "Adjust mask threshold used for apical detection (blue overlay)."
        )
        self.threshold_slider.valueChanged.connect(self._slider_changed)
        th_row = QHBoxLayout()
        th_row.setSpacing(6)
        th_row.setContentsMargins(0, 0, 0, 0)
        th_row.addWidget(self.threshold_label, 0)
        th_row.addWidget(self.threshold_slider, 1)
        footer_layout.addLayout(th_row)

        self.brightness_label = QLabel("brightness")
        self.brightness_label.setToolTip(
            "Display-only brightness for both kymograph panels. Does not alter saved data."
        )
        self.brightness_label.setMinimumWidth(72)
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setMinimum(20)
        self.brightness_slider.setMaximum(300)
        self.brightness_slider.setValue(100)
        self.brightness_slider.setToolTip(
            "Display-only brightness for both kymograph panels. Does not alter saved data."
        )
        self.brightness_slider.valueChanged.connect(self._brightness_changed)
        br_row = QHBoxLayout()
        br_row.setSpacing(6)
        br_row.setContentsMargins(0, 0, 0, 0)
        br_row.addWidget(self.brightness_label, 0)
        br_row.addWidget(self.brightness_slider, 1)
        footer_layout.addLayout(br_row)
        self.layout_main.addWidget(self.footer)

        self.kymo = None
        self.mask = None
        self._dt_min = 10.0 / 60.0
        self.selected_island_mask = None
        self.base_vmin = 0.0
        self.base_vmax = 1.0
        self.use_island_mode = True
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

    def set_data(
        self,
        kymo: np.ndarray,
        threshold: float,
        mask: np.ndarray,
        movie_time_interval_sec: float,
        use_island_mode: bool = True,
    ):
        self.kymo = kymo
        self.mask = mask
        self.use_island_mode = bool(use_island_mode)
        self._dt_min = max(float(movie_time_interval_sec) / 60.0, 1e-12)
        if self.selected_island_mask is not None and self.selected_island_mask.shape != kymo.shape:
            self.selected_island_mask = None
        dmin = float(np.min(kymo))
        dmax = float(np.max(kymo))
        self.base_vmin = float(np.percentile(kymo, 1))
        self.base_vmax = float(np.percentile(kymo, 99))
        if self.base_vmax <= self.base_vmin:
            self.base_vmin = dmin
            self.base_vmax = dmax

        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setMinimum(int(np.floor(dmin)))
        self.threshold_slider.setMaximum(int(np.ceil(dmax)))
        self.threshold_slider.setValue(int(round(threshold)))
        self.threshold_slider.blockSignals(False)
        self.redraw()

    def _slider_changed(self, val: int):
        self.threshold_label.setText(f"threshold: {val}")
        self.threshold_changed.emit(float(val))

    def redraw(self):
        if self.kymo is None:
            return
        self.ax.clear()
        brightness = float(self.brightness_slider.value()) / 100.0
        vmax = self.base_vmin + (self.base_vmax - self.base_vmin) / max(brightness, 1e-6)
        h, w = self.kymo.shape
        x1 = float(max(w - 1, 0)) * self._dt_min
        if w == 1:
            x1 = self._dt_min
        extent = (0.0, x1, float(h - 1), 0.0)
        self.ax.imshow(
            self.kymo,
            cmap="gray",
            aspect="auto",
            origin="upper",
            extent=extent,
            vmin=self.base_vmin,
            vmax=vmax,
        )

        if self.mask is not None:
            cmap = ListedColormap([(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0)])
            self.ax.imshow(
                self.mask.astype(np.uint8),
                cmap=cmap,
                aspect="auto",
                origin="upper",
                extent=extent,
                vmin=0,
                vmax=1,
            )
        if self.use_island_mode and self.selected_island_mask is not None:
            island_cmap = ListedColormap([(0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.5)])
            self.ax.imshow(
                self.selected_island_mask.astype(np.uint8),
                cmap=island_cmap,
                aspect="auto",
                origin="upper",
                extent=extent,
                vmin=0,
                vmax=1,
            )

        self.ax.set_title("Apical Border Detection")
        self.ax.set_xlabel("Time (min)")
        self.ax.set_ylabel("Depth (px)")
        self.canvas.draw_idle()

    def _brightness_changed(self, _val: int):
        self.redraw()
        self.brightness_changed.emit(self.brightness_factor())

    def _on_canvas_click(self, event):
        if not self.use_island_mode:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        w = self.kymo.shape[1]
        x_col = float(np.clip(np.rint(float(event.xdata) / self._dt_min), 0, w - 1))
        self.island_clicked.emit(x_col, float(event.ydata))

    def set_selected_island_mask(self, selected_island_mask: np.ndarray | None):
        self.selected_island_mask = selected_island_mask
        self.redraw()

    def footer_height(self) -> int:
        return max(0, self.footer.sizeHint().height())

    def brightness_factor(self) -> float:
        return float(self.brightness_slider.value()) / 100.0

    def detach_footer_widget(self) -> QWidget:
        self.layout_main.removeWidget(self.footer)
        self.footer.setParent(None)
        return self.footer

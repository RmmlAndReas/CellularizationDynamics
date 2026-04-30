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
    QCheckBox,
    QToolButton,
)
from PyQt6.QtCore import Qt


class ThresholdPanel(QWidget):
    threshold_changed = pyqtSignal(float)
    brightness_changed = pyqtSignal(float)
    island_mode_changed = pyqtSignal(bool)
    island_clicked = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.layout_main = QVBoxLayout(self)
        help_row = QHBoxLayout()
        help_row.addStretch(1)
        self.help_button = QToolButton(self)
        self.help_button.setText("?")
        self.help_button.setAutoRaise(True)
        self.help_button.setToolTip(
            "Apical Border Detection\n"
            "- Threshold: adjusts mask detection for the blue overlay.\n"
            "- Brightness: display-only intensity scaling.\n"
            "- Island mode: click a threshold island to use its basal edge as apex."
        )
        self.help_button.setToolTipDuration(20000)
        help_row.addWidget(self.help_button)
        self.layout_main.addLayout(help_row)
        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.96)
        self.layout_main.addWidget(self.canvas)

        self.footer = QWidget(self)
        footer_layout = QVBoxLayout(self.footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(6)

        self.threshold_label = QLabel("threshold")
        self.threshold_label.setToolTip(
            "Intensity threshold for apical/cytoplasm mask. "
            "Changing this moves the detected apical border."
        )
        footer_layout.addWidget(self.threshold_label)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setToolTip(
            "Adjust mask threshold used for apical detection (blue overlay)."
        )
        self.threshold_slider.valueChanged.connect(self._slider_changed)
        footer_layout.addWidget(self.threshold_slider)

        self.brightness_label = QLabel("brightness")
        self.brightness_label.setToolTip(
            "Display-only brightness for both kymograph panels. Does not alter saved data."
        )
        footer_layout.addWidget(self.brightness_label)
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setMinimum(20)
        self.brightness_slider.setMaximum(300)
        self.brightness_slider.setValue(100)
        self.brightness_slider.setToolTip(
            "Display-only brightness for both kymograph panels. Does not alter saved data."
        )
        self.brightness_slider.valueChanged.connect(self._brightness_changed)
        footer_layout.addWidget(self.brightness_slider)

        self.island_mode_checkbox = QCheckBox("Use island mode")
        self.island_mode_checkbox.setToolTip(
            "If enabled, click a threshold island and use its basal edge as apex."
        )
        self.island_mode_checkbox.toggled.connect(self._island_mode_toggled)
        footer_layout.addWidget(self.island_mode_checkbox)
        self.layout_main.addWidget(self.footer)

        self.kymo = None
        self.mask = None
        self.selected_island_mask = None
        self.base_vmin = 0.0
        self.base_vmax = 1.0
        self.use_island_mode = False
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

    def set_data(self, kymo: np.ndarray, threshold: float, mask: np.ndarray):
        self.kymo = kymo
        self.mask = mask
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
        self.ax.imshow(self.kymo, cmap="gray", aspect="auto", vmin=self.base_vmin, vmax=vmax)

        if self.mask is not None:
            cmap = ListedColormap([(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0)])
            self.ax.imshow(self.mask.astype(np.uint8), cmap=cmap, aspect="auto", vmin=0, vmax=1)
        if self.use_island_mode and self.selected_island_mask is not None:
            island_cmap = ListedColormap([(0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.5)])
            self.ax.imshow(
                self.selected_island_mask.astype(np.uint8),
                cmap=island_cmap,
                aspect="auto",
                vmin=0,
                vmax=1,
            )

        self.ax.set_title("Apical Border Detection")
        self.ax.set_xlabel("time")
        self.ax.set_ylabel("depth")
        self.fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.96)
        self.canvas.draw_idle()

    def _brightness_changed(self, _val: int):
        self.redraw()
        self.brightness_changed.emit(self.brightness_factor())

    def _island_mode_toggled(self, checked: bool):
        self.use_island_mode = bool(checked)
        self.redraw()
        self.island_mode_changed.emit(self.use_island_mode)

    def _on_canvas_click(self, event):
        if not self.use_island_mode:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.island_clicked.emit(float(event.xdata), float(event.ydata))

    def set_selected_island_mask(self, selected_island_mask: np.ndarray | None):
        self.selected_island_mask = selected_island_mask
        self.redraw()

    def is_island_mode_enabled(self) -> bool:
        return bool(self.island_mode_checkbox.isChecked())

    def footer_height(self) -> int:
        return max(0, self.footer.sizeHint().height())

    def brightness_factor(self) -> float:
        return float(self.brightness_slider.value()) / 100.0

    def detach_footer_widget(self) -> QWidget:
        self.layout_main.removeWidget(self.footer)
        self.footer.setParent(None)
        return self.footer

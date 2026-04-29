from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout


class FrontPanel(QWidget):
    raw_point_added = pyqtSignal(float, float)
    undo_requested = pyqtSignal()
    clear_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.96)
        layout.addWidget(self.canvas)
        self.footer_spacer = QWidget(self)
        self.footer_spacer.setMinimumHeight(0)
        self.footer_spacer.setMaximumHeight(0)
        layout.addWidget(self.footer_spacer)

        self.straight_kymo = None
        self.points_display = np.empty((0, 2), dtype=float)
        self.shifts = None
        self.crop_top = 0
        self.num_cols = 0
        self.brightness = 1.0
        self.base_vmin = 0.0
        self.base_vmax = 1.0

        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("key_press_event", self._on_key)

    def set_data(self, straight_kymo, points_display, shifts, crop_top):
        self.straight_kymo = straight_kymo
        self.points_display = points_display
        self.shifts = shifts
        self.crop_top = crop_top
        self.num_cols = int(straight_kymo.shape[1]) if straight_kymo is not None else 0
        if self.straight_kymo is not None:
            view = self.straight_kymo[self.crop_top :, :]
            self.base_vmin = float(np.percentile(view, 1))
            self.base_vmax = float(np.percentile(view, 99))
            if self.base_vmax <= self.base_vmin:
                self.base_vmin = float(np.min(view))
                self.base_vmax = float(np.max(view))
                if self.base_vmax <= self.base_vmin:
                    self.base_vmax = self.base_vmin + 1.0
        self.redraw()

    def redraw(self):
        self.ax.clear()
        if self.straight_kymo is not None:
            view = self.straight_kymo[self.crop_top :, :]
            vmax = self.base_vmin + (self.base_vmax - self.base_vmin) / max(self.brightness, 1e-6)
            self.ax.imshow(view, cmap="gray", aspect="auto", vmin=self.base_vmin, vmax=vmax)
        if self.points_display.size:
            self.ax.plot(self.points_display[:, 0], self.points_display[:, 1], "r-", linewidth=2)
            self.ax.scatter(self.points_display[:, 0], self.points_display[:, 1], c="yellow", s=18)
        self.ax.set_title("Front annotation (click add, backspace undo, esc clear)")
        self.ax.set_xlabel("time")
        self.ax.set_ylabel("depth")
        self.fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.96)
        self.canvas.draw_idle()

    def _on_click(self, event):
        self.canvas.setFocus()
        if self.straight_kymo is None or self.shifts is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y_display = float(event.ydata)
        x_idx = int(np.clip(np.rint(x), 0, self.num_cols - 1))
        y_corr = y_display + self.crop_top
        y_raw = y_corr - float(self.shifts[x_idx])
        self.raw_point_added.emit(x, y_raw)

    def _on_key(self, event):
        key = (event.key or "").lower()
        if key == "backspace":
            self.undo_requested.emit()
        elif key == "escape":
            self.clear_requested.emit()

    def set_footer_height(self, height: int):
        h = max(0, int(height))
        self.footer_spacer.setMinimumHeight(h)
        self.footer_spacer.setMaximumHeight(h)

    def set_brightness(self, brightness: float):
        self.brightness = max(0.2, min(3.0, float(brightness)))
        self.redraw()

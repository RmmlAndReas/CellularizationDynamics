from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backend_bases import MouseButton

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QToolButton


class FrontPanel(QWidget):
    raw_point_added = pyqtSignal(float, float)
    raw_point_moved = pyqtSignal(int, float, float)
    undo_requested = pyqtSignal()
    clear_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        help_row = QHBoxLayout()
        help_row.addStretch(1)
        self.help_button = QToolButton(self)
        self.help_button.setText("?")
        self.help_button.setAutoRaise(True)
        self.help_button.setToolTip(
            "Cellularization Front Annotation\n"
            "- Click: add a front point.\n"
            "- Drag point: move an existing front point.\n"
            "- Scroll: zoom in/out horizontally around cursor.\n"
            "- Pan slider: move the zoomed window.\n"
            "- Backspace: undo last point.\n"
            "- Esc: clear all points.\n"
            "- R: reset zoom."
        )
        self.help_button.setToolTipDuration(20000)
        help_row.addWidget(self.help_button)
        layout.addLayout(help_row)
        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.96)
        layout.addWidget(self.canvas)
        self.footer_spacer = QWidget(self)
        self.footer_spacer.setMinimumHeight(0)
        self.footer_spacer.setMaximumHeight(0)
        self.pan_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.pan_slider.setMinimum(0)
        self.pan_slider.setMaximum(1000)
        self.pan_slider.setSingleStep(1)
        self.pan_slider.setPageStep(25)
        self.pan_slider.setVisible(False)
        self.pan_slider.valueChanged.connect(self._on_pan_slider_changed)
        self._updating_slider = False
        layout.addWidget(self.pan_slider)
        layout.addWidget(self.footer_spacer)

        self.straight_kymo = None
        self.points_display = np.empty((0, 2), dtype=float)
        self.shifts = None
        self.crop_top = 0
        self.num_cols = 0
        self.brightness = 1.0
        self.base_vmin = 0.0
        self.base_vmax = 1.0

        # Persistent horizontal zoom window in data coordinates. None means full extent.
        self.x_zoom: tuple[float, float] | None = None
        # Minimum zoom width in pixels to avoid degenerate views.
        self._min_zoom_width = 4.0
        # Multiplicative factor applied per scroll notch.
        self._zoom_step = 1.25

        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("key_press_event", self._on_key)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._dragging_idx: int | None = None
        self._pick_radius_px = 10.0

    def set_data(self, straight_kymo, points_display, shifts, crop_top):
        prev_num_cols = self.num_cols
        self.straight_kymo = straight_kymo
        self.points_display = points_display
        self.shifts = shifts
        self.crop_top = crop_top
        self.num_cols = int(straight_kymo.shape[1]) if straight_kymo is not None else 0
        # Drop a stale zoom window if the data dimensions changed (new sample loaded).
        if self.num_cols != prev_num_cols:
            self.x_zoom = None
            self._updating_slider = True
            self.pan_slider.setVisible(False)
            self.pan_slider.setValue(0)
            self._updating_slider = False
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
        self.ax.set_title("Cellularization Front Annotation")
        self.ax.set_xlabel("time")
        self.ax.set_ylabel("depth")
        self.fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.96)
        # Apply persistent horizontal zoom last so it survives state-triggered redraws.
        if self.x_zoom is not None and self.num_cols > 0:
            xmin, xmax = self._clamp_zoom(self.x_zoom[0], self.x_zoom[1])
            self.x_zoom = (xmin, xmax)
            self.ax.set_xlim(xmin, xmax)
        self._sync_pan_slider()
        self.canvas.draw_idle()

    def _on_click(self, event):
        self.canvas.setFocus()
        if self.straight_kymo is None or self.shifts is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button != MouseButton.LEFT:
            return

        picked_idx = self._nearest_point_index(event)
        if picked_idx is not None:
            self._dragging_idx = int(picked_idx)
            return

        x = float(event.xdata)
        y_display = float(event.ydata)
        x_idx = int(np.clip(np.rint(x), 0, self.num_cols - 1))
        y_corr = y_display + self.crop_top
        y_raw = y_corr - float(self.shifts[x_idx])
        self.raw_point_added.emit(x, y_raw)

    def _on_motion(self, event):
        if self._dragging_idx is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self._emit_moved_point(self._dragging_idx, float(event.xdata), float(event.ydata))

    def _on_release(self, event):
        if self._dragging_idx is None:
            return
        idx = self._dragging_idx
        self._dragging_idx = None
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self._emit_moved_point(idx, float(event.xdata), float(event.ydata))

    def _nearest_point_index(self, event) -> int | None:
        if not self.points_display.size or event.xdata is None or event.ydata is None:
            return None
        click_xy = self.ax.transData.transform((float(event.xdata), float(event.ydata)))
        points_xy = self.ax.transData.transform(self.points_display[:, :2])
        distances = np.sqrt(np.sum((points_xy - click_xy) ** 2, axis=1))
        idx = int(np.argmin(distances))
        if float(distances[idx]) > self._pick_radius_px:
            return None
        return idx

    def _emit_moved_point(self, idx: int, x: float, y_display: float):
        if self.straight_kymo is None or self.shifts is None or self.num_cols <= 0:
            return
        x_clamped = float(np.clip(x, 0.0, float(self.num_cols - 1)))
        visible_rows = max(1, int(self.straight_kymo.shape[0] - self.crop_top))
        y_display_clamped = float(np.clip(y_display, 0.0, float(visible_rows - 1)))
        x_idx = int(np.clip(np.rint(x_clamped), 0, self.num_cols - 1))
        y_corr = y_display_clamped + self.crop_top
        y_raw = y_corr - float(self.shifts[x_idx])
        self.raw_point_moved.emit(int(idx), x_clamped, y_raw)

    def _on_key(self, event):
        key = (event.key or "").lower()
        if key == "backspace":
            self.undo_requested.emit()
        elif key == "escape":
            self.clear_requested.emit()
        elif key == "r":
            self.reset_zoom()

    def _on_scroll(self, event):
        if self.straight_kymo is None or self.num_cols <= 0:
            return
        if event.inaxes != self.ax:
            return

        full_min = 0.0
        full_max = float(self.num_cols - 1)
        cur_min, cur_max = self.x_zoom if self.x_zoom is not None else (full_min, full_max)
        cur_width = cur_max - cur_min
        if cur_width <= 0:
            return

        if event.button == "up":
            scale = 1.0 / self._zoom_step
        elif event.button == "down":
            scale = self._zoom_step
        else:
            return

        center = float(event.xdata) if event.xdata is not None else (cur_min + cur_max) / 2.0
        center = min(max(center, cur_min), cur_max)
        left_frac = (center - cur_min) / cur_width
        right_frac = (cur_max - center) / cur_width
        new_width = cur_width * scale
        new_min = center - left_frac * new_width
        new_max = center + right_frac * new_width

        new_min, new_max = self._clamp_zoom(new_min, new_max)
        if new_min >= full_min and new_max <= full_max and (new_max - new_min) >= (full_max - full_min):
            # Zoomed back out to full extent; clear state to keep default behavior.
            self.x_zoom = None
        else:
            self.x_zoom = (new_min, new_max)
        self.redraw()

    def _clamp_zoom(self, xmin: float, xmax: float) -> tuple[float, float]:
        full_min = 0.0
        full_max = float(max(self.num_cols - 1, 0))
        min_width = min(self._min_zoom_width, max(full_max - full_min, 0.0))
        if xmax - xmin < min_width:
            center = (xmin + xmax) / 2.0
            xmin = center - min_width / 2.0
            xmax = center + min_width / 2.0
        if xmin < full_min:
            xmax += full_min - xmin
            xmin = full_min
        if xmax > full_max:
            xmin -= xmax - full_max
            xmax = full_max
        xmin = max(xmin, full_min)
        xmax = min(xmax, full_max)
        return xmin, xmax

    def reset_zoom(self):
        if self.x_zoom is None:
            return
        self.x_zoom = None
        self.redraw()

    def _sync_pan_slider(self):
        if self.num_cols <= 1 or self.x_zoom is None:
            self._updating_slider = True
            self.pan_slider.setVisible(False)
            self.pan_slider.setValue(0)
            self._updating_slider = False
            return
        xmin, xmax = self.x_zoom
        full_max = float(self.num_cols - 1)
        window_width = xmax - xmin
        max_shift = full_max - window_width
        if max_shift <= 0:
            self._updating_slider = True
            self.pan_slider.setVisible(False)
            self.pan_slider.setValue(0)
            self._updating_slider = False
            return
        slider_val = int(round((xmin / max_shift) * 1000.0))
        slider_val = max(0, min(1000, slider_val))
        self._updating_slider = True
        self.pan_slider.setVisible(True)
        self.pan_slider.setValue(slider_val)
        self._updating_slider = False

    def _on_pan_slider_changed(self, value: int):
        if self._updating_slider or self.x_zoom is None or self.num_cols <= 1:
            return
        full_max = float(self.num_cols - 1)
        xmin, xmax = self.x_zoom
        window_width = xmax - xmin
        max_shift = full_max - window_width
        if max_shift <= 0:
            return
        new_min = (float(value) / 1000.0) * max_shift
        new_max = new_min + window_width
        self.x_zoom = self._clamp_zoom(new_min, new_max)
        self.redraw()

    def set_footer_height(self, height: int):
        h = max(0, int(height))
        self.footer_spacer.setMinimumHeight(h)
        self.footer_spacer.setMaximumHeight(h)

    def set_brightness(self, brightness: float):
        self.brightness = max(0.2, min(3.0, float(brightness)))
        self.redraw()


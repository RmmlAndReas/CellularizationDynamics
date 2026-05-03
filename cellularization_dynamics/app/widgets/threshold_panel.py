from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backend_bases import MouseButton
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

APICAL_MODE_ISLAND = "island"
APICAL_MODE_MANUAL = "manual"


class ThresholdPanel(QWidget):
    threshold_changed = pyqtSignal(float)
    brightness_changed = pyqtSignal(float)
    island_clicked = pyqtSignal(float, float)

    manual_point_added = pyqtSignal(float, float)
    manual_point_moved = pyqtSignal(int, float, float, bool)
    manual_undo_requested = pyqtSignal()
    manual_clear_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.layout_main = QVBoxLayout(self)
        self.layout_main.setContentsMargins(0, 0, 0, 0)
        self.layout_main.setSpacing(4)

        self.fig = Figure(figsize=(5, 4), layout="constrained")
        apply_window_background_to_figure(self, self.fig)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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

        self.threshold_row = QWidget(self.footer)
        th_row = QHBoxLayout(self.threshold_row)
        th_row.setSpacing(6)
        th_row.setContentsMargins(0, 0, 0, 0)
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
        th_row.addWidget(self.threshold_label, 0)
        th_row.addWidget(self.threshold_slider, 1)
        footer_layout.addWidget(self.threshold_row)

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
        self.mode: str = APICAL_MODE_ISLAND
        self._manual_points_display = np.empty((0, 2), dtype=float)
        self._manual_curve_display: np.ndarray | None = None
        self._dragging_idx: int | None = None
        self._pick_radius_px = 10.0

        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("key_press_event", self._on_key)

    def set_mode(self, mode: str):
        if mode not in (APICAL_MODE_ISLAND, APICAL_MODE_MANUAL):
            raise ValueError(f"Unknown apical mode: {mode!r}")
        if mode == self.mode:
            return
        self.mode = mode
        self.threshold_row.setVisible(mode == APICAL_MODE_ISLAND)
        self.redraw()

    def set_data(
        self,
        kymo: np.ndarray,
        threshold: float,
        mask: np.ndarray,
        movie_time_interval_sec: float,
    ):
        self.kymo = kymo
        self.mask = mask
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

    def set_manual_curves(
        self,
        points_display: np.ndarray,
        smoothed_curve: np.ndarray | None,
    ) -> None:
        """Set manual polyline markers and the smoothed apical curve (both in display coords)."""
        self._manual_points_display = np.asarray(points_display, dtype=float).reshape(-1, 2)
        if smoothed_curve is None or np.asarray(smoothed_curve).size == 0:
            self._manual_curve_display = None
        else:
            self._manual_curve_display = np.asarray(smoothed_curve, dtype=float).reshape(-1, 2)
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

        if self.mode == APICAL_MODE_ISLAND:
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
            if self.selected_island_mask is not None:
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
        else:
            if self._manual_curve_display is not None and self._manual_curve_display.size:
                self.ax.plot(
                    self._manual_curve_display[:, 0],
                    self._manual_curve_display[:, 1],
                    color="#FFD700",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.9,
                )
            if self._manual_points_display.size:
                self.ax.plot(
                    self._manual_points_display[:, 0],
                    self._manual_points_display[:, 1],
                    "r-",
                    linewidth=2,
                )
                self.ax.scatter(
                    self._manual_points_display[:, 0],
                    self._manual_points_display[:, 1],
                    c="yellow",
                    s=18,
                )

        self.ax.set_title("Apical Border Detection")
        self.ax.set_xlabel("Time (min)")
        self.ax.set_ylabel("Depth (px)")
        self.canvas.draw_idle()

    def _brightness_changed(self, _val: int):
        self.redraw()
        self.brightness_changed.emit(self.brightness_factor())

    def _col_idx_from_xdata(self, xdata: float) -> int:
        if self.kymo is None:
            return 0
        w = int(self.kymo.shape[1])
        return int(np.clip(np.rint(float(xdata) / self._dt_min), 0, max(w - 1, 0)))

    def _nearest_manual_point_index(self, event) -> int | None:
        if not self._manual_points_display.size or event.xdata is None or event.ydata is None:
            return None
        click_xy = self.ax.transData.transform((float(event.xdata), float(event.ydata)))
        points_xy = self.ax.transData.transform(self._manual_points_display[:, :2])
        distances = np.sqrt(np.sum((points_xy - click_xy) ** 2, axis=1))
        idx = int(np.argmin(distances))
        if float(distances[idx]) > self._pick_radius_px:
            return None
        return idx

    def _on_click(self, event):
        self.canvas.setFocus()
        if self.kymo is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if self.mode == APICAL_MODE_ISLAND:
            x_col = float(self._col_idx_from_xdata(event.xdata))
            self.island_clicked.emit(x_col, float(event.ydata))
            return

        if event.button != MouseButton.LEFT:
            return
        picked_idx = self._nearest_manual_point_index(event)
        if picked_idx is not None:
            self._dragging_idx = int(picked_idx)
            return
        x_col = float(self._col_idx_from_xdata(event.xdata))
        y_raw = float(event.ydata)
        self.manual_point_added.emit(x_col, y_raw)

    def _on_motion(self, event):
        if self.mode != APICAL_MODE_MANUAL or self._dragging_idx is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x_col = float(self._col_idx_from_xdata(event.xdata))
        y_raw = float(np.clip(float(event.ydata), 0.0, float(self.kymo.shape[0] - 1)))
        self.manual_point_moved.emit(self._dragging_idx, x_col, y_raw, False)

    def _on_release(self, event):
        if self.mode != APICAL_MODE_MANUAL or self._dragging_idx is None:
            return
        idx = self._dragging_idx
        self._dragging_idx = None
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x_col = float(self._col_idx_from_xdata(event.xdata))
        y_raw = float(np.clip(float(event.ydata), 0.0, float(self.kymo.shape[0] - 1)))
        self.manual_point_moved.emit(int(idx), x_col, y_raw, True)

    def _on_key(self, event):
        if self.mode != APICAL_MODE_MANUAL:
            return
        key = (event.key or "").lower()
        if key == "backspace":
            self.manual_undo_requested.emit()
        elif key == "escape":
            self.manual_clear_requested.emit()

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

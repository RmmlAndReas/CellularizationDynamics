from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from .geometry_transform import straight_from_raw, um_from_straight
from .pipeline_adapter import compute_apical_column_positions
from .straighten_fast import straighten


@dataclass
class AcquisitionParams:
    px2micron: float
    movie_time_interval_sec: float
    smoothing: float = 0.0
    degree: int = 1


class SampleState(QObject):
    state_changed = pyqtSignal()

    def __init__(self, raw_movie_path: Path, work_dir: Path, acq_params: AcquisitionParams):
        super().__init__()
        self.raw_movie_path = raw_movie_path
        self.work_dir = work_dir
        self.acq_params = acq_params

        self.kymograph: np.ndarray | None = None
        self.threshold: float | None = None
        self.mask: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.ref_row: int = 0
        self.shifts: np.ndarray | None = None
        self.straight_kymo: np.ndarray | None = None
        self.front_points_raw: list[tuple[float, float]] = []
        self.use_island_mode: bool = True
        self.selected_island_labels: set[int] = set()
        self.dirty: bool = False

    def set_kymograph(self, kymo: np.ndarray) -> None:
        """Replace kymograph and initialize threshold from median intensity (full recompute)."""
        self.assign_kymograph_only(kymo)
        self.init_threshold_from_percentile_and_recompute()

    def assign_kymograph_only(self, kymo: np.ndarray) -> None:
        """Set raw kymograph array only; clears derived masks/shifts until restore or recompute."""
        self.kymograph = kymo
        self.threshold = None
        self.mask = None
        self.labels = None
        self.shifts = None
        self.straight_kymo = None

    def init_threshold_from_percentile_and_recompute(self) -> None:
        if self.kymograph is None:
            return
        self.threshold = float(np.percentile(self.kymograph, 50))
        self.recompute_from_threshold()

    def apply_apical_from_saved(
        self,
        threshold: float,
        use_island: bool,
        island_labels: list[int],
    ) -> None:
        """Restore threshold + island mode without clearing labels (unlike set_threshold)."""
        self.use_island_mode = bool(use_island)
        self.selected_island_labels = {int(x) for x in island_labels} if use_island else set()
        self.threshold = float(threshold)
        self.recompute_from_threshold()

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self.selected_island_labels.clear()
        self.recompute_from_threshold()

    def set_use_island_mode(self, use_island_mode: bool) -> None:
        self.use_island_mode = bool(use_island_mode)
        if not self.use_island_mode:
            self.selected_island_labels.clear()
        self.recompute_from_threshold()

    def select_island_at(self, x: float, y: float) -> bool:
        if self.labels is None:
            return False
        row = int(np.clip(np.rint(y), 0, self.labels.shape[0] - 1))
        col = int(np.clip(np.rint(x), 0, self.labels.shape[1] - 1))
        picked_label = int(self.labels[row, col])
        if picked_label <= 0:
            if self.selected_island_labels:
                self.selected_island_labels.clear()
                self.recompute_from_threshold()
            return False
        self.selected_island_labels.add(picked_label)
        self.recompute_from_threshold()
        return True

    def selected_island_mask(self) -> np.ndarray | None:
        if self.labels is None or not self.selected_island_labels:
            return None
        return np.isin(self.labels, list(self.selected_island_labels))

    def recompute_from_threshold(self) -> None:
        if self.kymograph is None or self.threshold is None:
            return

        self.mask = self.kymograph <= self.threshold
        depth, num_timepoints = self.mask.shape

        if self.use_island_mode:
            apical_px, self.labels = compute_apical_column_positions(
                self.mask,
                mode="island",
                island_labels=self.selected_island_labels,
                min_run_length_px=5,
            )
        else:
            apical_px, self.labels = compute_apical_column_positions(
                self.mask,
                mode="longest_run",
                island_labels=None,
                min_run_length_px=5,
            )

        valid = ~np.isnan(apical_px)
        self.shifts = np.zeros(num_timepoints, dtype=int)
        if np.any(valid):
            # Reserve ~2 µm headroom above apical.
            margin_px = int(round(2.0 / max(self.acq_params.px2micron, 1e-9)))
            self.ref_row = int(np.nanmin(apical_px[valid])) + max(0, margin_px)
            self.shifts[valid] = (self.ref_row - apical_px[valid]).astype(int)
        else:
            self.ref_row = 0

        self.straight_kymo = straighten(self.kymograph, self.shifts)
        self.dirty = True
        self.state_changed.emit()

    def add_front_point_raw(self, x: float, y: float) -> None:
        self.front_points_raw.append((x, y))
        self.dirty = True
        self.state_changed.emit()

    def update_front_point_raw(
        self,
        index: int,
        x: float,
        y: float,
        *,
        emit_state_changed: bool = True,
    ) -> None:
        if index < 0 or index >= len(self.front_points_raw):
            return
        self.front_points_raw[index] = (float(x), float(y))
        self.dirty = True
        if emit_state_changed:
            self.state_changed.emit()

    def undo_front_point(self) -> None:
        if self.front_points_raw:
            self.front_points_raw.pop()
            self.dirty = True
            self.state_changed.emit()

    def clear_front_points(self) -> None:
        self.front_points_raw = []
        self.dirty = True
        self.state_changed.emit()

    def display_points(self, ref_row: int, px2micron: float, movie_time_interval_sec: float) -> np.ndarray:
        """Map stored points (x = col index, y = raw depth row) to plot (time min, depth µm)."""
        if not self.front_points_raw:
            return np.empty((0, 2), dtype=float)

        pts = np.asarray(self.front_points_raw, dtype=float)
        dt_min = max(float(movie_time_interval_sec) / 60.0, 1e-12)
        if self.shifts is None or self.kymograph is None:
            out = pts.copy()
            out[:, 0] = out[:, 0] * dt_min
            return out

        s = max(float(px2micron), 1e-12)
        num_cols = self.kymograph.shape[1]
        display = pts.copy()
        rr = float(ref_row)
        for i in range(display.shape[0]):
            x_col = float(display[i, 0])
            x_idx = int(np.clip(np.rint(x_col), 0, num_cols - 1))
            straight_row = straight_from_raw(display[i, 1], float(self.shifts[x_idx]))
            display[i, 0] = x_col * dt_min
            display[i, 1] = um_from_straight(straight_row, rr, s)
        return display

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from .pipeline_adapter import select_cytoplasm_run
from .straighten_fast import straighten


@dataclass
class AcquisitionParams:
    px2micron: float
    movie_time_interval_sec: float
    keep_every: int
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
        self.ref_row: int = 0
        self.shifts: np.ndarray | None = None
        self.straight_kymo: np.ndarray | None = None
        self.front_points_raw: list[tuple[float, float]] = []
        self.dirty: bool = False

    def set_kymograph(self, kymo: np.ndarray) -> None:
        self.kymograph = kymo
        self.threshold = float(np.percentile(kymo, 50))
        self.recompute_from_threshold()

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self.recompute_from_threshold()

    def recompute_from_threshold(self) -> None:
        if self.kymograph is None or self.threshold is None:
            return

        self.mask = self.kymograph <= self.threshold
        depth, num_timepoints = self.mask.shape

        apical_px = np.full(num_timepoints, np.nan, dtype=float)
        for t in range(num_timepoints):
            best_start, _ = select_cytoplasm_run(self.mask[:, t], min_run_length_px=5)
            if best_start is not None:
                apical_px[t] = float(best_start)

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

    def undo_front_point(self) -> None:
        if self.front_points_raw:
            self.front_points_raw.pop()
            self.dirty = True
            self.state_changed.emit()

    def clear_front_points(self) -> None:
        self.front_points_raw = []
        self.dirty = True
        self.state_changed.emit()

    def display_points(self, crop_top: int) -> np.ndarray:
        if not self.front_points_raw:
            return np.empty((0, 2), dtype=float)

        pts = np.asarray(self.front_points_raw, dtype=float)
        if self.shifts is None or self.kymograph is None:
            return pts

        num_cols = self.kymograph.shape[1]
        display = pts.copy()
        for i in range(display.shape[0]):
            x = float(display[i, 0])
            x_idx = int(np.clip(np.rint(x), 0, num_cols - 1))
            display[i, 1] = display[i, 1] + float(self.shifts[x_idx]) - crop_top
        return display

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PyQt6.QtCore import QThread, pyqtSignal

from .pipeline_adapter import (
    trim_movie,
    create_single_kymograph,
    fit_and_save,
    export_geometry_timeseries,
    generate_outputs,
    straighten_kymograph_run,
)
from .output_layout import ensure_work_tree


class AnalyzeWorker(QThread):
    progress = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, raw_movie_path: Path, work_dir: Path, sample_path: str):
        super().__init__()
        self.raw_movie_path = raw_movie_path
        self.work_dir = work_dir
        self.sample_path = sample_path

    def run(self):
        try:
            ensure_work_tree(self.work_dir)
            self.progress.emit("Trimming movie...")
            trim_movie(str(self.work_dir), str(self.raw_movie_path.parent))

            self.progress.emit("Building kymograph...")
            movie_path = self.work_dir / "Cellularization_trimmed.tif"
            movie = tifffile.imread(str(movie_path))
            if movie.ndim == 2:
                movie = movie[np.newaxis, :, :]
            width = int(movie.shape[2])
            create_single_kymograph(movie, width, width, 0, width, "Kymograph.tif", str(self.work_dir))

            kymo_path = self.work_dir / "track" / "Kymograph.tif"
            kymo = tifffile.imread(str(kymo_path))
            if kymo.ndim > 2:
                kymo = np.squeeze(kymo)
            self.done.emit(kymo)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


class GenerateFigureWorker(QThread):
    progress = pyqtSignal(str)
    done = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, work_dir: Path, smoothing: float, degree: int, time_interval_min: float):
        super().__init__()
        self.work_dir = work_dir
        self.smoothing = smoothing
        self.degree = degree
        self.time_interval_min = time_interval_min

    def run(self):
        try:
            self.progress.emit("Straightening kymograph...")
            straighten_kymograph_run(str(self.work_dir))
            self.progress.emit("Fitting spline...")
            fit_and_save(str(self.work_dir), self.smoothing, self.degree, self.time_interval_min)
            self.progress.emit("Exporting geometry...")
            export_geometry_timeseries(str(self.work_dir))
            self.progress.emit("Generating figure...")
            generate_outputs(str(self.work_dir))
            self.done.emit()
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))

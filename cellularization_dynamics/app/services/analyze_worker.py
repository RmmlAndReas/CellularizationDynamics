from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PyQt6.QtCore import QThread, pyqtSignal

from .pipeline_adapter import (
    create_single_kymograph,
    fit_and_save,
    export_geometry_timeseries,
    generate_outputs,
    horizontal_roi_from_averaging_pct,
    straighten_kymograph_run,
)
from .output_layout import ensure_work_tree

from cellularization_dynamics.core.annotation_source import load_apical_session_v2_doc
from cellularization_dynamics.core.work_state import (
    get_movie_path,
    load_state,
    pipeline_config_flat,
)


def try_load_saved_kymograph(work_dir: str, averaging_pct: float) -> np.ndarray | None:
    """
    Load ``track/Kymograph.tif`` without reading the full movie when it matches the
    saved v2 session shape and was built with ``averaging_pct``.
    """
    wd = Path(work_dir)
    kpath = wd / "track" / "Kymograph.tif"
    if not kpath.is_file():
        return None
    doc = load_apical_session_v2_doc(str(wd / "track"))
    if doc is None:
        return None
    exp_h = int(doc["kymograph_height_px"])
    exp_w = int(doc["kymograph_width_px"])
    try:
        with tifffile.TiffFile(kpath) as tf:
            sh = tf.series[0].shape
    except Exception:
        return None
    if tuple(sh) != (exp_h, exp_w):
        return None
    state = load_state(str(wd), migrate_if_needed=True)
    ksec = state.get("kymograph") or {}
    pct_i = int(max(1, min(100, round(float(averaging_pct)))))
    last_built = ksec.get("averaging_width_pct_last_built")
    if last_built is not None:
        try:
            if int(max(1, min(100, round(float(last_built))))) != pct_i:
                return None
        except (TypeError, ValueError):
            return None
    else:
        try:
            committed = int(
                max(1, min(100, round(float(ksec.get("averaging_width_pct", 50)))))
            )
        except (TypeError, ValueError):
            committed = 50
        if committed != pct_i:
            return None
    kymo = tifffile.imread(kpath)
    if kymo.ndim > 2:
        kymo = np.squeeze(kymo)
    return kymo


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
            cfg = pipeline_config_flat(str(self.work_dir))
            k = cfg.get("kymograph") or {}
            try:
                pct = float(k.get("averaging_width_pct", 50))
            except (TypeError, ValueError):
                pct = 50.0
            cached = try_load_saved_kymograph(str(self.work_dir), pct)
            if cached is not None:
                self.progress.emit("Loaded saved kymograph…")
                self.done.emit(cached)
                return
            self.progress.emit("Building kymograph...")
            movie_path = get_movie_path(str(self.work_dir))
            movie = tifffile.imread(movie_path)
            if movie.ndim == 2:
                movie = movie[np.newaxis, :, :]
            width = int(movie.shape[2])
            w0, w1 = horizontal_roi_from_averaging_pct(width, pct)
            pct_i = int(max(1, min(100, round(float(pct)))))
            kymo = create_single_kymograph(
                movie,
                width,
                w1 - w0,
                w0,
                w1,
                "Kymograph.tif",
                str(self.work_dir),
                record_averaging_width_pct=pct_i,
            )
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
            movie_path = get_movie_path(str(self.work_dir))
            movie = tifffile.imread(movie_path)
            if movie.ndim == 2:
                movie = movie[np.newaxis, :, :]
            generate_outputs(str(self.work_dir), movie=movie)
            self.done.emit()
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))

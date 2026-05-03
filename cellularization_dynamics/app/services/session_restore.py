"""Reload saved desktop session from apical_alignment v2 (YAML scalars + track/ TSVs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cellularization_dynamics.core.annotation_source import (
    load_apical_session_v2_doc,
    session_front_time_depth,
    time_depth_to_raw_clicks,
)
from cellularization_dynamics.core.track_tabular import read_apical_manual_tsv

from .sample_state import (
    APICAL_MODE_ISLAND,
    APICAL_MODE_MANUAL,
    DEFAULT_MANUAL_SIGMA_UM,
    SampleState,
)

_MISMATCH = "Saved session does not match this kymograph (size changed)."


def track_folder(work_dir: Path) -> Path:
    return work_dir / "track"


def _manual_polyline_raw(
    work_dir: Path, movie_sec: float
) -> list[tuple[float, float]]:
    pts = read_apical_manual_tsv(work_dir)
    if pts is None:
        return []
    time_min, depth_px_raw = pts
    dt_min = max(float(movie_sec) / 60.0, 1e-12)
    cols = np.asarray(time_min, dtype=float) / dt_min
    out: list[tuple[float, float]] = []
    for c, d in zip(cols, np.asarray(depth_px_raw, dtype=float)):
        out.append((float(c), float(d)))
    return out


def _restore_from_v2_doc(state: SampleState, doc: dict) -> tuple[bool, str]:
    h, w = state.kymograph.shape
    if int(doc["kymograph_height_px"]) != int(h) or int(doc["kymograph_width_px"]) != int(w):
        return False, _MISMATCH

    movie_sec = float(doc.get("movie_time_interval_sec", state.acq_params.movie_time_interval_sec))
    state.acq_params.movie_time_interval_sec = movie_sec

    mode = str(doc.get("mode", APICAL_MODE_ISLAND)).strip()
    if mode == APICAL_MODE_MANUAL:
        polyline = _manual_polyline_raw(state.work_dir, movie_sec)
        if len(polyline) < 2:
            return False, _MISMATCH
        state.apply_apical_from_saved(
            mode=APICAL_MODE_MANUAL,
            threshold=float(doc.get("threshold", 0.0)),
            island_labels=[int(x) for x in (doc.get("island_labels") or [])],
            manual_polyline_raw=polyline,
            manual_sigma_um=float(doc.get("manual_sigma_um", DEFAULT_MANUAL_SIGMA_UM)),
        )
    else:
        state.apply_apical_from_saved(
            mode=APICAL_MODE_ISLAND,
            threshold=float(doc["threshold"]),
            island_labels=[int(x) for x in (doc.get("island_labels") or [])],
        )

    if state.shifts is None:
        return False, ""

    time_min, depth_px = session_front_time_depth(doc)
    state.front_points_raw = time_depth_to_raw_clicks(
        time_min,
        depth_px,
        np.asarray(state.shifts, dtype=float),
        w,
        movie_sec,
    )
    state.dirty = False
    return True, "Restored saved session"


def restore_interactive_session(state: SampleState) -> tuple[bool, str]:
    """
    After kymograph is loaded, reapply saved threshold / islands / manual polyline / front
    line when track data matches the current kymograph shape.
    """
    if state.kymograph is None:
        return False, ""

    tf = str(track_folder(state.work_dir))
    doc_v2 = load_apical_session_v2_doc(tf)
    if doc_v2 is None:
        return False, ""
    return _restore_from_v2_doc(state, doc_v2)


def has_restorable_session(work_dir: Path) -> bool:
    return load_apical_session_v2_doc(str(track_folder(work_dir))) is not None

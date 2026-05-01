"""Reload saved desktop session from apical_alignment v2 (YAML scalars + track/apical_front.tsv)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.annotation_source import (
    load_apical_session_v2_doc,
    session_front_time_depth,
    time_depth_to_raw_clicks,
)

from .sample_state import SampleState

_MISMATCH = "Saved session does not match this kymograph (size changed)."


def track_folder(work_dir: Path) -> Path:
    return work_dir / "track"


def _restore_from_v2_doc(state: SampleState, doc: dict) -> tuple[bool, str]:
    h, w = state.kymograph.shape
    if int(doc["kymograph_height_px"]) != int(h) or int(doc["kymograph_width_px"]) != int(w):
        return False, _MISMATCH

    movie_sec = float(doc.get("movie_time_interval_sec", state.acq_params.movie_time_interval_sec))
    state.acq_params.movie_time_interval_sec = movie_sec

    mode = str(doc.get("mode", "longest_run")).strip()
    use_island = mode == "island"
    labels = [int(x) for x in (doc.get("island_labels") or [])]
    state.apply_apical_from_saved(float(doc["threshold"]), use_island, labels)

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
    After kymograph is loaded, reapply saved threshold / islands / front line when
    track data matches the current kymograph shape.
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

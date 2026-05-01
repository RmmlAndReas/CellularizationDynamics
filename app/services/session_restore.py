"""Reload saved desktop session from track/apical_alignment.yaml and/or legacy TSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.annotation_source import (
    has_any_restorable_session,
    load_alignment_yaml_optional,
    load_apical_session_v2_doc,
    load_v2_apical_shell,
    session_front_time_depth,
    time_depth_to_raw_clicks,
    try_load_tsv_time_depth,
    tsv_time_depth_compatible_with_kymograph,
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


def _apply_yaml_apical_and_tsv_line(
    state: SampleState,
    ydoc: dict,
    time_min: np.ndarray,
    depth_px: np.ndarray,
) -> tuple[bool, str]:
    h, w = state.kymograph.shape
    movie_sec = float(ydoc.get("movie_time_interval_sec", state.acq_params.movie_time_interval_sec))
    state.acq_params.movie_time_interval_sec = movie_sec

    if not tsv_time_depth_compatible_with_kymograph(time_min, w, movie_sec):
        return False, _MISMATCH

    if int(ydoc.get("version", 1)) >= 2 and "threshold" in ydoc:
        if int(ydoc["kymograph_height_px"]) != int(h) or int(ydoc["kymograph_width_px"]) != int(w):
            return False, _MISMATCH
        mode = str(ydoc.get("mode", "longest_run")).strip()
        use_island = mode == "island"
        labels = [int(x) for x in (ydoc.get("island_labels") or [])]
        if use_island and not labels:
            use_island = False
        state.apply_apical_from_saved(float(ydoc["threshold"]), use_island, labels)
    else:
        mode = str(ydoc.get("mode", "longest_run")).strip()
        use_island = mode == "island"
        labels = [int(x) for x in (ydoc.get("island_labels") or [])]
        if use_island and not labels:
            use_island = False
        state.apply_apical_from_saved(float(state.threshold), use_island, labels)

    if state.shifts is None:
        return False, ""

    state.front_points_raw = time_depth_to_raw_clicks(
        time_min,
        depth_px,
        np.asarray(state.shifts, dtype=float),
        w,
        float(state.acq_params.movie_time_interval_sec),
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
    if doc_v2 is not None:
        return _restore_from_v2_doc(state, doc_v2)

    td = try_load_tsv_time_depth(tf)
    if td is None:
        return False, ""

    time_min, depth_px = td
    h, w = state.kymograph.shape

    shell = load_v2_apical_shell(tf)
    if shell is not None:
        if int(shell["kymograph_height_px"]) == int(h) and int(shell["kymograph_width_px"]) == int(w):
            return _apply_yaml_apical_and_tsv_line(state, shell, time_min, depth_px)

    ydoc = load_alignment_yaml_optional(tf)
    if ydoc is not None:
        return _apply_yaml_apical_and_tsv_line(state, ydoc, time_min, depth_px)

    movie_sec = float(state.acq_params.movie_time_interval_sec)
    if not tsv_time_depth_compatible_with_kymograph(time_min, w, movie_sec):
        return False, _MISMATCH
    if state.shifts is None:
        return False, ""
    state.front_points_raw = time_depth_to_raw_clicks(
        time_min,
        depth_px,
        np.asarray(state.shifts, dtype=float),
        w,
        movie_sec,
    )
    state.dirty = False
    return True, "Restored saved session"


def has_restorable_session(work_dir: Path) -> bool:
    return has_any_restorable_session(str(track_folder(work_dir)))

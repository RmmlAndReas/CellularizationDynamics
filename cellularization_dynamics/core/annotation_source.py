"""
Load cellularization front annotation: scalars in ``config.yaml`` (``apical_alignment``),
polyline in ``track/apical_front.tsv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

def _work_dir_from_track(track_folder: str) -> str:
    return str(Path(track_folder).resolve().parent)


def load_apical_alignment_doc(track_folder: str) -> dict[str, Any] | None:
    """
    ``apical_alignment`` from unified config, with ``front_points`` materialized from
    ``track/apical_front.tsv`` when not stored inline (after externalization).
    """
    wd = _work_dir_from_track(track_folder)
    from .track_tabular import read_apical_front_tsv
    from .work_state import load_state

    state = load_state(wd, migrate_if_needed=True)
    doc = state.get("apical_alignment")
    if not isinstance(doc, dict) or not doc:
        return None
    out = dict(doc)
    if not out.get("front_points"):
        td = read_apical_front_tsv(wd)
        if td is not None:
            tm, dp = td
            out["front_points"] = time_depth_to_yaml_front_points(tm, dp)
    return out


def raw_clicks_to_time_depth(
    points_raw: np.ndarray,
    shifts: np.ndarray | None,
    num_cols: int,
    time_interval_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match desktop save semantics: Time in minutes, Depth in straightened kymograph px."""
    order = np.argsort(points_raw[:, 0])
    pts = points_raw[order].astype(float)
    time_min = pts[:, 0] * float(time_interval_min)
    depth_px = pts[:, 1].copy()
    if shifts is not None and num_cols > 0:
        for i in range(depth_px.shape[0]):
            x_idx = int(np.clip(np.rint(float(pts[i, 0])), 0, num_cols - 1))
            depth_px[i] = float(depth_px[i]) + float(shifts[x_idx])
    return time_min, depth_px


def time_depth_to_yaml_front_points(
    time_min: np.ndarray,
    depth_px: np.ndarray,
) -> list[list[float]]:
    return [[float(t), float(d)] for t, d in zip(time_min, depth_px)]


def build_apical_alignment_v2(
    *,
    mode: str,
    island_labels: list[int],
    threshold: float,
    kymograph_shape: tuple[int, int],
    movie_time_interval_sec: float,
    manual_sigma_um: float | None = None,
) -> dict[str, Any]:
    """Scalar fields only; write polyline with ``persist_apical_alignment`` / ``write_apical_front_tsv``."""
    h, w = kymograph_shape
    doc: dict[str, Any] = {
        "version": 2,
        "mode": mode,
        "island_labels": island_labels,
        "threshold": float(threshold),
        "kymograph_height_px": int(h),
        "kymograph_width_px": int(w),
        "movie_time_interval_sec": float(movie_time_interval_sec),
    }
    if manual_sigma_um is not None:
        doc["manual_sigma_um"] = float(manual_sigma_um)
    return doc


def persist_apical_alignment(
    work_dir: str | Path,
    alignment_scalars: dict[str, Any],
    time_min: np.ndarray,
    depth_px: np.ndarray,
) -> None:
    """Write ``track/apical_front.tsv`` and merge ``apical_alignment`` scalars into config."""
    from .track_tabular import write_apical_front_tsv
    from .work_state import merge_patch

    write_apical_front_tsv(work_dir, time_min, depth_px)
    merge_patch(work_dir, {"apical_alignment": dict(alignment_scalars)})


def _front_points_from_yaml_value(raw: Any) -> np.ndarray:
    if not raw:
        raise ValueError("front_points missing or empty in apical_alignment")
    rows: list[tuple[float, float]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            rows.append((float(item[0]), float(item[1])))
        elif isinstance(item, dict):
            t = item.get("time_min", item.get("Time"))
            d = item.get("depth_px", item.get("Depth"))
            if t is None or d is None:
                raise ValueError(f"Invalid front_points entry: {item!r}")
            rows.append((float(t), float(d)))
        else:
            raise ValueError(f"Invalid front_points entry: {item!r}")
    return np.asarray(rows, dtype=float)


def load_annotation_time_depth(track_folder: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (time_min, depth_px_straightened) sorted by time from v2 apical_alignment
    and ``track/apical_front.tsv`` (or legacy inline ``front_points``).
    """
    doc = load_apical_alignment_doc(track_folder)
    if not doc:
        raise FileNotFoundError(
            f"No apical_alignment in config for work dir parent of {track_folder!r}"
        )
    ver = int(doc.get("version", 1))
    fp = doc.get("front_points")
    if fp is None or ver < 2:
        wd = _work_dir_from_track(track_folder)
        raise FileNotFoundError(
            f"Need apical_alignment v2 and track/apical_front.tsv (or inline front_points); "
            f"version={ver!r}, front_points={'set' if fp else 'missing'}, work_dir={wd!r}"
        )
    arr = _front_points_from_yaml_value(fp)
    if arr.shape[0] < 2:
        raise ValueError("Need at least two front_points in apical_alignment")
    time_min = arr[:, 0]
    depth_px = arr[:, 1]
    order = np.argsort(time_min)
    return time_min[order], depth_px[order]


def session_front_time_depth(doc: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Sorted Time (min) and Depth (straight px) from a v2 session document."""
    arr = _front_points_from_yaml_value(doc["front_points"])
    order = np.argsort(arr[:, 0])
    return arr[order, 0].copy(), arr[order, 1].copy()


def load_apical_session_v2_doc(track_folder: str) -> dict[str, Any] | None:
    """
    Return the alignment document if it contains a restorable desktop session.

    Accepts ``mode in {"island", "manual"}``:
      - island: requires ``threshold`` and non-empty ``island_labels``.
      - manual: requires ``track/apical_manual.tsv`` with at least two points.

    Both modes require the v2 cellularization front (``front_points`` or the
    externalized ``apical_front.tsv``).
    """
    doc = load_apical_alignment_doc(track_folder)
    if not doc:
        return None
    if int(doc.get("version", 1)) < 2:
        return None
    fp = doc.get("front_points")
    if not fp:
        return None
    try:
        arr = _front_points_from_yaml_value(fp)
    except ValueError:
        return None
    if arr.shape[0] < 2:
        return None
    mode = str(doc.get("mode", "island")).strip()
    if mode == "island":
        if "threshold" not in doc:
            return None
        raw_labels = doc.get("island_labels") or []
        if not raw_labels:
            return None
        return doc
    if mode == "manual":
        from .track_tabular import read_apical_manual_tsv

        wd = _work_dir_from_track(track_folder)
        pts = read_apical_manual_tsv(wd)
        if pts is None:
            return None
        return doc
    return None


def time_depth_to_raw_clicks(
    time_min: np.ndarray,
    depth_px: np.ndarray,
    shifts: np.ndarray,
    num_cols: int,
    movie_time_interval_sec: float,
) -> list[tuple[float, float]]:
    """Invert raw_clicks_to_time_depth: (col_index, y_raw) for SampleState.front_points_raw."""
    dt_min = max(float(movie_time_interval_sec) / 60.0, 1e-12)
    order = np.argsort(time_min.astype(float))
    out: list[tuple[float, float]] = []
    nc = max(int(num_cols), 1)
    for i in order:
        t = float(time_min[i])
        d = float(depth_px[i])
        x_idx = int(np.clip(np.rint(t / dt_min), 0, nc - 1))
        y_raw = d - float(shifts[x_idx])
        out.append((float(x_idx), float(y_raw)))
    return out

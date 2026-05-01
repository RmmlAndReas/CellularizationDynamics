"""
Load/save cellularization front annotation for the desktop and fit script.

Desktop Save writes ``track/apical_alignment.yaml`` (version >= 2) with embedded
``front_points`` and also writes ``VerticalKymoCelluSelection.tsv`` (Time, Depth)
for Snakemake / legacy compatibility.

Restore prefers full v2 YAML; otherwise applies apical metadata from YAML (v2 or v1)
when present and loads the polyline from the TSV.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import yaml

ALIGNMENT_FILENAME = "apical_alignment.yaml"
LEGACY_TSV = "VerticalKymoCelluSelection.tsv"


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
    time_min: np.ndarray,
    depth_px: np.ndarray,
) -> dict[str, Any]:
    h, w = kymograph_shape
    return {
        "version": 2,
        "mode": mode,
        "island_labels": island_labels,
        "threshold": float(threshold),
        "kymograph_height_px": int(h),
        "kymograph_width_px": int(w),
        "movie_time_interval_sec": float(movie_time_interval_sec),
        "front_points": time_depth_to_yaml_front_points(time_min, depth_px),
    }


def _front_points_from_yaml_value(raw: Any) -> np.ndarray:
    if not raw:
        raise ValueError("front_points missing or empty in apical_alignment.yaml")
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
    Return (time_min, depth_px_straightened) sorted by time.
    Prefers apical_alignment.yaml v2+ with front_points; else legacy TSV.
    """
    align_path = os.path.join(track_folder, ALIGNMENT_FILENAME)
    tsv_path = os.path.join(track_folder, LEGACY_TSV)

    if os.path.isfile(align_path):
        with open(align_path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        ver = int(doc.get("version", 1))
        fp = doc.get("front_points")
        if fp is not None and ver >= 2:
            arr = _front_points_from_yaml_value(fp)
            if arr.shape[0] < 2:
                raise ValueError("Need at least two front_points in apical_alignment.yaml")
            time_min = arr[:, 0]
            depth_px = arr[:, 1]
            order = np.argsort(time_min)
            return time_min[order], depth_px[order]

    if os.path.isfile(tsv_path):
        data = np.loadtxt(tsv_path, delimiter="\t", skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        time_min = data[:, 0]
        depth_px = data[:, 1]
        order = np.argsort(time_min)
        return time_min[order], depth_px[order]

    raise FileNotFoundError(
        f"No annotation found in {track_folder!r}: need {ALIGNMENT_FILENAME} (v2+ with "
        f"front_points) or {LEGACY_TSV}."
    )


def session_front_time_depth(doc: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Sorted Time (min) and Depth (straight px) from a v2 session document."""
    arr = _front_points_from_yaml_value(doc["front_points"])
    order = np.argsort(arr[:, 0])
    return arr[order, 0].copy(), arr[order, 1].copy()


def load_apical_session_v2_doc(track_folder: str) -> dict[str, Any] | None:
    """
    Return the alignment YAML document if it contains a restorable desktop session
    (version >= 2, threshold, front line, valid island labels when mode is island).
    """
    align_path = os.path.join(track_folder, ALIGNMENT_FILENAME)
    if not os.path.isfile(align_path):
        return None
    with open(align_path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    if int(doc.get("version", 1)) < 2:
        return None
    if "threshold" not in doc:
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
    mode = str(doc.get("mode", "longest_run")).strip()
    if mode not in ("longest_run", "island"):
        return None
    if mode == "island":
        raw_labels = doc.get("island_labels") or []
        if not raw_labels:
            return None
    return doc


def write_annotation_tsv(tsv_path: str, time_min: np.ndarray, depth_px: np.ndarray) -> None:
    """Write Time (min) / Depth (straight px) annotation file."""
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("Time\tDepth\n")
        for t, d in zip(time_min, depth_px):
            f.write(f"{float(t):.6f}\t{float(d):.6f}\n")


def load_alignment_yaml_optional(track_folder: str) -> dict[str, Any] | None:
    path = os.path.join(track_folder, ALIGNMENT_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def try_load_tsv_time_depth(track_folder: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load (time_min, depth_px) from legacy TSV, or None if missing/invalid."""
    tsv_path = os.path.join(track_folder, LEGACY_TSV)
    if not os.path.isfile(tsv_path):
        return None
    try:
        data = np.loadtxt(tsv_path, delimiter="\t", skiprows=1)
    except OSError:
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] < 2 or data.shape[1] < 2:
        return None
    time_min = data[:, 0].astype(float)
    depth_px = data[:, 1].astype(float)
    order = np.argsort(time_min)
    return time_min[order].copy(), depth_px[order].copy()


def tsv_time_depth_compatible_with_kymograph(
    time_min: np.ndarray,
    kymograph_width: int,
    movie_time_interval_sec: float,
) -> bool:
    """Heuristic: inferred max column index must fit the kymograph width."""
    w = max(int(kymograph_width), 1)
    dt_min = max(float(movie_time_interval_sec) / 60.0, 1e-12)
    max_idx = int(np.max(np.rint(time_min / dt_min))) if time_min.size else -1
    return max_idx < w + 2


def tsv_annotation_point_count(track_folder: str) -> int:
    td = try_load_tsv_time_depth(track_folder)
    if td is None:
        return 0
    return int(td[0].shape[0])


def load_v2_apical_shell(track_folder: str) -> dict[str, Any] | None:
    """
    v2 document with threshold + kymograph shape + mode/islands, possibly without
    ``front_points`` (line supplied via TSV).
    """
    doc = load_alignment_yaml_optional(track_folder)
    if not doc:
        return None
    if int(doc.get("version", 1)) < 2:
        return None
    if "threshold" not in doc:
        return None
    if "kymograph_height_px" not in doc or "kymograph_width_px" not in doc:
        return None
    mode = str(doc.get("mode", "longest_run")).strip()
    if mode not in ("longest_run", "island"):
        return None
    if mode == "island" and not (doc.get("island_labels") or []):
        return None
    return doc


def has_any_restorable_session(track_folder: str) -> bool:
    """True if auto-analyze + restore can run: full v2, v2 shell+TSV, v1+TSV, or TSV-only."""
    if load_apical_session_v2_doc(track_folder) is not None:
        return True
    if tsv_annotation_point_count(track_folder) < 2:
        return False
    if load_v2_apical_shell(track_folder) is not None:
        return True
    if load_alignment_yaml_optional(track_folder) is not None:
        return True
    return True


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

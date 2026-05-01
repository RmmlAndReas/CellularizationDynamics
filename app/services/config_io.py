from __future__ import annotations

from pathlib import Path
import os
import sys
from typing import Any

import numpy as np

# Ensure repo root and core are importable when running as app package
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_CORE = _REPO_ROOT / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from work_state import (  # noqa: E402
    deep_merge,
    default_v2_shell,
    load_state,
    merge_patch,
    normalize_state,
    save_state,
)


def save_apical_alignment(
    work_dir: Path,
    alignment: dict,
    time_min: np.ndarray,
    depth_px: np.ndarray,
) -> None:
    from annotation_source import persist_apical_alignment

    persist_apical_alignment(work_dir, alignment, time_min, depth_px)


def merge_kymograph_fields(work_dir: Path, **fields: Any) -> None:
    """Deep-merge scalar fields into the ``kymograph`` section of config.yaml."""
    merge_patch(work_dir, {"kymograph": dict(fields)})


def merge_visualization_fields(work_dir: Path, **fields: Any) -> None:
    """Deep-merge fields into the ``visualization`` section of config.yaml."""
    merge_patch(work_dir, {"visualization": dict(fields)})


def read_kymograph_brightness(work_dir: Path, default: float = 1.0) -> float:
    """Saved GUI kymograph brightness (0.2–3.0), for export parity with the desktop."""
    state = load_state(work_dir, migrate_if_needed=True)
    vis = state.get("visualization")
    if not isinstance(vis, dict):
        return default
    v = vis.get("kymograph_brightness")
    if v is None:
        return default
    try:
        return max(0.2, min(3.0, float(v)))
    except (TypeError, ValueError):
        return default


def read_averaging_width_pct(work_dir: Path, default: int = 50) -> int:
    """``kymograph.averaging_width_pct`` clamped to 1–100."""
    state = load_state(work_dir, migrate_if_needed=True)
    k = state.get("kymograph") or {}
    v = k.get("averaging_width_pct", default)
    try:
        return int(max(1, min(100, round(float(v)))))
    except (TypeError, ValueError):
        return default


def read_averaging_width_pct_last_built(work_dir: Path) -> int | None:
    """``kymograph.averaging_width_pct_last_built`` if set (width % used when ``Kymograph.tif`` was written)."""
    state = load_state(work_dir, migrate_if_needed=True)
    k = state.get("kymograph") or {}
    v = k.get("averaging_width_pct_last_built")
    if v is None:
        return None
    try:
        return int(max(1, min(100, round(float(v)))))
    except (TypeError, ValueError):
        return None


def read_averaging_width_pct_for_ui(work_dir: Path) -> int:
    lb = read_averaging_width_pct_last_built(work_dir)
    if lb is not None:
        return lb
    return read_averaging_width_pct(work_dir)


def build_runtime_config(params: dict) -> dict:
    px2micron = float(params["px2micron"])
    movie_time_interval_sec = float(params["movie_time_interval_sec"])
    smoothing = float(params.get("smoothing", 0.0))
    degree = int(params.get("degree", 1))
    time_interval_sec = movie_time_interval_sec

    return {
        "schema_version": 2,
        "acquisition": {
            "px2micron": px2micron,
            "movie_time_interval_sec": movie_time_interval_sec,
        },
        "spline": {
            "smoothing": smoothing,
            "degree": degree,
        },
        "kymograph": {
            "time_interval_sec": float(time_interval_sec),
        },
    }


def load_or_create_config(
    work_dir: Path,
    params: dict,
    *,
    source_movie: str | os.PathLike[str] | None = None,
    averaging_width_pct: int | None = None,
) -> dict:
    """
    Merge GUI acquisition params into unified config.yaml (schema v2).
    Optionally set acquisition.source_movie (absolute path to raw TIFF).
    """
    config_path = work_dir / "config.yaml"
    cfg: dict = {}
    if config_path.exists():
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    runtime = build_runtime_config(params)
    merged = deep_merge(default_v2_shell(), cfg)
    merged = deep_merge(merged, runtime)
    if averaging_width_pct is not None:
        merged["kymograph"] = deep_merge(
            merged.get("kymograph") or {},
            {
                "averaging_width_pct": int(
                    max(1, min(100, round(float(averaging_width_pct))))
                )
            },
        )
    if source_movie is not None:
        merged = deep_merge(
            merged,
            {"acquisition": {"source_movie": str(os.path.abspath(str(source_movie)))}},
        )

    merged.pop("time_window", None)
    # Keep visualization (e.g. kymograph_brightness) for export/GUI parity.
    prep = merged.get("preprocessing")
    if isinstance(prep, dict):
        prep = {k: v for k, v in prep.items() if k != "keep_every"}
        if prep:
            merged["preprocessing"] = prep
        else:
            merged.pop("preprocessing", None)

    save_state(work_dir, merged)
    return normalize_state(merged)


def save_config(work_dir: Path, config: dict) -> None:
    save_state(work_dir, config)

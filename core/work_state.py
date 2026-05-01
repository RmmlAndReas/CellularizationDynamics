"""
Single unified config.yaml (schema v2) per work directory.

All sample YAML state lives in work_dir/config.yaml. Legacy track/*.yaml files
are migrated on first load and then removed.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 2

CONFIG_FILENAME = "config.yaml"

# Final MP4 with front markers (replaces Cellularization_trimmed_delta.mp4).
FRONT_MARKERS_MP4 = "Cellularization_front_markers.mp4"


def config_path(work_dir: str | Path) -> Path:
    return Path(work_dir).resolve() / CONFIG_FILENAME


def default_v2_shell() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "acquisition": {
            "source_movie": "",
            "px2micron": 1.0,
            "movie_time_interval_sec": 10.0,
        },
        "kymograph": {"time_interval_sec": 10.0, "averaging_width_pct": 50},
        "spline": {"smoothing": 0.0, "degree": 1},
        "apical_alignment": {},
        "straightening": {},
        "spline_fit": {},
        "derived": {},
    }


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
            and not isinstance(v, type(None))
        ):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def normalize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Ensure v2 shape; lift legacy top-level keys into nested sections."""
    s = deep_merge(default_v2_shell(), {k: v for k, v in state.items() if k != "manual"})
    s["schema_version"] = SCHEMA_VERSION

    manual = state.get("manual")
    if isinstance(manual, dict):
        if "px2micron" in manual:
            s["acquisition"]["px2micron"] = float(manual["px2micron"])
        if "movie_time_interval_sec" in manual:
            s["acquisition"]["movie_time_interval_sec"] = float(manual["movie_time_interval_sec"])

    for key in ("apical_detection", "cellularization_front"):
        if key in state and key not in (s.get("derived") or {}):
            s.setdefault("derived", {})
            if isinstance(state[key], dict):
                s["derived"][key] = state[key]

    for sec in ("kymograph", "spline", "apical_alignment", "straightening", "spline_fit", "derived"):
        if sec in state and isinstance(state[sec], dict):
            s[sec] = deep_merge(s.get(sec) or {}, state[sec])

    return s


def _migrate_from_legacy_files(work_dir: str, raw_config: dict[str, Any]) -> dict[str, Any]:
    """Build v2 state from v1 config.yaml body + optional track YAMLs."""
    state = default_v2_shell()
    if isinstance(raw_config, dict) and raw_config:
        if "manual" in raw_config:
            m = raw_config["manual"]
            if isinstance(m, dict):
                if "px2micron" in m:
                    state["acquisition"]["px2micron"] = float(m["px2micron"])
                if "movie_time_interval_sec" in m:
                    state["acquisition"]["movie_time_interval_sec"] = float(m["movie_time_interval_sec"])
        if isinstance(raw_config.get("kymograph"), dict):
            state["kymograph"] = deep_merge(state["kymograph"], raw_config["kymograph"])
        if isinstance(raw_config.get("spline"), dict):
            state["spline"] = deep_merge(state["spline"], raw_config["spline"])
        for key in ("apical_detection", "cellularization_front"):
            if isinstance(raw_config.get(key), dict):
                state.setdefault("derived", {})
                state["derived"][key] = raw_config[key]

    trimmed = os.path.join(work_dir, "Cellularization_trimmed.tif")
    if os.path.isfile(trimmed):
        state["acquisition"]["source_movie"] = os.path.abspath(trimmed)

    track = os.path.join(work_dir, "track")
    ap_path = os.path.join(track, "apical_alignment.yaml")
    if os.path.isfile(ap_path):
        with open(ap_path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if isinstance(doc, dict):
            state["apical_alignment"] = doc
    st_path = os.path.join(track, "straighten_metadata.yaml")
    if os.path.isfile(st_path):
        with open(st_path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if isinstance(doc, dict):
            state["straightening"] = doc
    md_path = os.path.join(track, "metadata.yaml")
    if os.path.isfile(md_path):
        with open(md_path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if isinstance(doc, dict):
            state["spline_fit"] = doc

    for key in ("time_window", "visualization"):
        if key in raw_config and raw_config[key] is not None:
            state[key] = raw_config[key]

    return normalize_state(state)


def _remove_legacy_track_yamls(work_dir: str) -> None:
    track = os.path.join(work_dir, "track")
    for name in ("apical_alignment.yaml", "straighten_metadata.yaml", "metadata.yaml"):
        p = os.path.join(track, name)
        if os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass


def atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(
        data, default_flow_style=False, sort_keys=False, allow_unicode=True
    )
    fd, tmp = tempfile.mkstemp(
        suffix=".yaml.tmp", prefix=path.name + ".", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_state(work_dir: str | Path, state: dict[str, Any]) -> None:
    """Write full v2 state atomically."""
    s = normalize_state(state)
    s["schema_version"] = SCHEMA_VERSION
    atomic_write_yaml(config_path(work_dir), s)


def merge_patch(work_dir: str | Path, patch: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge patch into current state and save."""
    current = load_state(work_dir, migrate_if_needed=True)
    merged = deep_merge(current, patch)
    save_state(work_dir, merged)
    return merged


def _maybe_externalize_sequential_to_tsv(wd: str, norm: dict[str, Any]) -> bool:
    """Write inline front_points / straightening arrays to ``track/*.tsv``; strip from ``norm``."""
    from track_tabular import (
        externalize_apical_front_if_inline,
        externalize_straightening_if_inline,
    )

    changed = False
    changed |= externalize_apical_front_if_inline(wd, norm)
    changed |= externalize_straightening_if_inline(wd, norm)
    return changed


def load_state(work_dir: str | Path, *, migrate_if_needed: bool = True) -> dict[str, Any]:
    """
    Load unified state; migrate legacy multi-file layout to v2 on first read.
    """
    wd = os.path.abspath(str(work_dir))
    cp = config_path(wd)
    if not cp.is_file():
        return normalize_state(default_v2_shell())

    with cp.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    ver = int(raw.get("schema_version", 0) or 0)
    if ver >= SCHEMA_VERSION:
        norm = normalize_state(raw)
        _remove_legacy_track_yamls(wd)
        if _maybe_externalize_sequential_to_tsv(wd, norm):
            save_state(wd, norm)
        return norm

    if not migrate_if_needed:
        norm = normalize_state(_migrate_from_legacy_files(wd, raw))
        _maybe_externalize_sequential_to_tsv(wd, norm)
        return norm

    migrated = _migrate_from_legacy_files(wd, raw)
    _maybe_externalize_sequential_to_tsv(wd, migrated)
    save_state(wd, migrated)
    _remove_legacy_track_yamls(wd)
    return migrated


def pipeline_config_flat(work_dir: str | Path) -> dict[str, Any]:
    """
    Legacy-shaped dict: manual, kymograph, spline, apical_detection,
    cellularization_front — for core scripts that expect old config.yaml layout.
    """
    state = load_state(work_dir, migrate_if_needed=True)
    acq = state.get("acquisition") or {}
    out: dict[str, Any] = {
        "manual": {
            "px2micron": float(acq.get("px2micron", 1.0)),
            "movie_time_interval_sec": float(acq.get("movie_time_interval_sec", 10.0)),
        },
        "kymograph": dict(state.get("kymograph") or {}),
        "spline": dict(state.get("spline") or {}),
    }
    derived = state.get("derived") or {}
    if isinstance(derived.get("apical_detection"), dict):
        out["apical_detection"] = derived["apical_detection"]
    if isinstance(derived.get("cellularization_front"), dict):
        out["cellularization_front"] = derived["cellularization_front"]
    return out


def get_movie_path(work_dir: str | Path) -> str:
    """Absolute path to source timelapse TIFF."""
    state = load_state(work_dir, migrate_if_needed=True)
    acq = state.get("acquisition") or {}
    p = (acq.get("source_movie") or "").strip()
    if p and os.path.isfile(p):
        return os.path.abspath(p)
    trimmed = os.path.join(str(work_dir), "Cellularization_trimmed.tif")
    if os.path.isfile(trimmed):
        return os.path.abspath(trimmed)
    raise FileNotFoundError(
        "acquisition.source_movie is not set or file is missing, and no "
        "Cellularization_trimmed.tif fallback was found. "
        "Run Analyze or set source_movie in config.yaml."
    )


def set_source_movie(work_dir: str | Path, absolute_movie_path: str) -> None:
    merge_patch(
        work_dir,
        {"acquisition": {"source_movie": os.path.abspath(absolute_movie_path)}},
    )


def straightening_meta(work_dir: str | Path) -> dict[str, Any]:
    """Straightening metadata dict; loads ``shifts`` / ``apical_px_by_col`` from TSV when not inline."""
    import numpy as np

    from track_tabular import read_straightening_columns_tsv

    state = load_state(work_dir, migrate_if_needed=True)
    sec = dict(state.get("straightening") or {})
    if "shifts" in sec and "apical_px_by_col" in sec:
        return sec
    got = read_straightening_columns_tsv(work_dir)
    if got is not None:
        sh, ap = got
        sec["shifts"] = [int(x) for x in sh.tolist()]
        sec["apical_px_by_col"] = [None if np.isnan(x) else float(x) for x in ap.tolist()]
    return sec


def spline_fit_meta(work_dir: str | Path) -> dict[str, Any]:
    """Contents of former track/metadata.yaml under unified spline_fit."""
    state = load_state(work_dir, migrate_if_needed=True)
    return dict(state.get("spline_fit") or {})

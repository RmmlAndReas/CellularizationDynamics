#!/usr/bin/env python3
"""
Generate all final outputs for the cellularization pipeline from data products.

Inputs (per sample folder):
    - config.yaml (v2 unified; acquisition.source_movie for MP4)
    - track/Kymograph.tif
    - track/VerticalKymoCelluSelection_spline.tsv
    - output.csv (geometry timeseries; merged cytoplasm + front), or legacy track/geometry_timeseries.csv

Outputs:
    - Cellularization.png
    - Cellularization.pdf
        Straightened kymograph aligned so apical border = 0 µm, with:
          - apical/basal cytoplasm borders
          - cellularization front spline
          - no milestone triangle markers
    - Cellularization_front_markers.mp4
        H.264 preview of the acquisition movie with front markers.

This script combines core logic from:
    - create_kymographs.py
    - mark_delta_on_trimmed_movie.py
    - mark_kymograph_progress.py (style, milestones)
    - tmp/detect_yolk_border.py (straightened overlay geometry)
"""

import argparse
import csv
import os
import subprocess

from .work_state import (
    FRONT_MARKERS_MP4,
    get_movie_path,
    load_state,
    pipeline_config_flat,
    straightening_meta,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tifffile

from . import pipeline_diag


def format_experiment_hms(total_seconds: int) -> str:
    """HH:MM:SS from elapsed experiment time in whole seconds (may exceed 24 h)."""
    if total_seconds < 0:
        total_seconds = 0
    h, rem = divmod(int(total_seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def draw_timestamp_bottom_left_rgb(
    rgb: np.ndarray,
    text: str,
    *,
    color_rgb: tuple[int, int, int],
    size_px: int,
) -> None:
    """Draw ``text`` on uint8 RGB [H,W,3] in place, bottom-left (OpenCV)."""
    h, _w, _c = rgb.shape
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, float(size_px) / 22.0)
    thickness = max(1, int(round(float(size_px) / 14.0)))
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
    (_tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    margin = 8
    x = margin
    y = h - margin - baseline
    if y < th + margin:
        y = th + margin
    cv2.putText(bgr, text, (x, y), font, font_scale, color_bgr, thickness, cv2.LINE_AA)
    rgb[:] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def burn_timestamp_on_u16_rgb_frame(
    frame_u16: np.ndarray,
    text: str,
    *,
    color_rgb: tuple[int, int, int],
    size_px: int,
) -> None:
    """Burn timestamp into uint16 RGB frame in place (matches MP4 encode scaling)."""
    u8 = np.clip(frame_u16.astype(np.float32) / 257.0, 0.0, 255.0).astype(np.uint8).copy()
    draw_timestamp_bottom_left_rgb(u8, text, color_rgb=color_rgb, size_px=size_px)
    frame_u16[:] = np.minimum(65535, u8.astype(np.uint32) * 257).astype(np.uint16)


# Match PDF/text settings used previously for Affinity compatibility
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

FONT_SIZE_LABEL = 30
FONT_SIZE_TICKS = 23
SPINE_LINEWIDTH = 3
TICK_MAJOR_LENGTH = 8
TICK_MAJOR_WIDTH = 3
APICAL_HEADROOM_UM = 2.0


def _movie_intensity_percentiles(
    movie: np.ndarray, p_lo: float = 2.0, p_hi: float = 99.5
) -> tuple[float, float]:
    """Robust global intensity bounds for autoscaling the acquisition movie (subsampled)."""
    m = np.asarray(movie)
    n = int(m.size)
    stride = max(1, n // 2_000_000)
    flat = m.ravel()[::stride].astype(np.float64, copy=False)
    lo, hi = np.percentile(flat, [p_lo, p_hi])
    lo_f, hi_f = float(lo), float(hi)
    if hi_f <= lo_f + 1e-12:
        lo_f, hi_f = float(np.min(m)), float(np.max(m))
        if hi_f <= lo_f:
            hi_f = lo_f + 1.0
    return lo_f, hi_f


def _autoscale_acquisition_movie_gray_u16(
    movie: np.ndarray, kymograph_brightness: float
) -> np.ndarray:
    """
    Map acquisition stack to uint16 grayscale for MP4 encoding.

    Uses global percentile autoscale (2nd–99.5th by default), then multiplies by
    ``kymograph_brightness`` in the same sense as the kymograph GUI (>1 brightens).
    """
    lo, hi = _movie_intensity_percentiles(movie)
    m = (movie.astype(np.float64) - lo) / (hi - lo)
    m = np.clip(m, 0.0, 1.0)
    b = max(0.2, min(3.0, float(kymograph_brightness)))
    m = np.clip(m * b, 0.0, 1.0)
    print(
        f"  Movie autoscale: intensity {lo:.6g} .. {hi:.6g} "
        f"(saved kymograph_brightness ×{b:.2g} applied after linear stretch)"
    )
    return (m * 65535.0).astype(np.uint16)


def _rgb_u8_to_u16(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return (
        int(round(r * 65535 / 255)),
        int(round(g * 65535 / 255)),
        int(round(b * 65535 / 255)),
    )


def _draw_dashed_horizontal_u16(
    rgb_frame: np.ndarray,
    y: float,
    color_rgb_u16: tuple[int, int, int],
    *,
    dash_px: int = 14,
    gap_px: int = 8,
    thickness_px: int = 2,
) -> None:
    """Draw a full-width dashed horizontal line on a uint16 RGB frame [H,W,3]."""
    h, w, _c = rgb_frame.shape
    y0 = int(np.clip(int(round(y)), 0, h - 1))
    r16, g16, b16 = color_rgb_u16
    x = 0
    draw_segment = True
    while x < w:
        seg = dash_px if draw_segment else gap_px
        x_end = min(w, x + seg)
        if draw_segment:
            for xi in range(x, x_end):
                for dy in range(-(thickness_px // 2), thickness_px - (thickness_px // 2)):
                    yy = y0 + dy
                    if 0 <= yy < h:
                        rgb_frame[yy, xi, 0] = r16
                        rgb_frame[yy, xi, 1] = g16
                        rgb_frame[yy, xi, 2] = b16
        x = x_end
        draw_segment = not draw_segment


def _kymograph_limits_like_front_panel(raw_kymo: np.ndarray, brightness: float) -> tuple[float, float]:
    """
    Match ``FrontPanel`` / ``ThresholdPanel``: 1st–99th percentile on raw kymograph,
    then vmax scaled by brightness (display factor ≥ 0.2).
    """
    b = max(0.2, min(3.0, float(brightness)))
    dmin = float(np.min(raw_kymo))
    dmax = float(np.max(raw_kymo))
    base_vmin = float(np.percentile(raw_kymo, 1))
    base_vmax = float(np.percentile(raw_kymo, 99))
    if base_vmax <= base_vmin:
        base_vmin = dmin
        base_vmax = dmax
        if base_vmax <= base_vmin:
            base_vmax = base_vmin + 1.0
    vmax = base_vmin + (base_vmax - base_vmin) / max(b, 1e-6)
    return base_vmin, vmax


def load_config(folder):
    """Load unified config.yaml (v2) and return commonly used fields."""
    config_path = os.path.join(folder, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found: {config_path}")
    cfg = pipeline_config_flat(folder)
    state = load_state(folder, migrate_if_needed=True)

    if "manual" not in cfg:
        raise ValueError("config.yaml must contain acquisition / manual fields")
    manual = cfg["manual"]
    if "px2micron" not in manual:
        raise ValueError("manual.px2micron missing in config.yaml")

    movie_time_interval_sec = float(manual.get("movie_time_interval_sec", 10.0))

    if "kymograph" not in cfg:
        raise ValueError("'kymograph' section not found in config.yaml")
    kymo_cfg = cfg["kymograph"]
    time_interval_sec = float(
        kymo_cfg.get("time_interval_sec", kymo_cfg.get("time_interval_min", 1.0) * 60.0)
    )

    if (
        "apical_detection" not in cfg
        or "apical_height_px" not in cfg["apical_detection"]
    ):
        raise ValueError("apical_detection.apical_height_px missing in config.yaml")
    apical_height_px = float(cfg["apical_detection"]["apical_height_px"])

    if (
        "cellularization_front" not in cfg
        or "final_height_px" not in cfg["cellularization_front"]
    ):
        raise ValueError("cellularization_front.final_height_px missing in config.yaml")
    final_height_px = float(cfg["cellularization_front"]["final_height_px"])

    basal_smooth_window = 1
    kymo_marked_start_pct = 70
    kymograph_brightness = 1.0
    vis = state.get("visualization")
    if isinstance(vis, dict):
        basal_smooth_window = int(vis.get("basal_smooth_window", 1))
        kymo_marked_start_pct = int(vis.get("kymo_marked_start_pct", 70))
        try:
            kb = vis.get("kymograph_brightness")
            if kb is not None:
                kymograph_brightness = float(kb)
        except (TypeError, ValueError):
            pass
    kymograph_brightness = max(0.2, min(3.0, float(kymograph_brightness)))

    time_window = None
    tw = state.get("time_window")
    if isinstance(tw, dict):
        try:
            def _tw_float(key: str):
                if key not in tw or tw[key] is None:
                    return None
                return float(tw[key])

            start_min = _tw_float("start_min")
            end_min = _tw_float("end_min")
            if start_min is not None or end_min is not None:
                time_window = {"start_min": start_min, "end_min": end_min}
        except Exception as e:
            print(f"Warning: invalid time_window in config.yaml: {e}")

    return {
        "px2micron": float(manual["px2micron"]),
        "movie_time_interval_sec": movie_time_interval_sec,
        "kymograph_time_interval_sec": time_interval_sec,
        "apical_height_px": apical_height_px,
        "final_height_px": final_height_px,
        "time_window": time_window,
        "basal_smooth_window": basal_smooth_window,
        "kymo_marked_start_pct": kymo_marked_start_pct,
        "kymograph_brightness": kymograph_brightness,
    }


def load_spline(folder):
    """Load cellularization front spline TSV."""
    track_folder = os.path.join(folder, "track")
    spline_path = os.path.join(track_folder, "VerticalKymoCelluSelection_spline.tsv")
    if not os.path.exists(spline_path):
        raise FileNotFoundError(f"Spline TSV not found: {spline_path}")
    data = np.loadtxt(spline_path, delimiter="\t", skiprows=1)
    time_min = data[:, 0]
    front_px = data[:, 1]
    return time_min, front_px


def _parse_csv_float(cell):
    if cell is None:
        return np.nan
    s = str(cell).strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_csv_int(cell):
    if cell is None:
        return None
    s = str(cell).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def resolve_geometry_timeseries_csv_path(folder: str) -> str:
    """Return path to output.csv, or legacy track/geometry_timeseries.csv."""
    primary = os.path.join(folder, "output.csv")
    legacy = os.path.join(folder, "track", "geometry_timeseries.csv")
    if os.path.isfile(primary):
        return primary
    if os.path.isfile(legacy):
        return legacy
    raise FileNotFoundError(
        f"output.csv not found: {primary} (legacy {legacy} also missing). "
        "Run export_geometry_timeseries before generate_outputs."
    )


def load_geometry_timeseries_csv(folder, dt_min_kymo, path=None):
    """
    Load output.csv (geometry timeseries) from the sample folder root.

    Falls back to legacy track/geometry_timeseries.csv if the new file is absent.

    Returns dict with numpy arrays needed for geometry-only plotting.
    Raises if file is missing/empty.

    Parameters
    ----------
    path : str or None
        If given, read this file (must exist). Otherwise resolve via
        ``resolve_geometry_timeseries_csv_path``.
    """
    if path is None:
        path = resolve_geometry_timeseries_csv_path(folder)
    rows = []
    with open(path, newline="") as f:
        first_line = f.readline()
        if not first_line.startswith("filename,"):
            f.seek(0)
        reader = csv.DictReader(f)
        required_cols = {
            "time_min",
            "apical_px_raw",
            "front_raw_px",
            "front_minus_apical_px",
            "front_minus_apical_um",
        }
        has_col_idx = "col_idx" in (reader.fieldnames or [])
        has_time_abs = "time_abs_min" in (reader.fieldnames or [])
        missing = [c for c in required_cols if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{os.path.basename(path)} is missing required columns: "
                + ", ".join(sorted(missing))
            )
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(
            f"{os.path.basename(path)} is empty: {path}. "
            "Run export_geometry_timeseries before generate_outputs."
        )
    n = len(rows)
    col_idx = np.zeros(n, dtype=int)
    time_min = np.zeros(n, dtype=float)
    time_abs_min = np.full(n, np.nan, dtype=float)
    apical_px = np.full(n, np.nan)
    front_raw_px = np.full(n, np.nan)
    front_minus_apical_um = np.full(n, np.nan)
    for i, row in enumerate(rows):
        time_min[i] = _parse_csv_float(row.get("time_min"))
        if has_time_abs:
            time_abs_min[i] = _parse_csv_float(row.get("time_abs_min"))
        if has_col_idx:
            parsed_col_idx = _parse_csv_int(row.get("col_idx"))
            if parsed_col_idx is not None:
                col_idx[i] = parsed_col_idx
            else:
                col_idx[i] = int(np.rint(time_min[i] / dt_min_kymo)) if dt_min_kymo > 0 else i
        else:
            col_idx[i] = int(np.rint(time_min[i] / dt_min_kymo)) if dt_min_kymo > 0 else i
        apical_px[i] = _parse_csv_float(row.get("apical_px_raw"))
        front_raw_px[i] = _parse_csv_float(row.get("front_raw_px"))
        front_minus_apical_um[i] = _parse_csv_float(row.get("front_minus_apical_um"))
    return {
        "col_idx": col_idx,
        "time_min": time_min,
        "time_abs_min": time_abs_min,
        "apical_px": apical_px,
        "front_raw_px": front_raw_px,
        "front_minus_apical_um": front_minus_apical_um,
    }


def load_front_furrow_stamp_time_bounds_minutes(
    work_dir: str | os.PathLike[str],
) -> tuple[float, float] | None:
    """
    First and last geometry time (minutes) where ``front_raw_px`` is finite.

    Same time axis as ``mark_delta_on_trimmed_movie`` (prefers ``time_abs_min``
    when present). Used for furrow-relative timestamps (0 until first detection,
    then elapsed time, frozen after the last valid front).
    """
    wd = os.fspath(work_dir)
    try:
        cfg = load_config(wd)
        dt_min_kymo = cfg["kymograph_time_interval_sec"] / 60.0
        gs = load_geometry_timeseries_csv(wd, dt_min_kymo)
    except (FileNotFoundError, OSError, ValueError):
        return None
    geom_time = gs["time_min"]
    ta = gs.get("time_abs_min")
    if ta is not None and np.any(np.isfinite(ta)):
        geom_time = ta
    front = gs["front_raw_px"]
    valid = ~np.isnan(front) & np.isfinite(geom_time)
    if not np.any(valid):
        return None
    gv = np.asarray(geom_time[valid], dtype=float)
    return (float(gv.min()), float(gv.max()))


def furrow_relative_stamp_seconds(
    frame_idx: int,
    movie_time_interval_sec: float,
    t_first_valid_min: float,
    t_last_valid_min: float,
) -> int:
    """Elapsed whole seconds for the on-movie stamp (0 → rise → hold at span)."""
    mdt = float(movie_time_interval_sec)
    time_min = frame_idx * mdt / 60.0
    if time_min < t_first_valid_min:
        return 0
    span_sec = max(0.0, (t_last_valid_min - t_first_valid_min) * 60.0)
    if time_min > t_last_valid_min:
        return int(round(span_sec))
    return int(round(max(0.0, (time_min - t_first_valid_min) * 60.0)))


def load_apical_line_series_for_movie(
    work_dir: str | os.PathLike[str],
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """
    Return ``(time_min_series, apical_px_series, movie_time_interval_sec)`` for
    overlaying the apical border on acquisition movies, or None if unavailable.

    Uses the same time axis as ``mark_delta_on_trimmed_movie`` (prefers
    ``time_abs_min`` when present in output.csv).
    """
    wd = os.fspath(work_dir)
    try:
        cfg = load_config(wd)
        dt_min_kymo = cfg["kymograph_time_interval_sec"] / 60.0
        gs = load_geometry_timeseries_csv(wd, dt_min_kymo)
    except (FileNotFoundError, OSError, ValueError):
        return None
    geom_time = gs["time_min"]
    ta = gs.get("time_abs_min")
    if ta is not None and np.any(np.isfinite(ta)):
        geom_time = ta
    ap = gs["apical_px"]
    valid = np.isfinite(geom_time) & np.isfinite(ap)
    if not np.any(valid):
        return None
    return (
        np.asarray(geom_time[valid], dtype=float),
        np.asarray(ap[valid], dtype=float),
        float(cfg["movie_time_interval_sec"]),
    )


def compute_milestones(time_min, front_px, apical_px, final_px, movie_dt_sec):
    """
    Compute 10% milestones for cellularization progress.

    Returns dict[pct] -> {frame, time_min, position_px}.
    """
    total_distance = final_px - apical_px
    progress_pct = (front_px - apical_px) / total_distance * 100.0

    milestones = {}
    for pct in range(0, 101, 10):
        indices = np.where(progress_pct >= pct)[0]
        if len(indices) == 0:
            continue
        idx = indices[0]
        t_min = float(time_min[idx])
        frame_idx = int(t_min * 60.0 / movie_dt_sec)
        milestones[pct] = {
            "frame": frame_idx,
            "time_min": t_min,
            "position_px": float(front_px[idx]),
        }

    # If the front never reaches 100% (e.g. final_px slightly above last spline),
    # still place a 100% marker at the end of the spline.
    if 100 not in milestones and time_min.size > 0:
        idx = int(time_min.size - 1)
        t_min = float(time_min[idx])
        frame_idx = int(t_min * 60.0 / movie_dt_sec)
        milestones[100] = {
            "frame": frame_idx,
            "time_min": t_min,
            "position_px": float(front_px[idx]),
        }
    return milestones


def mark_delta_on_trimmed_movie(
    work_dir,
    cfg,
    spline_time_min,
    spline_front_px,
    movie=None,
    *,
    kymograph_brightness: float | None = None,
    show_apical_line: bool = False,
    apical_line_rgb: tuple[int, int, int] = (255, 255, 0),
    output_mp4_path: str | None = None,
    mp4_fps: float | None = None,
    mp4_crf: int | None = None,
    show_timestamp: bool = False,
    timestamp_rgb: tuple[int, int, int] = (255, 255, 255),
    timestamp_size_px: int = 8,
):
    """
    Encode Cellularization_front_markers.mp4 with front markers on each frame.

    The front position per frame is derived from output.csv (geometry timeseries)
    (front_raw_px), interpolated to the movie frame rate. The marked stack is
    held in memory; only the MP4 is written.

    Parameters
    ----------
    movie : np.ndarray or None
        Pre-loaded acquisition movie [T, Y, X].  When supplied the TIFF is not
        re-read from disk, avoiding a redundant I/O round-trip.
    kymograph_brightness : float or None
        If None, use ``cfg['kymograph_brightness']`` (from config.yaml).
    show_apical_line : bool
        When True, draw a dashed horizontal line at ``apical_px_raw`` from output.csv.
    apical_line_rgb : tuple[int, int, int]
        Line color (8-bit RGB), default yellow like kymograph scatter.
    output_mp4_path : str or None
        Destination MP4 path; default ``results/Cellularization_front_markers.mp4``.
    mp4_fps : float or None
        Encoded MP4 frame rate. If None, uses ``EXPORT_MP4_FPS`` (default 10).
    mp4_crf : int or None
        libx264 CRF (0–51). If None, uses ``EXPORT_MP4_CRF`` (default 20).
    show_timestamp : bool
        When True, burn ``HH:MM:SS`` bottom-left on every frame: ``00:00:00`` until the
        first valid ``front_raw_px`` time, then elapsed time since that detection,
        then frozen at the span through the last valid front time (same bounds as
        the front markers).
    timestamp_rgb : tuple[int, int, int]
        8-bit RGB text color for the timestamp.
    timestamp_size_px : int
        Nominal text height scale (same sense as preview Size spin).
    """
    movie_time_interval_sec = cfg["movie_time_interval_sec"]
    dt_min_kymo = cfg["kymograph_time_interval_sec"] / 60.0

    if movie is None:
        movie_path = get_movie_path(work_dir)
        movie = tifffile.imread(movie_path)
    if movie.ndim != 3:
        raise ValueError(f"Expected 3D movie, got shape {movie.shape}")
    num_frames, height, width = movie.shape

    kb = (
        float(kymograph_brightness)
        if kymograph_brightness is not None
        else float(cfg.get("kymograph_brightness", 1.0))
    )
    pipeline_diag.info(
        __name__,
        "mark_delta: movie shape T,Y,X=%s brightness_kb=%s",
        (num_frames, height, width),
        kb,
    )
    out_gray = _autoscale_acquisition_movie_gray_u16(movie, kb)

    out = np.stack([out_gray, out_gray, out_gray], axis=-1)  # [T, Y, X, C]

    # Load raw pixel positions from geometry CSV — no spline / back-conversion needed.
    gs = load_geometry_timeseries_csv(work_dir, dt_min_kymo)
    geom_time = gs["time_min"]
    ta = gs.get("time_abs_min")
    if ta is not None and np.any(np.isfinite(ta)):
        geom_time = ta
    geom_front_raw = gs["front_raw_px"]
    geom_apical_raw = gs["apical_px"]

    # Only valid (non-NaN) entries should be used for interpolation.
    valid = ~np.isnan(geom_front_raw) & np.isfinite(geom_time)
    if not np.any(valid):
        print("Warning: no valid front_raw_px entries; skipping movie markers.")
        return
    geom_time_v = geom_time[valid]
    geom_front_v = geom_front_raw[valid]
    t_min_lo = float(geom_time_v.min())
    t_min_hi = float(geom_time_v.max())

    valid_ap = ~np.isnan(geom_apical_raw) & np.isfinite(geom_time)
    geom_time_ap: np.ndarray | None
    geom_apical_v: np.ndarray | None
    if np.any(valid_ap):
        geom_time_ap = np.asarray(geom_time[valid_ap], dtype=float)
        geom_apical_v = np.asarray(geom_apical_raw[valid_ap], dtype=float)
    else:
        geom_time_ap = None
        geom_apical_v = None

    apical_u16 = _rgb_u8_to_u16(apical_line_rgb)

    def draw_arrow_right(rgb_frame, y_center, arrow_size_px, max_val):
        """Draw an opaque left-pointing triangle on the right edge (RGB = max_val)."""
        h, w, _c = rgb_frame.shape
        y_center = int(round(y_center))
        y_center = max(arrow_size_px, min(h - 1 - arrow_size_px, y_center))
        for dy in range(-arrow_size_px, arrow_size_px + 1):
            row = y_center + dy
            if row < 0 or row >= h:
                continue
            half = int(arrow_size_px * (1 - abs(dy) / max(arrow_size_px, 1)))
            for dx in range(0, half + 1):
                col = w - 1 - dx
                if col >= 0:
                    rgb_frame[row, col, :] = max_val

    # 1.5× prior size (~3% of frame height); solid uint16 white (not frame-relative).
    white_u16 = 65535
    arrow_size_px = max(2, int(round(height * 0.02 * 1.5)))
    for f in range(num_frames):
        time_min = f * movie_time_interval_sec / 60.0
        if time_min < t_min_lo:
            continue
        if time_min > t_min_hi:
            break

        front_px = float(np.interp(time_min, geom_time_v, geom_front_v))
        y_arrow = max(0, min(height - 1, front_px))
        draw_arrow_right(out[f], y_arrow, arrow_size_px, max_val=white_u16)

        if show_apical_line and geom_time_ap is not None and geom_apical_v is not None:
            ap_y = float(np.interp(time_min, geom_time_ap, geom_apical_v))
            if np.isfinite(ap_y):
                _draw_dashed_horizontal_u16(out[f], ap_y, apical_u16)

        pipeline_diag.overlay_frame_tick(__name__, f, num_frames)

    if show_timestamp:
        pipeline_diag.info(__name__, "mark_delta: burning timestamps on %s frames", num_frames)
        mdt = float(movie_time_interval_sec)
        tsz = int(np.clip(int(timestamp_size_px), 2, 96))
        tr, tg, tb = (int(timestamp_rgb[0]), int(timestamp_rgb[1]), int(timestamp_rgb[2]))
        for f2 in range(num_frames):
            sec = furrow_relative_stamp_seconds(f2, mdt, t_min_lo, t_min_hi)
            txt = format_experiment_hms(sec)
            burn_timestamp_on_u16_rgb_frame(
                out[f2],
                txt,
                color_rgb=(tr, tg, tb),
                size_px=tsz,
            )

    mp4_path = output_mp4_path or os.path.join(work_dir, "results", FRONT_MARKERS_MP4)
    os.makedirs(os.path.dirname(mp4_path), exist_ok=True)
    pipeline_diag.info(__name__, "mark_delta: overlay done; encoding MP4 -> %s", mp4_path)
    _write_delta_movie_mp4(out, mp4_path, mp4_fps, mp4_crf)


def _pad_rgb_u8_for_h264_yuv420p(u8: np.ndarray, macro: int = 2) -> np.ndarray:
    """
    Pad ``[T, Y, X, 3]`` uint8 so **Y** and **X** are multiples of ``macro`` (default 2).

    ``yuv420p`` requires even width and height. We no longer pad to 16-pixel macroblocks,
    so at most one row/column is added per axis (avoids a wide ``edge`` strip on narrow frames).
    """
    _t, h, w, _c = u8.shape
    nh = ((h + macro - 1) // macro) * macro
    nw = ((w + macro - 1) // macro) * macro
    ph, pw = nh - h, nw - w
    if ph == 0 and pw == 0:
        return u8
    return np.pad(
        u8,
        ((0, 0), (0, ph), (0, pw), (0, 0)),
        mode="edge",
    )


EXPORT_MP4_FPS = 10.0
EXPORT_MP4_CRF = 20


def _write_delta_movie_mp4(
    out_uint16: np.ndarray,
    mp4_path: str,
    fps: float | None = None,
    crf: int | None = None,
) -> None:
    """H.264 (high) + silent AAC MP4 via bundled ffmpeg. ``fps`` / ``crf`` default to module constants."""
    import imageio_ffmpeg

    u8 = np.clip(out_uint16.astype(np.float32) / 257.0, 0.0, 255.0).astype(np.uint8)
    u8 = _pad_rgb_u8_for_h264_yuv420p(u8)
    t, height, width, channels = u8.shape
    if t < 1:
        print(f"Warning: no frames for MP4; skipping {mp4_path}.")
        return
    if channels != 3:
        print(f"Warning: expected RGB stack for MP4; skipping {mp4_path}.")
        return

    eff_fps = float(EXPORT_MP4_FPS if fps is None else fps)
    if eff_fps <= 0:
        print(f"Warning: invalid MP4 fps {eff_fps}; skipping {mp4_path}.")
        return
    eff_crf = int(EXPORT_MP4_CRF if crf is None else crf)
    eff_crf = max(0, min(51, eff_crf))
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    out_abs = os.path.abspath(mp4_path)
    parent = os.path.dirname(out_abs)
    if parent:
        os.makedirs(parent, exist_ok=True)

    cmd = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{int(width)}x{int(height)}",
        "-framerate",
        str(eff_fps),
        "-i",
        "pipe:0",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(eff_crf),
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-shortest",
        "-movflags",
        "+faststart",
        out_abs,
    ]
    pipeline_diag.info(
        __name__,
        "ffmpeg: %s frames %sx%s fps=%s crf=%s -> %s",
        t,
        width,
        height,
        eff_fps,
        eff_crf,
        out_abs,
    )
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
    )
    if proc.stdin is None:
        raise RuntimeError("ffmpeg subprocess did not provide stdin")
    pipeline_diag.info(__name__, "ffmpeg: subprocess started, writing %s raw frames to stdin", t)
    try:
        for i in range(t):
            pipeline_diag.debug(__name__, "ffmpeg stdin frame %s/%s", i + 1, t)
            proc.stdin.write(np.ascontiguousarray(u8[i]).tobytes())
    except BrokenPipeError:
        pipeline_diag.info(__name__, "ffmpeg: BrokenPipeError while writing stdin (ffmpeg may have exited)")
    finally:
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        # Popen.communicate() flushes stdin first; stdin is already closed here, which
        # raises ValueError: flush of closed file on Python 3.11+ (macOS/Linux).
        proc.stdin = None
    pipeline_diag.info(__name__, "ffmpeg: stdin closed, waiting on communicate()")
    _, err_b = proc.communicate()
    pipeline_diag.info(__name__, "ffmpeg: communicate() done returncode=%s", proc.returncode)
    if proc.returncode != 0:
        msg = (err_b or b"").decode(errors="replace").strip() or "ffmpeg exited with error"
        raise RuntimeError(msg)
    pipeline_diag.user_line(__name__, f"Saved {FRONT_MARKERS_MP4} to: {mp4_path}")


def _prepare_cellularization_data(folder, cfg, spline_time_min, spline_front_px):
    """Load and compute all data needed for the cellularization figure."""
    px2micron = cfg["px2micron"]
    dt_min_kymo = cfg["kymograph_time_interval_sec"] / 60.0

    track_folder = os.path.join(folder, "track")

    # Load pre-computed straightened kymograph and its alignment metadata.
    straight_path = os.path.join(track_folder, "Kymograph_straightened.tif")
    if not os.path.exists(straight_path):
        raise FileNotFoundError(
            f"Kymograph_straightened.tif not found in: {track_folder}. "
            "Run straighten_kymograph before generate_outputs."
        )

    straight_kymo = tifffile.imread(straight_path)
    if straight_kymo.ndim != 2:
        straight_kymo = np.squeeze(straight_kymo)
    if straight_kymo.ndim != 2:
        raise ValueError(f"Kymograph_straightened.tif has unexpected shape {straight_kymo.shape}")
    depth, num_timepoints = straight_kymo.shape

    raw_path = os.path.join(track_folder, "Kymograph.tif")
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(
            f"Kymograph.tif not found in: {track_folder} (needed for export contrast matching the GUI)."
        )
    raw_kymo = tifffile.imread(raw_path)
    if raw_kymo.ndim != 2:
        raw_kymo = np.squeeze(raw_kymo)
    if raw_kymo.ndim != 2:
        raise ValueError(f"Kymograph.tif has unexpected shape {raw_kymo.shape}")
    if raw_kymo.shape != straight_kymo.shape:
        print(
            f"Warning: Kymograph.tif shape {raw_kymo.shape} != straightened {straight_kymo.shape}; "
            "using straightened stack for contrast limits."
        )
        raw_kymo = straight_kymo

    meta = straightening_meta(folder)
    if not meta or "ref_row" not in meta:
        raise FileNotFoundError(
            "straightening metadata missing in config.yaml. "
            "Run straighten_kymograph before generate_outputs."
        )
    ref_row = int(meta["ref_row"])
    crop_top = int(meta["crop_top_px"])

    gt_path = resolve_geometry_timeseries_csv_path(folder)
    print(f"Using geometry timeseries: {gt_path}")
    gs = load_geometry_timeseries_csv(folder, dt_min_kymo, path=gt_path)
    time_min_cyto = gs["time_min"]
    front_minus_apical_um = gs["front_minus_apical_um"]

    tw = cfg.get("time_window")
    if tw is not None:
        lo = tw.get("start_min", -np.inf) or -np.inf
        hi = tw.get("end_min", np.inf) or np.inf
        mask_tw = (time_min_cyto >= lo) & (time_min_cyto <= hi)
        if np.any(mask_tw):
            time_min_cyto = time_min_cyto[mask_tw]
            front_minus_apical_um = front_minus_apical_um[mask_tw]
            print(f"time_window: keeping {mask_tw.sum()} / {mask_tw.size} rows ({lo}–{hi} min)")
        else:
            print(f"Warning: time_window [{lo}, {hi}] min excluded all rows; ignoring.")

    valid_front = ~np.isnan(front_minus_apical_um)
    if not np.any(valid_front):
        raise ValueError("No rows with valid front data in geometry timeseries; cannot build figure.")

    num_timepoints_cyto = time_min_cyto.size

    y_top_um = -(ref_row - crop_top) * px2micron   # slightly above apical (negative)
    y_bottom_um = (depth - 1 - ref_row) * px2micron
    apical_um = np.zeros(num_timepoints_cyto, dtype=float)
    basal_um = np.full(num_timepoints_cyto, np.nan, dtype=float)
    front_um = front_minus_apical_um.astype(float)

    # Kymograph columns to display: use explicit indices from geometry CSV.
    # This keeps exact alignment even when time_min is re-zeroed for plotting.
    col_indices = gs["col_idx"].astype(int)
    col_indices = np.clip(col_indices, 0, num_timepoints - 1)
    straight_kymo_plot = straight_kymo[crop_top:, col_indices]

    vmin, vmax = _kymograph_limits_like_front_panel(
        raw_kymo, float(cfg.get("kymograph_brightness", 1.0))
    )

    basal_um_plot = basal_um.copy()
    w = cfg.get("basal_smooth_window", 1)
    if w > 1:
        kernel = np.ones(w)
        valid_basal = ~np.isnan(basal_um_plot)
        if np.any(valid_basal):
            data_b = basal_um_plot[valid_basal]
            smoothed = np.convolve(data_b, kernel, mode="same")
            counts = np.convolve(np.ones(len(data_b)), kernel, mode="same")
            basal_um_plot[valid_basal] = smoothed / counts

    return {
        "straight_kymo_plot": straight_kymo_plot,
        "time_min_cyto": time_min_cyto,
        "apical_um": apical_um,
        "valid_apical": np.ones(num_timepoints_cyto, dtype=bool),
        "basal_um_plot": basal_um_plot,
        "front_um": front_um,
        "y_bottom_um": y_bottom_um,
        "y_top_um": y_top_um,
        "vmin": vmin,
        "vmax": vmax,
    }


def _draw_cellularization_on_ax(ax, d):
    """Draw the cellularization figure onto an existing axes."""
    time_min_cyto = d["time_min_cyto"]
    y_bottom_um = d["y_bottom_um"]
    y_top_um = d["y_top_um"]

    ax.imshow(
        d["straight_kymo_plot"],
        cmap="gray",
        aspect="auto",
        origin="upper",
        extent=[time_min_cyto[0], time_min_cyto[-1], y_bottom_um, y_top_um],
        vmin=d["vmin"],
        vmax=d["vmax"],
    )

    # Explicit apical reference: y = 0 µm by definition.
    ax.axhline(
        y=0.0,
        linestyle="--",
        color="#FFD700",
        linewidth=3.0,
        alpha=0.8,
    )

    valid_basal_um = ~np.isnan(d["basal_um_plot"])
    ax.plot(
        time_min_cyto[valid_basal_um],
        d["basal_um_plot"][valid_basal_um],
        linestyle="--", color="0.7", linewidth=3.0, alpha=0.5,
    )

    valid_front = ~np.isnan(d["front_um"])
    if np.any(valid_front):
        front_line, = ax.plot(
            time_min_cyto[valid_front],
            d["front_um"][valid_front],
            linestyle="--",
            color="#FFFACD",
            linewidth=3.0,
            alpha=1.0,
        )
        # Dark halo keeps pastel front line visible on bright regions.
        front_line.set_path_effects([
            path_effects.Stroke(linewidth=5.0, foreground="black", alpha=0.45),
            path_effects.Normal(),
        ])

    t0, t100 = time_min_cyto[0], time_min_cyto[-1]

    ax.set_xlim(t0, t100)
    # Keep image mapping unchanged; show fixed headroom of -2 µm above apical.
    apical_headroom_um = -APICAL_HEADROOM_UM
    top_visible = min(apical_headroom_um, y_top_um)
    ax.set_ylim(y_bottom_um, top_visible)
    ax.set_facecolor("white")

    # Keep negative space visible but don't label negative tick values.
    tick_step = 5.0
    tick_start = 0.0
    tick_end = np.ceil(y_bottom_um / tick_step) * tick_step
    ticks = list(np.arange(tick_start, tick_end + 0.1, tick_step))
    ticks = sorted(set(round(t, 6) for t in ticks))
    ax.set_yticks(ticks)
    ax.set_xlabel("Time (min)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Depth (µm)", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICKS, length=TICK_MAJOR_LENGTH, width=TICK_MAJOR_WIDTH)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINEWIDTH)


def make_cellularization_figure(work_dir, cfg, spline_time_min, spline_front_px):
    """Create Cellularization.png and .pdf (standalone square figure)."""
    pipeline_diag.info(__name__, "make_cellularization_figure: prepare data + build axes")
    d = _prepare_cellularization_data(work_dir, cfg, spline_time_min, spline_front_px)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_position([0.12, 0.12, 0.76, 0.76])
    _draw_cellularization_on_ax(ax, d)
    plt.tight_layout(pad=0)
    ax.set_position([0.12, 0.12, 0.76, 0.76])
    out_png = os.path.join(work_dir, "results", "Cellularization.png")
    out_pdf = os.path.join(work_dir, "results", "Cellularization.pdf")
    pipeline_diag.info(__name__, "figure: savefig PNG -> %s", out_png)
    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.05)
    pipeline_diag.info(__name__, "figure: PNG savefig returned")
    pipeline_diag.info(__name__, "figure: savefig PDF -> %s", out_pdf)
    plt.savefig(out_pdf, dpi=150, bbox_inches="tight", pad_inches=0.05, format="pdf")
    pipeline_diag.info(__name__, "figure: PDF savefig returned")
    pipeline_diag.info(__name__, "figure: calling plt.close()")
    plt.close()
    pipeline_diag.info(__name__, "figure: plt.close() returned")
    pipeline_diag.user_line(__name__, f"Saved Cellularization.png to: {out_png}")
    pipeline_diag.user_line(__name__, f"Saved Cellularization.pdf to: {out_pdf}")
    return d


def generate_outputs(work_dir, movie=None):
    """
    Parameters
    ----------
    movie : np.ndarray or None
        Pre-loaded acquisition movie [T, Y, X].  When supplied the TIFF is not
        read from disk inside ``mark_delta_on_trimmed_movie``, avoiding a
        redundant I/O round-trip when the caller already holds the array.
    """
    pipeline_diag.configure()
    pipeline_diag.info(
        __name__,
        "generate_outputs start work_dir=%s movie_preloaded=%s",
        work_dir,
        movie is not None,
    )
    cfg = load_config(work_dir)
    time_min_spline, front_px = load_spline(work_dir)

    # Milestones (also used inside Cellularization figure)
    milestones = compute_milestones(
        time_min_spline,
        front_px,
        cfg["apical_height_px"],
        cfg["final_height_px"],
        cfg["movie_time_interval_sec"],
    )
    if 0 not in milestones or 100 not in milestones:
        print(
            "Warning: 0% or 100% milestone missing; Cellularization figure will "
            "still be generated but crop range may be suboptimal."
        )

    pipeline_diag.info(__name__, "generate_outputs step: mark_delta_on_trimmed_movie")
    mark_delta_on_trimmed_movie(work_dir, cfg, time_min_spline, front_px, movie=movie)
    pipeline_diag.info(__name__, "generate_outputs step: mark_delta_on_trimmed_movie done")

    pipeline_diag.info(__name__, "generate_outputs step: make_cellularization_figure")
    make_cellularization_figure(work_dir, cfg, time_min_spline, front_px)
    pipeline_diag.info(__name__, "generate_outputs step: make_cellularization_figure done")
    pipeline_diag.info(__name__, "generate_outputs finished OK")


def main():
    parser = argparse.ArgumentParser(
        description="Generate final Cellularization outputs from spline and cytoplasm TSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help="Working directory containing config and intermediate outputs",
    )
    args = parser.parse_args()
    if not os.path.isdir(args.work_dir):
        raise ValueError(f"Work directory does not exist: {args.work_dir}")
    pipeline_diag.configure()
    generate_outputs(args.work_dir)


if __name__ == "__main__":
    main()


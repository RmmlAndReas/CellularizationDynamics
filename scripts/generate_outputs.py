#!/usr/bin/env python3
"""
Generate all final outputs for the cellularization pipeline from data products.

Inputs (per sample folder):
    - config.yaml
    - Cellularization.tif
    - Cellularization_trimmed.tif
    - track/Kymograph.tif
    - track/VerticalKymoCelluSelection_spline.tsv
    - track/cytoplasm_region.tsv
    - track/geometry_timeseries.csv (preferred; merged cytoplasm + front)

Outputs:
    - Cellularization.png
    - Cellularization.pdf
        Straightened kymograph aligned so apical border = 0 µm, with:
          - apical/basal cytoplasm borders
          - cellularization front spline
          - 10% milestone arrows (style/box/fonts from Kymograph_delta_marked)
    - Kymograph_delta.tif
        Delta-offset kymograph strip extracted from Cellularization.tif.
    - Cellularization_trimmed_delta.tif
        Trimmed movie with white arrow on each frame at delta position.

This script combines core logic from:
    - create_kymographs.py
    - mark_delta_on_trimmed_movie.py
    - mark_kymograph_progress.py (style, milestones)
    - tmp/detect_yolk_border.py (straightened overlay geometry)
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tifffile
import yaml
from scipy.interpolate import UnivariateSpline
from cellularization_paths import resolve_input_movie_path


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

MILESTONE_FIRST_COLOR = "#0072B2"  # 0% milestone
MILESTONE_LAST_COLOR = "#D55E00"   # 100% milestone


def _auto_brightness_contrast_limits(image, saturated_pct=0.35):
    """
    Compute display limits using an ImageJ-like auto contrast stretch.

    `saturated_pct` is the percentage clipped in total (split between tails).
    """
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0

    half_tail = saturated_pct / 2.0
    lo = float(np.percentile(finite, half_tail))
    hi = float(np.percentile(finite, 100.0 - half_tail))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi <= lo:
            hi = lo + 1.0
    return lo, hi


def load_config(folder):
    """Load config.yaml and return commonly used fields."""
    config_path = os.path.join(folder, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    if "manual" not in cfg:
        raise ValueError("config.yaml must contain 'manual' section")
    manual = cfg["manual"]
    if "px2micron" not in manual:
        raise ValueError("manual.px2micron missing in config.yaml")

    # Support both 'delta' and legacy 'delta1'
    if "delta" in manual:
        delta_microns = float(manual["delta"])
    elif "delta1" in manual:
        delta_microns = float(manual["delta1"])
    else:
        raise ValueError("manual.delta or manual.delta1 not found in config.yaml")

    movie_time_interval_sec = float(manual.get("movie_time_interval_sec", 10.0))

    keep_every = None
    if "preprocessing" in cfg and isinstance(cfg["preprocessing"], dict):
        keep_every = int(cfg["preprocessing"].get("keep_every", 1))

    if "kymograph" not in cfg:
        raise ValueError("'kymograph' section not found in config.yaml")
    kymo_cfg = cfg["kymograph"]
    time_interval_sec = float(
        kymo_cfg.get("time_interval_sec", kymo_cfg.get("time_interval_min", 1.0) * 60.0)
    )

    # Apical and final heights (needed for milestones and coordinate transforms)
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
    if "visualization" in cfg and isinstance(cfg["visualization"], dict):
        vis = cfg["visualization"]
        basal_smooth_window = int(vis.get("basal_smooth_window", 1))
        kymo_marked_start_pct = int(vis.get("kymo_marked_start_pct", 70))

    time_window = None
    if "time_window" in cfg and isinstance(cfg["time_window"], dict):
        tw = cfg["time_window"]
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
        "delta_microns": delta_microns,
        "px2micron": float(manual["px2micron"]),
        "movie_time_interval_sec": movie_time_interval_sec,
        "keep_every": keep_every,
        "kymograph_time_interval_sec": time_interval_sec,
        "apical_height_px": apical_height_px,
        "final_height_px": final_height_px,
        "time_window": time_window,
        "basal_smooth_window": basal_smooth_window,
        "kymo_marked_start_pct": kymo_marked_start_pct,
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


def load_geometry_timeseries_csv(folder):
    """
    Load track/geometry_timeseries.csv if present.

    Returns dict with numpy arrays needed for geometry-only plotting.
    Raises if file is missing/empty.
    """
    path = os.path.join(folder, "track", "geometry_timeseries.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"geometry_timeseries.csv not found: {path}. "
            "Run export_geometry_timeseries before generate_outputs."
        )
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {
            "col_idx",
            "time_min",
            "apical_px_raw",
            "cytoplasm_height_um",
            "basal_minus_apical_um",
            "front_minus_apical_um",
        }
        missing = [c for c in required_cols if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                "geometry_timeseries.csv is missing required columns: "
                + ", ".join(sorted(missing))
            )
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(
            f"geometry_timeseries.csv is empty: {path}. "
            "Run export_geometry_timeseries before generate_outputs."
        )
    n = len(rows)
    col_idx = np.zeros(n, dtype=int)
    time_min = np.zeros(n, dtype=float)
    apical_px = np.full(n, np.nan)
    basal_px = np.full(n, np.nan)
    height_um = np.full(n, np.nan)
    front_px = np.full(n, np.nan)
    basal_minus_apical_um = np.full(n, np.nan)
    front_minus_apical_um = np.full(n, np.nan)
    for i, row in enumerate(rows):
        col_idx[i] = int(row["col_idx"])
        time_min[i] = _parse_csv_float(row.get("time_min"))
        apical_px[i] = _parse_csv_float(row.get("apical_px_raw"))
        basal_px[i] = _parse_csv_float(row.get("basal_px_raw"))
        height_um[i] = _parse_csv_float(row.get("cytoplasm_height_um"))
        front_px[i] = _parse_csv_float(row.get("front_px_raw"))
        basal_minus_apical_um[i] = _parse_csv_float(row.get("basal_minus_apical_um"))
        front_minus_apical_um[i] = _parse_csv_float(row.get("front_minus_apical_um"))
    return {
        "col_idx": col_idx,
        "time_min": time_min,
        "apical_px": apical_px,
        "basal_px": basal_px,
        "height_um": height_um,
        "front_px": front_px,
        "basal_minus_apical_um": basal_minus_apical_um,
        "front_minus_apical_um": front_minus_apical_um,
    }


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


def create_kymograph_delta(work_dir, data_dir, cfg, time_min_spline, front_positions_px):
    """
    Re-implementation of create_kymographs.py, but only for Kymograph_delta.tif.
    """
    track_folder = os.path.join(work_dir, "track")
    # Spline is already loaded; create interpolation function
    spline_func = UnivariateSpline(time_min_spline, front_positions_px, s=0, k=3)

    movie_path = resolve_input_movie_path(data_dir)

    movie = tifffile.imread(movie_path)
    if movie.ndim != 3:
        raise ValueError(f"Expected 3D movie, got shape {movie.shape}")
    num_frames, height, width = movie.shape

    delta_px = cfg["delta_microns"] / cfg["px2micron"]

    kymograph = []
    for frame_idx in range(num_frames):
        time_sec = frame_idx * cfg["movie_time_interval_sec"]
        time_min = time_sec / 60.0
        t_clamped = float(
            max(time_min_spline.min(), min(time_min, time_min_spline.max()))
        )
        front_pos_px = float(spline_func(t_clamped))
        extract_pos_px = front_pos_px - delta_px
        y_pos = int(round(max(0, min(height - 1, extract_pos_px))))
        line = movie[frame_idx, y_pos, :]
        kymograph.append(line)

    kymograph = np.array(kymograph)
    out_path = os.path.join(work_dir, "results", "Kymograph_delta.tif")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tifffile.imwrite(out_path, kymograph)
    print(f"Saved Kymograph_delta.tif to: {out_path}")


def mark_delta_on_trimmed_movie(work_dir, cfg, spline_time_min, spline_front_px):
    """
    Re-implementation of mark_delta_on_trimmed_movie.py, writing Cellularization_trimmed_delta.tif.

    Adds:
      - white arrow at (cellu front - delta) on the right edge
      - red triangle at the cellularization front tip
    Both markers are removed once spline tip data is no longer available.
    """
    delta_microns = cfg["delta_microns"]
    px2micron = cfg["px2micron"]
    movie_time_interval_sec = cfg["movie_time_interval_sec"]
    keep_every = cfg["keep_every"] if cfg["keep_every"] is not None else 1
    delta_px = delta_microns / px2micron

    trimmed_path = os.path.join(work_dir, "Cellularization_trimmed.tif")
    if not os.path.exists(trimmed_path):
        raise FileNotFoundError(f"Cellularization_trimmed.tif not found: {trimmed_path}")
    movie = tifffile.imread(trimmed_path)
    if movie.ndim != 3:
        raise ValueError(f"Expected 3D movie, got shape {movie.shape}")
    num_frames, height, width = movie.shape

    if np.issubdtype(movie.dtype, np.integer):
        mx = np.iinfo(movie.dtype).max
        out_gray = (movie.astype(np.float64) / mx * 65535).astype(np.uint16)
    else:
        out_gray = (np.clip(movie.astype(np.float64), 0.0, 1.0) * 65535).astype(np.uint16)

    # Create RGB stack so we can overlay a red triangle on the right edge.
    # Base image stays grayscale (R=G=B).
    out = np.stack([out_gray, out_gray, out_gray], axis=-1)  # [T, Y, X, C]

    spline_func = UnivariateSpline(spline_time_min, spline_front_px, s=0, k=3)
    t_min_lo, t_min_hi = float(spline_time_min.min()), float(spline_time_min.max())

    def draw_arrow_right(rgb_frame, y_center, arrow_size_px, max_val):
        """Draw a white left-pointing triangle on the right edge."""
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

    def draw_red_triangle_right(rgb_frame, y_center, arrow_size_px, red_val):
        """Draw a red left-pointing triangle on the right edge."""
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
                    rgb_frame[row, col, 0] = red_val  # R
                    rgb_frame[row, col, 1] = 0       # G
                    rgb_frame[row, col, 2] = 0       # B

    arrow_size_px = max(2, int(height * 0.02))
    for f in range(num_frames):
        arrow_val = int(np.max(out_gray[f]))
        time_min = (f * keep_every) * movie_time_interval_sec / 60.0
        # Only draw while spline tip data exists; once we pass the spline's
        # maximum time, both markers should disappear.
        if time_min < t_min_lo:
            continue
        if time_min > t_min_hi:
            break

        front_px = float(spline_func(time_min))

        # Tip triangle at the cellularization front (no delta offset)
        y_tip = max(0, min(height - 1, front_px))
        draw_red_triangle_right(out[f], y_tip, arrow_size_px, red_val=arrow_val)

        # White delta arrow at (front - delta)
        y_arrow = max(0, min(height - 1, front_px - delta_px))
        draw_arrow_right(out[f], y_arrow, arrow_size_px, max_val=arrow_val)
        if (f + 1) % 50 == 0:
            print(f"  Frame {f + 1}/{num_frames}")

    out_path = os.path.join(work_dir, "results", "Cellularization_trimmed_delta.tif")
    metadata = {
        # TYXC: time, y, x, channel (RGB)
        "axes": "TYXC",
        "PhysicalSizeX": px2micron,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": px2micron,
        "PhysicalSizeYUnit": "µm",
    }
    with tifffile.TiffWriter(out_path, bigtiff=False, ome=True) as tif:
        tif.write(out, metadata=metadata)
    print(f"Saved Cellularization_trimmed_delta.tif to: {out_path}")


def _prepare_cellularization_data(folder, cfg, spline_time_min, spline_front_px):
    """Load and compute all data needed for the cellularization figure."""
    px2micron = cfg["px2micron"]
    apical_px = cfg["apical_height_px"]
    final_px = cfg["final_height_px"]
    dt_min_kymo = cfg["kymograph_time_interval_sec"] / 60.0

    track_folder = os.path.join(folder, "track")
    kymo_path = os.path.join(track_folder, "Kymograph.tif")
    if not os.path.exists(kymo_path):
        raise FileNotFoundError(f"Kymograph.tif not found in: {track_folder}")
    kymo = tifffile.imread(kymo_path)
    if kymo.ndim != 2:
        kymo = np.squeeze(kymo)
    if kymo.ndim != 2:
        raise ValueError(f"Kymograph.tif has unexpected shape {kymo.shape}")
    depth, num_timepoints = kymo.shape

    gs = load_geometry_timeseries_csv(folder)
    print("Using track/geometry_timeseries.csv for cellularization overlays")
    col_idx_cyto = gs["col_idx"]
    time_min_cyto = gs["time_min"]
    apical_px_cyto = gs["apical_px"]
    height_um_cyto = gs["height_um"]
    basal_minus_apical_um = gs["basal_minus_apical_um"]
    front_minus_apical_um = gs["front_minus_apical_um"]

    tw = cfg.get("time_window")
    if tw is not None:
        lo = tw.get("start_min", -np.inf) or -np.inf
        hi = tw.get("end_min", np.inf) or np.inf
        mask_tw = (time_min_cyto >= lo) & (time_min_cyto <= hi)
        if np.any(mask_tw):
            col_idx_cyto = col_idx_cyto[mask_tw]
            time_min_cyto = time_min_cyto[mask_tw]
            apical_px_cyto = apical_px_cyto[mask_tw]
            height_um_cyto = height_um_cyto[mask_tw]
            basal_minus_apical_um = basal_minus_apical_um[mask_tw]
            front_minus_apical_um = front_minus_apical_um[mask_tw]
            print(f"time_window: keeping {mask_tw.sum()} / {mask_tw.size} cytoplasm rows ({lo}–{hi} min)")
        else:
            print(f"Warning: time_window [{lo}, {hi}] min excluded all rows; ignoring.")

    valid = ~np.isnan(height_um_cyto)
    if not np.any(valid):
        raise ValueError("No valid cytoplasm measurements in cytoplasm_region.tsv")
    valid_apical = ~np.isnan(apical_px_cyto)
    valid_cols = valid & valid_apical
    if not np.any(valid_cols):
        raise ValueError("No columns with valid apical cytoplasm border; cannot build figure.")

    col_idx_cyto = col_idx_cyto[valid_cols]
    time_min_cyto = time_min_cyto[valid_cols]
    apical_px_cyto = apical_px_cyto[valid_cols]
    basal_minus_apical_um = basal_minus_apical_um[valid_cols]
    front_minus_apical_um = front_minus_apical_um[valid_cols]

    valid_apical = ~np.isnan(apical_px_cyto)
    ref_row = int(np.nanmin(apical_px_cyto[valid_apical]))
    straight_kymo = np.zeros_like(kymo)
    num_timepoints_cyto = apical_px_cyto.size
    for t in range(num_timepoints_cyto):
        if not valid_apical[t]:
            continue
        shift = int(ref_row - apical_px_cyto[t])
        src_col = max(0, min(col_idx_cyto[t], num_timepoints - 1))
        col = kymo[:, src_col]
        for y in range(depth):
            new_y = y + shift
            if 0 <= new_y < depth:
                straight_kymo[int(new_y), t] = col[y]

    y_bottom_um = (depth - 1 - ref_row) * px2micron
    y_top_um = (0 - ref_row) * px2micron
    apical_um = np.zeros_like(apical_px_cyto, dtype=float)
    basal_um = basal_minus_apical_um.astype(float)
    front_um = front_minus_apical_um.astype(float)

    milestones = compute_milestones(spline_time_min, spline_front_px, apical_px, final_px, cfg["movie_time_interval_sec"])

    straight_kymo_plot = straight_kymo[:, :num_timepoints_cyto]
    # Apply one-pass auto brightness/contrast for display.
    vmin, vmax = _auto_brightness_contrast_limits(straight_kymo_plot, saturated_pct=0.35)

    basal_um_plot = basal_um.copy()
    w = cfg.get("basal_smooth_window", 1)
    if w > 1:
        kernel = np.ones(w)
        valid_basal = ~np.isnan(basal_um_plot)
        if np.any(valid_basal):
            data = basal_um_plot[valid_basal]
            smoothed = np.convolve(data, kernel, mode="same")
            counts = np.convolve(np.ones(len(data)), kernel, mode="same")
            basal_um_plot[valid_basal] = smoothed / counts

    return {
        "straight_kymo_plot": straight_kymo_plot,
        "time_min_cyto": time_min_cyto,
        "apical_um": apical_um,
        "valid_apical": valid_apical,
        "basal_um_plot": basal_um_plot,
        "front_um": front_um,
        "milestones": milestones,
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

    ax.plot(
        time_min_cyto[d["valid_apical"]],
        d["apical_um"][d["valid_apical"]],
        linestyle="--", color="0.7", linewidth=3.0, alpha=0.5,
    )

    valid_basal_um = ~np.isnan(d["basal_um_plot"])
    ax.plot(
        time_min_cyto[valid_basal_um],
        d["basal_um_plot"][valid_basal_um],
        linestyle="--", color="0.7", linewidth=3.0, alpha=0.5,
    )

    valid_front = ~np.isnan(d["front_um"])
    if np.any(valid_front):
        ax.plot(
            time_min_cyto[valid_front],
            d["front_um"][valid_front],
            "g--",
            linewidth=3.0,
            alpha=0.3,
        )

    t0, t100 = time_min_cyto[0], time_min_cyto[-1]
    duration_min = max(0.0, t100 - t0)
    tri_w = 0.04 * duration_min
    tri_h = (0.06 / 1.3) * (y_bottom_um if y_bottom_um > 0 else -y_bottom_um)
    for pct in sorted(d["milestones"].keys()):
        x_pos = max(t0, min(d["milestones"][pct]["time_min"], t100))
        facecolor = "white"
        if pct == 0:
            facecolor = MILESTONE_FIRST_COLOR
        elif pct == 100:
            facecolor = MILESTONE_LAST_COLOR
        ax.add_patch(patches.Polygon(
            [[x_pos, y_bottom_um - tri_h], [x_pos - tri_w / 2, y_bottom_um], [x_pos + tri_w / 2, y_bottom_um]],
            closed=True, facecolor=facecolor, edgecolor="black", linewidth=1.5,
        ))

    ax.set_xlim(t0, t100)
    ax.set_xlabel("Time (min)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Depth (µm)", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICKS, length=TICK_MAJOR_LENGTH, width=TICK_MAJOR_WIDTH)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINEWIDTH)


def _prepare_kymo_delta_data(folder, cfg, milestones):
    """Load and crop Kymograph_delta.tif for the marked plot."""
    kymo_path = os.path.join(folder, "results", "Kymograph_delta.tif")
    if not os.path.exists(kymo_path):
        return None

    kymo = tifffile.imread(kymo_path)
    if kymo.ndim != 2:
        kymo = np.squeeze(kymo)
    num_rows, width = kymo.shape
    width_um = width * cfg["px2micron"]
    dt_min = cfg["movie_time_interval_sec"] / 60.0

    start_pct = cfg.get("kymo_marked_start_pct", 70)
    start_pct_snapped = max((p for p in milestones if p <= start_pct), default=0)
    frame_0 = milestones.get(start_pct_snapped, milestones.get(0, {})).get("frame", 0)
    frame_100 = milestones.get(100, {}).get("frame", num_rows - 1)
    frame_0 = max(0, min(frame_0, num_rows - 1))
    frame_100 = max(frame_0, min(frame_100, num_rows - 1))

    return {
        "kymo_crop": kymo[frame_0 : frame_100 + 1, :],
        "width_um": width_um,
        "duration_min": (frame_100 - frame_0) * dt_min,
        "frame_0": frame_0,
        "frame_100": frame_100,
        "start_pct_snapped": start_pct_snapped,
        "dt_min": dt_min,
    }


def _draw_kymo_delta_on_ax(ax, d, milestones):
    """Draw the marked delta kymograph onto an existing axes."""
    from matplotlib.ticker import MultipleLocator

    ax.imshow(
        d["kymo_crop"],
        aspect="auto",
        cmap="gray",
        origin="upper",
        extent=[0, d["width_um"], d["duration_min"], 0],
    )

    tri_w = 0.033 * d["width_um"]
    tri_h = (0.050 / 1.3) * d["duration_min"]
    for pct in sorted(milestones.keys()):
        if pct < d["start_pct_snapped"]:
            continue
        frame_idx = milestones[pct]["frame"]
        if frame_idx < d["frame_0"] or frame_idx > d["frame_100"]:
            continue
        y_pos = (frame_idx - d["frame_0"]) * d["dt_min"]
        facecolor = "white"
        if pct == 0:
            facecolor = MILESTONE_FIRST_COLOR
        elif pct == 100:
            facecolor = MILESTONE_LAST_COLOR
        ax.add_patch(patches.Polygon(
            [[0, y_pos - tri_h / 2], [0, y_pos + tri_h / 2], [tri_w, y_pos]],
            closed=True,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=1.5,
            zorder=10,
            clip_on=False,
        ))

    ax.set_xlim(0, d["width_um"])
    ax.set_ylim(d["duration_min"], 0)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("Width (µm)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Time (min)", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICKS, length=TICK_MAJOR_LENGTH, width=TICK_MAJOR_WIDTH)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINEWIDTH)


def make_cellularization_figure(work_dir, cfg, spline_time_min, spline_front_px):
    """Create Cellularization.png and .pdf (standalone square figure)."""
    d = _prepare_cellularization_data(work_dir, cfg, spline_time_min, spline_front_px)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_position([0.12, 0.12, 0.76, 0.76])
    _draw_cellularization_on_ax(ax, d)
    plt.tight_layout(pad=0)
    ax.set_position([0.12, 0.12, 0.76, 0.76])
    out_png = os.path.join(work_dir, "results", "Cellularization.png")
    out_pdf = os.path.join(work_dir, "results", "Cellularization.pdf")
    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(out_pdf, dpi=150, bbox_inches="tight", pad_inches=0.05, format="pdf")
    plt.close()
    print(f"Saved Cellularization.png to: {out_png}")
    print(f"Saved Cellularization.pdf to: {out_pdf}")
    return d


def make_kymograph_delta_marked(work_dir, cfg, milestones):
    """
    Create Kymograph_delta_marked.png and .pdf (annotated figure) and
    Kymograph_delta_marked.tif as the corresponding unannotated kymograph crop.
    """
    d = _prepare_kymo_delta_data(work_dir, cfg, milestones)
    if d is None:
        print("Warning: Kymograph_delta.tif not found, skipping marked version.")
        return None
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_position([0.12, 0.12, 0.76, 0.76])
    _draw_kymo_delta_on_ax(ax, d, milestones)
    plt.tight_layout(pad=0)
    ax.set_position([0.12, 0.12, 0.76, 0.76])
    out_png = os.path.join(work_dir, "results", "Kymograph_delta_marked.png")
    out_pdf = os.path.join(work_dir, "results", "Kymograph_delta_marked.pdf")
    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(out_pdf, dpi=150, bbox_inches="tight", pad_inches=0.05, format="pdf")
    plt.close()

    # Save the underlying unannotated kymograph crop as TIFF
    out_tif = os.path.join(work_dir, "results", "Kymograph_delta_marked.tif")
    tifffile.imwrite(out_tif, d["kymo_crop"])

    print(f"Saved Kymograph_delta_marked.png to: {out_png}")
    print(f"Saved Kymograph_delta_marked.pdf to: {out_pdf}")
    print(f"Saved Kymograph_delta_marked.tif (unannotated kymograph) to: {out_tif}")
    return d


def make_combined_figure(work_dir, cfg, spline_time_min, spline_front_px, milestones):
    """
    Create Cellularization_combined.png and .pdf:
    two-column figure with the cellularization (left) and delta kymograph (right).
    """
    cellu_d = _prepare_cellularization_data(work_dir, cfg, spline_time_min, spline_front_px)
    kymo_d = _prepare_kymo_delta_data(work_dir, cfg, milestones)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    _draw_cellularization_on_ax(ax1, cellu_d)
    if kymo_d is not None:
        _draw_kymo_delta_on_ax(ax2, kymo_d, milestones)
    else:
        ax2.set_visible(False)

    plt.tight_layout(pad=1.0)

    out_png = os.path.join(work_dir, "results", "Cellularization_combined.png")
    out_pdf = os.path.join(work_dir, "results", "Cellularization_combined.pdf")
    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(out_pdf, dpi=150, bbox_inches="tight", pad_inches=0.05, format="pdf")
    plt.close()
    print(f"Saved Cellularization_combined.png to: {out_png}")
    print(f"Saved Cellularization_combined.pdf to: {out_pdf}")


def generate_outputs(work_dir, data_dir):
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

    create_kymograph_delta(work_dir, data_dir, cfg, time_min_spline, front_px)
    mark_delta_on_trimmed_movie(work_dir, cfg, time_min_spline, front_px)
    make_cellularization_figure(work_dir, cfg, time_min_spline, front_px)
    make_kymograph_delta_marked(work_dir, cfg, milestones)
    make_combined_figure(work_dir, cfg, time_min_spline, front_px, milestones)


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
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Raw data directory containing full-resolution movie (default: work-dir)",
    )
    args = parser.parse_args()
    if not os.path.isdir(args.work_dir):
        raise ValueError(f"Work directory does not exist: {args.work_dir}")
    data_dir = args.data_dir if args.data_dir else args.work_dir
    if not os.path.isdir(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")

    generate_outputs(args.work_dir, data_dir)


if __name__ == "__main__":
    main()


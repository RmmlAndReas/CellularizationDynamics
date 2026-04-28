#!/usr/bin/env python3
"""
Generate all final outputs for the cellularization pipeline from data products.

Inputs (per sample folder):
    - config.yaml
    - Cellularization.tif
    - Cellularization_trimmed.tif
    - track/Kymograph.tif
    - track/VerticalKymoCelluSelection_spline.tsv
    - track/geometry_timeseries.csv (preferred; merged cytoplasm + front)

Outputs:
    - Cellularization.png
    - Cellularization.pdf
        Straightened kymograph aligned so apical border = 0 µm, with:
          - apical/basal cytoplasm borders
          - cellularization front spline
          - no milestone triangle markers
    - Cellularization_trimmed_delta.tif
        Trimmed movie with white arrow on each frame at front position.

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
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import yaml


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


def load_geometry_timeseries_csv(folder, dt_min_kymo):
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
            "time_min",
            "apical_px_raw",
            "front_raw_px",
            "front_minus_apical_px",
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
    front_raw_px = np.full(n, np.nan)
    front_minus_apical_um = np.full(n, np.nan)
    for i, row in enumerate(rows):
        time_min[i] = _parse_csv_float(row.get("time_min"))
        col_idx[i] = int(np.rint(time_min[i] / dt_min_kymo)) if dt_min_kymo > 0 else i
        apical_px[i] = _parse_csv_float(row.get("apical_px_raw"))
        front_raw_px[i] = _parse_csv_float(row.get("front_raw_px"))
        front_minus_apical_um[i] = _parse_csv_float(row.get("front_minus_apical_um"))
    return {
        "col_idx": col_idx,
        "time_min": time_min,
        "apical_px": apical_px,
        "front_raw_px": front_raw_px,
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


def mark_delta_on_trimmed_movie(work_dir, cfg, spline_time_min, spline_front_px):
    """
    Write Cellularization_trimmed_delta.tif with front markers on each frame.

    The front position per frame is derived from geometry_timeseries.csv
    (front_raw_px = apical_px_raw + front_minus_apical_px), interpolated to
    the movie frame rate. This avoids any coordinate-space back-conversion.
    """
    px2micron = cfg["px2micron"]
    movie_time_interval_sec = cfg["movie_time_interval_sec"]
    keep_every = cfg["keep_every"] if cfg["keep_every"] is not None else 1
    dt_min_kymo = cfg["kymograph_time_interval_sec"] / 60.0

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

    out = np.stack([out_gray, out_gray, out_gray], axis=-1)  # [T, Y, X, C]

    # Load raw pixel positions from geometry CSV — no spline / back-conversion needed.
    gs = load_geometry_timeseries_csv(work_dir, dt_min_kymo)
    geom_time = gs["time_min"]
    geom_front_raw = gs["front_raw_px"]

    # Only valid (non-NaN) entries should be used for interpolation.
    valid = ~np.isnan(geom_front_raw)
    if not np.any(valid):
        print("Warning: no valid front_raw_px entries; skipping movie markers.")
        return
    geom_time_v = geom_time[valid]
    geom_front_v = geom_front_raw[valid]
    t_min_lo = float(geom_time_v.min())
    t_min_hi = float(geom_time_v.max())

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
        if time_min < t_min_lo:
            continue
        if time_min > t_min_hi:
            break

        front_px = float(np.interp(time_min, geom_time_v, geom_front_v))

        # Tip triangle at the cellularization front (no delta offset)
        y_tip = max(0, min(height - 1, front_px))
        draw_red_triangle_right(out[f], y_tip, arrow_size_px, red_val=arrow_val)

        # White arrow at front position (no delta offset)
        y_arrow = max(0, min(height - 1, front_px))
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
    dt_min_kymo = cfg["kymograph_time_interval_sec"] / 60.0

    track_folder = os.path.join(folder, "track")

    # Load pre-computed straightened kymograph and its alignment metadata.
    straight_path = os.path.join(track_folder, "Kymograph_straightened.tif")
    meta_path = os.path.join(track_folder, "straighten_metadata.yaml")
    if not os.path.exists(straight_path):
        raise FileNotFoundError(
            f"Kymograph_straightened.tif not found in: {track_folder}. "
            "Run straighten_kymograph before generate_outputs."
        )
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"straighten_metadata.yaml not found in: {track_folder}. "
            "Run straighten_kymograph before generate_outputs."
        )

    straight_kymo = tifffile.imread(straight_path)
    if straight_kymo.ndim != 2:
        straight_kymo = np.squeeze(straight_kymo)
    if straight_kymo.ndim != 2:
        raise ValueError(f"Kymograph_straightened.tif has unexpected shape {straight_kymo.shape}")
    depth, num_timepoints = straight_kymo.shape

    with open(meta_path) as f:
        meta = yaml.safe_load(f) or {}
    ref_row = int(meta["ref_row"])
    crop_top = int(meta["crop_top_px"])

    gs = load_geometry_timeseries_csv(folder, dt_min_kymo)
    print("Using track/geometry_timeseries.csv for cellularization overlays")
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
        raise ValueError("No rows with valid front data in geometry_timeseries.csv; cannot build figure.")

    num_timepoints_cyto = time_min_cyto.size

    y_top_um = -(ref_row - crop_top) * px2micron   # slightly above apical (negative)
    y_bottom_um = (depth - 1 - ref_row) * px2micron
    apical_um = np.zeros(num_timepoints_cyto, dtype=float)
    basal_um = np.full(num_timepoints_cyto, np.nan, dtype=float)
    front_um = front_minus_apical_um.astype(float)

    # Kymograph columns to display: map time_min_cyto back to column indices.
    col_indices = np.round(time_min_cyto / dt_min_kymo).astype(int)
    col_indices = np.clip(col_indices, 0, num_timepoints - 1)
    straight_kymo_plot = straight_kymo[crop_top:, col_indices]

    vmin, vmax = _auto_brightness_contrast_limits(straight_kymo_plot, saturated_pct=0.35)

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

    ax.plot(
        time_min_cyto[d["valid_apical"]],
        d["apical_um"][d["valid_apical"]],
        linestyle="--", color="0.25", linewidth=3.0, alpha=0.9,
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
    ax.set_xlabel("Time (min)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Depth (µm)", fontsize=FONT_SIZE_LABEL)
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


def generate_outputs(work_dir):
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

    mark_delta_on_trimmed_movie(work_dir, cfg, time_min_spline, front_px)
    make_cellularization_figure(work_dir, cfg, time_min_spline, front_px)


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
    generate_outputs(args.work_dir)


if __name__ == "__main__":
    main()


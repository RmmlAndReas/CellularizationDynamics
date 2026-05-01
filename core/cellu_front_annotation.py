#!/usr/bin/env python3
"""
Manual cellularization-front annotation with matplotlib.

Writes apical scalars to ``config.yaml`` and the front polyline to ``track/apical_front.tsv``.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from scipy.interpolate import UnivariateSpline
import tifffile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from annotation_source import build_apical_alignment_v2, persist_apical_alignment
from work_state import load_state, pipeline_config_flat, straightening_meta


def load_config(work_dir):
    cfg_path = os.path.join(work_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise ValueError(f"config.yaml not found at: {cfg_path}")
    config = pipeline_config_flat(work_dir)
    time_interval_sec = 60.0
    k = config.get("kymograph") or {}
    if "time_interval_sec" in k:
        time_interval_sec = float(k["time_interval_sec"])
    elif "time_interval_min" in k:
        time_interval_sec = float(k["time_interval_min"]) * 60.0
    px2micron = float(config.get("manual", {}).get("px2micron", 1.0))
    return time_interval_sec / 60.0, px2micron


def load_straightened_kymograph(work_dir):
    """
    Load the pre-computed straightened kymograph and its alignment metadata.

    Requires that straighten_kymograph.py has already been run for this sample.
    """
    track_folder = os.path.join(work_dir, "track")
    tif_path = os.path.join(track_folder, "Kymograph_straightened.tif")
    if not os.path.exists(tif_path):
        raise FileNotFoundError(
            f"{os.path.basename(tif_path)} not found in {track_folder}. "
            "Run straighten_kymograph before front annotation."
        )

    meta = straightening_meta(work_dir)
    if not meta or "ref_row" not in meta or "crop_top_px" not in meta:
        raise FileNotFoundError(
            "Straightening metadata missing in config.yaml (ref_row, crop_top_px). "
            "Run straighten_kymograph before front annotation."
        )

    straight_kymo = tifffile.imread(tif_path)
    if straight_kymo.ndim > 2:
        straight_kymo = np.squeeze(straight_kymo)
    if straight_kymo.ndim != 2:
        raise ValueError(f"Kymograph_straightened.tif has unexpected shape: {straight_kymo.shape}")

    ref_row = int(meta["ref_row"])
    crop_top = int(meta["crop_top_px"])

    return straight_kymo, ref_row, crop_top


def collect_polyline_points(image, crop_top_px=0, ref_row=0, px2micron=1.0):
    """
    Interactive polyline annotation on a cropped apical-corrected kymograph.

    Parameters
    ----------
    image : 2-D array
        Cropped display image (rows start at crop_top_px in straight_kymo).
    crop_top_px : int
        Row in straight_kymo corresponding to the top of `image`.
    ref_row : int
        Row in straight_kymo where the apical border sits (depth = 0).
    px2micron : float
        Pixel-to-micron conversion for the y-axis labels.

    Returns
    -------
    points : (N, 2) array in straight_kymo pixel coordinates (not display coordinates).
    """
    points = []
    accepted = {"value": False}
    state = {"brightness": 1.0}

    dmin = float(np.min(image))
    dmax = float(np.max(image))
    base_vmin = float(np.percentile(image, 1))
    base_vmax = float(np.percentile(image, 99))
    if base_vmax <= base_vmin:
        base_vmin = dmin
        base_vmax = dmax

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.16)
    img_artist = ax.imshow(
        image, cmap="gray", aspect="auto", vmin=base_vmin, vmax=base_vmax
    )
    ax.set_title(
        "Click to add front points | Backspace: undo | Enter: save | Esc: cancel"
    )
    ax.set_xlabel("Time (kymograph column)")
    ax.set_ylabel("Depth relative to apical (µm)")

    # Y-axis labels in µm relative to apical (display_row r → µm = (r + crop_top_px - ref_row) * px2micron).
    num_rows = image.shape[0]
    tick_display_rows = np.arange(0, num_rows, max(1, num_rows // 8))
    tick_um = (tick_display_rows + crop_top_px - ref_row) * px2micron
    ax.set_yticks(tick_display_rows)
    ax.set_yticklabels([f"{v:.1f}" for v in tick_um])
    line, = ax.plot([], [], "r-", linewidth=2)
    scatter = ax.scatter([], [], c="yellow", s=18)
    brightness_ax = fig.add_axes([0.15, 0.06, 0.70, 0.04])
    brightness_slider = Slider(
        brightness_ax, "brightness", 0.2, 3.0, valinit=state["brightness"]
    )

    def redraw(_=None):
        state["brightness"] = float(brightness_slider.val)
        bright_vmax = base_vmin + (base_vmax - base_vmin) / state["brightness"]
        if bright_vmax <= base_vmin:
            bright_vmax = base_vmin + 1e-6
        img_artist.set_clim(vmin=base_vmin, vmax=bright_vmax)
        if points:
            arr = np.asarray(points, dtype=float)
            line.set_data(arr[:, 0], arr[:, 1])
            scatter.set_offsets(arr)
        else:
            line.set_data([], [])
            scatter.set_offsets(np.empty((0, 2)))
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        points.append((float(event.xdata), float(event.ydata)))
        redraw()

    def on_key(event):
        key = (event.key or "").lower()
        if key == "backspace" and points:
            points.pop()
            redraw()
        elif key == "enter":
            if len(points) < 2:
                print("Need at least 2 points to save.")
                return
            accepted["value"] = True
            plt.close(fig)
        elif key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    brightness_slider.on_changed(redraw)
    redraw()
    plt.show()

    if not accepted["value"]:
        raise RuntimeError("Annotation window closed without saving.")
    return np.asarray(points, dtype=float)


def points_to_smooth_curve(points_xy, time_interval_min):
    order = np.argsort(points_xy[:, 0])
    x = points_xy[order, 0]
    y = points_xy[order, 1]

    x_unique, uniq_idx = np.unique(x, return_index=True)
    y_unique = y[uniq_idx]
    if x_unique.size < 2:
        raise ValueError("Need at least two distinct x positions.")

    times = x_unique * time_interval_min
    k = min(3, x_unique.size - 1)
    spline = UnivariateSpline(times, y_unique, s=0, k=k)
    num_samples = max(500, x_unique.size * 10)
    time_samples = np.linspace(times.min(), times.max(), num_samples)
    depth_samples = spline(time_samples)
    return time_samples, depth_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=str, required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.work_dir):
        raise ValueError(f"Work directory does not exist: {args.work_dir}")

    time_interval_min, px2micron = load_config(args.work_dir)

    track_folder = os.path.join(args.work_dir, "track")
    os.makedirs(track_folder, exist_ok=True)

    corrected_kymo, ref_row, crop_top = load_straightened_kymograph(args.work_dir)

    display_kymo = corrected_kymo[crop_top:, :]

    points_display = collect_polyline_points(
        display_kymo, crop_top_px=crop_top, ref_row=ref_row, px2micron=px2micron
    )

    # Convert display coords → straight_kymo coords (add crop_top offset).
    # No back-conversion to raw pixels — annotation stays in straightened space.
    points_straight = points_display.copy()
    points_straight[:, 1] = points_display[:, 1] + crop_top

    time_samples, depth_samples = points_to_smooth_curve(points_straight, time_interval_min)

    state = load_state(args.work_dir, migrate_if_needed=True)
    existing = state.get("apical_alignment") or {}
    mode = str(existing.get("mode", "longest_run")).strip()
    labels = [int(x) for x in (existing.get("island_labels") or [])]
    if mode == "island" and not labels:
        mode = "longest_run"
    threshold = float(existing.get("threshold", 0.5))
    movie_sec = float(time_interval_min) * 60.0
    h, w = corrected_kymo.shape
    alignment = build_apical_alignment_v2(
        mode=mode,
        island_labels=labels if mode == "island" else [],
        threshold=threshold,
        kymograph_shape=(int(h), int(w)),
        movie_time_interval_sec=movie_sec,
    )
    persist_apical_alignment(
        args.work_dir,
        alignment,
        np.asarray(time_samples, dtype=float),
        np.asarray(depth_samples, dtype=float),
    )
    print(
        f"Saved apical_alignment v2 and track/apical_front.tsv ({len(time_samples)} points, "
        f"threshold={threshold})"
    )


if __name__ == "__main__":
    main()


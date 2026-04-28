#!/usr/bin/env python3
"""
Manual cellularization-front annotation with matplotlib.

Replaces the ImageJ ROI drawing step.
Outputs:
  - track/VerticalKymoCelluSelection.tsv
  - track/VerticalKymoCelluSelection.roi (compat placeholder)
"""

import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from scipy.interpolate import UnivariateSpline
import tifffile
import yaml


def load_config(config_path):
    if not os.path.exists(config_path):
        raise ValueError(f"config.yaml not found at: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    time_interval_sec = 60.0
    if "kymograph" in config and "time_interval_sec" in config["kymograph"]:
        time_interval_sec = float(config["kymograph"]["time_interval_sec"])
    elif "kymograph" in config and "time_interval_min" in config["kymograph"]:
        time_interval_sec = float(config["kymograph"]["time_interval_min"]) * 60.0
    px2micron = float(config.get("manual", {}).get("px2micron", 1.0))
    return time_interval_sec / 60.0, px2micron


def load_straightened_kymograph(work_dir):
    """
    Load the pre-computed straightened kymograph and its alignment metadata.

    Requires that straighten_kymograph.py has already been run for this sample.
    """
    track_folder = os.path.join(work_dir, "track")
    tif_path = os.path.join(track_folder, "Kymograph_straightened.tif")
    meta_path = os.path.join(track_folder, "straighten_metadata.yaml")

    for p in [tif_path, meta_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{os.path.basename(p)} not found in {track_folder}. "
                "Run straighten_kymograph before front annotation."
            )

    straight_kymo = tifffile.imread(tif_path)
    if straight_kymo.ndim > 2:
        straight_kymo = np.squeeze(straight_kymo)
    if straight_kymo.ndim != 2:
        raise ValueError(f"Kymograph_straightened.tif has unexpected shape: {straight_kymo.shape}")

    with open(meta_path) as f:
        meta = yaml.safe_load(f) or {}

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


def save_tsv(track_folder, time_samples, depth_samples):
    tsv_path = os.path.join(track_folder, "VerticalKymoCelluSelection.tsv")
    with open(tsv_path, "w") as f:
        f.write("Time\tDepth\n")
        for t, d in zip(time_samples, depth_samples):
            f.write(f"{t:.6f}\t{d:.6f}\n")
    return tsv_path


def save_compat_roi_placeholder(track_folder):
    roi_path = os.path.join(track_folder, "VerticalKymoCelluSelection.roi")
    with open(roi_path, "wb") as f:
        f.write(b"Iout")
        f.write((0).to_bytes(1, "big"))
        f.write((5).to_bytes(1, "big"))  # polyline type marker placeholder
        f.write((0).to_bytes(2, "big", signed=True))
        f.write((0).to_bytes(2, "big", signed=True))
        f.write((0).to_bytes(2, "big", signed=True))
        f.write((0).to_bytes(2, "big", signed=True))
    return roi_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=str, required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.work_dir):
        raise ValueError(f"Work directory does not exist: {args.work_dir}")

    config_path = os.path.join(args.work_dir, "config.yaml")
    time_interval_min, px2micron = load_config(config_path)

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

    tsv_path = save_tsv(track_folder, time_samples, depth_samples)
    roi_path = save_compat_roi_placeholder(track_folder)
    print(f"Saved front TSV to: {tsv_path}")
    print(f"Saved ROI placeholder to: {roi_path}")


if __name__ == "__main__":
    main()


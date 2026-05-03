#!/usr/bin/env python3
"""
Interactive cytoplasm thresholding on Kymograph.tif with matplotlib.

This replaces the previous ImageJ-based threshold step.
It creates:
  - track/YolkMask.tif
and updates config.yaml with mean apical height.
"""

import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button, Slider
import numpy as np
import tifffile

from .apical_manual import apical_px_from_manual_polyline
from .mask_utils import compute_apical_column_positions
from .track_tabular import read_apical_manual_tsv
from .work_state import load_state, merge_patch, pipeline_config_flat


def load_px2micron(work_dir):
    cfg_path = os.path.join(work_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise ValueError(f"config.yaml not found at: {cfg_path}")
    cfg = pipeline_config_flat(work_dir)
    if "manual" not in cfg or "px2micron" not in cfg["manual"]:
        raise ValueError("px2micron not found in config.yaml under 'manual'")
    return {"px2micron": float(cfg["manual"]["px2micron"])}


def interactive_threshold_and_save_mask(work_dir):
    track_folder = os.path.join(work_dir, "track")
    kymograph_path = os.path.join(track_folder, "Kymograph.tif")
    if not os.path.exists(kymograph_path):
        raise FileNotFoundError(f"Kymograph.tif not found in: {track_folder}")

    kymo = tifffile.imread(kymograph_path)
    if kymo.ndim > 2:
        kymo = np.squeeze(kymo)
    if kymo.ndim != 2:
        raise ValueError(f"Kymograph has unexpected shape: {kymo.shape}")

    kymo = np.asarray(kymo)
    dmin = float(np.min(kymo))
    dmax = float(np.max(kymo))
    if dmax <= dmin:
        raise ValueError("Kymograph has no intensity range for thresholding.")

    init_thr = float(np.percentile(kymo, 50))
    init_brightness = 1.0
    accepted = {"value": False}
    state = {"threshold": init_thr, "brightness": init_brightness}

    # Robust display range that works well with interactive brightness scaling.
    base_vmin = float(np.percentile(kymo, 1))
    base_vmax = float(np.percentile(kymo, 99))
    if base_vmax <= base_vmin:
        base_vmin = dmin
        base_vmax = dmax

    fig, ax_overlay = plt.subplots(1, 1, figsize=(8, 6))
    plt.subplots_adjust(bottom=0.28, right=0.92)
    mask0 = kymo <= init_thr

    overlay_img_artist = ax_overlay.imshow(
        kymo, cmap="gray", aspect="auto", vmin=base_vmin, vmax=base_vmax
    )
    overlay_cmap = ListedColormap(
        [
            (0.0, 0.0, 0.0, 0.0),  # non-mask pixels: fully transparent
            (0.0, 0.0, 1.0, 1.0),  # mask pixels: blue
        ]
    )
    overlay_mask_artist = ax_overlay.imshow(
        mask0.astype(np.uint8), cmap=overlay_cmap, vmin=0, vmax=1, aspect="auto"
    )
    ax_overlay.set_title("Threshold overlay (blue)")
    ax_overlay.set_xlabel("time (col_idx)")
    ax_overlay.set_ylabel("depth (px)")

    threshold_ax = fig.add_axes([0.12, 0.17, 0.68, 0.04])
    threshold_slider = Slider(
        threshold_ax, "threshold", dmin, dmax, valinit=init_thr
    )
    brightness_ax = fig.add_axes([0.12, 0.10, 0.68, 0.04])
    brightness_slider = Slider(
        brightness_ax, "brightness", 0.2, 3.0, valinit=init_brightness
    )

    btn_ax = fig.add_axes([0.82, 0.03, 0.10, 0.05])
    save_button = Button(btn_ax, "save")

    def compute_mask():
        thr = float(threshold_slider.val)
        state["threshold"] = thr
        # Fixed polarity chosen to match downstream geometry extraction.
        return kymo <= thr

    def refresh(_=None):
        state["brightness"] = float(brightness_slider.val)
        mask = compute_mask()
        bright_vmax = base_vmin + (base_vmax - base_vmin) / state["brightness"]
        if bright_vmax <= base_vmin:
            bright_vmax = base_vmin + 1e-6
        overlay_img_artist.set_clim(vmin=base_vmin, vmax=bright_vmax)
        overlay_mask_artist.set_data(mask.astype(np.uint8))
        fig.canvas.draw_idle()

    def on_save(_event):
        accepted["value"] = True
        plt.close(fig)

    threshold_slider.on_changed(refresh)
    brightness_slider.on_changed(refresh)
    save_button.on_clicked(on_save)
    refresh()

    print("Adjust threshold slider and click 'save' to continue.")
    plt.show()

    if not accepted["value"]:
        raise RuntimeError("Threshold window was closed without saving.")

    mask = compute_mask().astype(np.uint8) * 255
    mask_path = os.path.join(track_folder, "YolkMask.tif")
    tifffile.imwrite(mask_path, mask)
    print(
        f"Saved YolkMask.tif to: {mask_path} "
        f"(threshold={state['threshold']:.3f}, polarity='kymo <= threshold')"
    )
    return mask_path


def apical_px_using_saved_islands(work_dir: str, mask_bool: np.ndarray) -> np.ndarray | None:
    """Mean apical row per column from ``apical_alignment`` island labels, or None if unavailable."""
    state = load_state(work_dir, migrate_if_needed=True)
    aa = state.get("apical_alignment") or {}
    if str(aa.get("mode", "")).strip() != "island":
        return None
    raw = aa.get("island_labels") or []
    labels = [int(x) for x in raw]
    if not labels:
        return None
    apical_px, _ = compute_apical_column_positions(mask_bool, island_labels=labels)
    return apical_px


def apical_px_using_saved_manual(
    work_dir: str, num_timepoints: int, dt_min: float, px2micron: float
) -> np.ndarray | None:
    """Per-column apical row from the saved manual polyline, or None if unavailable."""
    state = load_state(work_dir, migrate_if_needed=True)
    aa = state.get("apical_alignment") or {}
    if str(aa.get("mode", "")).strip() != "manual":
        return None
    poly = read_apical_manual_tsv(work_dir)
    if poly is None:
        return None
    time_min_pts, depth_px_pts = poly
    sigma_um = float(aa.get("manual_sigma_um", 0.5))
    return apical_px_from_manual_polyline(
        time_min_pts,
        depth_px_pts,
        num_timepoints=int(num_timepoints),
        dt_min=float(dt_min),
        sigma_um=sigma_um,
        px2micron=float(px2micron),
    )


def update_apical_height_in_config(folder, apical_px, px2micron):
    valid = ~np.isnan(apical_px)
    if not np.any(valid):
        print("Warning: no valid apical_px values; skipping apical_detection update.")
        return

    mean_apical_px = float(np.nanmean(apical_px[valid]))
    mean_apical_microns = mean_apical_px * px2micron
    config_path = os.path.join(folder, "config.yaml")
    if not os.path.exists(config_path):
        print(f"Warning: config.yaml not found at {config_path}; skipping update.")
        return

    saved_state = load_state(folder, migrate_if_needed=True)
    saved_mode = str((saved_state.get("apical_alignment") or {}).get("mode", "")).strip()
    run_selection = saved_mode if saved_mode in ("island", "manual") else "island"

    merge_patch(
        folder,
        {
            "derived": {
                "apical_detection": {
                    "run_selection": run_selection,
                    "apical_height_px": mean_apical_px,
                    "apical_height_microns": mean_apical_microns,
                }
            }
        },
    )
    print(
        "Updated config.yaml derived.apical_detection: "
        f"apical_height_px={mean_apical_px:.2f}, "
        f"apical_height_microns={mean_apical_microns:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument(
        "--skip-threshold",
        action="store_true",
        help="Skip interactive threshold and reuse existing YolkMask.tif",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.work_dir):
        raise ValueError(f"Work directory does not exist: {args.work_dir}")

    config = load_px2micron(args.work_dir)
    saved_state = load_state(args.work_dir, migrate_if_needed=True)
    saved_mode = str((saved_state.get("apical_alignment") or {}).get("mode", "")).strip()

    if saved_mode == "manual":
        print("Saved apical_alignment mode is 'manual'; skipping threshold UI.")
        track_folder = os.path.join(args.work_dir, "track")
        kymo_path = os.path.join(track_folder, "Kymograph.tif")
        if not os.path.exists(kymo_path):
            raise FileNotFoundError(f"Kymograph.tif not found in: {track_folder}")
        with tifffile.TiffFile(kymo_path) as tf:
            kshape = tf.series[0].shape
        if len(kshape) < 2:
            raise ValueError(f"Kymograph.tif has unexpected shape: {kshape}")
        num_timepoints = int(kshape[-1])
        kymo_cfg = pipeline_config_flat(args.work_dir).get("kymograph") or {}
        dt_sec = float(
            kymo_cfg.get(
                "time_interval_sec",
                float(kymo_cfg.get("time_interval_min", 1.0)) * 60.0,
            )
        )
        apical_px = apical_px_using_saved_manual(
            args.work_dir,
            num_timepoints=num_timepoints,
            dt_min=dt_sec / 60.0,
            px2micron=config["px2micron"],
        )
        if apical_px is None:
            print(
                "Skipping derived.apical_detection: manual apical_alignment is present "
                "but track/apical_manual.tsv is missing. Re-save from the desktop."
            )
        else:
            update_apical_height_in_config(args.work_dir, apical_px, config["px2micron"])
        return

    if not args.skip_threshold:
        interactive_threshold_and_save_mask(args.work_dir)
    else:
        print("Skipping interactive threshold step (using existing YolkMask.tif).")

    track_folder = os.path.join(args.work_dir, "track")
    mask_path = os.path.join(track_folder, "YolkMask.tif")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"YolkMask.tif not found in: {track_folder}")
    mask = tifffile.imread(mask_path)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Mask has unexpected shape: {mask.shape}")
    mask_bool = mask > 0

    apical_px = apical_px_using_saved_islands(args.work_dir, mask_bool)
    if apical_px is None:
        print(
            "Skipping derived.apical_detection: no island apical_alignment in config "
            "(need mode: island and non-empty island_labels). "
            "Use the desktop to select islands and Save, then re-run this script."
        )
    else:
        update_apical_height_in_config(args.work_dir, apical_px, config["px2micron"])


if __name__ == "__main__":
    main()


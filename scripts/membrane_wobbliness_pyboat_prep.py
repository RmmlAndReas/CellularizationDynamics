#!/usr/bin/env python3
"""
Interactive membrane wobbliness annotation for delta kymographs.

This script:
- Opens a delta kymograph TIFF (e.g. Kymograph_delta_marked.tif) in ImageJ via pyimagej.
- Lets you draw multiple segmented-line / polyline ROIs along the membrane.
- For each ROI, fits a spline of lateral position vs time and resamples it
  at every movie frame time step.
- Saves one TSV per ROI, suitable as input to pyBOAT (time series signal),
  into a unique subfolder inside the sample's results directory.

Usage (from repository root):

    python scripts/membrane_wobbliness_pyboat_prep.py --folder data/Hermetia/2

This expects:
- <folder>/config.yaml
- <folder>/results/Kymograph_delta_marked.tif
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Tuple

import numpy as np
import yaml
from scipy.interpolate import UnivariateSpline


def load_sample_timing_and_scale(config_path: str) -> Tuple[float, float]:
    """
    Load per-sample config.yaml and return:
      - movie_dt_min : minutes per movie frame
      - px2micron    : lateral microns per pixel
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found at: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError("config.yaml is empty")

    manual_cfg = cfg.get("manual") or {}

    movie_time_interval_sec = float(
        manual_cfg.get("movie_time_interval_sec", 10.0)
    )
    movie_dt_min = movie_time_interval_sec / 60.0

    px2micron = float(manual_cfg.get("px2micron", 1.0))

    print("Loaded timing / scale from config.yaml:")
    print(f"  movie_time_interval_sec : {movie_time_interval_sec} s")
    print(f"  movie_dt_min            : {movie_dt_min:.3f} min/frame")
    print(f"  px2micron               : {px2micron:.6f} µm/px")

    return movie_dt_min, px2micron


def init_imagej():
    """
    Initialise ImageJ in interactive mode via pyimagej.
    """
    try:
        import imagej
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("pyimagej is required. Install with: pip install pyimagej") from exc

    print("Initializing ImageJ (interactive mode)...")
    ij = imagej.init(mode="interactive")
    print("ImageJ initialized.")
    ij.ui().showUI()
    return ij


def open_kymograph(ij, kymo_path: str):
    """
    Open the kymograph in ImageJ and return the image plus its window title.
    """
    if not os.path.exists(kymo_path):
        raise FileNotFoundError(f"Kymograph not found at: {kymo_path}")

    print(f"\nOpening delta kymograph in ImageJ:")
    print(f"  File: {kymo_path}")

    img = ij.io().open(kymo_path)
    ij.ui().show(img)

    # Give ImageJ a moment to create the window and assign a title.
    import time as _time

    _time.sleep(0.3)
    try:
        if hasattr(img, "getTitle"):
            title = img.getTitle()
        else:
            title = os.path.basename(kymo_path)
    except Exception:
        title = os.path.basename(kymo_path)

    print(f"  Window title: {title}")
    return img, title


def get_current_roi(ij, img, title: str):
    """
    Retrieve the current ROI from the active ImageJ window.
    """
    import time as _time

    # Try to find the window by title; fall back to the image reference.
    try:
        img_win = ij.WindowManager.getImage(title)
        if img_win is None:
            img_win = img
            try:
                ij.WindowManager.setCurrentImage(img_win)
            except Exception:
                pass
    except Exception:
        img_win = img

    if img_win is None:
        print("  Could not find the ImageJ window.")
        return None

    try:
        ij.WindowManager.setCurrentImage(img_win)
    except Exception:
        try:
            ij.py.run_macro(f'selectWindow("{title}");')
        except Exception:
            pass

    _time.sleep(0.3)

    roi = None
    try:
        roi = img_win.getRoi()
    except Exception:
        try:
            current_img = ij.WindowManager.getCurrentImage()
            if current_img is not None:
                roi = current_img.getRoi()
        except Exception:
            roi = None

    return roi


def roi_to_time_series(
    roi,
    movie_dt_min: float,
    px2micron: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a segmented-line ROI on a delta kymograph into a regularly sampled
    time series (time_min, position_px).

    Assumes:
    - Time runs vertically (top -> bottom) in the kymograph.
    - ROI X coordinate is lateral membrane position (pixels).
    - ROI Y coordinate is movie frame index (pixels along time axis).
    """
    # Import ROI utilities from the scripts/ directory.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    scripts_dir = os.path.join(repo_root, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from roi_utils import extract_roi_xy  # type: ignore

    x_px, y_px = extract_roi_xy(roi)

    # Map vertical pixel index to time in minutes.
    times = y_px.astype(float) * float(movie_dt_min)
    pos_px = x_px.astype(float)

    if len(times) < 2:
        raise ValueError("Need at least 2 points to build a time series from ROI.")

    # Ensure strictly increasing time axis.
    order = np.argsort(times)
    times = times[order]
    pos_px = pos_px[order]

    # Fit a spline position(t) and resample at each movie frame step.
    t_min = float(times.min())
    t_max = float(times.max())

    # Regular grid matching the movie frame interval.
    n_steps = max(2, int(round((t_max - t_min) / movie_dt_min)) + 1)
    t_eval = t_min + np.arange(n_steps, dtype=float) * movie_dt_min

    # Use an exact or nearly exact spline through the points.
    k = min(3, len(times) - 1)
    spline = UnivariateSpline(times, pos_px, s=0, k=k)
    pos_eval_px = spline(t_eval)

    # Convert lateral displacement to microns for cross-sample comparability.
    pos_eval_um = pos_eval_px * float(px2micron)

    print(
        f"  ROI time series: {len(times)} points -> {len(t_eval)} samples "
        f"from {t_min:.3f} to {t_max:.3f} min "
        f"(px2micron={px2micron:.6f})"
    )

    return t_eval, pos_eval_um


def write_roi_tsv(path: str, times: np.ndarray, positions_um: np.ndarray) -> None:
    """
    Write a single ROI time series to TSV with columns:
    Time_min, Position_um
    """
    if times.size == 0:
        print(f"  No data to write for {path}, skipping.")
        return

    with open(path, "w") as f:
        f.write("Time_min\tPosition_um\n")
        for t, x_um in zip(times, positions_um):
            f.write(f"{float(t):.6f}\t{float(x_um):.6f}\n")

    print(f"  Saved {len(times)} samples to {path}")


def interactive_annotation(
    folder: str,
    movie_dt_min: float,
    px2micron: float,
) -> None:
    """
    Main interactive loop: open delta kymograph, let user draw ROIs,
    export each as a spline-resampled time series for pyBOAT.
    """
    # Delta kymograph path with marks.
    kymo_path = os.path.join(folder, "results", "Kymograph_delta_marked.tif")

    # Prepare results subfolder (unique per run).
    results_root = os.path.join(folder, "results")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(results_root, f"MembraneWobble_pyBOAT_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Membrane wobbliness annotation for pyBOAT")
    print("=" * 60)
    print(f"Sample folder : {folder}")
    print(f"Kymograph     : {kymo_path}")
    print(f"Output folder : {out_dir}")
    print(f"movie_dt_min  : {movie_dt_min:.3f} min/frame")
    print(f"px2micron     : {px2micron:.6f} µm/px")

    ij = init_imagej()
    img, title = open_kymograph(ij, kymo_path)

    # Instructions: write to /dev/tty if available so they are always visible.
    instructions = (
        "\n" + "=" * 60 + "\n"
        "INSTRUCTIONS (Membrane wobble lines for pyBOAT):\n"
        + "=" * 60
        + "\n"
        "A delta kymograph window has been opened in ImageJ.\n"
        "\n"
        "For each membrane track you want to analyse in pyBOAT:\n"
        "  1. Click on the kymograph window to make it active.\n"
        "  2. Use the Segmented Line / Polyline tool to trace the membrane\n"
        "     over time (time runs vertically).\n"
        "  3. Return to this terminal.\n"
        "\n"
        "In this terminal:\n"
        "  - Press Enter after drawing each ROI to record and export it.\n"
        "  - Type 'q' and press Enter when you are done with all lines.\n"
        + "=" * 60
        + "\n"
    )

    try:
        tty_out = open("/dev/tty", "w")
        tty_out.write(instructions)
        tty_out.flush()
        tty_out.close()
    except Exception:
        print(instructions, file=sys.stderr)

    roi_index = 1

    while True:
        # Prompt via /dev/tty when possible so it shows even under Snakemake.
        prompt = (
            "\nDraw a segmented line ROI in ImageJ, then press Enter here "
            "(or type 'q' + Enter to finish): "
        )
        try:
            with open("/dev/tty", "r") as tty_in:
                sys.stdout.write(prompt)
                sys.stdout.flush()
                line = tty_in.readline()
        except Exception:
            # Fallback to stdin
            line = input(prompt)

        if line is None:
            break
        if line.strip().lower().startswith("q"):
            print("Finished ROI annotation loop.")
            break

        print(f"\nProcessing ROI #{roi_index}...")
        roi = get_current_roi(ij, img, title)
        if roi is None:
            print("  No ROI found. Please draw a segmented line and try again.")
            continue

        try:
            roi_type = roi.getTypeAsString()
        except Exception:
            roi_type = "Unknown"
        print(f"  ROI type: {roi_type}")

        try:
            times, positions_um = roi_to_time_series(
                roi,
                movie_dt_min=movie_dt_min,
                px2micron=px2micron,
            )
        except Exception as exc:
            print(f"  Error converting ROI #{roi_index} to time series: {exc}")
            continue

        out_path = os.path.join(out_dir, f"MembraneWobble_pyBOAT_roi{roi_index:02d}.tsv")
        write_roi_tsv(out_path, times, positions_um)

        roi_index += 1

    print("\n" + "=" * 60)
    print("Membrane wobble annotation complete.")
    print(f"Results written under: {out_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Annotate membrane wobbliness on delta kymographs using ImageJ, "
            "and export spline-resampled time series for pyBOAT."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--folder",
        "-f",
        type=str,
        required=True,
        help="Sample folder containing config.yaml and results/Kymograph_delta_marked.tif",
    )

    args = parser.parse_args()

    # Resolve folder relative to repository root if needed.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, os.pardir))

    folder = args.folder
    if not os.path.isabs(folder):
        folder = os.path.join(repo_root, folder)

    if not os.path.isdir(folder):
        raise ValueError(f"Sample folder does not exist: {folder}")

    config_path = os.path.join(folder, "config.yaml")
    movie_dt_min, px2micron = load_sample_timing_and_scale(config_path)

    interactive_annotation(
        folder=folder,
        movie_dt_min=movie_dt_min,
        px2micron=px2micron,
    )


if __name__ == "__main__":
    main()


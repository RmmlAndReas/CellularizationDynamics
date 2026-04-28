#!/usr/bin/env python3
"""
Detect cytoplasm region on the vertical kymograph using an interactive
ImageJ threshold, then quantify apical/basal borders and cytoplasm
thickness over time.

This script is **data-only**: it produces a binary mask and a TSV with
cytoplasm region geometry; all visualization is handled elsewhere.

Workflow (per sample folder):
1. Opens track/Kymograph.tif in ImageJ via pyimagej.
2. You interactively threshold to create a binary mask
   (cytoplasm = white, yolk/background = black).
3. On confirmation in the terminal, the current ImageJ image is saved as
   track/YolkMask.tif.
4. The script reloads YolkMask.tif, finds apical/basal borders per time
   column, and computes cytoplasm thickness over time.
5. Outputs:
   - track/YolkMask.tif        (binary mask)
   - track/cytoplasm_region.tsv
       Columns:
         time_min
         apical_px, apical_microns
         basal_px,  basal_microns
         cytoplasm_height_px, cytoplasm_height_microns
   - Updates config.yaml:
       apical_detection.apical_height_px
       apical_detection.apical_height_microns

Usage:
    python scripts/detect_cytoplasm_region.py --work-dir <work_dir>
"""

import argparse
import os
import sys

import numpy as np
import tifffile
import yaml


def load_px2micron_and_time(config_path):
    """
    Read px2micron from the sample's config.yaml.

    Time interval and time-window are intentionally not loaded here — those
    concerns belong to downstream visualisation steps (generate_outputs.py).
    Storing column indices in the TSV keeps this step independent of the
    imaging interval, so changing ``time_interval_sec`` never requires
    rerunning the interactive mask step.
    """
    if not os.path.exists(config_path):
        raise ValueError(f"config.yaml not found at: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError("config.yaml is empty")
    except Exception as e:
        raise ValueError(f"Error reading config.yaml: {e}")

    if "manual" not in config:
        raise ValueError("'manual' section not found in config.yaml")
    manual = config["manual"]
    if "px2micron" not in manual:
        raise ValueError("px2micron not found in config.yaml under 'manual'")

    return {
        "px2micron": float(manual["px2micron"]),
    }


def interactive_threshold_and_save_mask(folder):
    """
    Open Kymograph.tif in ImageJ, let the user threshold to create a yolk/cytoplasm
    mask, then save it as track/YolkMask.tif.
    """
    try:
        import imagej
    except ImportError:
        raise ImportError("pyimagej is required. Install with: pip install pyimagej")

    track_folder = os.path.join(folder, "track")
    kymograph_path = os.path.join(track_folder, "Kymograph.tif")

    if not os.path.exists(kymograph_path):
        raise FileNotFoundError(f"Kymograph.tif not found in: {track_folder}")

    print(f"\nLoading kymograph from: {folder}")
    print("Initializing ImageJ...")
    ij = imagej.init(mode="interactive")
    print("ImageJ initialized")

    ij.ui().showUI()

    print("\nOpening kymograph in ImageJ...")
    print(f"  Opening {kymograph_path}...")
    img = ij.io().open(kymograph_path)
    ij.ui().show(img)

    # Give ImageJ a moment to create the window, then duplicate once (no dialog).
    import time

    time.sleep(0.3)
    ij.py.run_macro("run('Duplicate...', 'title=KymoReal');")

    try:
        if hasattr(img, "getTitle"):
            title = img.getTitle()
        else:
            title = os.path.basename(kymograph_path)
    except Exception:
        title = os.path.basename(kymograph_path)

    print(f"    Window title: {title}")

    instructions = [
        "",
        "=" * 60,
        "INSTRUCTIONS: Yolk / Cytoplasm Thresholding",
        "=" * 60,
        "A kymograph window has been opened in ImageJ.",
        "",
        "Goal: create a binary mask where",
        "  - cytoplasm is WHITE (foreground)",
        "  - yolk/background is BLACK.",
        "",
        "Suggested steps in ImageJ:",
        "  1. Make sure the kymograph window is active.",
        "  2. Use 'Image > Adjust > Threshold...' or similar tools",
        "     to adjust the mask until cytoplasm is white and yolk is black.",
        "  3. When satisfied, ensure the final binary mask image is the",
        "     active window (e.g. after 'Apply').",
        "",
        "When you are finished and the mask is visible:",
        "  - Return to this terminal and press Enter to continue.",
        "=" * 60,
        "",
        "Press Enter here when the yolk/cytoplasm mask is ready...",
    ]

    try:
        tty_out = open("/dev/tty", "w")
        for line in instructions:
            tty_out.write(line + "\n")
        tty_out.flush()
        tty_out.close()
    except Exception:
        for line in instructions:
            print(line, file=sys.stderr)

    # Wait for confirmation
    try:
        with open("/dev/tty", "r") as tty:
            tty.readline()
    except Exception:
        try:
            if sys.stdin.isatty():
                input()
            else:
                sys.stdin.readline()
        except Exception:
            print(
                "Waiting for marker file .yolk_mask_done in folder...",
                file=sys.stderr,
            )
            marker_file = os.path.join(folder, ".yolk_mask_done")
            while not os.path.exists(marker_file):
                time.sleep(1.0)
            os.remove(marker_file)

    # After confirmation, grab the current image (assumed to be the binary mask)
    try:
        current_img = ij.WindowManager.getCurrentImage()
    except Exception:
        current_img = None

    if current_img is None:
        raise RuntimeError(
            "Could not get current ImageJ image after thresholding. "
            "Make sure the binary mask window is active before pressing Enter."
        )

    # Convert to NumPy array
    try:
        mask_array = ij.py.from_java(current_img)
    except Exception as e:
        raise RuntimeError(f"Could not convert ImageJ image to NumPy array: {e}")

    # Ensure 2D (depth x time)
    if mask_array.ndim > 2:
        mask_array = np.squeeze(mask_array)
    if mask_array.ndim != 2:
        raise ValueError(f"Mask has unexpected shape: {mask_array.shape}")

    # Normalize to 0/255 uint8 for saving
    mask_norm = (mask_array > 0).astype(np.uint8) * 255

    os.makedirs(track_folder, exist_ok=True)
    mask_path = os.path.join(track_folder, "YolkMask.tif")
    tifffile.imwrite(mask_path, mask_norm)
    print(f"Saved yolk/cytoplasm mask to: {mask_path}")

    return mask_path


def compute_cytoplasm_size_over_time(folder, config, min_run_length_px=5):
    """
    Load YolkMask.tif and compute cytoplasm thickness over time.

    White = cytoplasm, Black = yolk/background.
    For each time column:
        - find contiguous white runs along depth
        - ignore runs shorter than min_run_length_px
        - define apical border as start of first valid run
        - define basal border as end of last valid run
        - cytoplasm thickness = basal - apical (pixels)
    """
    track_folder = os.path.join(folder, "track")
    mask_path = os.path.join(track_folder, "YolkMask.tif")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"YolkMask.tif not found in: {track_folder}. "
            "Run the thresholding step first."
        )

    mask = tifffile.imread(mask_path)

    if mask.ndim != 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Mask has unexpected shape: {mask.shape}")

    # Convert to boolean: True = cytoplasm (white)
    mask_bool = mask > 0

    depth, num_timepoints = mask_bool.shape
    print(f"Loaded YolkMask.tif with shape (depth, time): {mask_bool.shape}")

    px2micron = config["px2micron"]

    heights_px = np.full(num_timepoints, np.nan, dtype=float)
    apical_px = np.full(num_timepoints, np.nan, dtype=float)
    basal_px = np.full(num_timepoints, np.nan, dtype=float)

    for t in range(num_timepoints):
        col = mask_bool[:, t]

        best_start = None
        best_end = None
        best_len = 0

        in_run = False
        run_start = 0

        for y in range(depth):
            if col[y] and not in_run:
                in_run = True
                run_start = y
            elif not col[y] and in_run:
                in_run = False
                run_end = y - 1
                run_len = run_end - run_start + 1
                if run_len >= min_run_length_px and run_len > best_len:
                    best_len = run_len
                    best_start = run_start
                    best_end = run_end
        if in_run:
            run_end = depth - 1
            run_len = run_end - run_start + 1
            if run_len >= min_run_length_px and run_len > best_len:
                best_len = run_len
                best_start = run_start
                best_end = run_end

        if best_start is None or best_end is None:
            continue

        apical_px[t] = float(best_start)
        basal_px[t] = float(best_end)
        heights_px[t] = float(best_end - best_start)

    col_idx = np.arange(num_timepoints, dtype=float)

    heights_microns = heights_px * px2micron
    apical_microns = apical_px * px2micron
    basal_microns = basal_px * px2micron

    return (
        col_idx,
        heights_px,
        heights_microns,
        apical_px,
        basal_px,
        apical_microns,
        basal_microns,
    )


def save_cytoplasm_region_tsv(
    folder,
    col_idx,
    heights_px,
    heights_microns,
    apical_px,
    basal_px,
    apical_microns,
    basal_microns,
):
    """
    Save cytoplasm region geometry over time as TSV in track/.

    The first column is ``col_idx`` — the integer column index of the source
    kymograph — rather than time in minutes.  This keeps the output independent
    of ``time_interval_sec``; downstream scripts convert to real time by
    multiplying ``col_idx * dt_min``.
    """
    track_folder = os.path.join(folder, "track")
    os.makedirs(track_folder, exist_ok=True)

    tsv_path = os.path.join(track_folder, "cytoplasm_region.tsv")

    valid = ~np.isnan(heights_microns)

    with open(tsv_path, "w") as f:
        f.write(
            "col_idx"
            "\tapical_px"
            "\tapical_microns"
            "\tbasal_px"
            "\tbasal_microns"
            "\tcytoplasm_height_px"
            "\tcytoplasm_height_microns\n"
        )
        for (
            ci,
            h_px,
            h_um,
            a_px,
            a_um,
            b_px,
            b_um,
            v,
        ) in zip(
            col_idx,
            heights_px,
            heights_microns,
            apical_px,
            apical_microns,
            basal_px,
            basal_microns,
            valid,
        ):
            if not v:
                continue
            f.write(
                f"{int(ci)}\t{a_px:.6f}\t{a_um:.6f}"
                f"\t{b_px:.6f}\t{b_um:.6f}\t{h_px:.6f}\t{h_um:.6f}\n"
            )

    print(f"Saved cytoplasm region TSV to: {tsv_path}")


def update_apical_height_in_config(folder, apical_px, px2micron):
    """
    Compute mean apical height and store it in config.yaml under:

        apical_detection.apical_height_px
        apical_detection.apical_height_microns
    """
    valid = ~np.isnan(apical_px)
    if not np.any(valid):
        print("Warning: no valid apical_px values; skipping apical_detection update.")
        return

    mean_apical_px = float(np.nanmean(apical_px[valid]))
    mean_apical_microns = mean_apical_px * px2micron

    config_path = os.path.join(folder, "config.yaml")
    if not os.path.exists(config_path):
        print(f"Warning: config.yaml not found at {config_path}; skipping apical_detection update.")
        return

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: could not read config.yaml for apical update: {e}")
        config = {}

    if not isinstance(config, dict):
        config = {}

    config.setdefault("apical_detection", {})
    config["apical_detection"]["apical_height_px"] = mean_apical_px
    config["apical_detection"]["apical_height_microns"] = mean_apical_microns

    try:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(
            "Updated config.yaml apical_detection: "
            f"apical_height_px={mean_apical_px:.2f}, "
            f"apical_height_microns={mean_apical_microns:.2f}"
        )
    except Exception as e:
        print(f"Warning: could not write apical_detection to config.yaml: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect cytoplasm region on kymograph via ImageJ threshold "
        "and compute cytoplasm geometry over time (TSV only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help="Working directory containing track/Kymograph.tif and config.yaml",
    )
    parser.add_argument(
        "--skip-threshold",
        action="store_true",
        help="Skip the interactive ImageJ threshold step and reuse existing YolkMask.tif",
    )
    parser.add_argument(
        "--min-run-length-px",
        type=int,
        default=5,
        help="Minimum contiguous cytoplasm run length (in pixels) to treat as valid.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.work_dir):
        raise ValueError(f"Work directory does not exist: {args.work_dir}")

    config_path = os.path.join(args.work_dir, "config.yaml")
    config = load_px2micron_and_time(config_path)

    print("\n" + "=" * 60)
    print("Cytoplasm region detection and geometry quantification")
    print("=" * 60)

    if not args.skip_threshold:
        interactive_threshold_and_save_mask(args.work_dir)
    else:
        print("Skipping interactive threshold step (using existing YolkMask.tif).")

    (
        col_idx,
        heights_px,
        heights_microns,
        apical_px,
        basal_px,
        apical_microns,
        basal_microns,
    ) = compute_cytoplasm_size_over_time(
        args.work_dir,
        config,
        min_run_length_px=args.min_run_length_px,
    )

    save_cytoplasm_region_tsv(
        args.work_dir,
        col_idx,
        heights_px,
        heights_microns,
        apical_px,
        basal_px,
        apical_microns,
        basal_microns,
    )

    update_apical_height_in_config(args.work_dir, apical_px, config["px2micron"])

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


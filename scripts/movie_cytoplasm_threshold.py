#!/usr/bin/env python3
"""
Interactive cytoplasm mask for one 2D timepoint (single-slice pipeline).

Opens the original input movie from the data directory in ImageJ (pyimagej).
You choose/duplicate the frame manually, threshold so cytoplasm is white and
background is black, then make that **2D binary image** the active window and
press Enter in the terminal. The active image is saved as track/CytoplasmMask.tif,
and another open 2D window (the duplicate slice) is saved as
track/MovieSliceDup.tif.

Re-running quantification without ImageJ:
    python scripts/movie_cytoplasm_threshold.py --data-dir ... --work-dir ... --skip-threshold

Non-TTY environments: create an empty file work_dir/.cytoplasm_mask_done after
masking; the script will detect it and continue (then remove the marker).

Usage:
    python scripts/movie_cytoplasm_threshold.py --data-dir <data_dir> --work-dir <work_dir>
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import os
import sys
import time

import numpy as np
import tifffile

from cellularization_paths import resolve_input_movie_path


@contextmanager
def _imagej_session_lock():
    """
    Enforce a single interactive ImageJ session across concurrent processes.

    The lock path can be overridden with YOLK_PYIMAGEJ_LOCK.
    """
    lock_path = os.environ.get("YOLK_PYIMAGEJ_LOCK", "/tmp/yolk_pyimagej_session.lock")
    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    print(f"Waiting for ImageJ session lock: {lock_path}")
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    print("Acquired ImageJ session lock.")
    try:
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        print("Released ImageJ session lock.")


def _is_binary_2d(arr: np.ndarray) -> bool:
    vals = np.unique(arr)
    if vals.size > 3:
        return False
    return np.all(np.isin(vals, np.array([0, 1, 255], dtype=vals.dtype)))


def _image_title(img_obj: object) -> str:
    try:
        return str(img_obj.getTitle())
    except Exception:
        return ""


def interactive_threshold_and_save_mask(work_dir: str, data_dir: str) -> tuple[str, str]:
    try:
        import imagej
    except ImportError as exc:
        raise ImportError("pyimagej is required. Install with: pip install pyimagej") from exc

    track_folder = os.path.join(work_dir, "track")
    movie_path = resolve_input_movie_path(data_dir)
    if not os.path.exists(movie_path):
        raise FileNotFoundError(f"Input movie not found: {movie_path}")

    with _imagej_session_lock():
        print(f"\nLoading input movie: {movie_path}")
        print("Initializing ImageJ...")
        ij = imagej.init(mode="interactive")
        print("ImageJ initialized")

        ij.ui().showUI()

        print("\nOpening movie in ImageJ...")
        img = ij.io().open(movie_path)
        ij.ui().show(img)
        original_title = _image_title(img) or os.path.basename(movie_path)

        instructions = [
            "",
            "=" * 60,
            "INSTRUCTIONS: Single-frame cytoplasm mask",
            "=" * 60,
            "A movie (or single plane) window is open.",
            "",
            "Goal: one 2D binary mask where",
            "  - cytoplasm is WHITE (foreground)",
            "  - background / yolk is BLACK.",
            "",
            "Suggested steps:",
            "  1. Select the stack window and move to the timepoint you want.",
            "     (If needed, duplicate/slice manually in ImageJ for your overlay workflow.)",
            "  2. Image > Adjust > Threshold... (or your preferred tool). Apply to get",
            "     a binary image.",
            "  3. Keep two windows open besides the original movie:",
            "     - one thresholded mask (make THIS window active before continue)",
            "     - one duplicate grayscale slice for overlay export",
            "  4. The **active window** must be the final 2D mask before you continue.",
            "     (If you used the duplicate, threshold that image; avoid leaving a multi-slice",
            "     stack active unless the current slice is the only one that matters and",
            "     the image is effectively 2D.)",
            "",
            "When the 2D mask is ready: return here and press Enter.",
            "=" * 60,
            "",
            "Press Enter when CytoplasmMask is ready to save...",
        ]

        try:
            tty_out = open("/dev/tty", "w")
            for line in instructions:
                tty_out.write(line + "\n")
            tty_out.flush()
            tty_out.close()
        except OSError:
            for line in instructions:
                print(line, file=sys.stderr)

        try:
            with open("/dev/tty", "r") as tty:
                tty.readline()
        except OSError:
            try:
                if sys.stdin.isatty():
                    input()
                else:
                    sys.stdin.readline()
            except Exception:
                print(
                    "Waiting for marker file .cytoplasm_mask_done in work dir...",
                    file=sys.stderr,
                )
                marker_file = os.path.join(work_dir, ".cytoplasm_mask_done")
                while not os.path.exists(marker_file):
                    time.sleep(1.0)
                os.remove(marker_file)

        try:
            current_img = ij.WindowManager.getCurrentImage()
        except Exception:
            current_img = None

        if current_img is None:
            raise RuntimeError(
                "Could not get current ImageJ image. Activate the 2D binary mask window "
                "before pressing Enter."
            )

        try:
            mask_array = ij.py.from_java(current_img)
        except Exception as e:
            raise RuntimeError(f"Could not convert ImageJ image to NumPy array: {e}") from e

        if mask_array.ndim > 2:
            mask_array = np.squeeze(mask_array)
        if mask_array.ndim != 2:
            raise ValueError(
                f"Expected a 2D mask after squeeze; got shape {mask_array.shape}. "
                "Activate a single-slice binary image, not a multi-frame stack."
            )

        os.makedirs(track_folder, exist_ok=True)

        duplicate_candidates: list[tuple[bool, np.ndarray]] = []
        id_list = ij.WindowManager.getIDList()
        if id_list is not None:
            for img_id in id_list:
                try:
                    cand = ij.WindowManager.getImage(int(img_id))
                except Exception:
                    cand = None
                if cand is None:
                    continue
                try:
                    if int(cand.getID()) == int(current_img.getID()):
                        continue
                except Exception:
                    if cand is current_img:
                        continue
                title = _image_title(cand)
                if title == original_title:
                    continue
                try:
                    cand_arr = ij.py.from_java(cand)
                except Exception:
                    continue
                if cand_arr.ndim > 2:
                    cand_arr = np.squeeze(cand_arr)
                if cand_arr.ndim != 2:
                    continue
                duplicate_candidates.append((_is_binary_2d(cand_arr), cand_arr))

        if not duplicate_candidates:
            raise RuntimeError(
                "Could not find a second 2D window for duplicate slice export. "
                "Keep both the threshold mask window and the duplicate grayscale slice open."
            )

        # Prefer a non-binary candidate (typically the duplicate grayscale slice).
        duplicate_array = None
        for is_binary, arr in duplicate_candidates:
            if is_binary:
                continue
            duplicate_array = arr
            break
        if duplicate_array is None:
            duplicate_array = duplicate_candidates[0][1]

        duplicate_path = os.path.join(track_folder, "MovieSliceDup.tif")
        tifffile.imwrite(duplicate_path, duplicate_array)
        print(f"Saved duplicate slice to: {duplicate_path}")

        mask_norm = (mask_array > 0).astype(np.uint8) * 255
        mask_path = os.path.join(track_folder, "CytoplasmMask.tif")
        tifffile.imwrite(mask_path, mask_norm)
        print(f"Saved cytoplasm mask to: {mask_path}")

        return mask_path, duplicate_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Raw data directory containing the input movie TIFF",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Work directory for track outputs",
    )
    parser.add_argument(
        "--skip-threshold",
        action="store_true",
        help="Do not open ImageJ; require existing track/CytoplasmMask.tif",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.work_dir):
        raise SystemExit(f"Not a directory: {args.work_dir}")
    if not os.path.isdir(args.data_dir):
        raise SystemExit(f"Not a directory: {args.data_dir}")

    mask_path = os.path.join(args.work_dir, "track", "CytoplasmMask.tif")
    duplicate_path = os.path.join(args.work_dir, "track", "MovieSliceDup.tif")

    if args.skip_threshold:
        if not os.path.isfile(mask_path):
            raise SystemExit(f"--skip-threshold but missing: {mask_path}")
        if not os.path.isfile(duplicate_path):
            raise SystemExit(f"--skip-threshold but missing: {duplicate_path}")
        print(f"Skipping ImageJ; using existing mask: {mask_path}")
        return

    interactive_threshold_and_save_mask(args.work_dir, args.data_dir)


if __name__ == "__main__":
    main()

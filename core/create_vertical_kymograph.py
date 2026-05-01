#!/usr/bin/env python3
"""
Create Vertical Kymograph from Trimmed Movie

This script creates a single vertical kymograph by averaging intensity along
the horizontal axis for each timepoint and depth position.

Usage:
    python create_vertical_kymograph.py --work-dir <work_dir>
    
The folder should contain:
    - config.yaml with acquisition.source_movie, or legacy Cellularization_trimmed.tif
    
Outputs are saved to:
    - track/Kymograph.tif  (vertical kymograph)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import tifffile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from work_state import get_movie_path, load_state  # noqa: E402


def horizontal_roi_from_averaging_pct(movie_width: int, pct: float) -> tuple[int, int]:
    """
    Centered horizontal span for kymograph averaging: use ``pct`` percent of columns.

    ``pct`` in [1, 100]. 100 uses the full width (0, movie_width).
    """
    w = int(movie_width)
    if w <= 0:
        return 0, 0
    p = float(np.clip(pct, 1.0, 100.0))
    if p >= 100.0 - 1e-9:
        return 0, w
    span = max(1, int(round(w * (p / 100.0))))
    if span >= w:
        return 0, w
    start = (w - span) // 2
    end = start + span
    return start, end


def create_single_kymograph(
    movie,
    width_original,
    width_used,
    width_start,
    width_end,
    output_name,
    folder,
    *,
    record_averaging_width_pct: int | None = None,
):
    """
    Create a single vertical kymograph by averaging intensity along horizontal axis.
    
    Parameters:
    -----------
    movie : numpy array
        Movie array with shape [time, height, width]
    width_original : int
        Original width of the movie
    width_used : int
        Width of region being used
    width_start : int
        Start column index for region
    width_end : int
        End column index for region
    output_name : str
        Output filename (e.g., 'Kymograph.tif')
    folder : str
        Base folder path
    """
    num_frames = movie.shape[0]
    height = movie.shape[1]
    
    print(f"\nCreating kymograph: {output_name}")
    print(f"  Using width: {width_used} pixels (from {width_start} to {width_end})")
    
    # Create vertical kymograph by averaging along horizontal axis
    kymograph = np.mean(movie[:, :, width_start:width_end], axis=2)
    print(f"  Kymograph shape (before transform): {kymograph.shape}")
    print(f"    Time points: {kymograph.shape[0]}")
    print(f"    Depth: {kymograph.shape[1]} pixels")
    
    # Rotate 90 degrees right (clockwise) and flip horizontally
    kymograph = np.rot90(kymograph, k=-1)  # Rotate 90 degrees right (clockwise)
    kymograph = np.fliplr(kymograph)  # Flip horizontally
    
    print(f"  Kymograph shape (after transform): {kymograph.shape}")
    print(f"    Width: {kymograph.shape[1]} pixels")
    print(f"    Height: {kymograph.shape[0]} pixels")
    
    # Create track subfolder if it doesn't exist
    track_folder = os.path.join(folder, "track")
    os.makedirs(track_folder, exist_ok=True)
    
    # Save kymograph
    output_path = os.path.join(track_folder, output_name)
    tifffile.imwrite(output_path, kymograph)
    print(f"  Saved to: {output_path}")
    if record_averaging_width_pct is not None:
        from work_state import merge_patch

        wd = os.path.abspath(folder)
        pct_i = int(max(1, min(100, round(float(record_averaging_width_pct)))))
        merge_patch(wd, {"kymograph": {"averaging_width_pct_last_built": pct_i}})
    return kymograph


def create_vertical_kymographs(folder):
    """
    Create a single vertical kymograph.
    
    Parameters:
    -----------
    folder : str
        Base folder path
    """
    movie_path = get_movie_path(folder)
    print(f"\nLoading acquisition movie from: {movie_path}")
    movie = tifffile.imread(movie_path)
    
    # Inspect movie structure
    shape = movie.shape
    ndim = len(shape)
    
    print(f"Movie shape: {shape}")
    print(f"Movie dtype: {movie.dtype}")
    
    if ndim == 3:
        # Assume [time, height, width]
        num_frames = shape[0]
        height = shape[1]
        width_original = shape[2]
        print(f"  Frames: {num_frames}")
        print(f"  Height: {height} pixels")
        print(f"  Width: {width_original} pixels")
    elif ndim == 2:
        # Single frame, treat as [height, width]
        print("Warning: Movie appears to be 2D (single frame)")
        num_frames = 1
        height = shape[0]
        width_original = shape[1]
        movie = movie[np.newaxis, :, :]  # Add time dimension
    else:
        raise ValueError(f"Unexpected movie dimensions: {ndim}D")
    
    print("\n" + "="*60)
    print("Creating vertical kymograph")
    print("="*60)
    
    state = load_state(folder, migrate_if_needed=True)
    k = state.get("kymograph") or {}
    try:
        pct = float(k.get("averaging_width_pct", 50))
    except (TypeError, ValueError):
        pct = 50.0
    width_start_full, width_end_full = horizontal_roi_from_averaging_pct(width_original, pct)
    width_used = width_end_full - width_start_full
    pct_i = int(max(1, min(100, round(float(pct)))))
    create_single_kymograph(
        movie,
        width_original,
        width_used,
        width_start_full,
        width_end_full,
        "Kymograph.tif",
        folder,
        record_averaging_width_pct=pct_i,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Create vertical kymograph from trimmed movie',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help="Working directory with config.yaml (acquisition.source_movie)",
    )
    
    args = parser.parse_args()
    
    # Validate folder
    if not os.path.isdir(args.work_dir):
        raise ValueError(f"Work directory does not exist: {args.work_dir}")
    
    print("\n" + "="*60)
    print("Creating Vertical Kymographs")
    print("="*60)
    create_vertical_kymographs(args.work_dir)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == '__main__':
    main()


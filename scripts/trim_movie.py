#!/usr/bin/env python3
"""
Copy the timelapse movie into the work tree as Cellularization_trimmed.tif.

Usage:
    python trim_movie.py --work-dir <work_dir> --data-dir <data_dir>

The folder should contain:
    - A timelapse movie (*.tif/*.tiff). Preferably named Cellularization.tif,
      but other TIFF filenames are accepted (auto-detected).
    - config.yaml (validated; temporal spacing lives in manual / kymograph)

Outputs are saved to:
    - Cellularization_trimmed.tif (full frame stack, same as input timing)
"""

import argparse
import os
import tifffile
import yaml
from cellularization_paths import resolve_input_movie_path


def assert_config_readable(config_path):
    """Ensure config.yaml exists and is non-empty YAML."""
    if not os.path.exists(config_path):
        raise ValueError(f"config.yaml not found at: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError("config.yaml is empty")
    except Exception as e:
        raise ValueError(f"Error reading config.yaml: {e}") from e


def trim_movie(work_dir, data_dir):
    """
    Write the full movie stack to Cellularization_trimmed.tif (no frame skipping).

    Parameters
    ----------
    work_dir : str
        Working directory where derived files are written.
    data_dir : str
        Raw data directory containing the input movie.
    """
    movie_path = resolve_input_movie_path(data_dir)

    print(f"\nLoading timelapse movie from: {movie_path}")
    movie = tifffile.imread(movie_path)

    shape = movie.shape
    ndim = len(shape)

    print(f"Original movie shape: {shape}")
    print(f"Original movie dtype: {movie.dtype}")

    if ndim == 3:
        num_frames = shape[0]
        height = shape[1]
        width = shape[2]
        print(f"  Frames: {num_frames}")
        print(f"  Height: {height} pixels")
        print(f"  Width: {width} pixels")
        trimmed = movie
    elif ndim == 2:
        print("Warning: Movie appears to be 2D (single frame)")
        trimmed = movie
    else:
        raise ValueError(f"Unexpected movie dimensions: {ndim}D")

    print(f"\nWriting trimmed movie (all frames): shape {trimmed.shape}")

    os.makedirs(work_dir, exist_ok=True)
    output_path = os.path.join(work_dir, "Cellularization_trimmed.tif")
    tifffile.imwrite(output_path, trimmed)
    print(f"\nSaved trimmed movie to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy timelapse movie to Cellularization_trimmed.tif",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help="Working directory containing config.yaml and output products",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Raw data directory containing the input movie TIFF (default: work-dir)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.work_dir):
        raise ValueError(f"Work directory does not exist: {args.work_dir}")
    data_dir = args.data_dir if args.data_dir else args.work_dir
    if not os.path.isdir(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")

    config_path = os.path.join(args.work_dir, "config.yaml")
    assert_config_readable(config_path)

    print("\n" + "=" * 60)
    print("Trim Movie (copy full stack)")
    print("=" * 60)
    trim_movie(args.work_dir, data_dir)

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

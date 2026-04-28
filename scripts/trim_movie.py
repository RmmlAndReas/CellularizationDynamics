#!/usr/bin/env python3
"""
Trim Movie by Keeping Every Nth Frame

This script trims a timelapse movie by keeping every Nth frame, reducing
the temporal resolution while preserving spatial information.

Usage:
    python trim_movie.py --work-dir <work_dir> --data-dir <data_dir>
    python trim_movie.py --work-dir <work_dir> [--keep-every N]
    
The folder should contain:
    - A timelapse movie (*.tif/*.tiff). Preferably named Cellularization.tif,
      but other TIFF filenames are accepted (auto-detected).
    - config.yaml                 (contains preprocessing.keep_every, default: 6)
    
Outputs are saved to:
    - Cellularization_trimmed.tif (trimmed movie)
"""

import argparse
import os
import tifffile
import yaml
from cellularization_paths import resolve_input_movie_path


def load_config(config_path):
    """
    Load configuration from config.yaml.
    
    Parameters:
    -----------
    config_path : str
        Path to config.yaml file
    
    Returns:
    --------
    dict
        Configuration dictionary with keep_every value
    """
    if not os.path.exists(config_path):
        raise ValueError(f"config.yaml not found at: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError("config.yaml is empty")
    except Exception as e:
        raise ValueError(f"Error reading config.yaml: {e}")
    
    # Get keep_every from preprocessing section, default to 6
    keep_every = 6
    if 'preprocessing' in config and 'keep_every' in config['preprocessing']:
        keep_every = int(config['preprocessing']['keep_every'])
    
    result = {
        'keep_every': keep_every
    }
    
    print(f"Loaded configuration:")
    print(f"  keep_every: {result['keep_every']}")
    
    return result


def trim_movie(work_dir, data_dir, keep_every):
    """
    Trim movie by keeping every Nth frame.
    
    Parameters:
    -----------
    work_dir : str
        Working directory where derived files are written.
    data_dir : str
        Raw data directory containing the input movie.
    keep_every : int
        Keep every Nth frame (e.g., 6 means keep frames 0, 6, 12, ...)
    """
    # Load original movie
    movie_path = resolve_input_movie_path(data_dir)
    
    print(f"\nLoading timelapse movie from: {movie_path}")
    movie = tifffile.imread(movie_path)
    
    # Inspect movie structure
    shape = movie.shape
    ndim = len(shape)
    
    print(f"Original movie shape: {shape}")
    print(f"Original movie dtype: {movie.dtype}")
    
    if ndim == 3:
        # Assume [time, height, width]
        num_frames = shape[0]
        height = shape[1]
        width = shape[2]
        print(f"  Frames: {num_frames}")
        print(f"  Height: {height} pixels")
        print(f"  Width: {width} pixels")
    elif ndim == 2:
        # Single frame, treat as [height, width]
        print("Warning: Movie appears to be 2D (single frame)")
        num_frames = 1
        height = shape[0]
        width = shape[1]
    else:
        raise ValueError(f"Unexpected movie dimensions: {ndim}D")
    
    # Trim movie by keeping every Nth frame
    print(f"\nTrimming movie: keeping every {keep_every}th frame...")
    if ndim == 3:
        trimmed = movie[::keep_every, :, :]
    else:
        # Single frame, no trimming needed
        trimmed = movie
    
    trimmed_frames = trimmed.shape[0]
    print(f"Trimmed movie shape: {trimmed.shape}")
    print(f"  Original frames: {num_frames}")
    print(f"  Trimmed frames: {trimmed_frames}")
    print(f"  Reduction: {num_frames / trimmed_frames:.2f}x")
    
    # Save trimmed movie
    os.makedirs(work_dir, exist_ok=True)
    output_path = os.path.join(work_dir, "Cellularization_trimmed.tif")
    tifffile.imwrite(output_path, trimmed)
    print(f"\nSaved trimmed movie to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Trim movie by keeping every Nth frame',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
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
    
    parser.add_argument(
        '--keep-every',
        type=int,
        default=None,
        help='Keep every Nth frame (overrides config.yaml value)'
    )
    
    args = parser.parse_args()
    
    # Validate folder
    if not os.path.isdir(args.work_dir):
        raise ValueError(f"Work directory does not exist: {args.work_dir}")
    data_dir = args.data_dir if args.data_dir else args.work_dir
    if not os.path.isdir(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Load configuration
    config_path = os.path.join(args.work_dir, 'config.yaml')
    config = load_config(config_path)
    
    # Override with command-line argument if provided
    keep_every = args.keep_every if args.keep_every is not None else config['keep_every']
    
    print("\n" + "="*60)
    print("Trimming Movie")
    print("="*60)
    trim_movie(args.work_dir, data_dir, keep_every)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == '__main__':
    main()


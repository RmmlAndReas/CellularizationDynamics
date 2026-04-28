#!/usr/bin/env python3
"""
Create Vertical Kymograph from Trimmed Movie

This script creates a single vertical kymograph by averaging intensity along
the horizontal axis for each timepoint and depth position.

Usage:
    python create_vertical_kymograph.py --work-dir <work_dir>
    
The folder should contain:
    - Cellularization_trimmed.tif  (trimmed timelapse movie, in track/)
    
Outputs are saved to:
    - track/Kymograph.tif  (vertical kymograph)
"""

import argparse
import os
import numpy as np
import tifffile


def create_single_kymograph(movie, width_original, width_used, width_start, width_end, output_name, folder):
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
    kymograph = []
    
    for frame_idx in range(num_frames):
        frame = movie[frame_idx, :, :]
        
        # Extract region of interest
        frame_roi = frame[:, width_start:width_end]
        
        # Average along horizontal axis (axis=1)
        vertical_profile = np.mean(frame_roi, axis=1)
        kymograph.append(vertical_profile)
        
        if (frame_idx + 1) % 100 == 0:
            print(f"  Processed {frame_idx + 1}/{num_frames} frames...")
    
    # Convert to numpy array
    kymograph = np.array(kymograph)
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


def create_vertical_kymographs(folder):
    """
    Create a single vertical kymograph.
    
    Parameters:
    -----------
    folder : str
        Base folder path
    """
    # Load trimmed movie
    movie_path = os.path.join(folder, "Cellularization_trimmed.tif")
    if not os.path.exists(movie_path):
        raise FileNotFoundError(f"Cellularization_trimmed.tif not found in: {folder}")
    
    print(f"\nLoading trimmed movie from: {movie_path}")
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
    
    # Create single kymograph with full width
    width_full = width_original
    width_start_full = 0
    width_end_full = width_original
    create_single_kymograph(movie, width_original, width_full, width_start_full, width_end_full,
                           "Kymograph.tif", folder)


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
        help="Working directory containing Cellularization_trimmed.tif",
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


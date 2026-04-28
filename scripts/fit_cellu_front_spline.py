#!/usr/bin/env python3
"""
Fit cellularization front spline from VerticalKymoCelluSelection.tsv and save the fit.

This script is now **fit-only**: it reads the annotated cellularization front
points, fits a smoothing spline (front depth vs time), and saves the sampled
curve plus final-height metadata. All plotting/figure generation is handled
by a separate script.

Assumes a folder containing:
    - track/VerticalKymoCelluSelection.tsv   (Time, CelluFront)
    - config.yaml

Outputs:
    - track/VerticalKymoCelluSelection_spline.tsv
        Columns: Time (min), CelluFront_spline (px), CellHeight_microns
    - config.yaml updated with:
        cellularization_front.final_height_px

Usage:
    python fit_cellu_front_spline.py \\
        --work-dir /path/to/work_folder
"""

import argparse
import os

import numpy as np
from scipy.interpolate import UnivariateSpline
import yaml


def load_px2micron_from_config(config_path):
    """
    Load px2micron conversion value from config.yaml.
    
    Parameters:
    -----------
    config_path : str
        Path to config.yaml file
    
        Returns:
        --------
        float
            px2micron conversion value (microns per pixel)
    
    Raises:
    -------
    ValueError
        If config.yaml doesn't exist or doesn't contain px2micron
    """
    if not os.path.exists(config_path):
        raise ValueError(f"config.yaml not found at: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError("config.yaml is empty")
            
            # px2micron is nested under 'manual' section
            if 'manual' not in config:
                raise ValueError("'manual' section not found in config.yaml")
            
            if 'px2micron' not in config['manual']:
                raise ValueError("px2micron not found in config.yaml under 'manual' section")
            
            px2micron = float(config['manual']['px2micron'])
            print(f"Loaded px2micron from config: {px2micron} microns per pixel ({1/px2micron:.2f} pixels per micron)")
            return px2micron
    except Exception as e:
        raise ValueError(f"Error reading px2micron from config.yaml: {e}")


def fit_and_save(folder: str, smoothing: float = 0.0, degree: int = 3, time_interval_min: float = 1.0):
    """
    Fit a smoothing spline to the cellularization front and save the results.

    Parameters
    ----------
    folder : str
        Base folder containing track/VerticalKymoCelluSelection.tsv and track/Kymograph.tif.
    smoothing : float
        Spline smoothing factor `s` (0 = interpolate exactly).
    degree : int
        Spline degree `k` (1–5, typically 3).
    """
    # Use track subfolder
    track_folder = os.path.join(folder, "track")
    tsv_path = os.path.join(track_folder, "VerticalKymoCelluSelection.tsv")

    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(f"VerticalKymoCelluSelection.tsv not found in: {track_folder}")

    # Load px2micron and apical height from config.yaml (in the folder)
    config_path = os.path.join(folder, 'config.yaml')
    px2micron = load_px2micron_from_config(config_path)
    
    # Load apical height from config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config and 'apical_detection' in config and 'apical_height_microns' in config['apical_detection']:
                avg_height_microns = float(config['apical_detection']['apical_height_microns'])
                print(f"Loaded apical height from config: {avg_height_microns:.2f} microns")
            else:
                # Fallback: try to get from pixels if microns not available
                if config and 'apical_detection' in config and 'apical_height_px' in config['apical_detection']:
                    avg_height_px = float(config['apical_detection']['apical_height_px'])
                    avg_height_microns = avg_height_px * px2micron
                    print(f"Loaded apical height from config (converted from pixels): {avg_height_microns:.2f} microns")
                else:
                    raise ValueError("apical_height not found in config.yaml")
    except Exception as e:
        raise ValueError(f"Error reading apical height from config.yaml: {e}")

    print(f"Loading cellularization front data from: {tsv_path}")
    data = np.loadtxt(tsv_path, delimiter="\t", skiprows=1)
    time_min = data[:, 0]      # Time is already in minutes (from annotation/averaging)
    cellu_front = data[:, 1]    # front depth in pixels

    print(f"Loaded {len(time_min)} points")
    print(f"Time range: {time_min.min():.2f} to {time_min.max():.2f} minutes")

    # Time is already in minutes, no conversion needed

    # Sort by time to ensure strictly increasing order (required for s=0)
    sort_indices = np.argsort(time_min)
    time_min = time_min[sort_indices]
    cellu_front = cellu_front[sort_indices]
    
    # Remove duplicates by averaging (if any exist)
    unique_times, unique_indices = np.unique(time_min, return_index=True)
    if len(unique_times) < len(time_min):
        print(f"Warning: Found {len(time_min) - len(unique_times)} duplicate time values, averaging them")
        # Group by unique time and average the corresponding cellu_front values
        unique_front = np.zeros_like(unique_times)
        for i, t in enumerate(unique_times):
            mask = time_min == t
            unique_front[i] = np.mean(cellu_front[mask])
        time_min = unique_times
        cellu_front = unique_front
        print(f"After deduplication: {len(time_min)} unique points")

    # Fit smoothing spline: CelluFront = f(Time_min)
    print(f"Fitting UnivariateSpline with s={smoothing}, k={degree}")
    spline = UnivariateSpline(time_min, cellu_front, s=smoothing, k=degree)

    # Sample fitted curve for saving
    t_fit = np.linspace(time_min.min(), time_min.max(), 500)
    y_fit = spline(t_fit)
    
    # Find maximum depth from spline
    max_depth_px = np.max(y_fit)
    max_depth_microns = max_depth_px * px2micron
    max_depth_shifted = max_depth_microns - avg_height_microns
    print(f"Maximum depth from spline: {max_depth_microns:.2f} microns (relative to apical: {max_depth_shifted:.2f} microns)")

    # Save spline data to TSV file (Time in minutes, CelluFront_spline in pixels, CellHeight in microns)
    spline_tsv_path = os.path.join(track_folder, "VerticalKymoCelluSelection_spline.tsv")
    # Calculate cell height in microns (cellularization front - apical height)
    y_fit_microns = y_fit * px2micron
    cell_height_microns = y_fit_microns - avg_height_microns
    spline_data = np.column_stack([t_fit, y_fit, cell_height_microns])
    np.savetxt(spline_tsv_path, spline_data, delimiter="\t", 
               header="Time\tCelluFront_spline\tCellHeight_microns", 
               fmt="%.6f", comments="")
    print(f"Saved spline data to: {spline_tsv_path}")
    
    # Save final_height to metadata file (for Snakemake tracking)
    metadata_path = os.path.join(folder, 'metadata.yaml')
    
    # Read existing metadata if it exists
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not parse existing metadata.yaml: {e}, creating new")
            metadata = {}
    else:
        metadata = {}
    
    # Update with cellularization front results
    metadata["cellularization_front"] = {
        "final_height_px": float(max_depth_px),
    }
    
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    
    print(f"Saved cellularization front metadata to {metadata_path}")
    
    # Also update config.yaml for backward compatibility
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None:
                config = {}
            elif not isinstance(config, dict):
                config = {}
    except Exception as e:
        print(f"Warning: Could not parse config.yaml: {e}, skipping update")
        config = {}
    
    # Namespace results under 'cellularization_front' to avoid clobbering other keys
    config.setdefault("cellularization_front", {})
    config["cellularization_front"]["final_height_px"] = float(max_depth_px)
    
    # Write back to config file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Updated config.yaml with cellularization front results")

    return spline


def main():
    parser = argparse.ArgumentParser(
        description="Fit cellularization front function from VerticalKymoCelluSelection.tsv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help="Working directory containing track/VerticalKymoCelluSelection.tsv and config.yaml",
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="Spline smoothing factor s (0 = interpolate exactly, default: 0.0)",
    )

    parser.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Spline degree k (1–5, default: 3)",
    )

    parser.add_argument(
        "--time-interval-min",
        type=float,
        default=1.0,
        help="Time interval between kymograph columns in minutes (default: 1.0)",
    )

    args = parser.parse_args()

    fit_and_save(
        folder=args.work_dir,
        smoothing=args.smoothing,
        degree=args.degree,
        time_interval_min=args.time_interval_min,
    )


if __name__ == "__main__":
    main()



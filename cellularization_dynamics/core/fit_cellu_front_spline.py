#!/usr/bin/env python3
"""
Fit cellularization front spline from saved annotation and save the fit.

Assumes a folder containing:
    - config.yaml with apical_alignment (v2+) and track/apical_front.tsv
    - config.yaml (unified v2)
    - straightening metadata in config.yaml (run straighten_kymograph first)

Outputs:
    - track/VerticalKymoCelluSelection_spline.tsv
    - Updates config.yaml: spline_fit + derived.cellularization_front.final_height_px
"""

import argparse
import os

import numpy as np
from scipy.interpolate import UnivariateSpline

from .annotation_source import load_annotation_time_depth
from .work_state import merge_patch, pipeline_config_flat, straightening_meta


def load_px2micron_from_work_dir(folder: str) -> float:
    cfg = pipeline_config_flat(folder)
    manual = cfg.get("manual") or {}
    if "px2micron" not in manual:
        raise ValueError("manual.px2micron missing in config.yaml")
    px2micron = float(manual["px2micron"])
    print(
        f"Loaded px2micron from config: {px2micron} microns per pixel "
        f"({1 / px2micron:.2f} pixels per micron)"
    )
    return px2micron


def fit_and_save(folder: str, smoothing: float = 0.0, degree: int = 3, time_interval_min: float = 1.0):
    track_folder = os.path.join(folder, "track")

    px2micron = load_px2micron_from_work_dir(folder)

    time_min, cellu_front = load_annotation_time_depth(track_folder)
    print(f"Loaded cellularization front from apical_alignment ({len(time_min)} points)")

    meta = straightening_meta(folder)
    if not meta or "ref_row" not in meta:
        raise FileNotFoundError(
            "straightening metadata missing in config.yaml. "
            "Run straighten_kymograph before fit_cellu_front_spline."
        )
    ref_row = int(meta["ref_row"])
    print(f"Loaded ref_row from unified config: {ref_row} px")

    sort_indices = np.argsort(time_min)
    time_min = time_min[sort_indices]
    cellu_front = cellu_front[sort_indices]

    unique_times, unique_indices = np.unique(time_min, return_index=True)
    if len(unique_times) < len(time_min):
        print(f"Warning: Found {len(time_min) - len(unique_times)} duplicate time values, averaging them")
        unique_front = np.zeros_like(unique_times)
        for i, t in enumerate(unique_times):
            mask = time_min == t
            unique_front[i] = np.mean(cellu_front[mask])
        time_min = unique_times
        cellu_front = unique_front
        print(f"After deduplication: {len(time_min)} unique points")

    print(f"Fitting UnivariateSpline with s={smoothing}, k={degree}")
    spline = UnivariateSpline(time_min, cellu_front, s=smoothing, k=degree)

    t_fit = np.linspace(time_min.min(), time_min.max(), 500)
    y_fit = spline(t_fit)

    max_depth_px = np.max(y_fit)
    max_depth_um = (max_depth_px - ref_row) * px2micron
    print(f"Maximum depth from spline: {max_depth_um:.2f} µm below apical")

    spline_tsv_path = os.path.join(track_folder, "VerticalKymoCelluSelection_spline.tsv")
    cell_height_microns = (y_fit - ref_row) * px2micron
    spline_data = np.column_stack([t_fit, y_fit, cell_height_microns])
    np.savetxt(
        spline_tsv_path,
        spline_data,
        delimiter="\t",
        header="Time\tCelluFront_spline\tCellHeight_microns",
        fmt="%.6f",
        comments="",
    )
    print(f"Saved spline data to: {spline_tsv_path}")

    merge_patch(
        folder,
        {
            "spline_fit": {"cellularization_front": {"final_height_px": float(max_depth_px)}},
            "derived": {
                "cellularization_front": {"final_height_px": float(max_depth_px)},
            },
        },
    )
    print("Updated config.yaml (spline_fit + derived.cellularization_front)")

    return spline


def main():
    parser = argparse.ArgumentParser(
        description="Fit cellularization front from unified apical_alignment (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--work-dir", type=str, required=True)
    args = parser.parse_args()
    if not os.path.isdir(args.work_dir):
        raise SystemExit(f"Not a directory: {args.work_dir}")
    fit_and_save(args.work_dir)


if __name__ == "__main__":
    main()

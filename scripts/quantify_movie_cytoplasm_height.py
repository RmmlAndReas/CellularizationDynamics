#!/usr/bin/env python3
"""
Quantify cytoplasm thickness along X from a 2D CytoplasmMask.tif.

For each column x, scans along Y using the same run logic as detect_cytoplasm_region.

Outputs (under track/):
  - cytoplasm_height_vs_x.csv
  - cytoplasm_height_summary.csv

Usage:
    python scripts/quantify_movie_cytoplasm_height.py --work-dir <work_dir>
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List

import numpy as np
import tifffile
import yaml

from cytoplasm_column_geometry import longest_cytoplasm_run_along_y


def _load_px2micron(work_dir: str) -> float:
    config_path = os.path.join(work_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    manual = cfg.get("manual") or {}
    if "px2micron" not in manual:
        raise ValueError("manual.px2micron missing in config.yaml")
    return float(manual["px2micron"])


def quantify_from_mask(
    work_dir: str,
    min_run_length_px: int = 5,
) -> tuple[str, str]:
    track = os.path.join(work_dir, "track")
    mask_path = os.path.join(track, "CytoplasmMask.tif")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Missing mask: {mask_path}")

    mask = tifffile.imread(mask_path)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"CytoplasmMask.tif must be 2D (or squeezable to 2D); got {mask.shape}")

    height_px, width_px = mask.shape
    mask_bool = mask > 0
    px2um = _load_px2micron(work_dir)

    vs_x_path = os.path.join(track, "cytoplasm_height_vs_x.csv")
    summary_path = os.path.join(track, "cytoplasm_height_summary.csv")

    rows_vs_x: List[Dict[str, Any]] = []
    heights_valid: List[float] = []

    for x in range(width_px):
        col = mask_bool[:, x]
        ap, ba, h = longest_cytoplasm_run_along_y(col, min_run_length_px=min_run_length_px)
        if ap is None or ba is None or h is None:
            continue
        hum = h * px2um
        heights_valid.append(h)
        rows_vs_x.append(
            {
                "x_idx": x,
                "apical_px": ap,
                "basal_px": ba,
                "cytoplasm_height_px": h,
                "cytoplasm_height_um": hum,
            }
        )

    os.makedirs(track, exist_ok=True)
    with open(vs_x_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "x_idx",
                "apical_px",
                "basal_px",
                "cytoplasm_height_px",
                "cytoplasm_height_um",
            ],
        )
        w.writeheader()
        for r in rows_vs_x:
            w.writerow(r)

    n_x = len(heights_valid)
    if n_x == 0:
        summary = {
            "mean_height_px": "",
            "median_height_px": "",
            "std_height_px": "",
            "mean_height_um": "",
            "median_height_um": "",
            "std_height_um": "",
            "n_x_valid": 0,
            "px2micron": px2um,
            "min_run_length_px": min_run_length_px,
        }
    else:
        arr = np.array(heights_valid, dtype=float)
        summary = {
            "mean_height_px": float(np.mean(arr)),
            "median_height_px": float(np.median(arr)),
            "std_height_px": float(np.std(arr, ddof=0)),
            "mean_height_um": float(np.mean(arr) * px2um),
            "median_height_um": float(np.median(arr) * px2um),
            "std_height_um": float(np.std(arr, ddof=0) * px2um),
            "n_x_valid": n_x,
            "px2micron": px2um,
            "min_run_length_px": min_run_length_px,
        }

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print(f"Wrote {len(rows_vs_x)} rows to {vs_x_path}")
    print(f"Wrote summary to {summary_path}")
    return vs_x_path, summary_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", required=True)
    p.add_argument(
        "--min-run-length-px",
        type=int,
        default=5,
        help="Minimum contiguous cytoplasm run (pixels) per column.",
    )
    args = p.parse_args()
    if not os.path.isdir(args.work_dir):
        raise SystemExit(f"Not a directory: {args.work_dir}")
    quantify_from_mask(args.work_dir, min_run_length_px=args.min_run_length_px)


if __name__ == "__main__":
    main()

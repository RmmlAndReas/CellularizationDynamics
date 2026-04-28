#!/usr/bin/env python3
"""
Export a unified geometry timeseries CSV per sample.

Computes cytoplasm geometry directly from track/YolkMask.tif and merges it with
track/VerticalKymoCelluSelection_spline.tsv on the kymograph time grid.

Output: track/geometry_timeseries.csv

Columns:
  time_min, apical_px_raw, front_minus_apical_px, front_minus_apical_um
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import tifffile
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from mask_utils import select_cytoplasm_run


CSV_COLUMNS: List[str] = [
    "time_min",
    "apical_px_raw",
    "front_raw_px",
    "front_minus_apical_px",
    "front_minus_apical_um",
]


def _load_config(folder: str) -> Tuple[float, float, float]:
    """Return (px2micron, ref_row, kymo_time_interval_sec)."""
    path = os.path.join(folder, "config.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.yaml not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    manual = cfg.get("manual", {})
    if "px2micron" not in manual:
        raise ValueError("manual.px2micron missing in config.yaml")
    px2micron = float(manual["px2micron"])
    kymo = cfg.get("kymograph", {})
    dt_sec = float(
        kymo.get(
            "time_interval_sec",
            float(kymo.get("time_interval_min", 1.0)) * 60.0,
        )
    )

    # ref_row comes from the pre-computed straighten_metadata (single source of truth).
    meta_path = os.path.join(folder, "track", "straighten_metadata.yaml")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"straighten_metadata.yaml not found: {meta_path}. "
            "Run straighten_kymograph before export_geometry_timeseries."
        )
    with open(meta_path) as f:
        meta = yaml.safe_load(f) or {}
    ref_row = int(meta["ref_row"])

    return px2micron, ref_row, dt_sec


def export_geometry_timeseries(folder: str) -> str:
    track = os.path.join(folder, "track")
    mask_path = os.path.join(track, "YolkMask.tif")
    spline_path = os.path.join(track, "VerticalKymoCelluSelection_spline.tsv")
    out_path = os.path.join(track, "geometry_timeseries.csv")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"YolkMask.tif not found: {mask_path}")
    if not os.path.exists(spline_path):
        raise FileNotFoundError(f"VerticalKymoCelluSelection_spline.tsv not found: {spline_path}")

    px2micron, ref_row, kymo_dt_sec = _load_config(folder)
    dt_min = kymo_dt_sec / 60.0

    mask = tifffile.imread(mask_path)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"YolkMask.tif has unexpected shape: {mask.shape}")
    mask_bool = mask > 0
    depth, num_timepoints = mask_bool.shape

    col_idx = np.arange(num_timepoints, dtype=int)
    apical_px = np.full(num_timepoints, np.nan, dtype=float)
    basal_px = np.full(num_timepoints, np.nan, dtype=float)
    min_run_length_px = 5

    for t in range(num_timepoints):
        col = mask_bool[:, t]
        best_start, best_end = select_cytoplasm_run(col, min_run_length_px)
        if best_start is None or best_end is None:
            continue
        apical_px[t] = float(best_start)
        basal_px[t] = float(best_end)

    sp = np.loadtxt(spline_path, delimiter="\t", skiprows=1)
    if sp.ndim == 1:
        sp = np.expand_dims(sp, axis=0)
    spline_time = sp[:, 0].astype(float)
    spline_front = sp[:, 1].astype(float)

    time_min = col_idx.astype(float) * dt_min
    front_px = np.interp(
        time_min,
        spline_time,
        spline_front,
        left=np.nan,
        right=np.nan,
    )

    rows: List[Dict[str, Any]] = []
    for i in range(len(col_idx)):
        fpx = float(front_px[i])
        apx = apical_px[i]
        valid_apical = np.isfinite(apx) and not np.isnan(apx)
        valid_fr = np.isfinite(fpx) and not np.isnan(fpx)

        # Keep only timepoints where the front is present and apical is valid.
        if not (valid_fr and valid_apical):
            continue

        # front_minus_apical uses ref_row (the fixed apical reference from straightening),
        # so the value is identical to what was drawn and stored in straight-px space.
        fmapx = fpx - ref_row
        fmap_um = fmapx * px2micron
        front_raw = apx + fmapx   # raw movie pixel row of the front

        rows.append(
            {
                "time_min": float(time_min[i]),
                "apical_px_raw": apx,
                "front_raw_px": front_raw,
                "front_minus_apical_px": fmapx,
                "front_minus_apical_um": fmap_um,
            }
        )

    os.makedirs(track, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {k: r[k] for k in CSV_COLUMNS}
            for k, v in out.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    out[k] = ""
                elif v is None:
                    out[k] = ""
            w.writerow(out)

    print(f"Wrote {len(rows)} rows to {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Export track/geometry_timeseries.csv for one sample.")
    p.add_argument("--work-dir", required=True, help="Working directory with track/ and config.yaml")
    args = p.parse_args()
    if not os.path.isdir(args.work_dir):
        raise SystemExit(f"Not a directory: {args.work_dir}")
    export_geometry_timeseries(args.work_dir)


if __name__ == "__main__":
    main()

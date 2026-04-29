#!/usr/bin/env python3
"""
Pre-compute and persist the apical-straightened kymograph.

Each column of Kymograph.tif is shifted vertically so that the apical
cytoplasm border (from YolkMask.tif) aligns to a common reference row
(the topmost detected apical row across all timepoints).

Inputs (from work_dir):
    - track/Kymograph.tif
    - track/YolkMask.tif
    - config.yaml

Outputs (in work_dir/track/):
    - Kymograph_straightened.tif   – column-shifted kymograph
    - straighten_metadata.yaml     – ref_row, crop_top_px, per-column shifts
"""

import argparse
import os
import sys

import numpy as np
import tifffile
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from mask_utils import select_cytoplasm_run


def run(work_dir: str) -> None:
    track = os.path.join(work_dir, "track")
    kymo_path = os.path.join(track, "Kymograph.tif")
    mask_path = os.path.join(track, "YolkMask.tif")
    config_path = os.path.join(work_dir, "config.yaml")

    for p in [kymo_path, mask_path, config_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    kymo = tifffile.imread(kymo_path)
    if kymo.ndim > 2:
        kymo = np.squeeze(kymo)
    if kymo.ndim != 2:
        raise ValueError(f"Kymograph has unexpected shape: {kymo.shape}")

    mask = tifffile.imread(mask_path)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    mask_bool = mask > 0

    if mask_bool.shape != kymo.shape:
        raise ValueError(
            f"Mask shape {mask_bool.shape} does not match kymograph shape {kymo.shape}"
        )

    depth, num_timepoints = kymo.shape
    apical_px = np.full(num_timepoints, np.nan, dtype=float)
    for t in range(num_timepoints):
        best_start, _ = select_cytoplasm_run(mask_bool[:, t], min_run_length_px=5)
        if best_start is not None:
            apical_px[t] = float(best_start)

    valid = ~np.isnan(apical_px)
    if not np.any(valid):
        raise ValueError("No valid apical border detected in YolkMask.tif")

    # Reserve ~2 µm headroom above apical.
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    px2micron = float(cfg.get("manual", {}).get("px2micron", 1.0))
    margin_px = int(round(2.0 / max(px2micron, 1e-9)))
    ref_row = int(np.nanmin(apical_px[valid])) + margin_px
    shifts = np.zeros(num_timepoints, dtype=int)
    shifts[valid] = (ref_row - apical_px[valid]).astype(int)

    straight_kymo = np.zeros_like(kymo)
    for t in range(num_timepoints):
        if not valid[t]:
            continue
        shift = int(shifts[t])
        for y in range(depth):
            new_y = y + shift
            if 0 <= new_y < depth:
                straight_kymo[new_y, t] = kymo[y, t]

    margin_px = int(round(2.0 / max(px2micron, 1e-9)))
    crop_top = max(0, ref_row - margin_px)

    out_tif = os.path.join(track, "Kymograph_straightened.tif")
    tifffile.imwrite(out_tif, straight_kymo)
    print(f"Saved: {out_tif}")

    meta = {
        "ref_row": int(ref_row),
        "crop_top_px": int(crop_top),
        "shifts": [int(s) for s in shifts.tolist()],
    }
    meta_path = os.path.join(track, "straighten_metadata.yaml")
    with open(meta_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
    print(f"Saved: {meta_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--work-dir",
        required=True,
        help="Working directory containing track/ and config.yaml",
    )
    args = p.parse_args()
    if not os.path.isdir(args.work_dir):
        raise SystemExit(f"Not a directory: {args.work_dir}")
    run(args.work_dir)


if __name__ == "__main__":
    main()

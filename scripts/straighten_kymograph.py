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
    - track/apical_alignment.yaml (optional; written by the desktop on Save)
      If present with mode ``island``, uses the same per-column island-center rule as the UI.

Outputs (in work_dir/track/):
    - Kymograph_straightened.tif   – column-shifted kymograph
    - straighten_metadata.yaml     – ref_row, crop_top_px, per-column shifts, apical_px_by_col
"""

import argparse
import os
import sys
from typing import cast

import numpy as np
import tifffile
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from mask_utils import (
    STRAIGHTEN_METADATA_VERSION,
    ApicalMode,
    build_alignment,
    serialize_apical_px_for_yaml,
)


def _load_apical_mode(track: str) -> tuple[str, list[int] | None]:
    """Return (mode, island_labels or None). Default longest_run if file missing."""
    align_path = os.path.join(track, "apical_alignment.yaml")
    if not os.path.isfile(align_path):
        return "longest_run", None
    with open(align_path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    mode = str(doc.get("mode", "longest_run")).strip()
    if mode not in ("longest_run", "island"):
        mode = "longest_run"
    raw = doc.get("island_labels") or []
    island_labels = [int(x) for x in raw]
    if mode == "island" and not island_labels:
        raise ValueError(
            f"{align_path} has mode island but empty island_labels. "
            "Select islands in the desktop and Save, or set mode to longest_run."
        )
    return mode, island_labels if mode == "island" else None


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
    mode, island_labels = _load_apical_mode(track)

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    px2micron = float(cfg.get("manual", {}).get("px2micron", 1.0))

    alignment = build_alignment(
        mask_bool,
        px2micron=px2micron,
        mode=cast(ApicalMode, mode),
        island_labels=island_labels,
        min_run_length_px=5,
    )

    apical_px = alignment.apical_px_by_col
    valid = ~np.isnan(apical_px)
    shifts = alignment.shifts_by_col
    ref_row = alignment.ref_row
    crop_top = alignment.crop_top_px

    straight_kymo = np.zeros_like(kymo)
    for t in range(num_timepoints):
        if not valid[t]:
            continue
        shift = int(shifts[t])
        for y in range(depth):
            new_y = y + shift
            if 0 <= new_y < depth:
                straight_kymo[new_y, t] = kymo[y, t]

    out_tif = os.path.join(track, "Kymograph_straightened.tif")
    tifffile.imwrite(out_tif, straight_kymo)
    print(f"Saved: {out_tif}")

    meta = {
        "version": int(STRAIGHTEN_METADATA_VERSION),
        "ref_row": int(ref_row),
        "crop_top_px": int(crop_top),
        "shifts": [int(s) for s in shifts.tolist()],
        "apical_mode": mode,
        "apical_px_by_col": serialize_apical_px_for_yaml(apical_px),
    }
    if mode == "island":
        meta["island_labels"] = [int(x) for x in (island_labels or [])]
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

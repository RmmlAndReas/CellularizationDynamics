#!/usr/bin/env python3
"""
Pre-compute and persist the apical-straightened kymograph.

Each column of Kymograph.tif is shifted vertically so that the apical
cytoplasm border (from YolkMask.tif) aligns to a common reference row
(the topmost detected apical row across all timepoints).

Inputs (from work_dir):
    - track/Kymograph.tif
    - track/YolkMask.tif
    - config.yaml (unified v2; apical_alignment for island mode)

Outputs (in work_dir/track/):
    - Kymograph_straightened.tif   – column-shifted kymograph
    Per-column ``shifts`` / ``apical_px_by_col`` are stored in ``track/straightening_columns.tsv``;
    scalars (``ref_row``, ``crop_top_px``, etc.) in ``config.yaml`` under ``straightening``.
"""

import argparse
import os
import sys
from typing import cast

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(__file__))
from annotation_source import load_apical_alignment_doc  # noqa: E402
from mask_utils import (  # noqa: E402
    STRAIGHTEN_METADATA_VERSION,
    ApicalMode,
    build_alignment,
)
from track_tabular import write_straightening_columns_tsv  # noqa: E402
from work_state import merge_patch, pipeline_config_flat  # noqa: E402


def _load_apical_mode(track: str) -> tuple[str, list[int] | None]:
    """Return (mode, island_labels or None). Default longest_run if missing."""
    doc = load_apical_alignment_doc(track)
    if not doc:
        return "longest_run", None
    mode = str(doc.get("mode", "longest_run")).strip()
    if mode not in ("longest_run", "island"):
        mode = "longest_run"
    raw = doc.get("island_labels") or []
    island_labels = [int(x) for x in raw]
    if mode == "island" and not island_labels:
        raise ValueError(
            "apical_alignment has mode island but empty island_labels. "
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

    cfg_flat = pipeline_config_flat(work_dir)
    px2micron = float(cfg_flat["manual"]["px2micron"])

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
    for t in np.where(valid)[0]:
        s = int(shifts[t])
        if s >= 0:
            straight_kymo[s:, t] = kymo[:depth - s, t]
        else:
            straight_kymo[:depth + s, t] = kymo[-s:, t]

    out_tif = os.path.join(track, "Kymograph_straightened.tif")
    tifffile.imwrite(out_tif, straight_kymo)
    print(f"Saved: {out_tif}")

    write_straightening_columns_tsv(work_dir, shifts, apical_px)
    meta = {
        "version": int(STRAIGHTEN_METADATA_VERSION),
        "ref_row": int(ref_row),
        "crop_top_px": int(crop_top),
        "apical_mode": mode,
    }
    if mode == "island":
        meta["island_labels"] = [int(x) for x in (island_labels or [])]
    merge_patch(work_dir, {"straightening": meta})
    print("Updated track/straightening_columns.tsv and config.yaml straightening section")


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

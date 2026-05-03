#!/usr/bin/env python3
"""
Pre-compute and persist the apical-straightened kymograph.

Each column of Kymograph.tif is shifted vertically so that the apical border
aligns to a common reference row. Two apical-detection modes are supported:
  - island: row = mean of selected connected components in track/YolkMask.tif
  - manual: row = spline-smoothed user polyline in track/apical_manual.tsv

Inputs (from work_dir):
    - track/Kymograph.tif
    - track/YolkMask.tif           (island mode only)
    - track/apical_manual.tsv      (manual mode only)
    - config.yaml (unified v2; apical_alignment section)

Outputs (in work_dir/track/):
    - Kymograph_straightened.tif   – column-shifted kymograph
    Per-column ``shifts`` / ``apical_px_by_col`` are stored in ``track/straightening_columns.tsv``;
    scalars (``ref_row``, ``crop_top_px``, etc.) in ``config.yaml`` under ``straightening``.
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np
import tifffile

from .annotation_source import load_apical_alignment_doc
from .apical_manual import apical_px_from_manual_polyline
from .mask_utils import (
    STRAIGHTEN_METADATA_VERSION,
    alignment_from_apical_px,
    build_alignment,
)
from .track_tabular import read_apical_manual_tsv, write_straightening_columns_tsv
from .work_state import merge_patch, pipeline_config_flat


@dataclass
class _ApicalAlignmentSpec:
    mode: str
    island_labels: list[int]
    manual_sigma_um: float


def _load_apical_mode(track: str) -> _ApicalAlignmentSpec:
    """Return the saved apical-alignment mode and its mode-specific parameters."""
    doc = load_apical_alignment_doc(track)
    if not doc:
        raise ValueError(
            "config.yaml is missing apical_alignment. "
            "Use the desktop app: select islands or draw a manual polyline and Save."
        )
    mode = str(doc.get("mode", "island")).strip()
    if mode == "longest_run":
        raise ValueError(
            "Saved apical_alignment mode 'longest_run' is no longer supported. "
            "Re-save from the desktop using island or manual mode."
        )
    if mode == "island":
        raw = doc.get("island_labels") or []
        island_labels = [int(x) for x in raw]
        if not island_labels:
            raise ValueError(
                "apical_alignment has mode island but empty island_labels. "
                "Select islands in the desktop and Save."
            )
        return _ApicalAlignmentSpec(mode=mode, island_labels=island_labels, manual_sigma_um=0.0)
    if mode == "manual":
        sigma = float(doc.get("manual_sigma_um", 0.5))
        return _ApicalAlignmentSpec(mode=mode, island_labels=[], manual_sigma_um=sigma)
    raise ValueError(f"Unsupported apical_alignment mode: {mode!r}")


def run(work_dir: str) -> None:
    track = os.path.join(work_dir, "track")
    kymo_path = os.path.join(track, "Kymograph.tif")
    config_path = os.path.join(work_dir, "config.yaml")

    for p in [kymo_path, config_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    kymo = tifffile.imread(kymo_path)
    if kymo.ndim > 2:
        kymo = np.squeeze(kymo)
    if kymo.ndim != 2:
        raise ValueError(f"Kymograph has unexpected shape: {kymo.shape}")

    depth, num_timepoints = kymo.shape
    spec = _load_apical_mode(track)

    cfg_flat = pipeline_config_flat(work_dir)
    px2micron = float(cfg_flat["manual"]["px2micron"])

    if spec.mode == "island":
        mask_path = os.path.join(track, "YolkMask.tif")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(mask_path)
        mask = tifffile.imread(mask_path)
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        mask_bool = mask > 0
        if mask_bool.shape != kymo.shape:
            raise ValueError(
                f"Mask shape {mask_bool.shape} does not match kymograph shape {kymo.shape}"
            )
        alignment = build_alignment(
            mask_bool,
            px2micron=px2micron,
            mode="island",
            island_labels=spec.island_labels,
        )
    else:
        poly = read_apical_manual_tsv(work_dir)
        if poly is None:
            raise FileNotFoundError(
                "track/apical_manual.tsv missing or invalid. "
                "Draw a manual apical polyline in the desktop and Save."
            )
        time_min_pts, depth_px_pts = poly
        kymo_cfg = cfg_flat.get("kymograph") or {}
        dt_sec = float(
            kymo_cfg.get(
                "time_interval_sec",
                float(kymo_cfg.get("time_interval_min", 1.0)) * 60.0,
            )
        )
        dt_min = dt_sec / 60.0
        apical_px = apical_px_from_manual_polyline(
            time_min_pts,
            depth_px_pts,
            num_timepoints=num_timepoints,
            dt_min=dt_min,
            sigma_um=spec.manual_sigma_um,
            px2micron=px2micron,
        )
        alignment = alignment_from_apical_px(
            apical_px,
            px2micron=px2micron,
            mode="manual",
            mode_params={"manual_sigma_um": float(spec.manual_sigma_um)},
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
    meta: dict = {
        "version": int(STRAIGHTEN_METADATA_VERSION),
        "ref_row": int(ref_row),
        "crop_top_px": int(crop_top),
        "apical_mode": spec.mode,
    }
    if spec.mode == "island":
        meta["island_labels"] = [int(x) for x in spec.island_labels]
    else:
        meta["manual_sigma_um"] = float(spec.manual_sigma_um)
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

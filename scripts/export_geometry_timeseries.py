#!/usr/bin/env python3
"""
Export a unified geometry timeseries CSV per sample.

Merges track/cytoplasm_region.tsv and track/VerticalKymoCelluSelection_spline.tsv
on the kymograph column index / time grid for downstream plotting.

Output: track/geometry_timeseries.csv

Columns:
  col_idx, time_min,
  apical_px_raw, basal_px_raw, cytoplasm_height_px, cytoplasm_height_um (µm),
  front_px_raw,
  front_minus_apical_px, front_minus_apical_um,
  apical_px_ref, px2micron, kymo_time_interval_sec,
  valid_cytoplasm, valid_front
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml


CSV_COLUMNS: List[str] = [
    "col_idx",
    "time_min",
    "apical_px_raw",
    "basal_px_raw",
    "cytoplasm_height_px",
    "cytoplasm_height_um",
    "front_px_raw",
    "basal_minus_apical_px",
    "basal_minus_apical_um",
    "front_minus_apical_px",
    "front_minus_apical_um",
    "apical_px_ref",
    "px2micron",
    "kymo_time_interval_sec",
    "valid_cytoplasm",
    "valid_front",
]


def _load_config(folder: str) -> Tuple[float, float, float]:
    """Return (px2micron, apical_px_ref, kymo_time_interval_sec)."""
    path = os.path.join(folder, "config.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.yaml not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    manual = cfg.get("manual", {})
    if "px2micron" not in manual:
        raise ValueError("manual.px2micron missing in config.yaml")
    px2micron = float(manual["px2micron"])
    apical = cfg.get("apical_detection", {})
    if "apical_height_px" not in apical:
        raise ValueError("apical_detection.apical_height_px missing in config.yaml")
    apical_px_ref = float(apical["apical_height_px"])
    kymo = cfg.get("kymograph", {})
    dt_sec = float(
        kymo.get(
            "time_interval_sec",
            float(kymo.get("time_interval_min", 1.0)) * 60.0,
        )
    )
    return px2micron, apical_px_ref, dt_sec


def export_geometry_timeseries(folder: str) -> str:
    track = os.path.join(folder, "track")
    cyto_path = os.path.join(track, "cytoplasm_region.tsv")
    spline_path = os.path.join(track, "VerticalKymoCelluSelection_spline.tsv")
    out_path = os.path.join(track, "geometry_timeseries.csv")

    if not os.path.exists(cyto_path):
        raise FileNotFoundError(f"cytoplasm_region.tsv not found: {cyto_path}")
    if not os.path.exists(spline_path):
        raise FileNotFoundError(f"VerticalKymoCelluSelection_spline.tsv not found: {spline_path}")

    px2micron, apical_px_ref, kymo_dt_sec = _load_config(folder)
    dt_min = kymo_dt_sec / 60.0

    cyto = np.loadtxt(cyto_path, delimiter="\t", skiprows=1)
    if cyto.ndim == 1:
        cyto = np.expand_dims(cyto, axis=0)
    col_idx = cyto[:, 0].astype(int)
    apical_px = cyto[:, 1].astype(float)
    basal_px = cyto[:, 3].astype(float)
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
        bpx = basal_px[i]
        valid_apical = np.isfinite(apx) and not np.isnan(apx)
        valid_basal = np.isfinite(bpx) and not np.isnan(bpx)
        valid_cyto = valid_apical and valid_basal
        valid_fr = np.isfinite(fpx) and not np.isnan(fpx)

        hpx = np.nan
        hum = np.nan
        if valid_cyto:
            # Define cytoplasm thickness from raw apical and basal only.
            hpx = bpx - apx
            hum = hpx * px2micron

        bmapx = np.nan
        bma_um = np.nan
        if valid_cyto:
            bmapx = bpx - apx
            bma_um = bmapx * px2micron

        fmapx = np.nan
        fmap_um = np.nan
        if valid_fr and valid_apical:
            fmapx = fpx - apx
            fmap_um = fmapx * px2micron

        rows.append(
            {
                "col_idx": int(col_idx[i]),
                "time_min": float(time_min[i]),
                "apical_px_raw": apx,
                "basal_px_raw": bpx,
                "cytoplasm_height_px": hpx,
                "cytoplasm_height_um": hum,
                "front_px_raw": fpx,
                "basal_minus_apical_px": bmapx,
                "basal_minus_apical_um": bma_um,
                "front_minus_apical_px": fmapx,
                "front_minus_apical_um": fmap_um,
                "apical_px_ref": apical_px_ref,
                "px2micron": px2micron,
                "kymo_time_interval_sec": kymo_dt_sec,
                "valid_cytoplasm": 1 if valid_cyto else 0,
                "valid_front": 1 if valid_fr else 0,
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

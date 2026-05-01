#!/usr/bin/env python3
"""
Export a unified geometry timeseries CSV per sample.

Merges track/VerticalKymoCelluSelection_spline.tsv on the kymograph time grid with
apical positions from straighten_metadata (same source as straightening).

Output: output.csv (sample folder root; main tabular product for follow-up analysis)

Line 1 (metadata): filename,<movie filename>
Columns:
  col_idx, time_min, time_abs_min, apical_px_raw, front_raw_px, front_minus_apical_px, front_minus_apical_um
  (time_min = minutes from first valid row; time_abs_min = minutes from movie/kymograph t=0)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import tifffile

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

sys.path.insert(0, os.path.dirname(__file__))
from mask_utils import compute_apical_column_positions
from work_state import get_movie_path, pipeline_config_flat, straightening_meta

from app.services.geometry_transform import raw_from_straight

CSV_COLUMNS: List[str] = [
    "col_idx",
    "time_min",
    "time_abs_min",
    "apical_px_raw",
    "front_raw_px",
    "front_minus_apical_px",
    "front_minus_apical_um",
]


def _load_config_and_meta(folder: str) -> Tuple[float, int, float, dict]:
    """Return (px2micron, ref_row, kymo_time_interval_sec, straighten_metadata)."""
    cfg = pipeline_config_flat(folder)
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

    meta = straightening_meta(folder)
    if not meta or "ref_row" not in meta:
        raise FileNotFoundError(
            "straightening metadata missing in config.yaml. "
            "Run straighten_kymograph before export_geometry_timeseries."
        )
    ref_row = int(meta["ref_row"])

    return px2micron, ref_row, dt_sec, meta


def _apical_px_per_column(mask_bool: np.ndarray, meta: dict) -> np.ndarray:
    """Apical row in raw kymograph/movie space; matches straightening."""
    num_timepoints = int(mask_bool.shape[1])
    raw_list = meta.get("apical_px_by_col")
    if raw_list is not None and len(raw_list) == num_timepoints:
        return np.array(
            [np.nan if x is None else float(x) for x in raw_list],
            dtype=float,
        )

    mode = str(meta.get("apical_mode", "longest_run")).strip()
    if mode not in ("longest_run", "island"):
        mode = "longest_run"
    il = meta.get("island_labels") or []
    island_labels = il if mode == "island" else None
    apical_px, _ = compute_apical_column_positions(
        mask_bool,
        mode=mode,  # type: ignore[arg-type]
        island_labels=island_labels,
        min_run_length_px=5,
    )
    return apical_px


def export_geometry_timeseries(folder: str) -> str:
    track = os.path.join(folder, "track")
    mask_path = os.path.join(track, "YolkMask.tif")
    spline_path = os.path.join(track, "VerticalKymoCelluSelection_spline.tsv")
    out_path = os.path.join(folder, "output.csv")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"YolkMask.tif not found: {mask_path}")
    if not os.path.exists(spline_path):
        raise FileNotFoundError(f"VerticalKymoCelluSelection_spline.tsv not found: {spline_path}")

    px2micron, ref_row, kymo_dt_sec, meta = _load_config_and_meta(folder)
    dt_min = kymo_dt_sec / 60.0

    mask = tifffile.imread(mask_path)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"YolkMask.tif has unexpected shape: {mask.shape}")
    mask_bool = mask > 0
    _depth, num_timepoints = mask_bool.shape

    shifts = np.asarray(meta["shifts"], dtype=int)
    if shifts.size != num_timepoints:
        raise ValueError(
            f"straighten_metadata shifts length {shifts.size} != mask columns {num_timepoints}"
        )

    apical_px = _apical_px_per_column(mask_bool, meta)

    col_idx = np.arange(num_timepoints, dtype=int)

    sp = np.loadtxt(spline_path, delimiter="\t", skiprows=1)
    if sp.ndim == 1:
        sp = np.expand_dims(sp, axis=0)
    spline_time = sp[:, 0].astype(float)
    spline_front = sp[:, 1].astype(float)

    time_abs_min = col_idx.astype(float) * dt_min
    front_px = np.interp(
        time_abs_min,
        spline_time,
        spline_front,
        left=np.nan,
        right=np.nan,
    )

    rows: List[Dict[str, Any]] = []
    first_valid_time_abs_min: float | None = None
    for i in range(len(col_idx)):
        fpx = float(front_px[i])
        apx = float(apical_px[i])
        valid_apical = np.isfinite(apx) and not np.isnan(apx)
        valid_fr = np.isfinite(fpx) and not np.isnan(fpx)

        if not (valid_fr and valid_apical):
            continue

        t_abs_min = float(time_abs_min[i])
        if first_valid_time_abs_min is None:
            first_valid_time_abs_min = t_abs_min
        t_plot_min = t_abs_min - first_valid_time_abs_min

        fmapx = fpx - ref_row
        fmap_um = fmapx * px2micron
        front_raw = raw_from_straight(fpx, float(shifts[i]))

        rows.append(
            {
                "col_idx": int(col_idx[i]),
                "time_min": t_plot_min,
                "time_abs_min": t_abs_min,
                "apical_px_raw": apx,
                "front_raw_px": front_raw,
                "front_minus_apical_px": fmapx,
                "front_minus_apical_um": fmap_um,
            }
        )

    try:
        movie_filename = os.path.basename(get_movie_path(folder))
    except FileNotFoundError:
        movie_filename = ""

    with open(out_path, "w", newline="") as f:
        f.write(f"filename,{movie_filename}\n")
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
    p = argparse.ArgumentParser(description="Export output.csv (geometry timeseries) for one sample.")
    p.add_argument("--work-dir", required=True, help="Working directory with track/ and config.yaml")
    args = p.parse_args()
    if not os.path.isdir(args.work_dir):
        raise SystemExit(f"Not a directory: {args.work_dir}")
    export_geometry_timeseries(args.work_dir)


if __name__ == "__main__":
    main()

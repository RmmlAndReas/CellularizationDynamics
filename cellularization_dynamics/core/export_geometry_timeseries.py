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
from typing import Any, Dict, List, Tuple

import numpy as np
import tifffile

from cellularization_dynamics.app.services.geometry_transform import raw_from_straight

from .mask_utils import compute_apical_column_positions
from .work_state import get_movie_path, pipeline_config_flat, straightening_meta

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


def _apical_px_from_meta(meta: dict, num_timepoints: int) -> np.ndarray:
    """Apical row in raw kymograph/movie space from the straightening TSV."""
    raw_list = meta.get("apical_px_by_col")
    if raw_list is None or len(raw_list) != num_timepoints:
        raise ValueError(
            "straightening metadata is missing apical_px_by_col; "
            "re-run straighten_kymograph for this sample."
        )
    return np.array(
        [np.nan if x is None else float(x) for x in raw_list],
        dtype=float,
    )


def _apical_px_island_fallback(
    folder: str, meta: dict, num_timepoints: int
) -> np.ndarray:
    """Recompute island apical_px from YolkMask.tif + saved island labels."""
    mask_path = os.path.join(folder, "track", "YolkMask.tif")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"YolkMask.tif not found: {mask_path}")
    mask = tifffile.imread(mask_path)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"YolkMask.tif has unexpected shape: {mask.shape}")
    mask_bool = mask > 0
    if mask_bool.shape[1] != num_timepoints:
        raise ValueError(
            f"YolkMask.tif columns {mask_bool.shape[1]} != kymograph columns {num_timepoints}"
        )
    il = meta.get("island_labels") or []
    island_labels = [int(x) for x in il]
    if not island_labels:
        raise ValueError(
            "straightening metadata lacks island_labels; cannot recompute apical_px_by_col."
        )
    apical_px, _ = compute_apical_column_positions(
        mask_bool,
        island_labels=island_labels,
    )
    return apical_px


def export_geometry_timeseries(folder: str) -> str:
    track = os.path.join(folder, "track")
    kymo_path = os.path.join(track, "Kymograph.tif")
    spline_path = os.path.join(track, "VerticalKymoCelluSelection_spline.tsv")
    out_path = os.path.join(folder, "output.csv")

    if not os.path.exists(kymo_path):
        raise FileNotFoundError(f"Kymograph.tif not found: {kymo_path}")
    if not os.path.exists(spline_path):
        raise FileNotFoundError(f"VerticalKymoCelluSelection_spline.tsv not found: {spline_path}")

    px2micron, ref_row, kymo_dt_sec, meta = _load_config_and_meta(folder)
    dt_min = kymo_dt_sec / 60.0

    with tifffile.TiffFile(kymo_path) as tf:
        kymo_shape = tf.series[0].shape
    if len(kymo_shape) < 2:
        raise ValueError(f"Kymograph.tif has unexpected shape: {kymo_shape}")
    num_timepoints = int(kymo_shape[-1])

    mode = str(meta.get("apical_mode", "island")).strip()
    if mode == "longest_run":
        raise ValueError(
            "straightening.apical_mode 'longest_run' is no longer supported. "
            "Re-save apical alignment from the desktop."
        )
    if mode not in ("island", "manual"):
        raise ValueError(f"Unsupported straightening apical_mode: {mode!r}")

    shifts = np.asarray(meta["shifts"], dtype=int)
    if shifts.size != num_timepoints:
        raise ValueError(
            f"straighten_metadata shifts length {shifts.size} != kymograph columns {num_timepoints}"
        )

    try:
        apical_px = _apical_px_from_meta(meta, num_timepoints)
    except ValueError:
        if mode == "island":
            apical_px = _apical_px_island_fallback(folder, meta, num_timepoints)
        else:
            raise

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

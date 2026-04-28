#!/usr/bin/env python3
"""
Panel A: Species-level average cellularization front + cytoplasm region,
with a vertical "dance onset" marker.

Dance onset times are read from dance_onset_min.csv in each sample's results
folder.  Generate that file first by running plot_panels_bc_kymo_period.py,
then optionally edit it to correct any automatically detected onset values.

Example commands:

  python scripts/analysis/plot_panels_bc_kymo_period.py \
    --results-folder data/Hermetia/2/results

  python scripts/analysis/plot_panels_bc_kymo_period.py \
    --results-folder data/Hermetia/6/results

  python scripts/analysis/plot_panel_a_dance_alignment.py \
    --species-folder data/Hermetia \
    --sample-list 2,6 \
    --out-stem panel_a_dance_alignment

Time alignment:
  - Per sample: align each trace so the cellularization progress reaches 10%
    (t10 per sample) at x=0.
  - Global: shift again so x=0 corresponds to the earliest aligned front point
    among selected samples.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


FONT_SIZE_LABEL = 24


def _parse_sample_list(sample_list: str) -> List[str]:
    parts = [p.strip() for p in sample_list.split(",") if p.strip()]
    if not parts:
        raise ValueError("--sample-list is empty")
    return parts


def _natural_sample_folder(species_folder: str, sample_id: str) -> str:
    # Most of the repo uses numeric folder names (e.g. .../Hermetia/2).
    cand = os.path.join(species_folder, str(sample_id))
    if os.path.isdir(cand):
        return cand
    # Fallback: if the user passed a full relative path.
    cand2 = os.path.join(species_folder, str(sample_id).lstrip("./"))
    if os.path.isdir(cand2):
        return cand2
    raise FileNotFoundError(f"Could not resolve sample folder for id '{sample_id}' under {species_folder}")


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Unexpected YAML structure: {path}")
    return cfg


def load_time_window_from_config(sample_folder: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
    cfg_path = os.path.join(sample_folder, "config.yaml")
    if not os.path.exists(cfg_path):
        return None
    try:
        cfg = _load_yaml(cfg_path)
    except Exception:
        return None

    tw = cfg.get("time_window")
    if not isinstance(tw, dict):
        return None

    start_min = tw.get("start_min")
    end_min = tw.get("end_min")
    try:
        start_min = float(start_min) if start_min is not None else None
        end_min = float(end_min) if end_min is not None else None
    except Exception:
        return None
    if start_min is None and end_min is None:
        return None
    return start_min, end_min


def load_cytoplasm_region_tsv(sample_folder: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load cytoplasm height over time from track/cytoplasm_region.tsv.
    """
    tsv_path = os.path.join(sample_folder, "track", "cytoplasm_region.tsv")
    if not os.path.exists(tsv_path):
        return None

    cfg = _load_yaml(os.path.join(sample_folder, "config.yaml"))
    kymograph = cfg.get("kymograph", {})
    if not isinstance(kymograph, dict):
        return None
    dt_sec = kymograph.get("time_interval_sec")
    if dt_sec is None:
        return None

    try:
        dt_min = float(dt_sec) / 60.0
    except Exception:
        return None

    data = np.loadtxt(tsv_path, delimiter="\t", skiprows=1)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    if data.shape[1] < 7:
        return None

    col_idx = data[:, 0].astype(int)
    time_min = col_idx * dt_min
    height_um = data[:, 6].astype(float)
    return time_min, height_um


def load_cellularization_front_spline_raw(sample_folder: str) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Load cellularization front spline (time_min, front_px).
    Returns (time_min, front_px, px2micron).
    """
    cfg_path = os.path.join(sample_folder, "config.yaml")
    spline_path = os.path.join(sample_folder, "track", "VerticalKymoCelluSelection_spline.tsv")
    if not os.path.exists(cfg_path) or not os.path.exists(spline_path):
        return None

    cfg = _load_yaml(cfg_path)
    manual = cfg.get("manual", {})
    if not isinstance(manual, dict) or "px2micron" not in manual:
        return None
    px2micron = float(manual["px2micron"])

    try:
        data = np.loadtxt(spline_path, delimiter="\t", skiprows=1)
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        time_min = data[:, 0].astype(float)
        front_px = data[:, 1].astype(float)
    except Exception:
        return None

    return time_min, front_px, px2micron


def compute_t_at_progress_pct(
    time_min: np.ndarray,
    front_px: np.ndarray,
    apical_px: float,
    final_px: float,
    progress_pct: float,
) -> Optional[float]:
    """
    Find the first time where progress_pct is reached.
    Uses linear interpolation between the last below and first above points.
    """
    if time_min.size < 2:
        return None
    total_distance = float(final_px - apical_px)
    if total_distance == 0:
        return None

    progress = (front_px - apical_px) / total_distance * 100.0

    # If front isn't monotonic, this is still "first crossing" in time order.
    order = np.argsort(time_min)
    time_sorted = time_min[order]
    progress_sorted = progress[order]

    # Find indices where it crosses upward.
    target = float(progress_pct)
    above = progress_sorted >= target
    idxs = np.where(above)[0]
    if idxs.size == 0:
        return None

    i = int(idxs[0])
    if i == 0:
        return float(time_sorted[0])

    # Interpolate between i-1 and i
    p0 = progress_sorted[i - 1]
    p1 = progress_sorted[i]
    t0 = time_sorted[i - 1]
    t1 = time_sorted[i]
    if not np.isfinite(p0) or not np.isfinite(p1) or p1 == p0:
        return float(t1)

    frac = (target - p0) / (p1 - p0)
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(t0 + frac * (t1 - t0))


def interp_to_grid(x_grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Interpolate y(x) onto x_grid, returning NaN outside input bounds.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return np.full_like(x_grid, np.nan, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    lo = float(x.min())
    hi = float(x.max())
    out = np.full_like(x_grid, np.nan, dtype=float)

    mask = (x_grid >= lo) & (x_grid <= hi)
    out[mask] = np.interp(x_grid[mask], x, y)
    return out


def load_dance_onset_csv(results_folder: str) -> Optional[List[float]]:
    """
    Load per-membrane dance_start_min_global values from dance_onset_min.csv.

    The CSV is written by plot_panels_bc_kymo_period.py and has columns:
      membrane_nr, dance_start_min, dance_start_min_global

    dance_start_min_global is already on the global cellularization scale
    (minutes relative to t10, the 10% cellularization milestone), ready to
    use directly as the aligned onset in plot_panel_a_dance_alignment.py.

    Edit the file to correct automatically detected values before running
    this script.  Returns None if the file is absent or has no valid rows.
    """
    csv_path = os.path.join(results_folder, "dance_onset_min.csv")
    if not os.path.exists(csv_path):
        return None
    times: List[float] = []
    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
        # Support both old format (onset_time_min at col 1) and new (dance_start_min_global at col 2).
        col = 2 if len(header) >= 3 and "global" in header[2] else 1
        for line in f:
            parts = line.strip().split(",")
            if len(parts) > col and parts[col]:
                try:
                    times.append(float(parts[col]))
                except ValueError:
                    pass
    return times if times else None


@dataclass(frozen=True)
class SampleTraces:
    sample_id: str
    t_front_aligned: np.ndarray
    front_um: np.ndarray
    t_cyto_aligned: np.ndarray
    cytoplasm_um: np.ndarray
    t10_aligned_removed: float
    onset_aligned_per_sample: Optional[float]


def main() -> None:
    parser = argparse.ArgumentParser(description="Panel A: average front/cytoplasm with dance onset marker.")
    parser.add_argument("--species-folder", "-s", required=True, type=str, help="e.g. data/Hermetia")
    parser.add_argument("--sample-list", "-n", required=True, type=str, help="comma-separated ids, e.g. 1,2,6")
    parser.add_argument("--out-stem", type=str, default="panel_a_dance_onset_alignment")
    args = parser.parse_args()

    species_folder = os.path.abspath(args.species_folder)
    sample_ids = _parse_sample_list(args.sample_list)

    out_dir = os.path.join(species_folder, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{args.out_stem}.png")
    out_pdf = os.path.join(out_dir, f"{args.out_stem}.pdf")

    per_sample: List[SampleTraces] = []
    onset_per_sample: List[float] = []

    for sid in sample_ids:
        sample_folder = _natural_sample_folder(species_folder, sid)
        cfg = _load_yaml(os.path.join(sample_folder, "config.yaml"))
        manual = cfg.get("manual", {})
        if not isinstance(manual, dict) or "px2micron" not in manual:
            raise ValueError(f"{sample_folder}/config.yaml missing manual.px2micron")
        px2micron = float(manual["px2micron"])

        apical_px = float(cfg["apical_detection"]["apical_height_px"])
        final_px = float(cfg["cellularization_front"]["final_height_px"])

        # Load raw front spline (time + front_px)
        front_raw = load_cellularization_front_spline_raw(sample_folder)
        if front_raw is None:
            print(f"Warning: missing front spline for sample {sid}; skipping")
            continue
        t_front, front_px, _px2micron_from_spline = front_raw
        if not np.isfinite(px2micron):
            px2micron = _px2micron_from_spline

        t10 = compute_t_at_progress_pct(
            time_min=t_front,
            front_px=front_px,
            apical_px=apical_px,
            final_px=final_px,
            progress_pct=10.0,
        )
        if t10 is None:
            print(f"Warning: sample {sid} never reaches 10% cellularization progression; skipping")
            continue

        # Load cytoplasm trace
        cyto = load_cytoplasm_region_tsv(sample_folder)
        if cyto is None:
            print(f"Warning: missing cytoplasm_region.tsv for sample {sid}; skipping")
            continue
        t_cyto, cyto_um = cyto

        # Apply time_window cropping (absolute time)
        tw = load_time_window_from_config(sample_folder)
        if tw is not None:
            start_min, end_min = tw
            mask_f = np.ones_like(t_front, dtype=bool)
            mask_c = np.ones_like(t_cyto, dtype=bool)
            if start_min is not None:
                mask_f &= t_front >= start_min
                mask_c &= t_cyto >= start_min
            if end_min is not None:
                mask_f &= t_front <= end_min
                mask_c &= t_cyto <= end_min
            if np.any(mask_f):
                t_front = t_front[mask_f]
                front_px = front_px[mask_f]
            if np.any(mask_c):
                t_cyto = t_cyto[mask_c]
                cyto_um = cyto_um[mask_c]

        if t_front.size < 2 or t_cyto.size < 2:
            print(f"Warning: insufficient points after cropping for sample {sid}; skipping")
            continue

        # Shift to per-sample 10% milestone
        t_front_aligned = t_front - t10
        t_cyto_aligned = t_cyto - t10
        front_um = (front_px - apical_px) * px2micron

        # Load dance onset times (absolute movie minutes) written by
        # plot_panels_bc_kymo_period.py into dance_onset_min.csv.
        sample_results_folder = os.path.join(sample_folder, "results")
        onset_abs_list = load_dance_onset_csv(sample_results_folder)
        if onset_abs_list is None:
            print(
                f"Warning: no dance_onset_min.csv found for sample {sid}. "
                f"Run plot_panels_bc_kymo_period.py first to generate it."
            )
            onset_mean_aligned = None
        else:
            # dance_start_min_global is already relative to t10 (global scale).
            onset_times = [v for v in onset_abs_list if np.isfinite(v)]
            onset_mean_aligned = float(np.nanmean(onset_times)) if onset_times else None
            if onset_mean_aligned is None:
                print(f"Warning: no valid onset values in dance_onset_min.csv for sample {sid}")
            else:
                onset_per_sample.append(onset_mean_aligned)
            print(f"\nSample {sid} dance onsets (from dance_onset_min.csv, t10={t10:.2f} min):")
            for v in onset_abs_list:
                print(f"  - dance_start_min_global={v:.3f} min (t10 coords)")
            if onset_mean_aligned is not None:
                print(f"  -> sample onset mean (t10 coords): {onset_mean_aligned:.3f} min")

        per_sample.append(
            SampleTraces(
                sample_id=sid,
                t_front_aligned=t_front_aligned,
                front_um=front_um,
                t_cyto_aligned=t_cyto_aligned,
                cytoplasm_um=cyto_um,
                t10_aligned_removed=t10,
                onset_aligned_per_sample=onset_mean_aligned,
            )
        )

    if len(per_sample) < 2:
        raise ValueError(f"Need at least 2 usable samples for averaging; got {len(per_sample)}")

    # Global shift: x=0 at earliest aligned front time among selected samples.
    t_min_front_global = float(min(np.nanmin(ps.t_front_aligned) for ps in per_sample if ps.t_front_aligned.size))

    # Make common aligned x grid from overlap.
    # We include the earliest aligned front on the axis (x=0),
    # but the mean/std are still computed ignoring NaNs.
    x_lo = t_min_front_global
    x_hi = min(float(np.nanmax(ps.t_front_aligned)) for ps in per_sample)
    if x_hi <= x_lo:
        raise ValueError("Aligned front traces do not overlap in time; cannot average.")

    # Choose step size from median dt of the first sample (front spline).
    ref = per_sample[0].t_front_aligned
    finite_ref = ref[np.isfinite(ref)]
    if finite_ref.size >= 2:
        step = float(np.median(np.diff(np.sort(finite_ref))))
        if not np.isfinite(step) or step <= 0:
            step = 1.0
    else:
        step = 1.0
    # Avoid extremely dense grids.
    step = max(step, 0.1)
    x_grid = np.arange(x_lo, x_hi + 1e-9, step)

    # Compute mean±SD for front and cytoplasm.
    front_stack = []
    cyto_stack = []
    for ps in per_sample:
        # Interpolate into the per-sample aligned coordinates (t - t10),
        # which correspond directly to x_grid.
        front_stack.append(interp_to_grid(x_grid, ps.t_front_aligned, ps.front_um))
        cyto_stack.append(interp_to_grid(x_grid, ps.t_cyto_aligned, ps.cytoplasm_um))

    front_stack = np.stack(front_stack, axis=0)
    cyto_stack = np.stack(cyto_stack, axis=0)
    front_mean = np.nanmean(front_stack, axis=0)
    front_sd = np.nanstd(front_stack, axis=0)
    cyto_mean = np.nanmean(cyto_stack, axis=0)
    cyto_sd = np.nanstd(cyto_stack, axis=0)

    # Onset marker in final x coordinates.
    onset_aligned_vals = [ps.onset_aligned_per_sample for ps in per_sample if ps.onset_aligned_per_sample is not None]
    onset_aligned_vals = [float(v) for v in onset_aligned_vals if np.isfinite(v)]
    onset_mean = float(np.nanmean(onset_aligned_vals)) if onset_aligned_vals else None
    onset_sd = float(np.nanstd(onset_aligned_vals, ddof=0)) if onset_aligned_vals else None

    x_final = x_grid - t_min_front_global

    # Square figure/axes to match the visual style of other kymograph-derived plots.
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.plot(x_final, cyto_mean, color="tab:blue", linewidth=2.5, label="Cytoplasm region (mean)")
    ax.fill_between(x_final, cyto_mean - cyto_sd, cyto_mean + cyto_sd, color="tab:blue", alpha=0.15)

    ax.plot(x_final, front_mean, color="tab:orange", linewidth=2.5, label="Cellularization front (mean)")
    ax.fill_between(x_final, front_mean - front_sd, front_mean + front_sd, color="tab:orange", alpha=0.15)

    if onset_mean is not None and onset_sd is not None and np.isfinite(onset_mean) and np.isfinite(onset_sd):
        onset_final = onset_mean - t_min_front_global
        ax.axvline(onset_final, color="0.3", linewidth=2.0, alpha=0.5)
        ax.axvspan(onset_final - onset_sd, onset_final + onset_sd, color="0.3", alpha=0.08)

    ax.set_xlabel("Time (min) relative to earliest aligned front")
    ax.set_ylabel("Height (µm)")
    ax.legend(fontsize=8, loc="best")
    # Show y-axis from 0 (0 at top). Height is expected to be >= 0, so clamp lower bound.
    y_max = float(
        max(
            0.0,
            np.nanmax(cyto_mean + cyto_sd) if np.any(np.isfinite(cyto_mean + cyto_sd)) else 0.0,
            np.nanmax(front_mean + front_sd) if np.any(np.isfinite(front_mean + front_sd)) else 0.0,
        )
    )
    ax.set_ylim(y_max, 0.0)
    ax.set_box_aspect(1)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.95, bottom=0.14)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)

    print("\nPanel A dance onset alignment")
    print("-" * 60)
    print(f"species_folder          : {species_folder}")
    print(f"sample_ids              : {sample_ids}")
    print(f"t_min_front_global (min): {t_min_front_global:.3f}")
    if onset_mean is not None and onset_sd is not None:
        print(f"onset aligned mean±SD  : {onset_mean:.3f} ± {onset_sd:.3f} min (t10 coordinates)")
        print(f"onset final x          : {onset_mean - t_min_front_global:.3f} min")
    else:
        print("onset marker: not available (run plot_panels_bc_kymo_period.py first)")
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


if __name__ == "__main__":
    main()


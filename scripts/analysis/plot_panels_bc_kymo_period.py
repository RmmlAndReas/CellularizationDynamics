#!/usr/bin/env python3
"""
Panels B and C (single-sample detail):
  - Panel B: delta kymograph (with cellularization progression marks) + overlay
    all pyBOAT ROI trajectories, each in a distinct color.
  - Panel C: vertical "period vs time" traces for all pyBOAT runs, each in the
    matching color. Onset of each trace is marked with a triangle on the y-axis.

Example:

  python scripts/analysis/plot_panels_bc_kymo_period.py \
    --results-folder data/Hermetia/2/results \
    --out-stem panels_bc_kymo_period

The script auto-discovers all MembraneWobble_pyBOAT_* subfolders inside
--results-folder.  Each folder must contain exactly one *.tsv ROI file and a
Position_um_readout.csv.

Sample config.yaml and kymograph TIFF are read from:
  <results-folder>/../config.yaml
  <results-folder>/Kymograph_delta_marked.tif

This is designed to be composed later (e.g. in Affinity).
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml


FONT_SIZE_LABEL = 22
FONT_SIZE_TICKS = 18
SPINE_LINEWIDTH = 3
TICK_MAJOR_LENGTH = 7
TICK_MAJOR_WIDTH = 2

MILESTONE_FIRST_COLOR = "#0072B2"  # 0% milestone
MILESTONE_LAST_COLOR = "#D55E00"   # 100% milestone

# Color palette for pyBOAT overlays (tab10, skipping reserved milestone colors).
_OVERLAY_COLORS = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#000000",
    "#999999",
    "#882255",
    "#117733",
    "#332288",
]


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Unexpected YAML structure: {path}")
    return cfg


def _infer_time_scale_to_minutes(time: np.ndarray) -> float:
    """
    Infer whether pyBOAT `time` is in seconds or minutes using sampling cadence.

    Heuristic:
      - if median dt > ~1.5 -> interpret as seconds -> divide by 60
      - else -> interpret as minutes -> keep as-is
    """
    time = np.asarray(time, dtype=float)
    time = time[np.isfinite(time)]
    if time.size < 2:
        return 1.0

    diffs = np.diff(np.sort(time))
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 1.0

    dt = float(np.median(diffs))
    return 1.0 / 60.0 if dt > 1.5 else 1.0


def load_front_spline_px(sample_folder: str) -> Tuple[np.ndarray, np.ndarray]:
    spline_path = os.path.join(sample_folder, "track", "VerticalKymoCelluSelection_spline.tsv")
    if not os.path.exists(spline_path):
        raise FileNotFoundError(f"Missing spline TSV: {spline_path}")
    data = np.loadtxt(spline_path, delimiter="\t", skiprows=1)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    time_min = data[:, 0].astype(float)
    front_px = data[:, 1].astype(float)
    return time_min, front_px


def compute_milestones(
    time_min: np.ndarray,
    front_px: np.ndarray,
    apical_px: float,
    final_px: float,
    movie_dt_sec: float,
) -> dict:
    """
    Progress milestones every 10% from 0..100.

    Returns dict[pct] -> {frame, time_min, position_px}.
    """
    total_distance = float(final_px - apical_px)
    if total_distance == 0:
        raise ValueError("final_px == apical_px; cannot compute progress")

    progress_pct = (front_px - apical_px) / total_distance * 100.0

    milestones = {}
    for pct in range(0, 101, 10):
        indices = np.where(progress_pct >= pct)[0]
        if indices.size == 0:
            continue
        idx = int(indices[0])
        t_min = float(time_min[idx])
        frame_idx = int(t_min * 60.0 / movie_dt_sec)
        milestones[pct] = {"frame": frame_idx, "time_min": t_min, "position_px": float(front_px[idx])}

    if 100 not in milestones and time_min.size > 0:
        idx = int(time_min.size - 1)
        t_min = float(time_min[idx])
        frame_idx = int(t_min * 60.0 / movie_dt_sec)
        milestones[100] = {"frame": frame_idx, "time_min": t_min, "position_px": float(front_px[idx])}

    return milestones


def load_roi_tsv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ROI TSV written by membrane_wobbliness_pyboat_prep.py.

    Expected columns: Time_min\\tPosition_um
    """
    if not path.exists():
        raise FileNotFoundError(f"ROI TSV not found: {path}")
    data = np.loadtxt(path, delimiter="\t", skiprows=1)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    time_min = data[:, 0].astype(float)
    pos_um = data[:, 1].astype(float)
    return time_min, pos_um


def load_pyboat_readout(pyboat_folder: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Position_um_readout.csv from a pyBOAT run folder.

    Returns time (min), periods (min), power (raw, unconverted).
    """
    readout_path = pyboat_folder / "Position_um_readout.csv"
    if not readout_path.exists():
        raise FileNotFoundError(f"Missing pyBOAT wobbliness readout: {readout_path}")

    data = np.genfromtxt(readout_path, delimiter=",", names=True, dtype=float)
    if data is None:
        raise ValueError(f"Could not parse: {readout_path}")
    if getattr(data, "ndim", 0) == 0:
        data = np.expand_dims(data, axis=0)

    time_readout = np.asarray(data["time"], dtype=float)
    periods = np.asarray(data["periods"], dtype=float)
    power = np.asarray(data["power"], dtype=float)

    to_min = _infer_time_scale_to_minutes(time_readout)
    return time_readout * to_min, periods * to_min, power




def discover_pyboat_folders(results_folder: Path) -> List[Path]:
    folders = sorted(results_folder.glob("MembraneWobble_pyBOAT_*"))
    return [f for f in folders if f.is_dir()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Panels B/C: kymograph with all ROI overlays + period-vs-time traces."
    )
    parser.add_argument(
        "--results-folder",
        "-r",
        required=True,
        type=str,
        help="Sample results folder containing MembraneWobble_pyBOAT_* subfolders.",
    )
    parser.add_argument("--out-stem", type=str, default="panels_bc_kymo_period")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, os.pardir))

    results_arg = args.results_folder
    if os.path.isabs(results_arg):
        results_folder = Path(results_arg)
    else:
        results_folder = Path(repo_root) / results_arg

    if not results_folder.exists() or not results_folder.is_dir():
        raise FileNotFoundError(f"Results folder not found: {results_folder}")

    sample_folder = results_folder.parent
    cfg_path = sample_folder / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml: {cfg_path}")

    # Discover all pyBOAT run folders.
    pyboat_folders = discover_pyboat_folders(results_folder)
    if not pyboat_folders:
        raise FileNotFoundError(
            f"No MembraneWobble_pyBOAT_* folders found in: {results_folder}"
        )

    cfg = _load_yaml(str(cfg_path))
    manual = cfg.get("manual", {})
    px2micron = float(manual["px2micron"])
    movie_dt_sec = float(manual.get("movie_time_interval_sec", cfg.get("movie_time_interval_sec", 10.0)))

    apical_px = float(cfg["apical_detection"]["apical_height_px"])
    final_px = float(cfg["cellularization_front"]["final_height_px"])
    start_pct = int(cfg.get("visualization", {}).get("kymo_marked_start_pct", 70))

    kymo_path = results_folder / "Kymograph_delta_marked.tif"
    if not kymo_path.exists():
        raise FileNotFoundError(f"Missing marked Kymograph delta TIFF: {kymo_path}")

    import tifffile

    kymo_img = tifffile.imread(kymo_path)
    kymo_img = np.squeeze(kymo_img)
    if kymo_img.ndim != 2:
        raise ValueError(f"Unexpected kymo shape: {kymo_img.shape}")

    depth, width = kymo_img.shape
    width_um = float(width) * px2micron
    dt_min = movie_dt_sec / 60.0
    duration_min = float(depth - 1) * dt_min

    spline_time_min, spline_front_px = load_front_spline_px(str(sample_folder))
    milestones = compute_milestones(
        time_min=spline_time_min,
        front_px=spline_front_px,
        apical_px=apical_px,
        final_px=final_px,
        movie_dt_sec=movie_dt_sec,
    )

    start_pct_snapped = max((p for p in milestones if p <= start_pct), default=0)
    if start_pct_snapped not in milestones or 100 not in milestones:
        raise ValueError(f"Milestones missing start or end percent (got keys: {sorted(milestones.keys())})")

    frame_0 = int(milestones[start_pct_snapped]["frame"])

    fig, (ax_kymo, ax_period) = plt.subplots(
        1,
        2,
        figsize=(12.0, 6.5),
        constrained_layout=False,
        sharey=False,
        gridspec_kw={"width_ratios": [6, 1]},
    )
    fig.subplots_adjust(wspace=-0.20, left=0.08, right=0.98, bottom=0.12, top=0.98)

    for ax in (ax_kymo, ax_period):
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_LINEWIDTH)
        ax.tick_params(labelsize=FONT_SIZE_TICKS, length=TICK_MAJOR_LENGTH, width=TICK_MAJOR_WIDTH)

    # Panel B: kymograph.
    ax_kymo.imshow(
        kymo_img,
        cmap="gray",
        aspect="auto",
        origin="upper",
        extent=[0, width_um, duration_min, 0],
    )
    ax_kymo.set_xlim(0, width_um)
    ax_kymo.set_ylim(duration_min, 0)
    ax_kymo.set_xlabel("Width (µm)", fontsize=FONT_SIZE_LABEL)
    ax_kymo.set_ylabel("Time (min)", fontsize=FONT_SIZE_LABEL)
    ax_kymo.set_box_aspect(1)

    tri_w = 0.033 * width_um
    tri_h = (0.050 / 1.3) * duration_min

    # Cellularization milestone triangles on Panel B.
    for pct in sorted(milestones.keys()):
        if pct < start_pct_snapped:
            continue
        frame_idx = int(milestones[pct]["frame"])
        y_pos = (frame_idx - frame_0) * dt_min
        if y_pos < 0 or y_pos > duration_min:
            continue
        if pct == 0:
            facecolor = MILESTONE_FIRST_COLOR
        elif pct == 100:
            facecolor = MILESTONE_LAST_COLOR
        else:
            facecolor = "white"
        ax_kymo.add_patch(
            patches.Polygon(
                [[0, y_pos - tri_h / 2], [0, y_pos + tri_h / 2], [tri_w, y_pos]],
                closed=True,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=1.2,
                zorder=10,
                clip_on=False,
            )
        )

    # Per-run overlay: ROI on Panel B, period line + onset triangle on Panel C.
    # onset_runs: list of (color, membrane_nr, dance_start_min, dance_start_min_global)
    #   dance_start_min        = earliest time in the pyBOAT readout (kymograph-relative minutes).
    #                            pyBOAT only reports periods where the wavelet analysis is
    #                            confident, so min(time_pb) marks the first detectable oscillation.
    #   dance_start_min_global = dance_start_min + (t_kymo_start − t10)
    #                            (relative to t10, the global scale used for Panel A)
    all_periods: List[np.ndarray] = []
    onset_runs: List[tuple] = []

    t_kymo_start = milestones[start_pct_snapped]["time_min"]
    t10_abs: Optional[float] = milestones.get(10, {}).get("time_min")

    for i, pb_folder in enumerate(pyboat_folders):
        color = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]

        # Extract membrane number from ROI TSV filename (e.g. MembraneWobble_pyBOAT_roi02.tsv → 2).
        roi_tsvs = sorted(pb_folder.glob("*.tsv"))
        membrane_nr: int = i + 1
        if roi_tsvs:
            m = re.search(r"roi(\d+)", roi_tsvs[0].stem, re.IGNORECASE)
            if m:
                membrane_nr = int(m.group(1))

        # Panel B: ROI overlay on kymograph.
        if len(roi_tsvs) == 1:
            roi_time_min, roi_pos_um = load_roi_tsv(roi_tsvs[0])
            finite = np.isfinite(roi_time_min) & np.isfinite(roi_pos_um)
            ax_kymo.plot(
                roi_pos_um[finite],
                roi_time_min[finite],
                color=color,
                linewidth=4.0,
                linestyle="--",
                alpha=0.5,
                zorder=20,
            )

        # Panel C: period line.
        try:
            time_pb, periods_pb, _ = load_pyboat_readout(pb_folder)
        except FileNotFoundError:
            print(f"  Skipping {pb_folder.name}: no Position_um_readout.csv")
            continue

        finite2 = np.isfinite(time_pb) & np.isfinite(periods_pb)
        time_pb = time_pb[finite2]
        periods_pb = periods_pb[finite2]
        if time_pb.size == 0:
            continue

        all_periods.append(periods_pb)
        ax_period.plot(periods_pb, time_pb, color=color, linewidth=2.0)

        # Dance onset = first time point in the pyBOAT readout.
        dance_start_min = float(time_pb.min())
        dance_start_min_global = (
            dance_start_min + (t_kymo_start - t10_abs)
            if t10_abs is not None else None
        )
        onset_runs.append((color, membrane_nr, dance_start_min, dance_start_min_global))

        dsm_str = f"{dance_start_min:.2f}"
        print(f"  membrane {membrane_nr} ({pb_folder.name}): "
              f"t={time_pb[0]:.1f}..{time_pb[-1]:.1f} min, "
              f"period={float(periods_pb.min()):.2f}..{float(periods_pb.max()):.2f} min, "
              f"dance_start_min={dsm_str}")

    ax_period.set_ylim(duration_min, 0)
    ax_period.set_xlabel("Period (min)", fontsize=FONT_SIZE_LABEL)
    ax_period.set_ylabel("Time (min)", fontsize=FONT_SIZE_LABEL)

    # x-axis: 0 to 2× grand mean of all period values so traces sit centred.
    if all_periods:
        grand_mean = float(np.nanmean(np.concatenate(all_periods)))
        ax_period.set_xlim(0, 2.0 * grand_mean)

    ax_period.set_xticks([10, 20])

    # Onset triangles on Panel C (drawn after xlim is set so triangle width is correct).
    x_max_period = ax_period.get_xlim()[1]
    tri_w_p = 0.08 * x_max_period
    tri_h_p = (0.050 / 1.3) * duration_min
    for color, _, dance_start_min, _ in onset_runs:
        onset_kymo_rel = dance_start_min
        if -tri_h_p <= onset_kymo_rel <= duration_min + tri_h_p:
            ax_period.add_patch(
                patches.Polygon(
                    [
                        [0, onset_kymo_rel - tri_h_p / 2],
                        [0, onset_kymo_rel + tri_h_p / 2],
                        [tri_w_p, onset_kymo_rel],
                    ],
                    closed=True,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.2,
                    zorder=10,
                    clip_on=False,
                )
            )

    out_png = results_folder / f"{args.out_stem}.png"
    out_pdf = results_folder / f"{args.out_stem}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)

    # Save per-run onset times to CSV so plot_panel_a_dance_alignment.py can read them.
    # Columns:
    #   membrane_nr            : ROI number (from TSV filename)
    #   dance_start_min        : first pyBOAT readout time (kymograph-relative), minutes
    #   dance_start_min_global : onset relative to t10 (10% cellularization), minutes
    #                            — the same global scale used for Panel A alignment
    # Edit this file to correct any automatically detected onset values before
    # running plot_panel_a_dance_alignment.py.
    out_csv = results_folder / "dance_onset_min.csv"
    with open(out_csv, "w") as fcsv:
        fcsv.write("membrane_nr,dance_start_min,dance_start_min_global\n")
        for _, membrane_nr, dance_start_min, dance_start_global in onset_runs:
            global_str = f"{dance_start_global:.4f}" if dance_start_global is not None else ""
            fcsv.write(f"{membrane_nr},{dance_start_min:.4f},{global_str}\n")

    print("\nPanels B/C kymograph + period")
    print("-" * 60)
    print(f"sample_folder  : {sample_folder}")
    print(f"results_folder : {results_folder}")
    print(f"pyboat_runs    : {len(pyboat_folders)}")
    print(f"kymo_path      : {kymo_path}")
    print(f"width_um       : {width_um:.3f}")
    print(f"duration_min   : {duration_min:.3f}")
    print(f"t_kymo_start   : {t_kymo_start:.3f} min  (= {start_pct_snapped}% cellularization)")
    print(f"t10_abs        : {t10_abs:.3f} min" if t10_abs is not None else "t10_abs        : n/a")
    if onset_runs:
        print("Dance onset (= first pyBOAT readout time on kymograph):")
        for _, membrane_nr, dance_start_min, dance_start_global in onset_runs:
            gstr = f"{dance_start_global:.2f}" if dance_start_global is not None else "n/a"
            print(f"  membrane {membrane_nr}: dance_start_min={dance_start_min:.2f} min "
                  f"(rel. to 70%), dance_start_min_global={gstr} min (rel. to t10)")
    print(f"Saved PNG      : {out_png}")
    print(f"Saved PDF      : {out_pdf}")
    print(f"Saved CSV      : {out_csv}")


if __name__ == "__main__":
    main()

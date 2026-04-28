#!/usr/bin/env python3
"""
Plot averaged geometry traces for orientation subset groups (multi-panel figure).

Works for any species or experiment: point ``--sample-folder`` at a directory that
contains group subfolders (e.g. dorsal/*, ventral/*, lateral/*) with runs that have
``track/geometry_timeseries.csv``.

For each group, it loads ``track/geometry_timeseries.csv`` from each run,
aligns traces on a common time axis, and plots:
  - mean cytoplasm thickness (solid) with variability band
  - mean membrane front depth (dashed) with variability band

Related scripts (same folder):

  - ``scripts/analysis/plot_geometry_subset_means_split.py`` — one file per group
  - ``scripts/analysis/plot_geometry_subset_means_combined.py`` — all group means on shared axes
  - ``scripts/analysis/plot_geometry_subset_individuals.py`` — every run per group (outliers)

Example:

    python scripts/analysis/plot_geometry_subset_means_multipanel.py \\
        --sample-folder data/MySpecies/condition \\
        --out-stem my_exp_subset_means \\
        --variability sem

With time-axis limits:

    python scripts/analysis/plot_geometry_subset_means_multipanel.py \\
        --sample-folder data/MySpecies/wt \\
        --out-stem wt_subset_means \\
        --x-min 0 \\
        --x-max 60

With explicit y-axis limits:

    python scripts/analysis/plot_geometry_subset_means_multipanel.py \\
        --sample-folder data/MySpecies/wt \\
        --out-stem wt_subset_means \\
        --y-min 0 \\
        --y-max 25
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from subset_geometry_timeseries import (
    build_group_data,
    detect_groups,
    plot_mean_front_single_group_on_ax,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot subset-group averages from geometry_timeseries.csv (one panel per group). "
            "Auto-detects groups (e.g. dorsal, ventral, lateral)."
        )
    )
    parser.add_argument(
        "--sample-folder",
        required=True,
        help=(
            "Folder containing per-group subfolders. "
            "Each group subfolder should contain run dirs with track/geometry_timeseries.csv."
        ),
    )
    parser.add_argument(
        "--out-stem",
        default="geometry_subset_means_multipanel",
        help="Output filename stem in <sample-folder>/results/",
    )
    parser.add_argument(
        "--variability",
        choices=("sem", "std"),
        default="sem",
        help="Variability band type: standard error (sem) or standard deviation (std).",
    )
    parser.add_argument(
        "--orientation",
        choices=("vertical", "horizontal"),
        default="vertical",
        help="Panel layout direction: vertical stack or horizontal row.",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Optional minimum time (min) for x-axis.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional maximum time (min) for x-axis.",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Optional minimum depth (um) for y-axis (typically 0 at top).",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional maximum depth (um) for y-axis (deeper value at bottom).",
    )
    args = parser.parse_args()

    sample_folder = os.path.abspath(args.sample_folder)
    if not os.path.isdir(sample_folder):
        raise SystemExit(f"Sample folder does not exist: {sample_folder}")
    if (
        args.x_min is not None
        and args.x_max is not None
        and not (args.x_max > args.x_min)
    ):
        raise SystemExit("--x-max must be greater than --x-min")
    if (
        args.y_min is not None
        and args.y_max is not None
        and not (args.y_max > args.y_min)
    ):
        raise SystemExit("--y-max must be greater than --y-min")

    groups = detect_groups(sample_folder)
    print(f"Detected groups: {', '.join(groups)}")

    group_results: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
    group_stats: Dict[str, Dict[str, int]] = {}
    for group in groups:
        result, stats = build_group_data(sample_folder, group, variability=args.variability)
        group_results[group] = result
        group_stats[group] = stats

    for group in groups:
        s = group_stats[group]
        print(f"{group}: found={s['found_runs']}, used={s['used_runs']}, skipped={s['skipped_runs']}")

    valid_groups = [g for g in groups if group_results[g] is not None]
    if not valid_groups:
        raise SystemExit(
            "No plottable group data found. Ensure group runs contain "
            "track/geometry_timeseries.csv with required columns."
        )

    panel_in = (8.0 / 5.0) * 1.5
    if args.orientation == "horizontal":
        nrows, ncols = 1, len(groups)
        figsize = (len(groups) * panel_in, panel_in)
    else:
        nrows, ncols = len(groups), 1
        figsize = (panel_in, len(groups) * panel_in)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        constrained_layout=False,
        sharex=False,
    )
    axes = np.asarray(axes, dtype=object).ravel()

    global_depth_candidates: List[float] = []
    for group in valid_groups:
        data = group_results[group]
        if data is None:
            continue
        c_ok = np.isfinite(data["cyto_mean"]) & np.isfinite(data["cyto_spread"]) & (data["cyto_n"] > 0)
        if np.any(c_ok):
            global_depth_candidates.append(float(np.nanmax((data["cyto_mean"] + data["cyto_spread"])[c_ok])))
        f_ok = np.isfinite(data["front_mean"]) & np.isfinite(data["front_spread"]) & (data["front_n"] > 0)
        if np.any(f_ok):
            global_depth_candidates.append(float(np.nanmax((data["front_mean"] + data["front_spread"])[f_ok])))

    global_max_depth = float(np.nanmax(global_depth_candidates)) if global_depth_candidates else 1.0
    if not np.isfinite(global_max_depth) or global_max_depth <= 0:
        global_max_depth = 1.0

    for i, group in enumerate(groups):
        ax = axes[i]
        n_used = group_stats[group]["used_runs"]
        ax.set_title(f"{group.capitalize()} (n={n_used})", fontsize=11, fontweight="bold")
        data = group_results[group]
        if data is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        plot_mean_front_single_group_on_ax(ax, data)

        ax.set_ylabel("Depth (µm)", fontsize=10)
        y_min = args.y_min if args.y_min is not None else 0.0
        y_max = args.y_max if args.y_max is not None else global_max_depth * 1.05
        if y_max > y_min:
            ax.set_ylim(y_max, y_min)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.set_box_aspect(1)
        ax.set_xlabel("Time (min)", fontsize=10)
        if args.x_min is not None or args.x_max is not None:
            auto_min, auto_max = ax.get_xlim()
            x_min = args.x_min if args.x_min is not None else auto_min
            x_max = args.x_max if args.x_max is not None else auto_max
            if x_max > x_min:
                ax.set_xlim(x_min, x_max)

    if args.orientation == "horizontal":
        fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.18, wspace=0.35)
    else:
        fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.07, hspace=0.50)

    out_dir = os.path.join(sample_folder, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{args.out_stem}.png")
    out_pdf = os.path.join(out_dir, f"{args.out_stem}.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()

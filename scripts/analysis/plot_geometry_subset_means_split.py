#!/usr/bin/env python3
"""
One output figure per orientation group: mean cytoplasm + mean membrane front (± SEM or SD).

Species-agnostic: same layout as ``plot_geometry_subset_means_multipanel.py`` but writes
one PNG/PDF per group (dorsal, ventral, …) instead of a single multi-panel figure.

Writes ``<sample-folder>/results/{out-stem}_{group}.png`` (and .pdf) for each group with data.

Example:

    python scripts/analysis/plot_geometry_subset_means_split.py \\
        --sample-folder data/MySpecies/wt \\
        --out-stem wt_subset_means
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

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
        description="Save one mean±variability figure per group (dorsal, ventral, …)."
    )
    parser.add_argument("--sample-folder", required=True)
    parser.add_argument(
        "--out-stem",
        default="geometry_subset_means_split",
        help="Stem for <stem>_<group>.png under <sample-folder>/results/",
    )
    parser.add_argument(
        "--variability",
        choices=("sem", "std"),
        default="sem",
    )
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    args = parser.parse_args()

    sample_folder = os.path.abspath(args.sample_folder)
    if not os.path.isdir(sample_folder):
        raise SystemExit(f"Sample folder does not exist: {sample_folder}")
    if args.x_min is not None and args.x_max is not None and not (args.x_max > args.x_min):
        raise SystemExit("--x-max must be greater than --x-min")
    if args.y_min is not None and args.y_max is not None and not (args.y_max > args.y_min):
        raise SystemExit("--y-max must be greater than --y-min")

    groups = detect_groups(sample_folder)
    print(f"Detected groups: {', '.join(groups)}")

    out_dir = os.path.join(sample_folder, "results")
    os.makedirs(out_dir, exist_ok=True)

    panel_in = (8.0 / 5.0) * 1.5
    any_saved = False

    for group in groups:
        data: Optional[Dict[str, np.ndarray]]
        data, stats = build_group_data(sample_folder, group, variability=args.variability)
        print(
            f"{group}: found={stats['found_runs']}, used={stats['used_runs']}, skipped={stats['skipped_runs']}"
        )
        if data is None:
            continue

        depth_candidates: list[float] = []
        c_ok = (
            np.isfinite(data["cyto_mean"])
            & np.isfinite(data["cyto_spread"])
            & (data["cyto_n"] > 0)
        )
        if np.any(c_ok):
            depth_candidates.append(
                float(np.nanmax((data["cyto_mean"] + data["cyto_spread"])[c_ok]))
            )
        f_ok = (
            np.isfinite(data["front_mean"])
            & np.isfinite(data["front_spread"])
            & (data["front_n"] > 0)
        )
        if np.any(f_ok):
            depth_candidates.append(
                float(np.nanmax((data["front_mean"] + data["front_spread"])[f_ok]))
            )
        y_hi = float(np.nanmax(depth_candidates)) if depth_candidates else 1.0
        if not np.isfinite(y_hi) or y_hi <= 0:
            y_hi = 1.0

        fig, ax = plt.subplots(figsize=(panel_in, panel_in), constrained_layout=False)
        n_used = stats["used_runs"]
        ax.set_title(f"{group.capitalize()} (n={n_used})", fontsize=11, fontweight="bold")
        plot_mean_front_single_group_on_ax(ax, data)

        ax.set_ylabel("Depth (µm)", fontsize=10)
        y_min = args.y_min if args.y_min is not None else 0.0
        y_max = args.y_max if args.y_max is not None else y_hi * 1.05
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

        fig.subplots_adjust(left=0.14, right=0.96, top=0.92, bottom=0.12)
        stem = f"{args.out_stem}_{group}"
        out_png = os.path.join(out_dir, f"{stem}.png")
        out_pdf = os.path.join(out_dir, f"{stem}.pdf")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, dpi=300, bbox_inches="tight", format="pdf")
        plt.close(fig)
        print(f"Saved: {out_png}")
        any_saved = True

    if not any_saved:
        raise SystemExit("No plottable group data found.")


if __name__ == "__main__":
    main()

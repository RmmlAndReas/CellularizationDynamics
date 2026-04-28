#!/usr/bin/env python3
"""
Overlay mean ± variability for every group (dorsal, ventral, lateral, …) on shared axes.

Species-agnostic. Left panel: cytoplasm mean ± band per group (solid). Right panel: membrane
front (dashed). Each group gets a distinct color.

Example:

    python scripts/analysis/plot_geometry_subset_means_combined.py \\
        --sample-folder data/MySpecies/wt \\
        --out-stem orientations_compared
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
    common_time_grid_from_group_results,
    default_group_color,
    detect_groups,
    interp_on_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot all group means on two panels (cytoplasm | membrane front)."
    )
    parser.add_argument("--sample-folder", required=True)
    parser.add_argument(
        "--out-stem",
        default="geometry_subset_means_combined",
        help="Output stem under <sample-folder>/results/",
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

    group_results: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
    group_stats: Dict[str, Dict[str, int]] = {}
    for idx, group in enumerate(groups):
        result, stats = build_group_data(sample_folder, group, variability=args.variability)
        group_results[group] = result
        group_stats[group] = stats
        print(
            f"{group}: found={stats['found_runs']}, used={stats['used_runs']}, skipped={stats['skipped_runs']}"
        )

    valid_groups = [g for g in groups if group_results[g] is not None]
    if not valid_groups:
        raise SystemExit("No plottable group data found.")

    t_common = common_time_grid_from_group_results(group_results)
    if t_common.size == 0:
        raise SystemExit("Empty time grid.")

    fig, (ax_c, ax_f) = plt.subplots(1, 2, figsize=(10.5, 5.0), constrained_layout=False)

    global_depth_candidates: List[float] = []

    for idx, group in enumerate(valid_groups):
        data = group_results[group]
        assert data is not None
        color = default_group_color(group, idx)
        tg = data["time_min"]

        c_mean = interp_on_grid(t_common, tg, data["cyto_mean"])
        c_spread = interp_on_grid(t_common, tg, data["cyto_spread"])
        c_n = interp_on_grid(t_common, tg, data["cyto_n"].astype(float))
        f_mean = interp_on_grid(t_common, tg, data["front_mean"])
        f_spread = interp_on_grid(t_common, tg, data["front_spread"])
        f_n = interp_on_grid(t_common, tg, data["front_n"].astype(float))

        c_ok = np.isfinite(c_mean) & np.isfinite(c_spread) & np.isfinite(c_n) & (c_n > 0)
        if np.any(c_ok):
            ax_c.fill_between(
                t_common[c_ok],
                (c_mean - c_spread)[c_ok],
                (c_mean + c_spread)[c_ok],
                color=color,
                alpha=0.15,
                linewidth=0,
            )
            ax_c.plot(
                t_common[c_ok],
                c_mean[c_ok],
                color=color,
                linewidth=2.0,
                label=f"{group.capitalize()} (n={group_stats[group]['used_runs']})",
            )
            global_depth_candidates.append(
                float(np.nanmax((c_mean + c_spread)[c_ok]))
            )

        f_ok = np.isfinite(f_mean) & np.isfinite(f_spread) & np.isfinite(f_n) & (f_n > 0)
        if np.any(f_ok):
            ax_f.fill_between(
                t_common[f_ok],
                (f_mean - f_spread)[f_ok],
                (f_mean + f_spread)[f_ok],
                color=color,
                alpha=0.10,
                linewidth=0,
            )
            ax_f.plot(
                t_common[f_ok],
                f_mean[f_ok],
                color=color,
                linewidth=1.8,
                linestyle="--",
                label=f"{group.capitalize()} (n={group_stats[group]['used_runs']})",
            )
            global_depth_candidates.append(
                float(np.nanmax((f_mean + f_spread)[f_ok]))
            )

    y_hi = float(np.nanmax(global_depth_candidates)) if global_depth_candidates else 1.0
    if not np.isfinite(y_hi) or y_hi <= 0:
        y_hi = 1.0
    y_min = args.y_min if args.y_min is not None else 0.0
    y_max = args.y_max if args.y_max is not None else y_hi * 1.05

    for ax in (ax_c, ax_f):
        ax.set_ylabel("Depth (µm)", fontsize=10)
        if y_max > y_min:
            ax.set_ylim(y_max, y_min)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Time (min)", fontsize=10)
        ax.set_box_aspect(1)
        if ax.lines and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="best", framealpha=0.9)
        if args.x_min is not None or args.x_max is not None:
            auto_min, auto_max = ax.get_xlim()
            x_min = args.x_min if args.x_min is not None else auto_min
            x_max = args.x_max if args.x_max is not None else auto_max
            if x_max > x_min:
                ax.set_xlim(x_min, x_max)

    ax_c.set_title("Cytoplasm mean ± variability", fontsize=11)
    ax_f.set_title("Membrane front mean ± variability", fontsize=11)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.12, wspace=0.28)

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

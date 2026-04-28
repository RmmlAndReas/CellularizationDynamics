#!/usr/bin/env python3
"""
One figure per group with every run plotted (cytoplasm solid, membrane front dashed).

Species-agnostic: use to inspect spread and outliers within dorsal, ventral, etc.

Writes ``<sample-folder>/results/{out-stem}_{group}_individuals.png`` (and .pdf).

Example:

    python scripts/analysis/plot_geometry_subset_individuals.py \\
        --sample-folder data/MySpecies/wt \\
        --out-stem wt_runs

    Align time so the first frame where the cellularization front depth
    (``front_minus_apical_um``) reaches 20 µm is t = 0; cytoplasm uses the same shift:

        python scripts/analysis/plot_geometry_subset_individuals.py \\
            --sample-folder data/MySpecies/wt \\
            --out-stem wt_runs \\
            --align-front-depth-um 20
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from subset_geometry_timeseries import detect_groups, load_group_traces, sample_colors


def _first_time_front_reaches_depth(
    time_min: np.ndarray, front_um: np.ndarray, depth_um: float
) -> float | None:
    """First ``time_min`` where ``front_um >= depth_um`` (finite samples only)."""
    if time_min.size == 0 or front_um.size == 0 or time_min.shape != front_um.shape:
        return None
    try:
        target = float(depth_um)
    except (TypeError, ValueError):
        return None
    ok = np.isfinite(time_min) & np.isfinite(front_um)
    idxs = np.where(ok & (front_um >= target))[0]
    if idxs.size == 0:
        return None
    return float(time_min[int(idxs[0])])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot each run's trace in its own file per orientation group."
    )
    parser.add_argument("--sample-folder", required=True)
    parser.add_argument(
        "--out-stem",
        default="geometry_subset_individuals",
        help="Stem for <stem>_<group>_individuals.png under results/",
    )
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument(
        "--align-front-depth-um",
        type=float,
        default=None,
        metavar="UM",
        help=(
            "If set, shift each run's time axis so the first time the membrane front depth "
            "(front_minus_apical_um) reaches this value (µm) is t=0. Cytoplasm and front "
            "share the same shift. Runs that never reach this depth are plotted in absolute time "
            "with a warning."
        ),
    )
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
        traces, stats = load_group_traces(sample_folder, group)
        print(
            f"{group}: found={stats['found_runs']}, used={stats['used_runs']}, skipped={stats['skipped_runs']}"
        )
        if not traces:
            continue

        colors = sample_colors(len(traces))
        fig, ax = plt.subplots(figsize=(panel_in, panel_in), constrained_layout=False)
        title = f"{group.capitalize()} — individual runs (n={len(traces)})"
        if args.align_front_depth_um is not None:
            title += f"; t=0 when front ≥ {args.align_front_depth_um:g} µm"
        ax.set_title(title, fontsize=11, fontweight="bold")

        depth_vals: list[float] = []
        for i, tr in enumerate(traces):
            c = colors[i]
            lbl = tr.run_name
            t_plot = tr.time_min
            if args.align_front_depth_um is not None:
                t0 = _first_time_front_reaches_depth(tr.time_min, tr.front_um, args.align_front_depth_um)
                if t0 is None:
                    print(
                        f"Warning: {group}/{lbl} front never reaches {args.align_front_depth_um:g} µm; "
                        "plotting absolute time for this run."
                    )
                else:
                    t_plot = tr.time_min - t0
            if tr.time_min.size and np.any(np.isfinite(tr.cytoplasm_um)):
                ax.plot(t_plot, tr.cytoplasm_um, color=c, linewidth=1.0, label=f"{lbl} cytoplasm")
                depth_vals.extend(
                    list(tr.cytoplasm_um[np.isfinite(tr.cytoplasm_um)].ravel())
                )
            if tr.time_min.size and np.any(np.isfinite(tr.front_um)):
                ax.plot(
                    t_plot,
                    tr.front_um,
                    color=c,
                    linewidth=0.85,
                    linestyle="--",
                    label=f"{lbl} front",
                )
                depth_vals.extend(list(tr.front_um[np.isfinite(tr.front_um)].ravel()))

        ax.set_ylabel("Depth (µm)", fontsize=10)
        if args.align_front_depth_um is not None:
            ax.set_xlabel(f"Time (min); 0 = first front ≥ {args.align_front_depth_um:g} µm", fontsize=10)
        else:
            ax.set_xlabel("Time (min)", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.set_box_aspect(1)

        if depth_vals:
            lo, hi = float(np.nanmin(depth_vals)), float(np.nanmax(depth_vals))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                pad = 0.05 * (hi - lo)
                y_min = args.y_min if args.y_min is not None else lo - pad
                y_max = args.y_max if args.y_max is not None else hi + pad
                if y_max > y_min:
                    ax.set_ylim(y_max, y_min)
        elif args.y_min is not None and args.y_max is not None and args.y_max > args.y_min:
            ax.set_ylim(args.y_max, args.y_min)

        if args.x_min is not None or args.x_max is not None:
            auto_min, auto_max = ax.get_xlim()
            x_min = args.x_min if args.x_min is not None else auto_min
            x_max = args.x_max if args.x_max is not None else auto_max
            if x_max > x_min:
                ax.set_xlim(x_min, x_max)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                fontsize=6,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0,
                framealpha=0.92,
            )

        fig.subplots_adjust(left=0.11, right=0.70, top=0.90, bottom=0.14)
        stem = f"{args.out_stem}_{group}_individuals"
        out_png = os.path.join(out_dir, f"{stem}.png")
        out_pdf = os.path.join(out_dir, f"{stem}.pdf")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, dpi=300, bbox_inches="tight", format="pdf")
        plt.close(fig)
        print(f"Saved: {out_png}")
        any_saved = True

    if not any_saved:
        raise SystemExit("No plottable runs found in any group.")


if __name__ == "__main__":
    main()

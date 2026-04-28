#!/usr/bin/env python3
"""
Plot grouped boxplot + individual points for per-movie maximum cellularization height.

Input CSV is expected to contain:
    group,movie_id,movie_path,n_points,max_cellu_height_um
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
import yaml


DEFAULT_GROUPS: Tuple[str, ...] = ("dorsal", "ventral", "lateral")
GROUP_COLORS: Dict[str, str] = {
    "dorsal": "tab:blue",
    "ventral": "tab:green",
    "lateral": "tab:orange",
}


@dataclass
class HeightRow:
    group: str
    movie_id: str
    experiment: str
    value_um: float


def parse_float(value: Optional[str]) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def default_out_prefix(input_csv: str) -> str:
    parent = os.path.dirname(os.path.abspath(input_csv))
    return os.path.join(parent, "max_cellu_height_boxplot")


def load_rows(input_csv: str) -> List[HeightRow]:
    if not os.path.isfile(input_csv):
        raise SystemExit(f"Input CSV does not exist: {input_csv}")

    rows: List[HeightRow] = []
    try:
        with open(input_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            required = {"group", "movie_id", "max_cellu_height_um"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                missing = required.difference(set(reader.fieldnames or []))
                raise SystemExit(f"Input CSV missing required columns: {', '.join(sorted(missing))}")
            for row in reader:
                group = (row.get("group") or "").strip()
                movie_id = (row.get("movie_id") or "").strip()
                experiment = (row.get("experiment") or "unknown").strip() or "unknown"
                value_um = parse_float(row.get("max_cellu_height_um"))
                if not group or not movie_id or not np.isfinite(value_um):
                    continue
                rows.append(
                    HeightRow(
                        group=group,
                        movie_id=movie_id,
                        experiment=experiment,
                        value_um=float(value_um),
                    )
                )
    except OSError as exc:
        raise SystemExit(f"Failed reading input CSV: {exc}") from exc

    if not rows:
        raise SystemExit("No valid rows found in input CSV.")
    return rows


def resolve_groups(rows: Sequence[HeightRow], requested_groups: Sequence[str]) -> List[str]:
    present = {r.group for r in rows}
    ordered = [g for g in requested_groups if g in present]
    extras = sorted(g for g in present if g not in set(requested_groups))
    groups = ordered + extras
    if not groups:
        raise SystemExit("No groups available to plot.")
    return groups


def load_grouping_from_datasets_yaml(path: str) -> Tuple[List[str], List[str]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except OSError as exc:
        raise SystemExit(f"Failed reading datasets YAML: {exc}") from exc

    if not isinstance(cfg, dict):
        raise SystemExit(f"Datasets YAML root must be a mapping: {path}")
    datasets = cfg.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise SystemExit("Datasets YAML must contain non-empty 'datasets' list.")

    groups: List[str] = []
    experiments: List[str] = []
    seen_groups = set()
    seen_experiments = set()
    for idx, ds in enumerate(datasets):
        if not isinstance(ds, dict):
            raise SystemExit(f"datasets[{idx}] must be a mapping.")
        name = str(ds.get("name", "")).strip()
        if not name:
            raise SystemExit(f"datasets[{idx}].name is required.")
        if name not in seen_experiments:
            experiments.append(name)
            seen_experiments.add(name)

        ds_groups = ds.get("groups")
        if not isinstance(ds_groups, list) or not ds_groups:
            raise SystemExit(f"datasets[{idx}].groups must be a non-empty list.")
        alias = ds.get("group_alias", {}) or {}
        if not isinstance(alias, dict):
            raise SystemExit(f"datasets[{idx}].group_alias must be a mapping if provided.")
        for g in ds_groups:
            raw = str(g).strip()
            if not raw:
                continue
            canonical = str(alias.get(raw, raw)).strip() or raw
            if canonical not in seen_groups:
                groups.append(canonical)
                seen_groups.add(canonical)
    if not groups:
        raise SystemExit("No groups found in datasets YAML.")
    return groups, experiments


def _parse_layout(raw_layout: object) -> Dict[str, float]:
    layout: Dict[str, float] = {}
    if raw_layout is None:
        return layout
    if not isinstance(raw_layout, dict):
        raise SystemExit("plot_max_cellu_height.layout must be a mapping when provided.")
    if "box_width" in raw_layout:
        layout["box_width"] = float(raw_layout["box_width"])
    if "y_max_scale" in raw_layout:
        layout["y_max_scale"] = float(raw_layout["y_max_scale"])
    if "x_tick_fontsize" in raw_layout:
        layout["x_tick_fontsize"] = float(raw_layout["x_tick_fontsize"])
    if "figsize" in raw_layout:
        figsize = raw_layout["figsize"]
        if (
            not isinstance(figsize, (list, tuple))
            or len(figsize) != 2
        ):
            raise SystemExit("plot_max_cellu_height.layout.figsize must be [width, height].")
        layout["fig_w"] = float(figsize[0])
        layout["fig_h"] = float(figsize[1])
    return layout


def load_plot_config_from_datasets_yaml(path: str) -> Dict[str, object]:
    groups, experiments = load_grouping_from_datasets_yaml(path)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    plot_cfg = cfg.get("plot_max_cellu_height", {}) or {}
    if not isinstance(plot_cfg, dict):
        raise SystemExit("plot_max_cellu_height must be a mapping when provided.")

    mode = plot_cfg.get("mode")
    if mode is not None:
        mode = str(mode).strip()
        allowed_modes = {"boxplot", "boxplot_experiment", "by_experiment", "group_experiment_pairs"}
        if mode not in allowed_modes:
            raise SystemExit(
                f"Unsupported plot_max_cellu_height.mode '{mode}'. "
                f"Allowed: {', '.join(sorted(allowed_modes))}"
            )

    cfg_groups = plot_cfg.get("groups")
    if cfg_groups is not None:
        if not isinstance(cfg_groups, list) or not cfg_groups:
            raise SystemExit("plot_max_cellu_height.groups must be a non-empty list.")
        groups = [str(g).strip() for g in cfg_groups if str(g).strip()]

    cfg_exps = plot_cfg.get("experiments")
    if cfg_exps is not None:
        if not isinstance(cfg_exps, list) or not cfg_exps:
            raise SystemExit("plot_max_cellu_height.experiments must be a non-empty list.")
        experiments = [str(e).strip() for e in cfg_exps if str(e).strip()]

    show_experiment = plot_cfg.get("show_experiment")
    if show_experiment is not None:
        show_experiment = bool(show_experiment)

    out_prefix = plot_cfg.get("out_prefix")
    if out_prefix is not None:
        out_prefix = str(out_prefix).strip() or None

    layout = _parse_layout(plot_cfg.get("layout"))
    return {
        "groups": groups,
        "experiments": experiments,
        "mode": mode,
        "show_experiment": show_experiment,
        "out_prefix": out_prefix,
        "layout": layout,
    }


def color_for_group(group: str, idx: int) -> str:
    if group in GROUP_COLORS:
        return GROUP_COLORS[group]
    try:
        cmap = matplotlib.colormaps["tab10"]
    except (AttributeError, KeyError):
        cmap = matplotlib.cm.get_cmap("tab10")
    return cmap(idx % 10)


def marker_map(experiments: Sequence[str]) -> Dict[str, str]:
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h"]
    ordered = sorted(set(experiments))
    return {exp: markers[i % len(markers)] for i, exp in enumerate(ordered)}


def permutation_pvalue(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    n_perm: int = 10000,
    rng_seed: int = 123,
) -> float:
    a = values_a[np.isfinite(values_a)]
    b = values_b[np.isfinite(values_b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    observed = abs(float(np.mean(a) - np.mean(b)))
    combined = np.concatenate([a, b])
    n_a = a.size
    rng = np.random.default_rng(rng_seed)
    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        diff = abs(float(np.mean(perm[:n_a]) - np.mean(perm[n_a:])))
        if diff >= observed:
            extreme += 1
    # add-one correction to avoid exact zero p-values
    return (extreme + 1) / (n_perm + 1)


def format_pvalue(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def welch_ttest_pvalue(values_a: np.ndarray, values_b: np.ndarray) -> float:
    a = values_a[np.isfinite(values_a)]
    b = values_b[np.isfinite(values_b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    _stat, pval = ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(pval)


def plot_boxplot(rows: Sequence[HeightRow], groups: Sequence[str], out_prefix: str) -> None:
    grouped: Dict[str, List[float]] = {g: [] for g in groups}
    grouped_rows: Dict[str, List[HeightRow]] = {g: [] for g in groups}
    for row in rows:
        if row.group in grouped:
            grouped[row.group].append(row.value_um)
            grouped_rows[row.group].append(row)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=False)
    rng = np.random.default_rng(12345)
    exp_to_marker = marker_map([r.experiment for r in rows])

    data_arrays: List[np.ndarray] = []
    data_groups: List[str] = []
    max_candidates: List[float] = []
    for idx, group in enumerate(groups):
        vals = np.asarray(grouped[group], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        data_arrays.append(vals)
        data_groups.append(group)
        max_candidates.append(float(np.max(vals)))

    if not data_arrays:
        raise SystemExit("No finite values available to plot.")

    positions = np.arange(len(data_groups))
    bp = ax.boxplot(
        data_arrays,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        zorder=1,
    )
    for idx, patch in enumerate(bp["boxes"]):
        group = data_groups[idx]
        patch.set_facecolor(color_for_group(group, idx))
        patch.set_alpha(0.30)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
    for key in ("medians", "caps", "whiskers"):
        for artist in bp[key]:
            artist.set_color("black")
            artist.set_linewidth(1.0 if key != "medians" else 1.4)

    for idx, group in enumerate(data_groups):
        rows_for_group = [r for r in grouped_rows[group] if np.isfinite(r.value_um)]
        if not rows_for_group:
            continue
        vals = np.asarray([r.value_um for r in rows_for_group], dtype=float)
        jitter = rng.uniform(-0.18, 0.18, size=vals.size)
        x = np.full(vals.size, idx, dtype=float) + jitter
        color = color_for_group(group, idx)
        for j, row in enumerate(rows_for_group):
            marker = exp_to_marker.get(row.experiment, "o")
            ax.scatter(
                [x[j]],
                [row.value_um],
                s=44,
                marker=marker,
                color=color,
                alpha=0.9,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
    ax.set_xticks(np.arange(len(data_groups)))
    ax.set_xticklabels([g.capitalize() for g in data_groups], fontsize=10)
    ax.set_ylabel("Max cellularization height (um)", fontsize=10)
    ax.set_xlim(-0.6, len(data_groups) - 0.4)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(labelsize=9)
    ax.set_box_aspect(1)

    if max_candidates:
        y_top = max(max_candidates) * 1.1
        if y_top > 0:
            ax.set_ylim(0.0, y_top)

    group_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color_for_group(group, idx),
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=7,
            label=f"{group.capitalize()} (n={len(data_arrays[idx])})",
        )
        for idx, group in enumerate(data_groups)
    ]
    exp_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=exp_to_marker[exp],
            linestyle="",
            markerfacecolor="gray",
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=7,
            label=exp,
        )
        for exp in sorted(exp_to_marker)
    ]
    if group_handles:
        legend1 = ax.legend(handles=group_handles, fontsize=8, loc="lower left", framealpha=0.9, title="Group")
        ax.add_artist(legend1)
    if exp_handles:
        ax.legend(handles=exp_handles, fontsize=8, loc="lower right", framealpha=0.9, title="Experiment")

    # Show simple between-group permutation-test p-values on means.
    stat_lines: List[str] = []
    for (i, g1), (j, g2) in itertools.combinations(enumerate(data_groups), 2):
        pval = permutation_pvalue(data_arrays[i], data_arrays[j], n_perm=10000, rng_seed=1000 + i * 10 + j)
        stat_lines.append(f"{g1} vs {g2}: p={format_pvalue(pval)}")
    if stat_lines:
        ax.text(
            0.02,
            0.98,
            "Permutation test (mean difference)\n" + "\n".join(stat_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 4},
        )

    fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.14)

    out_dir = os.path.dirname(os.path.abspath(out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_png = f"{out_prefix}.png"
    out_pdf = f"{out_prefix}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def plot_experiment_boxplot(
    rows: Sequence[HeightRow],
    groups: Sequence[str],
    experiments: Sequence[str],
    out_prefix: str,
    show_experiment: bool = False,
) -> None:
    exps = [e for e in experiments if e]
    if len(exps) != 2:
        raise SystemExit("Barplot mode requires exactly two experiments.")

    grouped_exp_values: Dict[Tuple[str, str], np.ndarray] = {}
    for group in groups:
        for exp in exps:
            vals = np.asarray(
                [r.value_um for r in rows if r.group == group and r.experiment == exp],
                dtype=float,
            )
            vals = vals[np.isfinite(vals)]
            grouped_exp_values[(group, exp)] = vals

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=False)
    x = np.arange(len(groups), dtype=float)
    width = 0.34
    offsets = [-width / 2, width / 2]
    bar_colors = ["tab:blue", "tab:orange"]
    rng = np.random.default_rng(12345)

    ymax = 0.0
    for i, exp in enumerate(exps):
        ns: List[int] = []
        for group in groups:
            vals = grouped_exp_values[(group, exp)]
            n = int(vals.size)
            ns.append(n)
            if n > 0:
                ymax = max(ymax, float(np.max(vals)))

        xpos = x + offsets[i]
        for j, group in enumerate(groups):
            vals = grouped_exp_values[(group, exp)]
            if vals.size == 0:
                continue
            bp = ax.boxplot(
                [vals],
                positions=[xpos[j]],
                widths=width * 0.9,
                patch_artist=True,
                showfliers=False,
                zorder=1,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(bar_colors[i])
                patch.set_alpha(0.35)
                patch.set_edgecolor("black")
                patch.set_linewidth(1.0)
            for key in ("medians", "caps", "whiskers"):
                for artist in bp[key]:
                    artist.set_color("black")
                    artist.set_linewidth(1.0 if key != "medians" else 1.4)
            jitter = rng.uniform(-width * 0.18, width * 0.18, size=vals.size)
            ax.scatter(
                np.full(vals.size, xpos[j]) + jitter,
                vals,
                s=36,
                color=bar_colors[i],
                alpha=0.9,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
            if show_experiment:
                for k, yv in enumerate(vals):
                    ax.text(
                        xpos[j] + jitter[k] + 0.01,
                        yv,
                        exp,
                        fontsize=6.5,
                        ha="left",
                        va="center",
                        color="black",
                        alpha=0.9,
                    )

    stat_lines: List[str] = []
    for j, group in enumerate(groups):
        vals_a = grouped_exp_values[(group, exps[0])]
        vals_b = grouped_exp_values[(group, exps[1])]
        pval = welch_ttest_pvalue(vals_a, vals_b)
        stat_lines.append(
            f"{group}: {exps[0]}(n={vals_a.size}) vs {exps[1]}(n={vals_b.size}) p={format_pvalue(pval)}"
        )
        if np.isfinite(pval):
            y_pair = 0.0
            if vals_a.size:
                y_pair = max(y_pair, float(np.max(vals_a)))
            if vals_b.size:
                y_pair = max(y_pair, float(np.max(vals_b)))
            y_text = y_pair + max(0.5, 0.03 * max(ymax, 1.0))
            ax.text(
                x[j],
                y_text,
                f"p={format_pvalue(pval)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in groups], fontsize=10)
    ax.set_ylabel("Max cellularization height (um)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(labelsize=9)
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=bar_colors[i],
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=8,
            alpha=0.5,
            label=exps[i],
        )
        for i in range(2)
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right", framealpha=0.9, title="Experiment")

    if ymax > 0:
        ax.set_ylim(0.0, ymax * 1.25)

    if stat_lines:
        ax.text(
            0.02,
            0.98,
            "Welch t-test per group\n" + "\n".join(stat_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 4},
        )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.14)
    out_dir = os.path.dirname(os.path.abspath(out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_png = f"{out_prefix}.png"
    out_pdf = f"{out_prefix}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def plot_by_experiment(
    rows: Sequence[HeightRow],
    experiments: Sequence[str],
    out_prefix: str,
    show_experiment: bool = False,
) -> None:
    exps = [e for e in experiments if e]
    if not exps:
        raise SystemExit("by_experiment mode requires at least one experiment.")

    grouped_values: Dict[str, np.ndarray] = {}
    for exp in exps:
        vals = np.asarray([r.value_um for r in rows if r.experiment == exp], dtype=float)
        vals = vals[np.isfinite(vals)]
        grouped_values[exp] = vals

    fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=False)
    rng = np.random.default_rng(12345)
    x = np.arange(len(exps), dtype=float)
    width = 0.55
    ymax = 0.0

    for i, exp in enumerate(exps):
        vals = grouped_values[exp]
        if vals.size == 0:
            continue
        ymax = max(ymax, float(np.max(vals)))
        color = color_for_group(exp, i)
        bp = ax.boxplot(
            [vals],
            positions=[x[i]],
            widths=width,
            patch_artist=True,
            showfliers=False,
            zorder=1,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.0)
        for key in ("medians", "caps", "whiskers"):
            for artist in bp[key]:
                artist.set_color("black")
                artist.set_linewidth(1.0 if key != "medians" else 1.4)
        jitter = rng.uniform(-width * 0.22, width * 0.22, size=vals.size)
        ax.scatter(
            np.full(vals.size, x[i]) + jitter,
            vals,
            s=36,
            color=color,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        if show_experiment:
            for k, yv in enumerate(vals):
                ax.text(
                    x[i] + jitter[k] + 0.01,
                    yv,
                    exp,
                    fontsize=6.5,
                    ha="left",
                    va="center",
                    color="black",
                    alpha=0.9,
                )
        ax.text(
            x[i],
            np.max(vals) + max(0.5, 0.03 * max(ymax, 1.0)),
            f"n={vals.size}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(exps, fontsize=10)
    ax.set_ylabel("Max cellularization height (um)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(labelsize=9)
    if ymax > 0:
        ax.set_ylim(0.0, ymax * 1.25)

    stat_lines: List[str] = []
    for a_idx, b_idx in itertools.combinations(range(len(exps)), 2):
        exp_a = exps[a_idx]
        exp_b = exps[b_idx]
        vals_a = grouped_values[exp_a]
        vals_b = grouped_values[exp_b]
        pval = welch_ttest_pvalue(vals_a, vals_b)
        stat_lines.append(f"{exp_a} vs {exp_b}: p={format_pvalue(pval)}")
    if stat_lines:
        ax.text(
            0.02,
            0.98,
            "Welch t-test\n" + "\n".join(stat_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 4},
        )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.14)
    out_dir = os.path.dirname(os.path.abspath(out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_png = f"{out_prefix}.png"
    out_pdf = f"{out_prefix}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def plot_group_experiment_pairs(
    rows: Sequence[HeightRow],
    groups: Sequence[str],
    experiments: Sequence[str],
    out_prefix: str,
    show_experiment: bool = False,
    layout: Optional[Dict[str, float]] = None,
) -> None:
    style = layout or {}
    exps = [e for e in experiments if e]
    if not exps:
        exps = sorted({r.experiment for r in rows})

    combos: List[Tuple[str, str]] = []
    for group in groups:
        for exp in exps:
            vals = np.asarray(
                [r.value_um for r in rows if r.experiment == exp and r.group == group],
                dtype=float,
            )
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                combos.append((exp, group))
    if not combos:
        raise SystemExit("No non-empty (experiment, group) combinations to plot.")

    fig_w = float(style.get("fig_w", 8.4))
    fig_h = float(style.get("fig_h", 5.2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=False)
    rng = np.random.default_rng(12345)
    exp_to_marker = marker_map([r.experiment for r in rows])
    exp_to_color: Dict[str, str] = {exp: color_for_group(exp, i) for i, exp in enumerate(exps)}
    x = np.arange(len(combos), dtype=float)
    width = float(style.get("box_width", 0.58))
    ymax = 0.0
    group_to_positions: Dict[str, List[int]] = {g: [] for g in groups}
    group_to_values: Dict[str, Dict[str, np.ndarray]] = {g: {} for g in groups}

    for i, (exp, group) in enumerate(combos):
        vals = np.asarray(
            [r.value_um for r in rows if r.experiment == exp and r.group == group],
            dtype=float,
        )
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        ymax = max(ymax, float(np.max(vals)))
        color = exp_to_color.get(exp, color_for_group(exp, i))
        group_to_positions.setdefault(group, []).append(i)
        group_to_values.setdefault(group, {})[exp] = vals
        bp = ax.boxplot(
            [vals],
            positions=[x[i]],
            widths=width,
            patch_artist=True,
            showfliers=False,
            zorder=1,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.0)
        for key in ("medians", "caps", "whiskers"):
            for artist in bp[key]:
                artist.set_color("black")
                artist.set_linewidth(1.0 if key != "medians" else 1.4)
        jitter = rng.uniform(-width * 0.22, width * 0.22, size=vals.size)
        ax.scatter(
            np.full(vals.size, x[i]) + jitter,
            vals,
            s=34,
            marker=exp_to_marker.get(exp, "o"),
            color=color,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        if show_experiment:
            for k, yv in enumerate(vals):
                ax.text(
                    x[i] + jitter[k] + 0.01,
                    yv,
                    exp,
                    fontsize=6.5,
                    ha="left",
                    va="center",
                    color="black",
                    alpha=0.9,
                )

    labels = [f"{exp}\n{group}" for exp, group in combos]
    x_tick_fontsize = float(style.get("x_tick_fontsize", 9))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=x_tick_fontsize)
    ax.set_ylabel("Max cellularization height (um)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(labelsize=9)
    ax.set_ylim(20.0, 40.0)

    present_groups = [g for g in groups if group_to_positions.get(g)]
    present_exps = [exp for exp in exps if any(exp == e for e, _g in combos)]
    experiment_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=exp_to_color.get(exp, "gray"),
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=7,
            label=exp,
        )
        for exp in present_exps
    ]
    if experiment_handles:
        ax.legend(
            handles=experiment_handles,
            fontsize=8,
            loc="upper right",
            framealpha=0.9,
            title="Experiment",
        )

    # Within-group t-tests (dorsal/ventral/lateral): compare experiments that are present.
    # Place p-values above each side block.
    text_offset = max(0.5, 0.035 * max(ymax, 1.0))
    for group in present_groups:
        positions = group_to_positions.get(group, [])
        exp_vals = group_to_values.get(group, {})
        if not positions or len(exp_vals) < 2:
            continue
        x_center = float(np.mean([x[i] for i in positions]))
        local_ymax = max(float(np.max(v)) for v in exp_vals.values() if v.size > 0)
        y_text = local_ymax + text_offset
        lines: List[str] = []
        for exp_a, exp_b in itertools.combinations([e for e in exps if e in exp_vals], 2):
            pval = welch_ttest_pvalue(exp_vals[exp_a], exp_vals[exp_b])
            lines.append(f"{exp_a} vs {exp_b}: {format_pvalue(pval)}")
        if not lines:
            continue
        ax.text(
            x_center,
            y_text,
            "\n".join(lines),
            ha="center",
            va="bottom",
            fontsize=7.5,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 2},
        )

    fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.20)
    out_dir = os.path.dirname(os.path.abspath(out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_png = f"{out_prefix}.png"
    out_pdf = f"{out_prefix}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot grouped boxplot + individual samples of max cellularization height."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output path prefix without extension (default next to CSV)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        help="Preferred group order (default: dorsal ventral lateral)",
    )
    parser.add_argument(
        "--mode",
        choices=["boxplot", "boxplot_experiment", "by_experiment", "group_experiment_pairs"],
        default=None,
        help=(
            "Plot mode. Default behavior: use YAML plot_max_cellu_height.mode if available, "
            "otherwise group_experiment_pairs."
        ),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Experiment labels (e.g. wt centrifugation suction). For boxplot_experiment mode, provide exactly two.",
    )
    parser.add_argument(
        "--show-experiment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overlay experiment label text next to each sample point.",
    )
    parser.add_argument(
        "--datasets-yaml",
        default=None,
        help=(
            "Optional datasets YAML (same format as exporter config). "
            "When provided, uses YAML-defined group order and experiment order."
        ),
    )
    args = parser.parse_args()

    rows = load_rows(os.path.abspath(args.input_csv))
    yaml_cfg: Dict[str, object] = {}
    if args.datasets_yaml:
        yaml_cfg = load_plot_config_from_datasets_yaml(os.path.abspath(args.datasets_yaml))
        groups = resolve_groups(rows, list(yaml_cfg.get("groups", [])))
    else:
        groups = resolve_groups(rows, args.groups)
    yaml_out_prefix = yaml_cfg.get("out_prefix") if yaml_cfg else None
    if args.out_prefix:
        out_prefix = os.path.abspath(args.out_prefix)
    elif isinstance(yaml_out_prefix, str) and yaml_out_prefix.strip():
        out_prefix = os.path.abspath(yaml_out_prefix)
    else:
        out_prefix = default_out_prefix(args.input_csv)
    mode = args.mode or str(yaml_cfg.get("mode") or "group_experiment_pairs")
    show_experiment = (
        args.show_experiment
        if args.show_experiment is not None
        else bool(yaml_cfg.get("show_experiment", False))
    )
    yaml_experiments: Optional[List[str]] = None
    if yaml_cfg.get("experiments"):
        yaml_experiments = list(yaml_cfg["experiments"])  # type: ignore[index]
    layout: Optional[Dict[str, float]] = None
    if yaml_cfg.get("layout"):
        layout = dict(yaml_cfg["layout"])  # type: ignore[arg-type]
    print(f"Plot groups: {', '.join(groups)}")
    if mode == "boxplot":
        plot_boxplot(rows, groups, out_prefix)
        return
    if mode == "by_experiment":
        if args.experiments:
            experiments = list(args.experiments)
        elif yaml_experiments is not None:
            experiments = list(yaml_experiments)
        else:
            experiments = sorted({r.experiment for r in rows})
        plot_by_experiment(rows, experiments, out_prefix, show_experiment=show_experiment)
        return
    if mode == "group_experiment_pairs":
        if args.experiments:
            experiments = list(args.experiments)
        elif yaml_experiments is not None:
            experiments = list(yaml_experiments)
        else:
            experiments = sorted({r.experiment for r in rows})
        plot_group_experiment_pairs(
            rows,
            groups,
            experiments,
            out_prefix,
            show_experiment=show_experiment,
            layout=layout,
        )
        return

    if args.experiments:
        experiments = list(args.experiments)
    elif yaml_experiments is not None:
        experiments = list(yaml_experiments)
    else:
        experiments = sorted({r.experiment for r in rows})
    if len(experiments) != 2:
        raise SystemExit(
            "boxplot_experiment mode requires exactly two experiments. "
            "Provide them with --experiments EXP_A EXP_B."
        )
    plot_experiment_boxplot(
        rows,
        groups,
        experiments,
        out_prefix,
        show_experiment=show_experiment,
    )


if __name__ == "__main__":
    main()

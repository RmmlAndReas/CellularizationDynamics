#!/usr/bin/env python3
"""
Shared helpers for geometry_timeseries.csv grouped by orientation folders
(dorsal/*, ventral/*, lateral/*).

Used by ``scripts/analysis/plot_geometry_subset_*.py`` for any species whose data use this layout.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np

PREFERRED_GROUP_ORDER = ("dorsal", "ventral", "lateral")

CYTOPLASM_COLOR = "tab:blue"
FRONT_COLOR = "tab:red"


@dataclass
class RunTrace:
    run_name: str
    time_min: np.ndarray
    cytoplasm_um: np.ndarray
    front_um: np.ndarray


def parse_csv_float(cell: Optional[str]) -> float:
    if cell is None:
        return float("nan")
    s = str(cell).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def list_run_dirs(group_folder: str) -> List[str]:
    if not os.path.isdir(group_folder):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(group_folder), key=lambda x: (not x.isdigit(), x)):
        path = os.path.join(group_folder, name)
        if os.path.isdir(path):
            out.append(path)
    return out


def is_run_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    csv_path = os.path.join(path, "track", "geometry_timeseries.csv")
    return os.path.isfile(csv_path)


def detect_groups(sample_folder: str) -> List[str]:
    discovered: List[str] = []
    try:
        names = sorted(os.listdir(sample_folder), key=lambda x: (x not in PREFERRED_GROUP_ORDER, x))
    except OSError:
        return list(PREFERRED_GROUP_ORDER)

    for name in names:
        group_folder = os.path.join(sample_folder, name)
        if not os.path.isdir(group_folder):
            continue
        try:
            children = os.listdir(group_folder)
        except OSError:
            continue
        if any(is_run_dir(os.path.join(group_folder, child)) for child in children):
            discovered.append(name)

    if not discovered:
        return list(PREFERRED_GROUP_ORDER)

    pref_set = set(PREFERRED_GROUP_ORDER)
    preferred = [g for g in PREFERRED_GROUP_ORDER if g in discovered]
    extras = [g for g in discovered if g not in pref_set]
    return preferred + extras


def load_geometry_csv(csv_path: str) -> Optional[RunTrace]:
    if not os.path.isfile(csv_path):
        return None

    t: List[float] = []
    c: List[float] = []
    f: List[float] = []

    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            required = {"time_min", "cytoplasm_height_um", "front_minus_apical_um"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                return None

            for row in reader:
                t.append(parse_csv_float(row.get("time_min")))
                c.append(parse_csv_float(row.get("cytoplasm_height_um")))
                f.append(parse_csv_float(row.get("front_minus_apical_um")))
    except OSError:
        return None

    if not t:
        return None

    time = np.asarray(t, dtype=float)
    cyto = np.asarray(c, dtype=float)
    front = np.asarray(f, dtype=float)

    valid_time = np.isfinite(time)
    if not np.any(valid_time):
        return None
    time = time[valid_time]
    cyto = cyto[valid_time]
    front = front[valid_time]

    order = np.argsort(time)
    time = time[order]
    cyto = cyto[order]
    front = front[order]

    uniq_time, uniq_idx = np.unique(time, return_index=True)
    time = uniq_time
    cyto = cyto[uniq_idx]
    front = front[uniq_idx]

    return RunTrace(
        run_name=os.path.basename(os.path.dirname(os.path.dirname(csv_path))),
        time_min=time,
        cytoplasm_um=cyto,
        front_um=front,
    )


def interp_on_grid(time_grid: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.full(time_grid.shape, np.nan, dtype=float)
    valid = np.isfinite(y)
    if np.sum(valid) < 2:
        return out
    tv = t[valid]
    yv = y[valid]
    in_bounds = (time_grid >= tv[0]) & (time_grid <= tv[-1])
    if np.any(in_bounds):
        out[in_bounds] = np.interp(time_grid[in_bounds], tv, yv)
    return out


def mean_and_spread(arr2d: np.ndarray, variability: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_valid = np.sum(np.isfinite(arr2d), axis=0)
    mean = np.full(arr2d.shape[1], np.nan, dtype=float)
    std = np.full(arr2d.shape[1], np.nan, dtype=float)
    for j in range(arr2d.shape[1]):
        col = arr2d[:, j]
        valid = np.isfinite(col)
        if not np.any(valid):
            continue
        vals = col[valid]
        mean[j] = float(np.mean(vals))
        std[j] = float(np.std(vals, ddof=0))
    if variability == "std":
        spread = std
    else:
        spread = np.full_like(std, np.nan)
        good = n_valid > 0
        spread[good] = std[good] / np.sqrt(n_valid[good].astype(float))
    return mean, spread, n_valid


def aggregate_group(runs: List[RunTrace], variability: str) -> Dict[str, np.ndarray]:
    all_times = np.unique(np.concatenate([r.time_min for r in runs], axis=0))
    all_times = np.sort(all_times)

    cyto_mat = np.vstack([interp_on_grid(all_times, r.time_min, r.cytoplasm_um) for r in runs])
    front_mat = np.vstack([interp_on_grid(all_times, r.time_min, r.front_um) for r in runs])

    cyto_mean, cyto_spread, cyto_n = mean_and_spread(cyto_mat, variability)
    front_mean, front_spread, front_n = mean_and_spread(front_mat, variability)
    return {
        "time_min": all_times,
        "cyto_mean": cyto_mean,
        "cyto_spread": cyto_spread,
        "cyto_n": cyto_n,
        "front_mean": front_mean,
        "front_spread": front_spread,
        "front_n": front_n,
    }


def load_group_traces(sample_folder: str, group: str) -> Tuple[List[RunTrace], Dict[str, int]]:
    group_folder = os.path.join(sample_folder, group)
    run_dirs = list_run_dirs(group_folder)

    found_runs = len(run_dirs)
    used_runs = 0
    skipped_runs = 0
    traces: List[RunTrace] = []

    for run_dir in run_dirs:
        csv_path = os.path.join(run_dir, "track", "geometry_timeseries.csv")
        trace = load_geometry_csv(csv_path)
        if trace is None:
            skipped_runs += 1
            print(f"Warning: {group}/{os.path.basename(run_dir)} missing/invalid geometry_timeseries.csv")
            continue

        finite_cyto = np.any(np.isfinite(trace.cytoplasm_um))
        finite_front = np.any(np.isfinite(trace.front_um))
        if not (finite_cyto or finite_front):
            skipped_runs += 1
            print(f"Warning: {group}/{trace.run_name} has no finite cytoplasm/front data")
            continue

        traces.append(trace)
        used_runs += 1

    stats = {"found_runs": found_runs, "used_runs": used_runs, "skipped_runs": skipped_runs}
    return traces, stats


def build_group_data(
    sample_folder: str, group: str, variability: str
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, int]]:
    traces, stats = load_group_traces(sample_folder, group)
    if not traces:
        return None, stats
    return aggregate_group(traces, variability=variability), stats


def default_group_color(group: str, index: int) -> str:
    palette = {
        "dorsal": "tab:blue",
        "ventral": "tab:green",
        "lateral": "tab:orange",
    }
    if group in palette:
        return palette[group]
    try:
        cmap = matplotlib.colormaps["tab10"]
    except (AttributeError, KeyError):
        cmap = matplotlib.cm.get_cmap("tab10")
    return cmap(index % 10)


def sample_colors(n: int) -> List:
    try:
        cmap = matplotlib.colormaps["tab10"]
    except (AttributeError, KeyError):
        cmap = matplotlib.cm.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def common_time_grid_from_group_results(
    group_results: Dict[str, Optional[Dict[str, np.ndarray]]],
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for d in group_results.values():
        if d is not None and d["time_min"].size:
            chunks.append(d["time_min"])
    if not chunks:
        return np.array([], dtype=float)
    return np.sort(np.unique(np.concatenate(chunks)))


def plot_mean_front_single_group_on_ax(
    ax,
    data: Dict[str, np.ndarray],
    *,
    cyto_color: str = CYTOPLASM_COLOR,
    front_color: str = FRONT_COLOR,
) -> None:
    t = data["time_min"]
    c_mean = data["cyto_mean"]
    c_spread = data["cyto_spread"]
    c_n = data["cyto_n"]
    f_mean = data["front_mean"]
    f_spread = data["front_spread"]
    f_n = data["front_n"]

    c_ok = np.isfinite(c_mean) & np.isfinite(c_spread) & (c_n > 0)
    if np.any(c_ok):
        ax.fill_between(
            t[c_ok],
            (c_mean - c_spread)[c_ok],
            (c_mean + c_spread)[c_ok],
            color=cyto_color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            t[c_ok],
            c_mean[c_ok],
            color=cyto_color,
            linewidth=2.0,
            label="Cytoplasm mean",
        )

    f_ok = np.isfinite(f_mean) & np.isfinite(f_spread) & (f_n > 0)
    if np.any(f_ok):
        ax.fill_between(
            t[f_ok],
            (f_mean - f_spread)[f_ok],
            (f_mean + f_spread)[f_ok],
            color=front_color,
            alpha=0.10,
            linewidth=0,
        )
        ax.plot(
            t[f_ok],
            f_mean[f_ok],
            color=front_color,
            linewidth=1.8,
            linestyle="--",
            label="Membrane front mean",
        )

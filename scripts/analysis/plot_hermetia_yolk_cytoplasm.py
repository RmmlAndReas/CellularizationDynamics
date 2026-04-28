#!/usr/bin/env python3
"""
Plot yolk cytoplasm size over time for all Hermetia samples.

This script:
  - Takes a species folder (e.g. data/Hermetia)
  - Uses all sample subfolders inside it (e.g. data/Hermetia/1, /2, ...)
  - Crops traces using per-sample config.yaml time_window (if present)
  - Reads sample config (single file or species index) to group samples by orientation
  - Plots cytoplasm thickness vs time (µm) for each orientation + a grouped mean±SD panel
  - Optionally aligns time so that a chosen cytoplasm height (in µm) occurs at t = 0
    (see `--align-depth-um` below).

Usage (from repo root):

  - Without alignment (absolute time from movie start):

        python scripts/analysis/plot_hermetia_yolk_cytoplasm.py -s data/Hermetia

  - With depth alignment (e.g. align when the cytoplasm first reaches 20 µm):

        python scripts/analysis/plot_hermetia_yolk_cytoplasm.py -s data/Hermetia --align-depth-um 20

    Here, for each sample:
      - The cytoplasm thickness trace (µm) over time is read from
        `track/cytoplasm_region.tsv`.
      - The script finds the first time when `cytoplasm_height_microns >= align_depth_um`
        and shifts that time to t = 0. If the sample never reaches that depth,
        it is left unaligned and a warning is printed.
"""

import argparse
import csv
import os
from typing import List, Optional, Tuple

import analysis_paths  # noqa: F401 — adds scripts/ to sys.path
import matplotlib.pyplot as plt
import numpy as np
import yaml
from samples_loader import load_samples_config, default_samples_config_path


def _compute_align_time_from_depth(time_min: np.ndarray, height_um: np.ndarray, depth_um: float):
    """
    Compute the time (minutes) when cytoplasm height first reaches `depth_um`.

    Returns:
      - float time (min) of first frame where height >= depth_um, or
      - None if this depth is never reached or inputs are invalid.
    """
    if time_min is None or height_um is None:
        return None
    if time_min.size == 0 or height_um.size == 0:
        return None
    if time_min.shape != height_um.shape:
        return None

    try:
        depth_val = float(depth_um)
    except Exception:
        return None

    idxs = np.where(height_um >= depth_val)[0]
    if idxs.size == 0:
        return None
    return float(time_min[int(idxs[0])])


def load_orientation_map(samples_yaml_path: str, species_folder: str):
    """
    Build mapping: sample_folder_abs -> group ('dorsal'|'lateral'|'ventral'|'suction').

    Uses keys like:
      - hermetia2-lateral
      - hermetia6-ventral-suction
    from sample config keys and their 'path' entries.
    """
    cfg = load_samples_config(samples_yaml_path)
    samples = cfg.get("samples", {}) if isinstance(cfg, dict) else {}

    orientation_map = {}
    for name, entry in samples.items():
        if not isinstance(entry, dict):
            continue
        parts = name.split("-")
        if len(parts) < 2:
            continue
        suffixes = [p.lower() for p in parts[1:]]
        group = None
        if "suction" in suffixes:
            group = "suction"
        else:
            for candidate in suffixes:
                if candidate in ("dorsal", "lateral", "ventral"):
                    group = candidate
                    break
        if group is None:
            continue
        rel_path = entry.get("path")
        if not rel_path:
            continue
        abs_path = os.path.abspath(rel_path)
        # Only include entries under the species folder we are plotting
        if os.path.commonpath([abs_path, species_folder]) != species_folder:
            continue
        orientation_map[abs_path] = group
    return orientation_map


def discover_sample_folders(species_folder: str):
    """Return a sorted list of (sample_name, sample_folder_abs)."""
    entries = []
    try:
        for name in os.listdir(species_folder):
            if name.startswith("."):
                continue
            p = os.path.join(species_folder, name)
            if os.path.isdir(p):
                entries.append((name, p))
    except FileNotFoundError:
        return []
    # Natural-ish sort: numeric folders first by int, then others lexicographically
    def sort_key(item):
        n, _ = item
        try:
            return (0, int(n))
        except Exception:
            return (1, n)
    return sorted(entries, key=sort_key)


def load_time_window_from_config(sample_folder: str):
    cfg_path = os.path.join(sample_folder, "config.yaml")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return None
    tw = cfg.get("time_window")
    if not isinstance(tw, dict):
        return None
    start_min = tw.get("start_min", None)
    end_min = tw.get("end_min", None)
    try:
        start_min = float(start_min) if start_min is not None else None
        end_min = float(end_min) if end_min is not None else None
    except Exception:
        return None
    if start_min is None and end_min is None:
        return None
    return {"start_min": start_min, "end_min": end_min}


def load_yolk_tsv(tsv_path: str):
    time = []
    height_um = []
    with open(tsv_path, "r") as f:
        header = f.readline().strip().split("\t")
        # Expect at least: time_min, cytoplasm_height_px, cytoplasm_height_microns, ...
        try:
            idx_time = header.index("time_min")
            idx_h_um = header.index("cytoplasm_height_microns")
        except ValueError:
            raise ValueError(
                f"Unexpected header in {tsv_path}: {header}. "
                "Expected columns 'time_min' and 'cytoplasm_height_microns'."
            )
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            try:
                t = float(parts[idx_time])
                h_um = float(parts[idx_h_um])
            except (ValueError, IndexError):
                continue
            time.append(t)
            height_um.append(h_um)
    return np.array(time, dtype=float), np.array(height_um, dtype=float)


def _parse_csv_float_cell(cell: Optional[str]) -> float:
    if cell is None:
        return float("nan")
    s = str(cell).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_geometry_timeseries_traces(
    sample_folder: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prefer track/geometry_timeseries.csv: returns (time_min, cytoplasm_height_um,
    front_minus_apical_um) or (None, None, None).
    """
    csv_path = os.path.join(sample_folder, "track", "geometry_timeseries.csv")
    if not os.path.exists(csv_path):
        return None, None, None
    time_min: List[float] = []
    cyto: List[float] = []
    front: List[float] = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                time_min.append(_parse_csv_float_cell(row.get("time_min")))
                cyto.append(_parse_csv_float_cell(row.get("cytoplasm_height_um")))
                front.append(_parse_csv_float_cell(row.get("front_minus_apical_um")))
    except OSError:
        return None, None, None
    if not time_min:
        return None, None, None
    return (
        np.asarray(time_min, dtype=float),
        np.asarray(cyto, dtype=float),
        np.asarray(front, dtype=float),
    )


def load_cytoplasm_region_tsv(sample_folder: str):
    """
    Load cytoplasm height over time from track/cytoplasm_region.tsv.

    This file is produced by the pipeline and contains:
      col_idx, ..., cytoplasm_height_microns

    Time is derived from col_idx × config.yaml kymograph.time_interval_sec.
    """
    tsv_path = os.path.join(sample_folder, "track", "cytoplasm_region.tsv")
    if not os.path.exists(tsv_path):
        return None, None

    cfg_path = os.path.join(sample_folder, "config.yaml")
    if not os.path.exists(cfg_path):
        return None, None
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    kymo = cfg.get("kymograph", {}) if isinstance(cfg, dict) else {}
    dt_sec = None
    if isinstance(kymo, dict):
        dt_sec = kymo.get("time_interval_sec", None)
    if dt_sec is None:
        return None, None
    try:
        dt_min = float(dt_sec) / 60.0
    except Exception:
        return None, None

    data = np.loadtxt(tsv_path, delimiter="\t", skiprows=1)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    if data.shape[1] < 7:
        return None, None

    col_idx = data[:, 0].astype(int)
    time_min = col_idx * dt_min
    height_um = data[:, 6].astype(float)
    return time_min, height_um


def load_cellularization_front_spline(sample_folder: str):
    """
    Load cellularization front position over time from
    track/VerticalKymoCelluSelection_spline.tsv and convert to microns.

    Returns (time_min, front_height_um) or (None, None) if unavailable.
    """
    cfg_path = os.path.join(sample_folder, "config.yaml")
    spline_path = os.path.join(sample_folder, "track", "VerticalKymoCelluSelection_spline.tsv")
    if not os.path.exists(cfg_path) or not os.path.exists(spline_path):
        return None, None

    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        manual = cfg.get("manual", {}) if isinstance(cfg, dict) else {}
        px2micron = float(manual.get("px2micron"))
        apical_px = float(cfg["apical_detection"]["apical_height_px"])
    except Exception:
        return None, None

    try:
        data = np.loadtxt(spline_path, delimiter="\t", skiprows=1)
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        time_min = data[:, 0].astype(float)
        front_px = data[:, 1].astype(float)
    except Exception:
        return None, None

    # Express front height relative to apical surface, in microns.
    front_height_um = (front_px - apical_px) * px2micron
    return time_min, front_height_um

def apply_window_xlim(ax, windows, prefer_intersection: bool = True):
    if not windows:
        return
    starts = [w[0] for w in windows if w[0] is not None and np.isfinite(w[0])]
    ends = [w[1] for w in windows if w[1] is not None and np.isfinite(w[1])]
    if not starts and not ends:
        return
    if prefer_intersection:
        # Prefer intersection if it exists; otherwise fallback to union
        lo = max(starts) if starts else None
        hi = min(ends) if ends else None
        if lo is not None and hi is not None and hi > lo:
            ax.set_xlim(lo, hi)
            return
    # Union fallback (or only one-sided bounds)
    if starts:
        lo = min(starts)
    if ends:
        hi = max(ends)
    if lo is not None and hi is not None and hi > lo:
        ax.set_xlim(lo, hi)


def main():
    parser = argparse.ArgumentParser(description="Plot yolk cytoplasm size over time for a species folder.")
    parser.add_argument(
        "-s",
        "--species-folder",
        required=True,
        help="Species folder containing sample subfolders (e.g. data/Hermetia)",
    )
    parser.add_argument(
        "--samples-yaml",
        default=None,
        help=(
            "Path to sample config (single samples.yaml or species files in config/). "
            "Used to infer orientation groups."
        ),
    )
    parser.add_argument(
        "--align-depth-um",
        type=float,
        default=None,
        help="If set (e.g. 20), shift each sample time axis so that the first time cytoplasm_height_microns >= this value occurs at t=0.",
    )
    parser.add_argument(
        "--overlay-front",
        action="store_true",
        help="If set, overlay the cellularization front height (µm) on each orientation panel for troubleshooting.",
    )
    args = parser.parse_args()

    species_folder = os.path.abspath(args.species_folder)
    if not os.path.isdir(species_folder):
        raise ValueError(f"Species folder does not exist: {species_folder}")

    repo_root = str(analysis_paths.REPO_ROOT)
    samples_yaml_path = (
        os.path.abspath(args.samples_yaml)
        if args.samples_yaml is not None
        else default_samples_config_path(repo_root)
    )
    if not os.path.exists(samples_yaml_path):
        raise FileNotFoundError(f"Samples config not found: {samples_yaml_path}")

    samples = discover_sample_folders(species_folder)
    if not samples:
        print(f"No sample subfolders found in: {species_folder}")
        return

    orientation_map = load_orientation_map(samples_yaml_path, species_folder)

    group_order = ["dorsal", "lateral", "ventral", "suction"]
    groups = {g: [] for g in group_order}  # list[(sample_name, sample_folder)]
    for sample_name, sample_folder in samples:
        group = orientation_map.get(sample_folder)
        if group in groups:
            groups[group].append((sample_name, sample_folder))

    if not any(groups.values()):
        print("No samples matched dorsal/lateral/ventral/suction groups (check sample config paths).")
        return

    # Determine which groups actually have samples
    nonempty_groups = [g for g in group_order if groups[g]]
    n_rows = len(nonempty_groups) + 1  # plus one for the mean±SD panel

    # Layout: stacked panels (all square)
    # Use manual spacing (constrained_layout tends to add large gaps with square axes).
    fig = plt.figure(figsize=(10, 3 * n_rows), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=1,
        height_ratios=[1.0] * n_rows,
        hspace=0.20,
    )

    axes_by_orient: dict[str, plt.Axes] = {}
    first_ax = None
    for i, orient in enumerate(nonempty_groups):
        if first_ax is None:
            ax = fig.add_subplot(gs[i, 0])
            first_ax = ax
        else:
            ax = fig.add_subplot(gs[i, 0], sharex=first_ax)
        axes_by_orient[orient] = ax

    # Group (mean±SD) panel at the bottom
    ax_group = fig.add_subplot(gs[-1, 0], sharex=first_ax if first_ax is not None else None)

    for ax in list(axes_by_orient.values()) + [ax_group]:
        ax.set_box_aspect(1)

    # Per-orientation collections for later averaging and x-limit management.
    grouped_time = {g: [] for g in group_order}
    grouped_height = {g: [] for g in group_order}
    grouped_front_time = {g: [] for g in group_order}
    grouped_front_height = {g: [] for g in group_order}
    grouped_windows = {g: [] for g in group_order}  # list[(start,end)]
    grouped_data_ranges = {g: [] for g in group_order}  # list[(min_t,max_t)]

    # Consistent colors by orientation.
    colors = {"dorsal": "tab:blue", "lateral": "tab:orange", "ventral": "tab:green", "suction": "tab:purple"}

    for orient, ax in axes_by_orient.items():
        if not groups[orient]:
            # No samples for this group; hide the entire axis.
            ax.set_visible(False)
            continue
        for sample_name, sample_folder in groups[orient]:
            # Load full cytoplasm trace (before any cropping); prefer geometry_timeseries.csv.
            tg, hg, fg = load_geometry_timeseries_traces(sample_folder)
            if tg is not None and hg is not None and tg.size:
                time_full, height_full = tg, hg
                if fg is not None and np.any(np.isfinite(fg)):
                    front_time_full, front_height_full = tg, fg
                else:
                    front_time_full, front_height_full = load_cellularization_front_spline(
                        sample_folder
                    )
            else:
                time_full, height_full = load_cytoplasm_region_tsv(sample_folder)
                front_time_full, front_height_full = load_cellularization_front_spline(
                    sample_folder
                )
            if time_full is None or height_full is None or time_full.size == 0:
                continue

            # Start with full trace, then apply optional time_window cropping.
            time_min = time_full
            height_um = height_full

            tw = load_time_window_from_config(sample_folder)
            start_min, end_min = None, None
            if tw is not None:
                start_min = tw.get("start_min")
                end_min = tw.get("end_min")
                lo = start_min if start_min is not None else -np.inf
                hi = end_min if end_min is not None else np.inf
                mask_t = (time_min >= lo) & (time_min <= hi)
                if np.any(mask_t):
                    time_min = time_min[mask_t]
                    height_um = height_um[mask_t]

            if args.align_depth_um is not None and front_time_full is not None and front_height_full is not None:
                # Compute alignment time on the cellularization front (full spline),
                # then shift both cytoplasm and front so that this depth occurs at t = 0.
                t_align = _compute_align_time_from_depth(front_time_full, front_height_full, float(args.align_depth_um))
                if t_align is not None:
                    # Shift cytoplasm trace (keep both t<0 and t>0 so the full history is visible).
                    time_min = time_min - t_align

                    # Shift front trace in the same way if available.
                    front_time = front_time_full - t_align
                    front_height = front_height_full
                else:
                    print(
                        f"Warning: {sample_name} front never reaches {args.align_depth_um:g} µm; "
                        "leaving this trace unaligned in absolute-time coordinates."
                    )
                    front_time, front_height = None, None
            else:
                # No alignment (either no align-depth or no usable front); if overlaying,
                # keep the front in absolute time coordinates.
                if front_time_full is not None and front_height_full is not None:
                    front_time, front_height = front_time_full, front_height_full
                else:
                    front_time, front_height = None, None

            grouped_time[orient].append(time_min)
            grouped_height[orient].append(height_um)
            if front_time is not None and front_height is not None:
                grouped_front_time[orient].append(front_time)
                grouped_front_height[orient].append(front_height)

            # Track data range after any alignment/cropping
            if time_min.size:
                grouped_data_ranges[orient].append((float(np.nanmin(time_min)), float(np.nanmax(time_min))))

            if start_min is None and time_min.size:
                start_min = float(np.nanmin(time_min))
            if end_min is None and time_min.size:
                end_min = float(np.nanmax(time_min))
            if start_min is not None or end_min is not None:
                grouped_windows[orient].append((start_min, end_min))

        # For the `suction` orientation panel, show combined cellularization fronts
        # from dorsal/lateral/ventral instead of a suction cytoplasm trace.
        if orient == "suction":
            front_panel_orients = [g for g in ("dorsal", "lateral", "ventral") if g in nonempty_groups]
            for g in front_panel_orients:
                ft_list = grouped_front_time[g]
                fh_list = grouped_front_height[g]
                n_front = len(fh_list)
                if n_front >= 2 and ft_list:
                    min_len_f = min(arr.size for arr in fh_list)
                    if min_len_f > 0:
                        truncated_front = np.stack([arr[:min_len_f] for arr in fh_list], axis=0)
                        time_common_f = ft_list[0][:min_len_f]
                        mean_front = np.nanmean(truncated_front, axis=0)
                        sd_front = np.nanstd(truncated_front, axis=0)
                        c = colors.get(g, "black")
                        ax.plot(
                            time_common_f,
                            mean_front,
                            color=c,
                            linestyle="--",
                            linewidth=2.0,
                            label=f"{g.capitalize()}",
                        )
                        ax.fill_between(
                            time_common_f,
                            mean_front - sd_front,
                            mean_front + sd_front,
                            color=c,
                            alpha=0.10,
                        )
                elif n_front == 1 and ft_list:
                    t_f_single = ft_list[0]
                    h_f_single = fh_list[0]
                    if t_f_single.size and h_f_single.size:
                        c = colors.get(g, "black")
                        ax.plot(
                            t_f_single,
                            h_f_single,
                            color=c,
                            linestyle="--",
                            linewidth=1.8,
                            label=f"{g.capitalize()}",
                        )

            ax.set_title("Cellularization front")
            ax.set_ylabel("Cellularization front (µm)")
        else:
            # If multiple samples in this orientation, show mean±SD curves on this panel.
            t_list = grouped_time[orient]
            h_list = grouped_height[orient]
            n_yolk = len(h_list)
            if n_yolk >= 2 and t_list:
                min_len = min(arr.size for arr in h_list)
                if min_len > 0:
                    truncated_heights = np.stack([arr[:min_len] for arr in h_list], axis=0)
                    time_common = t_list[0][:min_len]
                    mean_um = np.nanmean(truncated_heights, axis=0)
                    sd_um = np.nanstd(truncated_heights, axis=0)
                    c = colors.get(orient, "black")
                    ax.plot(time_common, mean_um, color=c, linewidth=2.2, label=f"Yolk mean (n={n_yolk})")
                    ax.fill_between(time_common, mean_um - sd_um, mean_um + sd_um, color=c, alpha=0.15)
            elif n_yolk == 1:
                # Single sample: show its yolk trace directly so the panel is not empty.
                t_single = t_list[0]
                h_single = h_list[0]
                if t_single.size and h_single.size:
                    c = colors.get(orient, "black")
                    ax.plot(t_single, h_single, color=c, linewidth=2.0, label="Yolk (n=1)")

            # Optional mean±SD for the cellularization front when enough samples
            ft_list = grouped_front_time[orient]
            fh_list = grouped_front_height[orient]
            n_front = len(fh_list)
            if n_front >= 2 and ft_list:
                min_len_f = min(arr.size for arr in fh_list)
                if min_len_f > 0:
                    truncated_front = np.stack([arr[:min_len_f] for arr in fh_list], axis=0)
                    time_common_f = ft_list[0][:min_len_f]
                    mean_front = np.nanmean(truncated_front, axis=0)
                    sd_front = np.nanstd(truncated_front, axis=0)
                    c = colors.get(orient, "black")
                    ax.plot(
                        time_common_f,
                        mean_front,
                        color=c,
                        linestyle="--",
                        linewidth=2.0,
                        label=f"Front mean (n={n_front})",
                    )
                    ax.fill_between(
                        time_common_f,
                        mean_front - sd_front,
                        mean_front + sd_front,
                        color=c,
                        alpha=0.10,
                    )
            elif n_front == 1:
                # Single sample front: show its trace directly when overlay is enabled.
                if ft_list:
                    t_f_single = ft_list[0]
                    h_f_single = fh_list[0]
                    if t_f_single.size and h_f_single.size:
                        c = colors.get(orient, "black")
                        ax.plot(
                            t_f_single,
                            h_f_single,
                            color=c,
                            linestyle="--",
                            linewidth=1.8,
                            label="Front (n=1)",
                        )

            ax.set_title(orient.capitalize())
            ax.set_ylabel("Cytoplasm (µm)")
        # Show apical surface (0 µm) at the top and deeper cytoplasm at the bottom.
        ax.invert_yaxis()
        if ax.lines:
            ax.legend(fontsize=8, loc="best")

        if args.align_depth_um is None:
            if orient == "suction":
                front_panel_orients = [g for g in ("dorsal", "lateral", "ventral") if g in nonempty_groups]
                windows = []
                for g in front_panel_orients:
                    windows.extend(grouped_windows[g])
                apply_window_xlim(ax, windows, prefer_intersection=True)
            else:
                apply_window_xlim(ax, grouped_windows[orient], prefer_intersection=True)
        else:
            # In alignment mode, show full aligned history: min_t .. max_t (can include t<0).
            if orient == "suction":
                front_panel_orients = [g for g in ("dorsal", "lateral", "ventral") if g in nonempty_groups]
                all_ranges = []
                for g in front_panel_orients:
                    for t_arr in grouped_front_time[g]:
                        if t_arr is None or t_arr.size == 0:
                            continue
                        all_ranges.append((float(np.nanmin(t_arr)), float(np.nanmax(t_arr))))
                if all_ranges:
                    mins, maxs = zip(*all_ranges)
                    lo = min(mins)
                    hi = max(maxs)
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        ax.set_xlim(lo, hi)
            elif grouped_data_ranges[orient]:
                mins, maxs = zip(*grouped_data_ranges[orient])
                lo = min(mins)
                hi = max(maxs)
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    ax.set_xlim(lo, hi)

    # Group panel: mean ± SD per orientation cytoplasm border.
    any_data = False
    for orient in nonempty_groups:
        t_list = grouped_time[orient]
        h_list = grouped_height[orient]
        if not t_list or not h_list:
            continue
        min_len = min(arr.size for arr in t_list)
        if min_len <= 0:
            continue
        any_data = True
        truncated_heights = np.stack([h[:min_len] for h in h_list], axis=0)
        time_common = t_list[0][:min_len]
        mean_um = np.nanmean(truncated_heights, axis=0)
        sd_um = np.nanstd(truncated_heights, axis=0)

        c = colors.get(orient, "black")
        # Compact legend in the combined panel: just orientation name.
        ax_group.plot(
            time_common,
            mean_um,
            color=c,
            linewidth=2,
            label=orient.capitalize(),
        )
        ax_group.fill_between(
            time_common,
            mean_um - sd_um,
            mean_um + sd_um,
            color=c,
            alpha=0.2,
        )

    ax_group.set_title(f"{os.path.basename(species_folder)} orientations (mean ± SD)" if any_data else "Orientations (no data)")
    ax_group.set_xlabel("Time (min)")
    ax_group.set_ylabel("Cytoplasm (µm)")
    # Keep the same convention (0 µm at the top).
    ax_group.invert_yaxis()
    if ax_group.lines:
        ax_group.legend(fontsize=8, loc="best")

    if args.align_depth_um is None:
        all_windows = []
        for g in nonempty_groups:
            all_windows.extend(grouped_windows[g])
        apply_window_xlim(ax_group, all_windows, prefer_intersection=True)
    else:
        # In alignment mode, show full aligned history across all groups.
        all_ranges = []
        for g in nonempty_groups:
            all_ranges.extend(grouped_data_ranges[g])
        if all_ranges:
            mins, maxs = zip(*all_ranges)
            lo = min(mins)
            hi = max(maxs)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ax_group.set_xlim(lo, hi)

    out_dir = os.path.join(species_folder, "results")
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{os.path.basename(species_folder)}_yolk_cytoplasm_all_samples"
    out_png = os.path.join(out_dir, f"{stem}.png")
    out_pdf = os.path.join(out_dir, f"{stem}.pdf")
    fig.subplots_adjust(top=0.97, bottom=0.06, left=0.10, right=0.98, hspace=0.22)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)

    print(f"Saved species yolk cytoplasm plot to: {out_png}")
    print(f"Saved species yolk cytoplasm plot to: {out_pdf}")


if __name__ == "__main__":
    main()


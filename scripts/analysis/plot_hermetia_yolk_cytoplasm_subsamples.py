#!/usr/bin/env python3
"""
Single-figure comparison of selected Hermetia subsamples.

Layout (2 columns):
  - One row per sample: left = cytoplasm thickness + cellularization front (µm vs time);
    right = the same cropped delta kymograph as Kymograph_delta_marked.pdf (``generate_outputs``
    ``_prepare_kymo_delta_data`` on ``results/Kymograph_delta.tif`` + milestones).
  - Final row: left = all cytoplasm traces overlaid; right = all membrane/front traces overlaid.

Each subplot uses a square axes box (set_box_aspect(1)). Figure size is scaled from the
standalone PDF style (8 in per panel) by 1/5, then ×1.5 for readability.
Kymograph panels use ``scripts/generate_outputs.py`` (``_prepare_kymo_delta_data``) so the crop
matches ``Kymograph_delta_marked.pdf`` exactly; time is shifted by the milestone ``time_min``
at the crop start for alignment with traces.

Traces (cytoplasm thickness and membrane front depth) are read from ``track/geometry_timeseries.csv``
(produced by ``scripts/export_geometry_timeseries.py`` or the Snakemake ``export_geometry_timeseries`` rule).
Run that export (or the full pipeline through ``generate_outputs``) before plotting.

Subplot titles use the sample **keys** from the sample config
(e.g. ``hermetia8-suction``), resolved by matching each sample folder to the ``path:`` entry.

Example:

    python scripts/analysis/plot_hermetia_yolk_cytoplasm_subsamples.py \\
        --species-folder data/Hermetia/wt \\
        --sample-list 2,6 \\
        --out-stem Hermetia_subsamples_2_6
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
import analysis_paths  # noqa: F401 — adds scripts/ to sys.path
from samples_loader import load_samples_config, default_samples_config_path

# Same kymograph crop / extent as Kymograph_delta_marked.pdf (see scripts/generate_outputs.py)
_GEN_PATH = os.path.join(str(analysis_paths.SCRIPTS_DIR), "generate_outputs.py")
_GEN_SPEC = importlib.util.spec_from_file_location("generate_outputs", _GEN_PATH)
if _GEN_SPEC is None or _GEN_SPEC.loader is None:
    raise RuntimeError("Could not load generate_outputs.py")
gen_out = importlib.util.module_from_spec(_GEN_SPEC)
_GEN_SPEC.loader.exec_module(gen_out)


def _parse_sample_list(sample_list: str) -> List[str]:
    parts = [p.strip() for p in sample_list.split(",") if p.strip()]
    if not parts:
        raise ValueError("--sample-list is empty")
    return parts


def _natural_sample_folder(species_folder: str, sample_id: str) -> str:
    cand = os.path.join(species_folder, str(sample_id))
    if os.path.isdir(cand):
        return cand
    cand2 = os.path.join(species_folder, str(sample_id).lstrip("./"))
    if os.path.isdir(cand2):
        return cand2
    raise FileNotFoundError(
        f"Could not resolve sample folder for id '{sample_id}' under {species_folder}"
    )


def _repo_root() -> str:
    """Project root (parent of ``scripts/``)."""
    return str(analysis_paths.REPO_ROOT)


def load_path_to_sample_title(samples_yaml_path: str, species_folder_abs: str) -> Dict[str, str]:
    """
    Map absolute sample directory path -> sample key from sample config
    (e.g. ``data/Hermetia/8`` -> ``hermetia8-suction``).

    Only entries whose ``path`` lies under ``species_folder_abs`` are included.
    """
    out: Dict[str, str] = {}
    if not os.path.isfile(samples_yaml_path):
        return out
    try:
        cfg = load_samples_config(samples_yaml_path)
    except Exception:
        return out
    samples = cfg.get("samples", {})
    if not isinstance(samples, dict):
        return out
    repo_root = _repo_root()
    species_folder_abs = os.path.normpath(os.path.abspath(species_folder_abs))
    for title_key, entry in samples.items():
        if not isinstance(entry, dict):
            continue
        path_rel = entry.get("path")
        if not path_rel or not isinstance(path_rel, str):
            continue
        abs_p = os.path.normpath(os.path.join(repo_root, path_rel))
        if abs_p == species_folder_abs or abs_p.startswith(species_folder_abs + os.sep):
            out[abs_p] = str(title_key)
    return out


def load_time_window_from_config(sample_folder: str) -> Optional[dict]:
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


def _parse_csv_float(cell: Optional[str]) -> float:
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
    Load ``track/geometry_timeseries.csv`` (sole source for traces in this script).

    Returns:
        time_min — same time grid for cytoplasm and membrane
        cytoplasm_height_um — thickness (µm)
        front_minus_apical_um — membrane depth relative to per-column apical border (µm)

    Returns (None, None, None) if the file is missing or unreadable.
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
                time_min.append(_parse_csv_float(row.get("time_min")))
                cyto.append(_parse_csv_float(row.get("cytoplasm_height_um")))
                front.append(_parse_csv_float(row.get("front_minus_apical_um")))
    except OSError:
        return None, None, None
    if not time_min:
        return None, None, None
    return (
        np.asarray(time_min, dtype=float),
        np.asarray(cyto, dtype=float),
        np.asarray(front, dtype=float),
    )


def apply_time_window_mask(
    time_min: np.ndarray,
    y: np.ndarray,
    tw: Optional[dict],
) -> Tuple[np.ndarray, np.ndarray]:
    if tw is None or time_min.size == 0:
        return time_min, y
    lo = tw.get("start_min")
    hi = tw.get("end_min")
    lo = -np.inf if lo is None else float(lo)
    hi = np.inf if hi is None else float(hi)
    mask = (time_min >= lo) & (time_min <= hi)
    if not np.any(mask):
        return time_min, y
    return time_min[mask], y[mask]


def load_config_dict(sample_folder: str) -> Optional[dict]:
    cfg_path = os.path.join(sample_folder, "config.yaml")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def prepare_kymograph_from_pipeline(sample_folder: str) -> Optional[Dict[str, Any]]:
    """
    Use the same ``_prepare_kymo_delta_data`` as Kymograph_delta_marked.pdf (generate_outputs).

    Requires ``results/Kymograph_delta.tif`` and a valid ``config.yaml`` (as for the pipeline).
    """
    try:
        cfg = gen_out.load_config(sample_folder)
    except Exception:
        return None
    try:
        time_min, front_px = gen_out.load_spline(sample_folder)
    except Exception:
        return None
    milestones = gen_out.compute_milestones(
        time_min,
        front_px,
        cfg["apical_height_px"],
        cfg["final_height_px"],
        cfg["movie_time_interval_sec"],
    )
    d = gen_out._prepare_kymo_delta_data(sample_folder, cfg, milestones)
    if d is None:
        return None
    start_pct = int(cfg.get("kymo_marked_start_pct", 70))
    start_pct_snapped = max((p for p in milestones if p <= start_pct), default=0)
    # Align kymograph time axis with spline traces: use milestone time at crop start (not frame_idx * dt).
    t0_crop = float(milestones[start_pct_snapped]["time_min"])
    return {
        "d": d,
        "milestones": milestones,
        "t0_crop": t0_crop,
        "start_pct_snapped": start_pct_snapped,
    }


def plot_kymograph_delta_marked_on_ax(ax: plt.Axes, k: dict) -> None:
    """
    Same ``imshow`` geometry as ``generate_outputs._draw_kymo_delta_on_ax``, with y shifted to
    absolute time (minutes) using the milestone time at the crop start.
    """
    d = k["d"]
    t0 = float(k["t0_crop"])
    img = d["kymo_crop"]
    width_um = float(d["width_um"])
    duration_min = float(d["duration_min"])
    # Relative: extent=[0, width_um, duration_min, 0]  ->  absolute: bottom=t0+duration, top=t0
    ax.imshow(
        img,
        aspect="auto",
        cmap="gray",
        origin="upper",
        extent=[0, width_um, t0 + duration_min, t0],
    )
    ax.set_xlim(0, width_um)
    ax.set_ylim(t0 + duration_min, t0)
    ax.set_xlabel("Width (µm)", fontsize=9)
    ax.set_ylabel("Time (min)", fontsize=9)


def sample_colors(n: int) -> List:
    try:
        cmap = matplotlib.colormaps["tab10"]
    except AttributeError:
        cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cytoplasm + membrane traces and kymographs for selected Hermetia subsamples."
    )
    parser.add_argument(
        "-s",
        "--species-folder",
        required=True,
        help="Species folder containing numeric sample subfolders (e.g. data/Hermetia)",
    )
    parser.add_argument(
        "--sample-list",
        required=True,
        help="Comma-separated sample folder names (e.g. 2,6)",
    )
    parser.add_argument(
        "--out-stem",
        default="Hermetia_yolk_cytoplasm_subsamples",
        help="Output filename stem (PNG/PDF) under <species-folder>/results/",
    )
    parser.add_argument(
        "--samples-yaml",
        default=None,
        help=(
            "Path to sample config (single samples.yaml or species files in config/). "
            "Maps folder -> title from sample keys."
        ),
    )
    args = parser.parse_args()

    species_folder = os.path.abspath(args.species_folder)
    if not os.path.isdir(species_folder):
        raise SystemExit(f"Species folder does not exist: {species_folder}")

    samples_yaml = (
        os.path.abspath(args.samples_yaml)
        if args.samples_yaml is not None
        else default_samples_config_path(_repo_root())
    )
    path_to_title = load_path_to_sample_title(samples_yaml, species_folder)

    sample_ids = _parse_sample_list(args.sample_list)
    n = len(sample_ids)
    colors = sample_colors(n)

    # Resolve folders and load data
    rows: List[dict] = []
    for i, sid in enumerate(sample_ids):
        try:
            folder = _natural_sample_folder(species_folder, sid)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            rows.append(
                {
                    "id": sid,
                    "title": f"Sample {sid}",
                    "folder": None,
                    "cfg": None,
                    "time_c": None,
                    "cyto": None,
                    "time_f": None,
                    "front": None,
                    "kymo_data": None,
                    "color": colors[i],
                }
            )
            continue

        cfg = load_config_dict(folder)
        tw = load_time_window_from_config(folder)

        t_g, h_c_g, h_f_g = load_geometry_timeseries_traces(folder)
        if t_g is not None and h_c_g is not None:
            t_c, h_c = t_g, h_c_g
            t_f, h_f = t_g, h_f_g
        else:
            t_c, h_c = None, None
            t_f, h_f = None, None

        if t_c is not None and h_c is not None:
            t_c, h_c = apply_time_window_mask(t_c, h_c, tw)
        if t_f is not None and h_f is not None:
            t_f, h_f = apply_time_window_mask(t_f, h_f, tw)

        kymo_data = prepare_kymograph_from_pipeline(folder)

        if t_c is None or h_c is None:
            print(
                f"Warning: sample {sid}: missing or invalid track/geometry_timeseries.csv "
                "(run scripts/export_geometry_timeseries.py or the pipeline export step)"
            )
        elif h_f is not None and not np.any(np.isfinite(h_f)):
            print(
                f"Warning: sample {sid}: no finite front_minus_apical_um in geometry_timeseries.csv "
                "(spline may not overlap this time range)"
            )
        if kymo_data is None:
            print(
                f"Warning: sample {sid}: could not load Kymograph_delta kymograph "
                "(need results/Kymograph_delta.tif + spline; run generate_outputs for this sample)"
            )

        folder_abs = os.path.abspath(folder)
        sample_title = path_to_title.get(folder_abs, f"Sample {sid}")

        rows.append(
            {
                "id": sid,
                "title": sample_title,
                "folder": folder,
                "cfg": cfg,
                "time_c": t_c,
                "cyto": h_c,
                "time_f": t_f,
                "front": h_f,
                "kymo_data": kymo_data,
                "color": colors[i],
            }
        )

    # Same square layout as Kymograph_delta_marked.pdf (8×8 in per panel), scaled down 5× then ×1.5.
    panel_in = (8.0 / 5.0) * 1.5
    n_rows = n + 1
    fig = plt.figure(figsize=(2 * panel_in, n_rows * panel_in), constrained_layout=False)
    gs = fig.add_gridspec(
        n_rows,
        2,
        height_ratios=[1.0] * n + [1.0],
        hspace=0.22,
        wspace=0.18,
    )

    trace_axes: List[plt.Axes] = []
    kymo_axes: List[plt.Axes] = []

    for i, row in enumerate(rows):
        ax_t = fig.add_subplot(gs[i, 0])
        ax_k = fig.add_subplot(gs[i, 1])
        trace_axes.append(ax_t)
        kymo_axes.append(ax_k)

        sid = row["id"]
        ax_t.set_title(row.get("title", f"Sample {sid}"), fontsize=11, fontweight="bold")

        if row["time_c"] is not None and row["cyto"] is not None and row["time_c"].size:
            ax_t.plot(
                row["time_c"],
                row["cyto"],
                color=row["color"],
                linewidth=1.8,
            )
        if row["time_f"] is not None and row["front"] is not None and row["time_f"].size:
            ax_t.plot(
                row["time_f"],
                row["front"],
                color=row["color"],
                linewidth=1.5,
                linestyle="--",
            )

        ax_t.set_ylabel("Depth (µm)", fontsize=10)
        ax_t.invert_yaxis()
        ax_t.grid(True, alpha=0.25)
        ax_t.tick_params(labelsize=8)

        if row.get("kymo_data"):
            try:
                plot_kymograph_delta_marked_on_ax(ax_k, row["kymo_data"])
            except Exception as e:
                print(f"Warning: sample {sid}: could not plot kymograph ({e})")
                ax_k.text(0.5, 0.5, "Kymograph error", ha="center", va="center", transform=ax_k.transAxes)
                ax_k.set_xticks([])
                ax_k.set_yticks([])
        else:
            ax_k.text(
                0.5,
                0.5,
                "No Kymograph_delta_marked",
                ha="center",
                va="center",
                transform=ax_k.transAxes,
                fontsize=10,
            )
            ax_k.set_xticks([])
            ax_k.set_yticks([])

        ax_k.tick_params(labelsize=8)

        # Square axes (same as standalone Kymograph_delta_marked.pdf panels).
        ax_t.set_box_aspect(1)
        ax_k.set_box_aspect(1)

        # Delta kymograph y-axis is absolute time (min); match trace x-limits (clip to crop).
        if ax_t.lines and row.get("kymo_data"):
            kd = row["kymo_data"]
            t0 = float(kd["t0_crop"])
            d = kd["d"]
            t_end = t0 + float(d["duration_min"])
            x0, x1 = ax_t.get_xlim()
            lo = min(x0, x1)
            hi = max(x0, x1)
            v0 = max(lo, t0)
            v1 = min(hi, t_end)
            if v1 > v0:
                ax_k.set_ylim(v1, v0)

    # Share x on trace column
    for j in range(1, len(trace_axes)):
        trace_axes[j].sharex(trace_axes[0])

    # Summary row (share time axis with per-sample traces)
    ax_cyto_all = fig.add_subplot(gs[n, 0], sharex=trace_axes[0])
    ax_front_all = fig.add_subplot(gs[n, 1], sharex=trace_axes[0])

    for row in rows:
        if row["time_c"] is None or row["cyto"] is None or not row["time_c"].size:
            continue
        ax_cyto_all.plot(
            row["time_c"],
            row["cyto"],
            color=row["color"],
            linewidth=1.8,
        )
    ax_cyto_all.set_title("Cytoplasm (all samples)", fontsize=11)
    ax_cyto_all.set_ylabel("Depth (µm)", fontsize=10)
    ax_cyto_all.set_xlabel("Time (min)", fontsize=10)
    ax_cyto_all.invert_yaxis()
    ax_cyto_all.grid(True, alpha=0.25)
    ax_cyto_all.tick_params(labelsize=8)

    for row in rows:
        if row["time_f"] is None or row["front"] is None or not row["time_f"].size:
            continue
        ax_front_all.plot(
            row["time_f"],
            row["front"],
            color=row["color"],
            linewidth=1.8,
            linestyle="--",
        )
    ax_front_all.set_title("Membrane front (all samples)", fontsize=11)
    ax_front_all.set_ylabel("Depth (µm)", fontsize=10)
    ax_front_all.set_xlabel("Time (min)", fontsize=10)
    ax_front_all.invert_yaxis()
    ax_front_all.grid(True, alpha=0.25)
    ax_front_all.tick_params(labelsize=8)

    ax_cyto_all.set_box_aspect(1)
    ax_front_all.set_box_aspect(1)

    # Match x-limits on summary to data
    for ax in (ax_cyto_all, ax_front_all):
        if ax.lines:
            ax.relim()
            ax.autoscale_view()

    fig.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.05)

    out_dir = os.path.join(species_folder, "results")
    os.makedirs(out_dir, exist_ok=True)
    stem = args.out_stem
    out_png = os.path.join(out_dir, f"{stem}.png")
    out_pdf = os.path.join(out_dir, f"{stem}.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()

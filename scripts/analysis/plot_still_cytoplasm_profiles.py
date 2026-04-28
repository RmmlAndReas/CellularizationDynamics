#!/usr/bin/env python3
"""
Plot still cytoplasm profiles: single-file line plot or group mean ± SD.
Group/compare averages support per-cohort optional peak alignment (x − x_peak).

CSV format: from draw_still_cytoplasm_boundaries.py (header rows start with #).

Usage — per file:
    python scripts/analysis/plot_still_cytoplasm_profiles.py per-file \\
        --csv track/cytoplasm_profile.csv --out-png out.png --out-pdf out.pdf \\
        --title "Sample" [--input-image source.tif]

Usage — group average:
    python scripts/analysis/plot_still_cytoplasm_profiles.py group \\
        --csvs a.csv b.csv --out-png mean.png --out-pdf mean.pdf --label condition \\
        [--hline Y,COLOR ...] [--condition-hline-center]

For condition cohorts, optional profile hlines: the orange (darkorange) line is placed at the
cytoplasm height at x − x_peak = 0 (aligned peak) on each trace or on the cohort mean curve.

Usage — condition vs control on one axes:
    python scripts/analysis/plot_still_cytoplasm_profiles.py group-compare \\
        --condition-csvs a.csv --control-csvs b.csv \\
        --out-png both.png --out-pdf both.pdf
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


def _image_hw(path: str) -> Tuple[int, int]:
    """Return (height, width) in pixels for a TIFF or compatible image."""
    img = tifffile.imread(path)
    if img.ndim == 3:
        if img.shape[0] <= 8 and img.shape[0] != img.shape[-1]:
            img = np.squeeze(img[0])
        elif img.shape[-1] <= 4:
            img = np.squeeze(img[..., 0])
        else:
            img = np.squeeze(img[0])
    if img.ndim != 2:
        img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image after squeeze, got shape {getattr(img, 'shape', None)}")
    h, w = img.shape
    return int(h), int(w)


def _read_profile_csv(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, float, Optional[int], Optional[int]]:
    """Returns (x_um, height_um, px2micron, image_width_px, image_height_px).

    image_* may be None for older CSVs. Keeps NaN heights for smoothing.
    """
    px2 = float("nan")
    img_w: Optional[int] = None
    img_h: Optional[int] = None
    x_list: List[float] = []
    h_list: List[float] = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                if len(row) >= 2 and "px2micron" in row[0]:
                    try:
                        px2 = float(row[1].strip())
                    except ValueError:
                        pass
                if len(row) >= 2 and "image_width_px" in row[0]:
                    try:
                        img_w = int(float(row[1].strip()))
                    except ValueError:
                        pass
                if len(row) >= 2 and "image_height_px" in row[0]:
                    try:
                        img_h = int(float(row[1].strip()))
                    except ValueError:
                        pass
                continue
            if row[0].strip() == "x_px":
                continue
            try:
                x_um = float(row[1])
            except (ValueError, IndexError):
                continue
            if not np.isfinite(x_um):
                continue
            try:
                height_um = float(row[5]) if len(row) > 5 else float(row[4])
            except (ValueError, IndexError):
                height_um = float("nan")
            if not np.isfinite(height_um):
                height_um = float("nan")
            x_list.append(x_um)
            h_list.append(height_um)

    x = np.asarray(x_list, dtype=float)
    h = np.asarray(h_list, dtype=float)
    order = np.argsort(x)
    return x[order], h[order], px2, img_w, img_h


def _normalize_smooth_window(window_px: int) -> int:
    w = int(window_px)
    if w < 1:
        return 1
    if w % 2 == 0:
        w += 1
    return w


def sliding_window_nanmean(y: np.ndarray, window_px: int) -> np.ndarray:
    """
    Centered moving average along the 1D array (one sample per column).
    window_px is in column indices; values < 2 disable smoothing.
    NaNs are ignored inside each window; all-NaN windows stay NaN.
    """
    w = _normalize_smooth_window(window_px)
    if w <= 1:
        return y.astype(float, copy=True)
    half = w // 2
    n = len(y)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = y[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size > 0:
            out[i] = float(np.mean(seg))
    return out


def _interp_on_grid(grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.full(grid.shape, np.nan, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2:
        return out
    xv = x[valid]
    yv = y[valid]
    order = np.argsort(xv)
    xv = xv[order]
    yv = yv[order]
    in_bounds = (grid >= xv[0]) & (grid <= xv[-1])
    if np.any(in_bounds):
        out[in_bounds] = np.interp(grid[in_bounds], xv, yv)
    return out


def _mean_std(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mat: n_profiles x n_grid → column mean, population SD, count per column."""
    n = np.sum(np.isfinite(mat), axis=0)
    mean = np.full(mat.shape[1], np.nan, dtype=float)
    std = np.full(mat.shape[1], np.nan, dtype=float)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        ok = np.isfinite(col)
        if not np.any(ok):
            continue
        vals = col[ok]
        mean[j] = float(np.mean(vals))
        std[j] = float(np.std(vals, ddof=0))
    return mean, std, n


def _x_at_max_height(x_um: np.ndarray, h_um: np.ndarray) -> float:
    """x (µm) where cytoplasm height is maximal (finite samples only)."""
    valid = np.isfinite(x_um) & np.isfinite(h_um)
    if not np.any(valid):
        return float("nan")
    xv = x_um[valid]
    hv = h_um[valid]
    i = int(np.argmax(hv))
    return float(xv[i])


def _peak_align_trace(
    x_um: np.ndarray, h_um: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Shift x so that the height maximum lies at 0 (relative position, µm)."""
    xp = _x_at_max_height(x_um, h_um)
    if not np.isfinite(xp):
        return x_um.astype(float), h_um.astype(float)
    return x_um.astype(float) - xp, h_um.astype(float)


def _peak_align_traces(
    traces: List[Tuple[np.ndarray, np.ndarray]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    return [_peak_align_trace(x, h) for x, h in traces]


def _finite_x_bounds(x: np.ndarray, h: np.ndarray) -> Tuple[float, float]:
    """Min/max x where both x and h are finite."""
    ok = np.isfinite(x) & np.isfinite(h)
    if not np.any(ok):
        return float("nan"), float("nan")
    xv = x[ok]
    return float(np.min(xv)), float(np.max(xv))


def _overlap_x_range(traces: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
    """
    Intersection of per-trace x-intervals with finite height (after alignment).
    Average/SD then include every trace at every grid point (tips with partial n are dropped).
    """
    bounds: List[Tuple[float, float]] = []
    for x, h in traces:
        lo, hi = _finite_x_bounds(x, h)
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            bounds.append((lo, hi))
    if not bounds:
        return float("nan"), float("nan")
    x_lo = max(b[0] for b in bounds)
    x_hi = min(b[1] for b in bounds)
    if x_lo >= x_hi:
        return float("nan"), float("nan")
    return x_lo, x_hi


def _grid_linspace(xmin: float, xmax: float) -> np.ndarray:
    return np.linspace(xmin, xmax, max(64, int((xmax - xmin) / 0.05) + 1))


def _x_at_max_y(grid: np.ndarray, y: np.ndarray) -> float:
    """x where y is maximal (finite samples only)."""
    ok = np.isfinite(grid) & np.isfinite(y)
    if not np.any(ok):
        return float("nan")
    g = grid[ok]
    v = y[ok]
    i = int(np.argmax(v))
    return float(g[i])


def _parse_hline_spec(spec: str) -> Tuple[float, str]:
    """Parse 'Y,color' for --hline (color may contain commas if quoted in shell)."""
    s = spec.strip()
    if "," not in s:
        raise SystemExit(f"--hline expects Y,color (e.g. 24,lightblue); got {spec!r}")
    y_str, color = s.split(",", 1)
    return float(y_str.strip()), color.strip()


def _is_orange_profile_hline(color: str) -> bool:
    """True for darkorange / orange reference lines from profile_hlines."""
    c = color.lower().replace(" ", "")
    return "orange" in c


def _y_at_peak_zero_peak_aligned(
    x_um: np.ndarray, h_um: np.ndarray
) -> Optional[float]:
    """Peak-aligned trace; cytoplasm height at x − x_peak = 0."""
    xr, hr = _peak_align_trace(x_um, h_um)
    ok = np.isfinite(xr) & np.isfinite(hr)
    if not np.any(ok):
        return None
    xv = xr[ok]
    hv = hr[ok]
    order = np.argsort(xv)
    xv, hv = xv[order], hv[order]
    return float(np.interp(0.0, xv, hv))


def _y_mean_at_x_zero(grid: np.ndarray, mean: np.ndarray) -> Optional[float]:
    """Cohort mean on peak-aligned overlap grid; height at x − x_peak = 0."""
    ok = np.isfinite(grid) & np.isfinite(mean)
    if not np.any(ok):
        return None
    g = grid[ok]
    m = mean[ok]
    order = np.argsort(g)
    g, m = g[order], m[order]
    return float(np.interp(0.0, g, m))


def _resolve_orange_hline_y(
    hlines: List[Tuple[float, str]], y_center: Optional[float]
) -> List[Tuple[float, str]]:
    """Replace orange profile hline y with y_center when finite; else keep config y."""
    if y_center is None or not np.isfinite(y_center):
        return list(hlines)
    out: List[Tuple[float, str]] = []
    for y, c in hlines:
        if _is_orange_profile_hline(c):
            out.append((y_center, c))
        else:
            out.append((y, c))
    return out


def plot_per_file(
    csv_path: str,
    out_png: str,
    out_pdf: str,
    title: Optional[str],
    smooth_window_px: int = 1,
    input_image: Optional[str] = None,
    hlines: Optional[List[Tuple[float, str]]] = None,
    group: Optional[str] = None,
) -> None:
    x_um, height_um, _, img_w, img_h = _read_profile_csv(csv_path)
    h_plot = sliding_window_nanmean(height_um, smooth_window_px)
    w_px, h_px = img_w, img_h
    if (w_px is None or h_px is None) and input_image and os.path.isfile(input_image):
        h_px, w_px = _image_hw(input_image)
    # Match export_still_cytoplasm_overlay.py base width (10 in) and image aspect w:h.
    base_w = 10.0
    if w_px is not None and h_px is not None and h_px > 0 and w_px > 0:
        fig, ax = plt.subplots(figsize=(base_w, base_w * float(h_px) / float(w_px)))
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
    hlines_draw = hlines
    if hlines and group == "condition":
        yc = _y_at_peak_zero_peak_aligned(x_um, h_plot)
        hlines_draw = _resolve_orange_hline_y(hlines, yc)
    if hlines_draw:
        for y, c in hlines_draw:
            ax.axhline(
                y,
                color=c,
                linewidth=1.4,
                linestyle="-",
                zorder=1,
                alpha=0.95,
            )
    ax.plot(x_um, h_plot, color="C0", linewidth=1.5, zorder=2)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("Cytoplasm height (µm)")
    ax.set_title(title or os.path.basename(csv_path))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved {out_png} and {out_pdf}")


def plot_group(
    csv_paths: List[str],
    out_png: str,
    out_pdf: str,
    label: str,
    smooth_windows: Optional[List[int]] = None,
    peak_align: bool = True,
    hlines: Optional[List[Tuple[float, str]]] = None,
    condition_hline_center: bool = False,
) -> None:
    if smooth_windows is None or len(smooth_windows) == 0:
        smooth_windows = [1] * len(csv_paths)
    if len(smooth_windows) != len(csv_paths):
        raise SystemExit(
            f"--smooth-windows count ({len(smooth_windows)}) must match --csvs ({len(csv_paths)})."
        )

    traces = _load_group_traces(csv_paths, smooth_windows)

    if not traces:
        raise SystemExit("No valid CSV traces to plot.")

    if peak_align:
        traces = _peak_align_traces(traces)
    x_lo, x_hi = _overlap_x_range(traces)
    if np.isfinite(x_lo) and np.isfinite(x_hi) and x_lo < x_hi:
        xmin, xmax = x_lo, x_hi
    else:
        print(
            "Warning: no common x overlap with finite height; using union of x ranges.",
        )
        xmin = min(float(np.nanmin(t[0])) for t in traces)
        xmax = max(float(np.nanmax(t[0])) for t in traces)
    grid = _grid_linspace(xmin, xmax)

    mat = np.vstack([_interp_on_grid(grid, t[0], t[1]) for t in traces])
    mean, std, n = _mean_std(mat)

    fig, ax = plt.subplots(figsize=(8, 5))
    if hlines:
        yc: Optional[float] = None
        if condition_hline_center and peak_align:
            yc = _y_mean_at_x_zero(grid, mean)
        h_draw = _resolve_orange_hline_y(hlines, yc) if yc is not None else hlines
        for y, c in h_draw:
            ax.axhline(
                y,
                color=c,
                linewidth=1.4,
                linestyle="-",
                zorder=1,
                alpha=0.95,
            )
    ok = np.isfinite(mean) & np.isfinite(std) & (n > 0)
    if np.any(ok):
        ax.fill_between(
            grid[ok],
            (mean - std)[ok],
            (mean + std)[ok],
            alpha=0.25,
            linewidth=0,
            label="± SD",
        )
        ax.plot(grid[ok], mean[ok], linewidth=2.0, label=f"Mean (n={len(traces)})")
    ax.set_xlabel("x − x_peak (µm)" if peak_align else "x (µm)")
    ax.set_ylabel("Cytoplasm height (µm)")
    ax.set_title(label)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved {out_png} and {out_pdf}")


def _load_group_traces(
    csv_paths: List[str],
    smooth_windows: List[int],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    traces: List[Tuple[np.ndarray, np.ndarray]] = []
    for p, sw in zip(csv_paths, smooth_windows):
        if not os.path.isfile(p):
            print(f"Warning: skip missing {p}")
            continue
        x_um, h_um, _, _, _ = _read_profile_csv(p)
        if x_um.size == 0:
            print(f"Warning: no data in {p}")
            continue
        h_sm = sliding_window_nanmean(h_um, sw)
        traces.append((x_um, h_sm))
    return traces


def plot_two_groups(
    csv_paths_condition: List[str],
    smooth_windows_condition: List[int],
    label_condition: str,
    csv_paths_control: List[str],
    smooth_windows_control: List[int],
    label_control: str,
    out_png: str,
    out_pdf: str,
    title: Optional[str] = None,
    hlines: Optional[List[Tuple[float, str]]] = None,
    condition_hline_center: bool = False,
    peak_align_condition: bool = True,
    peak_align_control: bool = False,
    center_control_midpoint: bool = True,
) -> None:
    """Mean ± SD on one axes.

    Each cohort can optionally use per-embryo peak alignment (x − x_peak). Optionally, for
    overlay the control curve can be shifted so the midpoint of its finite x-span is at 0.
    """
    tc = _load_group_traces(csv_paths_condition, smooth_windows_condition)
    tx = _load_group_traces(csv_paths_control, smooth_windows_control)
    if not tc:
        raise SystemExit("No valid condition CSV traces to plot.")
    if not tx:
        raise SystemExit("No valid control CSV traces to plot.")

    if peak_align_condition:
        tc = _peak_align_traces(tc)
    if peak_align_control:
        tx = _peak_align_traces(tx)

    lo_c, hi_c = _overlap_x_range(tc)
    lo_x, hi_x = _overlap_x_range(tx)
    c_ok = np.isfinite(lo_c) and np.isfinite(hi_c) and lo_c < hi_c
    x_ok = np.isfinite(lo_x) and np.isfinite(hi_x) and lo_x < hi_x

    if not c_ok:
        print(
            "Warning: no common x overlap for condition cohort; using union of condition x.",
        )
        grid_c = _grid_linspace(
            min(float(np.nanmin(t[0])) for t in tc),
            max(float(np.nanmax(t[0])) for t in tc),
        )
    else:
        grid_c = _grid_linspace(lo_c, hi_c)

    if not x_ok:
        print(
            "Warning: no common x overlap for control cohort; using union of control x.",
        )
        grid_x = _grid_linspace(
            min(float(np.nanmin(t[0])) for t in tx),
            max(float(np.nanmax(t[0])) for t in tx),
        )
    else:
        grid_x = _grid_linspace(lo_x, hi_x)

    mat_c = np.vstack([_interp_on_grid(grid_c, t[0], t[1]) for t in tc])
    mat_x = np.vstack([_interp_on_grid(grid_x, t[0], t[1]) for t in tx])
    mean_c, std_c, _ = _mean_std(mat_c)
    mean_x, std_x, _ = _mean_std(mat_x)

    if center_control_midpoint:
        ok_x_tmp = np.isfinite(grid_x) & np.isfinite(mean_x)
        if np.any(ok_x_tmp):
            x_mid_ctrl = 0.5 * (
                float(np.min(grid_x[ok_x_tmp])) + float(np.max(grid_x[ok_x_tmp]))
            )
            grid_ctrl = grid_x - x_mid_ctrl
        else:
            grid_ctrl = grid_x
    else:
        grid_ctrl = grid_x

    fig, ax = plt.subplots(figsize=(8, 2.5))
    color_c, color_x = "C0", "C1"

    if hlines:
        yc: Optional[float] = None
        if condition_hline_center:
            yc = _y_mean_at_x_zero(grid_c, mean_c)
        h_draw = _resolve_orange_hline_y(hlines, yc) if yc is not None else hlines
        for y, c in h_draw:
            ax.axhline(
                y,
                color=c,
                linewidth=1.4,
                linestyle="-",
                zorder=1,
                alpha=0.95,
            )

    ok_c = np.isfinite(mean_c) & np.isfinite(std_c)
    ok_x = np.isfinite(mean_x) & np.isfinite(std_x)

    if np.any(ok_c):
        ax.fill_between(
            grid_c[ok_c],
            (mean_c - std_c)[ok_c],
            (mean_c + std_c)[ok_c],
            alpha=0.22,
            linewidth=0,
            color=color_c,
        )
        ax.plot(
            grid_c[ok_c],
            mean_c[ok_c],
            linewidth=2.0,
            color=color_c,
            label=f"{label_condition} (n={len(tc)}, mean ± SD)",
        )
    if np.any(ok_x):
        ax.fill_between(
            grid_ctrl[ok_x],
            (mean_x - std_x)[ok_x],
            (mean_x + std_x)[ok_x],
            alpha=0.22,
            linewidth=0,
            color=color_x,
        )
        ax.plot(
            grid_ctrl[ok_x],
            mean_x[ok_x],
            linewidth=2.0,
            color=color_x,
            label=f"{label_control} (n={len(tx)}, mean ± SD)",
        )

    # Use symmetric x-limits only when at least one cohort is transformed around 0.
    # For fully absolute-x mode (no peak align, no midpoint-centering), keep natural bounds.
    xs: List[float] = []
    if np.any(ok_c):
        xs.extend(np.asarray(grid_c[ok_c], dtype=float).ravel().tolist())
    if np.any(ok_x):
        xs.extend(np.asarray(grid_ctrl[ok_x], dtype=float).ravel().tolist())
    if xs:
        lo, hi = min(xs), max(xs)
        if peak_align_condition or peak_align_control or center_control_midpoint:
            w = max(abs(lo), abs(hi))
            if w > 0 and np.isfinite(w):
                ax.set_xlim(-w, w)
        else:
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                ax.set_xlim(lo, hi)

    if peak_align_condition and peak_align_control:
        ax.set_xlabel("x − x_peak (µm)")
    elif peak_align_condition and not peak_align_control and center_control_midpoint:
        ax.set_xlabel("condition: x − x_peak (µm); control: midpoint-centered x (µm)")
    elif peak_align_condition and not peak_align_control and not center_control_midpoint:
        ax.set_xlabel("condition: x − x_peak (µm); control: x (µm)")
    elif not peak_align_condition and peak_align_control:
        ax.set_xlabel("condition: x (µm); control: x − x_peak (µm)")
    elif not peak_align_condition and not peak_align_control and center_control_midpoint:
        ax.set_xlabel("x (µm), control midpoint-centered for display")
    else:
        ax.set_xlabel("x (µm)")
    ax.set_ylabel("Cytoplasm height (µm)")
    ax.set_title(title or "Condition vs control")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved {out_png} and {out_pdf}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot still cytoplasm profiles.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("per-file", help="One CSV → PNG/PDF")
    p1.add_argument("--csv", required=True)
    p1.add_argument("--out-png", required=True)
    p1.add_argument("--out-pdf", required=True)
    p1.add_argument("--title", default=None)
    p1.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving average window in column pixels (along x); 1 = off. Even values use next odd.",
    )
    p1.add_argument(
        "--input-image",
        default=None,
        help="Source TIFF (same as overlay). Used for figure aspect ratio when CSV lacks image_*_px comments.",
    )
    p1.add_argument(
        "--hline",
        action="append",
        default=None,
        metavar="Y,COLOR",
        help="Horizontal reference at y (µm) with matplotlib color, e.g. 24,lightblue. Repeatable.",
    )
    p1.add_argument(
        "--group",
        choices=("condition", "control"),
        default=None,
        help="Cohort: for condition, orange profile hline is set to height at x − x_peak = 0 (peak-aligned).",
    )

    p2 = sub.add_parser("group", help="Many CSVs → mean ± SEM")
    p2.add_argument("--csvs", nargs="+", required=True)
    p2.add_argument(
        "--smooth-windows",
        nargs="*",
        type=int,
        default=None,
        help="One window per CSV (same order as --csvs); omit for all 1 (no smoothing).",
    )
    p2.add_argument("--out-png", required=True)
    p2.add_argument("--out-pdf", required=True)
    p2.add_argument("--label", default="group")
    p2.add_argument(
        "--no-peak-align",
        action="store_true",
        help="Use absolute x (µm); default is peak-aligned x (condition-style).",
    )
    p2.add_argument(
        "--hline",
        action="append",
        default=None,
        metavar="Y,COLOR",
        help="Horizontal reference (µm); repeat. Orange line snaps to mean height at x=0 with --condition-hline-center.",
    )
    p2.add_argument(
        "--condition-hline-center",
        action="store_true",
        help="With peak-aligned condition plot, set orange profile hline y to mean height at x − x_peak = 0.",
    )

    p3 = sub.add_parser("group-compare", help="Two cohorts → one mean ± SEM plot")
    p3.add_argument("--condition-csvs", nargs="+", required=True)
    p3.add_argument("--control-csvs", nargs="+", required=True)
    p3.add_argument(
        "--condition-smooth-windows",
        nargs="*",
        type=int,
        default=None,
        help="One window per condition CSV; omit for all 1.",
    )
    p3.add_argument(
        "--control-smooth-windows",
        nargs="*",
        type=int,
        default=None,
        help="One window per control CSV; omit for all 1.",
    )
    p3.add_argument("--condition-label", default="condition")
    p3.add_argument("--control-label", default="control")
    p3.add_argument("--out-png", required=True)
    p3.add_argument("--out-pdf", required=True)
    p3.add_argument("--title", default=None)
    p3.add_argument(
        "--hline",
        action="append",
        default=None,
        metavar="Y,COLOR",
        help="Horizontal reference on condition mean (µm); orange line snaps to height at x=0 with --condition-hline-center.",
    )
    p3.add_argument(
        "--condition-hline-center",
        action="store_true",
        help="With --hline, set orange profile hline y to condition mean height at x − x_peak = 0.",
    )
    p3.add_argument(
        "--condition-no-peak-align",
        action="store_true",
        help="Use absolute x (µm) for condition traces in compare mode.",
    )
    p3.add_argument(
        "--control-peak-align",
        action="store_true",
        help="Apply per-embryo peak alignment to control traces in compare mode.",
    )
    p3.add_argument(
        "--no-control-midpoint-center",
        action="store_true",
        help="Do not shift control mean to x-midpoint 0 in compare mode.",
    )

    args = p.parse_args()
    if args.cmd == "per-file":
        hl: Optional[List[Tuple[float, str]]] = None
        if args.hline:
            hl = [_parse_hline_spec(s) for s in args.hline]
        plot_per_file(
            args.csv,
            args.out_png,
            args.out_pdf,
            args.title,
            smooth_window_px=args.smooth_window,
            input_image=args.input_image,
            hlines=hl,
            group=args.group,
        )
    elif args.cmd == "group":
        sw = args.smooth_windows
        if sw is None or len(sw) == 0:
            sw = None
        hl2: Optional[List[Tuple[float, str]]] = None
        if args.hline:
            hl2 = [_parse_hline_spec(s) for s in args.hline]
        plot_group(
            args.csvs,
            args.out_png,
            args.out_pdf,
            args.label,
            smooth_windows=sw,
            peak_align=not args.no_peak_align,
            hlines=hl2,
            condition_hline_center=args.condition_hline_center,
        )
    else:
        swc = args.condition_smooth_windows
        if swc is None or len(swc) == 0:
            swc = None
        if swc is None:
            swc = [1] * len(args.condition_csvs)
        elif len(swc) != len(args.condition_csvs):
            raise SystemExit(
                "--condition-smooth-windows count must match --condition-csvs."
            )
        swx = args.control_smooth_windows
        if swx is None or len(swx) == 0:
            swx = None
        if swx is None:
            swx = [1] * len(args.control_csvs)
        elif len(swx) != len(args.control_csvs):
            raise SystemExit("--control-smooth-windows count must match --control-csvs.")
        hl3: Optional[List[Tuple[float, str]]] = None
        if args.hline:
            hl3 = [_parse_hline_spec(s) for s in args.hline]
        plot_two_groups(
            args.condition_csvs,
            swc,
            args.condition_label,
            args.control_csvs,
            swx,
            args.control_label,
            args.out_png,
            args.out_pdf,
            title=args.title,
            hlines=hl3,
            condition_hline_center=args.condition_hline_center,
            peak_align_condition=not args.condition_no_peak_align,
            peak_align_control=args.control_peak_align,
            center_control_midpoint=not args.no_control_midpoint_center,
        )


if __name__ == "__main__":
    main()

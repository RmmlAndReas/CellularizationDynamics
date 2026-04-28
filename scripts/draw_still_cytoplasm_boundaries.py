#!/usr/bin/env python3
"""
Interactive apical + basal polylines on a still TIFF (no ImageJ).

Click to place vertices: first polyline = apical, second = basal.
Splines: cubic along x where possible; linear if too few points.

Keys (image window must have focus):
  Enter   Finish apical → start basal, or finish basal and save
  r       Clear current polyline (apical or basal)
  q       Quit without saving

Writes cytoplasm_profile.csv (same columns as downstream tools) and
boundary_polylines.json (raw vertices for reproducibility).

Usage:
    python scripts/draw_still_cytoplasm_boundaries.py -i in.tif -o track/out.csv \\
        --output-json track/boundaries.json --px2micron 0.27
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import List, Optional, Tuple

import matplotlib

# Interactive GUI backend (must be before pyplot)
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.interpolate import CubicSpline


def _load_gray_2d(path: str) -> np.ndarray:
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
        raise ValueError(f"Expected 2D image, got shape {img.shape}")
    return img


def _aggregate_sorted_unique_x(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort by x; average y when x repeats."""
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if xs.size == 0:
        return xs, ys
    out_x: List[float] = []
    out_y: List[float] = []
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        seg = ys[i : j + 1]
        out_x.append(float(xs[i]))
        out_y.append(float(np.mean(seg)))
        i = j + 1
    return np.asarray(out_x, dtype=float), np.asarray(out_y, dtype=float)


def _smooth_along_x(
    xs: np.ndarray,
    ys: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    """
    Interpolate y(x) with cubic spline if len>=4, else linear.
    Outside support -> NaN.
    """
    xs, ys = _aggregate_sorted_unique_x(xs, ys)
    if xs.size < 2:
        return np.full(x_grid.shape, np.nan, dtype=float)
    mask = (x_grid >= xs[0]) & (x_grid <= xs[-1])
    out = np.full(x_grid.shape, np.nan, dtype=float)
    x_in = x_grid[mask]
    if xs.size < 4:
        out[mask] = np.interp(x_in, xs, ys)
        return out
    cs = CubicSpline(xs, ys, extrapolate=False)
    out[mask] = cs(x_in)
    return out


def _write_csv(
    path: str,
    x_px: np.ndarray,
    y_apical: np.ndarray,
    y_basal: np.ndarray,
    height_px: np.ndarray,
    px2micron: float,
    img_shape_yx: Tuple[int, int],
) -> None:
    px2 = float(px2micron)
    x_um = x_px * px2
    height_um = height_px * px2
    img_h, img_w = int(img_shape_yx[0]), int(img_shape_yx[1])

    od = os.path.dirname(os.path.abspath(path))
    if od:
        os.makedirs(od, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(["# px2micron", f"{px2:.10g}"])
        cw.writerow(["# source", "manual_polylines_cubic"])
        cw.writerow(["# image_width_px", str(img_w)])
        cw.writerow(["# image_height_px", str(img_h)])
        cw.writerow(
            [
                "x_px",
                "x_um",
                "y_apical_px",
                "y_basal_px",
                "height_px",
                "height_um",
            ]
        )
        for i in range(len(x_px)):
            cw.writerow(
                [
                    f"{x_px[i]:.0f}",
                    f"{x_um[i]:.10g}",
                    f"{y_apical[i]:.10g}",
                    f"{y_basal[i]:.10g}",
                    f"{height_px[i]:.10g}",
                    f"{height_um[i]:.10g}",
                ]
            )


def _write_json(
    path: str,
    apical: List[Tuple[float, float]],
    basal: List[Tuple[float, float]],
    shape: Tuple[int, int],
    px2micron: float,
) -> None:
    od = os.path.dirname(os.path.abspath(path))
    if od:
        os.makedirs(od, exist_ok=True)
    data = {
        "apical_xy": [list(p) for p in apical],
        "basal_xy": [list(p) for p in basal],
        "image_shape_yx": [int(shape[0]), int(shape[1])],
        "px2micron": px2micron,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def run_interactive(
    input_path: str,
    out_csv: str,
    out_json: str,
    px2micron: float,
) -> None:
    img = _load_gray_2d(input_path)
    h, w = img.shape

    apical: List[Tuple[float, float]] = []
    basal: List[Tuple[float, float]] = []
    phase = "apical"
    state = {"saved": False, "user_abort": False}

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img, cmap="gray", origin="upper", interpolation="nearest")
    (line_a,) = ax.plot([], [], "c-", lw=2, marker="o", ms=5, label="apical (click)")
    (line_b,) = ax.plot([], [], "m-", lw=2, marker="o", ms=5, label="basal (click)")
    ax.set_title(
        "Left-click: apical boundary  → Enter  → basal boundary  → Enter to save",
        fontsize=11,
    )
    ax.legend(loc="upper right")

    def _sync_lines() -> None:
        if apical:
            xa, ya = zip(*apical)
            line_a.set_data(xa, ya)
        else:
            line_a.set_data([], [])
        if basal:
            xb, yb = zip(*basal)
            line_b.set_data(xb, yb)
        else:
            line_b.set_data([], [])
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        nonlocal phase
        if event.inaxes != ax or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        x = float(np.clip(event.xdata, 0, w - 1))
        y = float(np.clip(event.ydata, 0, h - 1))
        if phase == "apical":
            apical.append((x, y))
        else:
            basal.append((x, y))
        _sync_lines()

    def on_key(event) -> None:
        nonlocal phase
        k = event.key
        if k is None:
            return
        if k == "r":
            if phase == "apical":
                apical.clear()
            else:
                basal.clear()
            _sync_lines()
            print("Cleared current polyline.", file=sys.stderr)
            return
        if k == "q":
            state["user_abort"] = True
            plt.close(fig)
            return
        if k in ("enter", "return"):
            if phase == "apical":
                if len(apical) < 2:
                    print("Need at least 2 apical points.", file=sys.stderr)
                    return
                phase = "basal"
                print("Apical locked. Click basal boundary, then Enter to save.", file=sys.stderr)
                return
            if len(basal) < 2:
                print("Need at least 2 basal points.", file=sys.stderr)
                return
            state["saved"] = True
            plt.close(fig)

    def on_close(_event) -> None:
        if not state["saved"]:
            state["user_abort"] = True

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    print(
        "\n"
        "  Click the image window, then:\n"
        "  Left-click: add vertices to the cyan (apical) line.\n"
        "  Enter:      lock apical, then add magenta (basal) points.\n"
        "  Enter:      save (after basal).\n"
        "  r:          clear current line (apical or basal).\n"
        "  q:          abort without saving.\n",
        file=sys.stderr,
    )
    plt.tight_layout()
    plt.show()

    if not state["saved"]:
        raise SystemExit("Aborted or window closed before saving (finish basal, then Enter).")

    if len(apical) < 2 or len(basal) < 2:
        raise RuntimeError("Incomplete polylines (need ≥2 points each).")

    xa = np.array([p[0] for p in apical], dtype=float)
    ya = np.array([p[1] for p in apical], dtype=float)
    xb = np.array([p[0] for p in basal], dtype=float)
    yb = np.array([p[1] for p in basal], dtype=float)

    x_grid = np.arange(w, dtype=float)
    y_a_s = _smooth_along_x(xa, ya, x_grid)
    y_b_s = _smooth_along_x(xb, yb, x_grid)

    # First polyline = apical, second = basal (do not min/max — that swaps anatomy).
    y_apical = y_a_s
    y_basal = y_b_s
    height_px = np.abs(y_basal - y_apical)

    _write_csv(out_csv, x_grid, y_apical, y_basal, height_px, px2micron, (h, w))
    _write_json(out_json, apical, basal, (h, w), px2micron)
    print(f"Wrote {out_csv} and {out_json}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description="Draw apical/basal polylines; export CSV (cubic spline).")
    p.add_argument("-i", "--input", required=True, help="Input TIFF")
    p.add_argument("-o", "--output-csv", required=True, help="Output cytoplasm_profile.csv")
    p.add_argument("--output-json", required=True, help="Output boundary_polylines.json")
    p.add_argument("--px2micron", type=float, required=True, help="µm per pixel")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Not found: {args.input}")

    run_interactive(args.input, args.output_csv, args.output_json, args.px2micron)


if __name__ == "__main__":
    main()

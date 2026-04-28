#!/usr/bin/env python3
"""
Overlay apical and basal border polylines on the original still image.

Reads cytoplasm_profile.csv with y_apical_px / y_basal_px (or legacy y_top_px / y_bottom_px)
per column x_px, and draws the two boundaries on the
input TIFF. Exports PNG, PDF, and an RGB TIFF with overlays baked in.

Usage:
    python scripts/export_still_cytoplasm_overlay.py \\
        --input-image path.tif --csv track/cytoplasm_profile.csv \\
        --out-png out.png --out-pdf out.pdf --out-tif out.tif
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
from PIL import Image, ImageDraw


def _load_gray_2d(path: str) -> np.ndarray:
    img = tifffile.imread(path)
    if img.ndim == 3:
        # (planes, y, x) or (y, x, c) — take first plane / channel
        if img.shape[0] <= 8 and img.shape[0] != img.shape[-1]:
            img = np.squeeze(img[0])
        elif img.shape[-1] <= 4:
            img = np.squeeze(img[..., 0])
        else:
            img = np.squeeze(img[0])
    if img.ndim != 2:
        img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image after squeeze, got shape {img.shape}")
    return img


def _read_borders_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (x_px, y_apical_px, y_basal_px) with NaN where missing."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    header_idx = None
    for i, row in enumerate(rows):
        if row and row[0].strip() == "x_px":
            header_idx = i
            break
    if header_idx is None:
        raise SystemExit(f"No header row starting with x_px in {path}")

    hdr = [c.strip() for c in rows[header_idx]]
    if "y_apical_px" in hdr:
        ia = hdr.index("y_apical_px")
        ib = hdr.index("y_basal_px")
    elif "y_top_px" in hdr:
        ia = hdr.index("y_top_px")
        ib = hdr.index("y_bottom_px")
    else:
        ia, ib = 2, 3

    x_list: List[float] = []
    a_list: List[float] = []
    b_list: List[float] = []

    for row in rows[header_idx + 1 :]:
        if not row or row[0].startswith("#"):
            continue
        try:
            x_px = float(row[0])
        except (ValueError, IndexError):
            continue
        try:
            ya = float(row[ia]) if len(row) > ia else float("nan")
        except ValueError:
            ya = float("nan")
        try:
            yb = float(row[ib]) if len(row) > ib else float("nan")
        except ValueError:
            yb = float("nan")
        if not np.isfinite(ya):
            ya = float("nan")
        if not np.isfinite(yb):
            yb = float("nan")
        x_list.append(x_px)
        a_list.append(ya)
        b_list.append(yb)

    x = np.asarray(x_list, dtype=float)
    ya = np.asarray(a_list, dtype=float)
    yb = np.asarray(b_list, dtype=float)
    order = np.argsort(x)
    return x[order], ya[order], yb[order]


def _to_uint8_rgb(base: np.ndarray) -> np.ndarray:
    """2D float or int -> uint8 RGB (H, W, 3)."""
    v = base.astype(np.float64)
    finite = np.isfinite(v)
    if not np.any(finite):
        v = np.zeros_like(v)
    else:
        lo, hi = np.percentile(v[finite], (1.0, 99.0))
        if hi <= lo:
            lo, hi = float(np.min(v[finite])), float(np.max(v[finite]))
        v = np.clip((v - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    gray = (v * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _draw_polyline_pil(
    draw: ImageDraw.ImageDraw,
    x: np.ndarray,
    y: np.ndarray,
    fill: Tuple[int, int, int],
    width: int = 2,
) -> None:
    pts = []
    for i in range(len(x)):
        if np.isfinite(x[i]) and np.isfinite(y[i]):
            pts.append((int(round(float(x[i]))), int(round(float(y[i])))))
    if len(pts) < 2:
        return
    draw.line(pts, fill=fill, width=width)


def export_overlay(
    input_image: str,
    csv_path: str,
    out_png: str,
    out_pdf: str,
    out_tif: Optional[str],
    title: Optional[str],
    mirror_y: bool = False,
) -> None:
    img = _load_gray_2d(input_image)
    h, w = img.shape
    x_px, y_apical, y_basal = _read_borders_csv(csv_path)

    if x_px.size == 0:
        raise SystemExit(f"No data rows in {csv_path}")

    # Clip coordinates for display safety (y: 0 = top of image, increasing downward)
    x_plot = np.clip(x_px, 0, w - 1)
    if mirror_y:
        # Reflect across vertical axis (standard y-axis mirror): x -> (w-1) - x
        img = np.fliplr(img)
        x_plot = (w - 1) - x_plot
    y_a = np.where(np.isfinite(y_apical), np.clip(y_apical, 0, h - 1), np.nan)
    y_b = np.where(np.isfinite(y_basal), np.clip(y_basal, 0, h - 1), np.nan)

    # extent=(left, right, bottom, top): y=0 at top, y=h at bottom (matches row index)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(
        img,
        cmap="gray",
        origin="upper",
        extent=(0, w, h, 0),
        aspect="equal",
        interpolation="nearest",
    )
    ax.plot(
        x_plot,
        y_a,
        color="cyan",
        linewidth=1.8,
        label="Apical (up)",
        alpha=0.95,
    )
    ax.plot(
        x_plot,
        y_b,
        color="magenta",
        linewidth=1.8,
        label="Basal (down)",
        alpha=0.95,
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels, down)")
    ax.set_title(title or os.path.basename(input_image))
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved {out_png} and {out_pdf}")

    if out_tif:
        rgb = _to_uint8_rgb(img)
        pil_im = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(pil_im)
        apical_rgb = (0, 255, 255)
        basal_rgb = (255, 0, 255)
        _draw_polyline_pil(draw, x_plot, y_a, apical_rgb, width=2)
        _draw_polyline_pil(draw, x_plot, y_b, basal_rgb, width=2)
        pil_im.save(out_tif)
        print(f"Saved {out_tif}")


def main() -> None:
    p = argparse.ArgumentParser(description="Apical/basal overlay on still image.")
    p.add_argument("--input-image", "-i", required=True, help="Original TIFF path")
    p.add_argument("--csv", "-c", required=True, help="cytoplasm_profile.csv")
    p.add_argument("--out-png", required=True)
    p.add_argument("--out-pdf", required=True)
    p.add_argument("--out-tif", default=None, help="Optional RGB TIFF with overlays")
    p.add_argument("--title", default=None)
    p.add_argument(
        "--mirror-y",
        action="store_true",
        help="Mirror horizontally (reflect across vertical axis: x -> width-1-x).",
    )
    args = p.parse_args()

    if not os.path.isfile(args.input_image):
        raise SystemExit(f"Missing image: {args.input_image}")
    if not os.path.isfile(args.csv):
        raise SystemExit(f"Missing CSV: {args.csv}")

    export_overlay(
        args.input_image,
        args.csv,
        args.out_png,
        args.out_pdf,
        args.out_tif,
        args.title,
        mirror_y=args.mirror_y,
    )


if __name__ == "__main__":
    main()

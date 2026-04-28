#!/usr/bin/env python3
"""
Export duplicated-slice and threshold-outline visualizations for movie cytoplasm
width measurements.

Reads:
  - a duplicated grayscale slice image (typically track/MovieSliceDup.tif)
  - threshold mask image (typically track/CytoplasmMask.tif)
  - track/cytoplasm_height_vs_x.csv with apical_px and basal_px columns
  - px2micron scale from config.yaml

Writes:
  - duplicated slice export
  - threshold-outline export
  - PNG/PDF overlays with white dashed apical + cytoplasm-border lines
  - Optional RGB TIFF with the same overlays baked in
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image, ImageDraw
import yaml


def _load_gray_2d(path: str) -> np.ndarray:
    img = tifffile.imread(path)
    if img.ndim > 2:
        img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image after squeeze, got shape {img.shape}")
    return img


def _load_profile_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    y_apical = []
    y_basal = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"x_idx", "apical_px", "basal_px"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"{path} must contain columns: {sorted(required)}; "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            try:
                xs.append(float(row["x_idx"]))
                y_apical.append(float(row["apical_px"]))
                y_basal.append(float(row["basal_px"]))
            except (TypeError, ValueError):
                continue

    if not xs:
        raise ValueError(f"No valid rows in {path}")

    x = np.asarray(xs, dtype=float)
    ap = np.asarray(y_apical, dtype=float)
    yb = np.asarray(y_basal, dtype=float)
    order = np.argsort(x)
    return x[order], ap[order], yb[order]


def _load_px2micron(config_path: str) -> float:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    manual = cfg.get("manual") or {}
    if "px2micron" not in manual:
        raise ValueError(f"manual.px2micron missing in {config_path}")
    return float(manual["px2micron"])


def _to_uint8_rgb(base: np.ndarray) -> np.ndarray:
    v = base.astype(np.float64)
    finite = np.isfinite(v)
    if not np.any(finite):
        v = np.zeros_like(v)
    else:
        lo, hi = np.percentile(v[finite], (1.0, 99.0))
        if hi <= lo:
            lo = float(np.min(v[finite]))
            hi = float(np.max(v[finite]))
        v = np.clip((v - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    gray = (v * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _mask_outline(mask_img: np.ndarray) -> np.ndarray:
    mask = mask_img > 0
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got shape {mask.shape}")
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    center = p[1:-1, 1:-1]
    up = p[:-2, 1:-1]
    down = p[2:, 1:-1]
    left = p[1:-1, :-2]
    right = p[1:-1, 2:]
    interior = center & up & down & left & right
    outline = center & (~interior)
    return outline.astype(np.uint8) * 255


def _draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    x: np.ndarray,
    y: np.ndarray,
    color: Tuple[int, int, int],
    width: int = 2,
    dash_px: int = 7,
    gap_px: int = 5,
) -> None:
    pts = []
    for i in range(len(x)):
        if np.isfinite(x[i]) and np.isfinite(y[i]):
            pts.append((int(round(float(x[i]))), int(round(float(y[i])))))
    if len(pts) < 2:
        return
    for p0, p1 in zip(pts[:-1], pts[1:]):
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        seg_len = float(np.hypot(dx, dy))
        if seg_len <= 0:
            continue
        ux = dx / seg_len
        uy = dy / seg_len
        t = 0.0
        while t < seg_len:
            t_end = min(t + dash_px, seg_len)
            a = (int(round(p0[0] + ux * t)), int(round(p0[1] + uy * t)))
            b = (int(round(p0[0] + ux * t_end)), int(round(p0[1] + uy * t_end)))
            draw.line([a, b], fill=color, width=width)
            t += dash_px + gap_px


def export_overlay(
    input_image: str,
    input_mask: str,
    input_csv: str,
    config_path: str,
    out_duplicate_slice: str,
    out_threshold_outline: str,
    out_png: str,
    out_pdf: str,
    out_tif: str | None = None,
    title: str | None = None,
) -> None:
    img = _load_gray_2d(input_image)
    mask_img = _load_gray_2d(input_mask)
    h, w = img.shape
    x, y_apical, y_basal = _load_profile_csv(input_csv)
    px2um = _load_px2micron(config_path)

    x = np.clip(x, 0, w - 1)
    y_apical = np.clip(y_apical, 0, h - 1)
    y_basal = np.clip(y_basal, 0, h - 1)
    x_um = x * px2um
    y_apical_um = y_apical * px2um
    y_basal_um = y_basal * px2um
    w_um = w * px2um
    h_um = h * px2um

    os.makedirs(os.path.dirname(os.path.abspath(out_duplicate_slice)) or ".", exist_ok=True)
    tifffile.imwrite(out_duplicate_slice, img)
    print(f"Saved duplicated slice to {out_duplicate_slice}")

    outline = _mask_outline(mask_img)
    os.makedirs(os.path.dirname(os.path.abspath(out_threshold_outline)) or ".", exist_ok=True)
    tifffile.imwrite(out_threshold_outline, outline)
    print(f"Saved threshold outline to {out_threshold_outline}")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(
        img,
        cmap="gray",
        origin="upper",
        extent=(0, w_um, h_um, 0),
        aspect="equal",
        interpolation="nearest",
    )
    ax.plot(
        x_um,
        y_apical_um,
        color="white",
        linestyle="--",
        linewidth=2.0,
        label="Apical border",
    )
    ax.plot(
        x_um,
        y_basal_um,
        color="white",
        linestyle="--",
        linewidth=2.0,
        alpha=0.9,
        label="Basal border",
    )
    ax.set_xlim(0, w_um)
    ax.set_ylim(h_um, 0)
    ax.set_xlabel("x (micron)")
    ax.set_ylabel("y (micron, down)")
    ax.set_title(title or os.path.basename(input_image))
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=220, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved {out_png} and {out_pdf}")

    if out_tif:
        rgb = _to_uint8_rgb(img)
        pil_im = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(pil_im)
        white = (255, 255, 255)
        _draw_dashed_polyline(draw, x, y_apical, white, width=2)
        _draw_dashed_polyline(draw, x, y_basal, white, width=2)
        pil_im.save(out_tif)
        print(f"Saved {out_tif}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--input-mask", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-duplicate-slice", required=True)
    parser.add_argument("--out-threshold-outline", required=True)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--out-pdf", required=True)
    parser.add_argument("--out-tif", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.input_image):
        raise SystemExit(f"Missing image: {args.input_image}")
    if not os.path.isfile(args.input_mask):
        raise SystemExit(f"Missing mask: {args.input_mask}")
    if not os.path.isfile(args.input_csv):
        raise SystemExit(f"Missing CSV: {args.input_csv}")
    if not os.path.isfile(args.config):
        raise SystemExit(f"Missing config: {args.config}")

    export_overlay(
        input_image=args.input_image,
        input_mask=args.input_mask,
        input_csv=args.input_csv,
        config_path=args.config,
        out_duplicate_slice=args.out_duplicate_slice,
        out_threshold_outline=args.out_threshold_outline,
        out_png=args.out_png,
        out_pdf=args.out_pdf,
        out_tif=args.out_tif,
        title=args.title,
    )


if __name__ == "__main__":
    main()

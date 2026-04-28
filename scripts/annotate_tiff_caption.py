#!/usr/bin/env python3
"""Annotate an 8-bit grayscale TIFF stack: suction label (selected frames), timestamp, scale bar."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont


FONT_SIZE = 28  # suction label (top right)
FONT_SIZE_SMALL = FONT_SIZE // 2  # timestamp + scale bar (½ size)


def load_font(size: int = FONT_SIZE) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size
        )
    except OSError:
        return ImageFont.load_default()


def infer_um_per_pixel(tif: tifffile.TiffFile) -> Optional[float]:
    """Try ImageJ-style resolution: micron calibration + XResolution = pixels/µm."""
    desc = tif.pages[0].description
    if desc:
        for part in str(desc).replace("\r", "\n").split("\n"):
            m = re.search(
                r"pixel[_\s]?width\s*=\s*([0-9.eE+-]+)", part, re.I
            ) or re.search(r"PhysicalSizeX\s*=\s*([0-9.eE+-]+)", part, re.I)
            if m:
                return float(m.group(1))
    ij = tif.imagej_metadata or {}
    xr = tif.pages[0].tags.get("XResolution")
    if xr is not None and str(ij.get("unit", "")).lower() == "micron":
        num, den = xr.value
        if den and num:
            pix_per_um = float(num) / float(den)
            if pix_per_um > 0:
                return 1.0 / pix_per_um
    return None


def draw_timestamp(
    draw: ImageDraw.ImageDraw,
    frame_index: int,
    seconds_per_frame: float,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    margin_px: int,
) -> None:
    t_sec = frame_index * seconds_per_frame
    label = f"t = {t_sec:g} s"
    draw.text((margin_px, margin_px), label, fill=255, font=font)


def draw_scale_bar_20um(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    um_per_pixel: float,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    margin_px: int,
    bar_um: float = 20.0,
) -> None:
    length_px = int(round(bar_um / um_per_pixel))
    length_px = max(3, min(length_px, width - 2 * margin_px))

    bbox = draw.textbbox((0, 0), f"{bar_um:g} µm", font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    gap = 4
    line_y = height - margin_px - text_h - gap - 2
    # Bar flush to bottom-right; label centered under bar
    x2 = width - margin_px
    x1 = x2 - length_px
    draw.line([(x1, line_y), (x2, line_y)], fill=255, width=2)
    label = f"{bar_um:g} µm"
    tx = (x1 + x2 - text_w) // 2
    if tx < margin_px:
        tx = margin_px
    draw.text((tx, line_y + gap), label, fill=255, font=font)


def annotate_stack(
    in_path: Path,
    out_path: Path,
    text: str,
    frame_start: int,
    frame_end: int,
    zero_based: bool,
    seconds_per_frame: float,
    um_per_pixel: float,
    margin_px: int = 8,
) -> None:
    stack = tifffile.imread(str(in_path))
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack (Z/T, Y, X), got shape {stack.shape}")

    n = stack.shape[0]
    if zero_based:
        idx_lo, idx_hi = frame_start, frame_end
    else:
        idx_lo, idx_hi = frame_start - 1, frame_end - 1

    if idx_lo < 0 or idx_hi < idx_lo or idx_hi >= n:
        raise ValueError(
            f"Invalid frame range [{frame_start}, {frame_end}] "
            f"({'0' if zero_based else '1'}-based): stack has {n} frame(s)"
        )

    annotate_indices = set(range(idx_lo, idx_hi + 1))
    out = np.zeros((n, stack.shape[1], stack.shape[2]), dtype=np.uint8)
    font_label = load_font(FONT_SIZE)
    font_small = load_font(FONT_SIZE_SMALL)

    for i in range(n):
        plane = stack[i]
        if plane.dtype != np.uint8:
            p = np.clip(plane, 0, 255).astype(np.uint8)
        else:
            p = np.asarray(plane)
        img = Image.fromarray(p, mode="L")
        draw = ImageDraw.Draw(img)

        draw_timestamp(draw, i, seconds_per_frame, font_small, margin_px)
        draw_scale_bar_20um(
            draw,
            img.width,
            img.height,
            um_per_pixel,
            font_small,
            margin_px,
            bar_um=20.0,
        )
        if i in annotate_indices:
            bbox = draw.textbbox((0, 0), text, font=font_label)
            tw = bbox[2] - bbox[0]
            x = img.width - tw - margin_px
            y = margin_px
            draw.text((x, y), text, fill=255, font=font_label)

        out[i] = np.asarray(img)

    tifffile.imwrite(
        str(out_path),
        out,
        bigtiff=((n * stack.shape[1] * stack.shape[2]) > 2**31 - 1),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <stem>_suction_caption.tif next to input)",
    )
    ap.add_argument(
        "--text",
        type=str,
        default="Suction",
        help="Label text on selected frames only (top right)",
    )
    ap.add_argument(
        "--frame-start",
        type=int,
        default=11,
        help="First frame to label (default 11, see --zero-based)",
    )
    ap.add_argument(
        "--frame-end",
        type=int,
        default=55,
        help="Last frame to label (default 55, see --zero-based)",
    )
    ap.add_argument(
        "--zero-based",
        action="store_true",
        help="Interpret --frame-start/--frame-end as 0-based indices (default is 1-based)",
    )
    ap.add_argument(
        "--seconds-per-frame",
        type=float,
        default=1.0,
        help="Time step for timestamp in seconds (default: 1 s per frame)",
    )
    ap.add_argument(
        "--um-per-pixel",
        type=float,
        default=None,
        help="Micrometers per pixel (XY). If omitted, inferred from TIFF when possible.",
    )
    args = ap.parse_args()
    inp = args.input.resolve()
    if not inp.exists():
        raise FileNotFoundError(inp)

    with tifffile.TiffFile(str(inp)) as tf:
        um = args.um_per_pixel
        if um is None:
            um = infer_um_per_pixel(tf)
        if um is None or um <= 0:
            raise SystemExit(
                "Could not infer µm/pixel from TIFF. Set --um-per-pixel explicitly."
            )

    out = (
        args.output.resolve()
        if args.output
        else inp.with_name(f"{inp.stem}_suction_caption.tif")
    )
    annotate_stack(
        inp,
        out,
        text=args.text,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        zero_based=args.zero_based,
        seconds_per_frame=args.seconds_per_frame,
        um_per_pixel=um,
    )
    mode = "0-based" if args.zero_based else "1-based"
    print(
        f"Wrote {out}\n  µm/pixel={um:.6g}\n  label '{args.text}' on frames "
        f"{args.frame_start}–{args.frame_end} inclusive ({mode}); "
        f"timestamp step {args.seconds_per_frame} s"
    )


if __name__ == "__main__":
    main()

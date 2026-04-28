#!/usr/bin/env python3
"""Build a single-row channel × time collage from a multi-channel TIFF (Hermetia yolk compaction).

PNG and PDF are raster-only (no on-image labels); add typography in Affinity or similar.

Each channel×time false-color panel and each timepoint overlay are also written as PNGs under
``<input_stem>_collage_panels/`` next to the collage outputs (for manual alignment in Affinity).

PNG outputs use **16-bit RGB** (lossless zlib-compressed PNG) at the collage’s native pixel grid;
the collage PNG is no longer resampled via a matplotlib figure.
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import zoom


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "data" / "Hermetia" / "yolk_compaction").is_dir():
            return p
    return start


def resolve_input_path(image: str | Path, repo: Path) -> Path:
    p = Path(image).expanduser()
    if not p.is_absolute():
        p = (repo / p).resolve()
    else:
        p = p.resolve()
    return p


def _percentile_norm(gray: np.ndarray, p_lo: float = 2.0, p_hi: float = 99.5) -> np.ndarray:
    g = gray.astype(np.float64, copy=False)
    lo, hi = np.percentile(g, (p_lo, p_hi))
    if hi <= lo:
        hi = lo + 1e-9
    return np.clip((g - lo) / (hi - lo), 0.0, 1.0)


def _norm_group_over_time(
    frames_thw: np.ndarray, p_lo: float = 2.0, p_hi: float = 99.5
) -> np.ndarray:
    g = frames_thw.astype(np.float64, copy=False)
    lo, hi = np.percentile(g, (p_lo, p_hi))
    if hi <= lo:
        hi = lo + 1e-9
    return np.clip((g - lo) / (hi - lo), 0.0, 1.0)


def _norm_group_over_time_rgb(
    rgb_thwc: np.ndarray, p_lo: float = 2.0, p_hi: float = 99.5
) -> np.ndarray:
    x = rgb_thwc.astype(np.float64, copy=False)
    out = np.empty_like(x)
    for j in range(3):
        plane = x[..., j]
        lo, hi = np.percentile(plane, (p_lo, p_hi))
        if hi <= lo:
            hi = lo + 1e-9
        out[..., j] = np.clip((plane - lo) / (hi - lo), 0.0, 1.0)
    return out


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    new_w = max(1, int(round(w * target_h / h)))
    zf = (target_h / h, new_w / w)
    if img.dtype == np.uint8:
        work = img.astype(np.float32) / 255.0
        out = zoom(work, zf, order=1)
        return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)
    return zoom(img.astype(np.float64), zf, order=1)


def _ij_hyperstack_reshape(
    data: np.ndarray, ij: dict, n_time: int, n_ch: int
) -> tuple[np.ndarray, str]:
    ch = int(ij.get("channels", 0) or 0)
    frames = int(ij.get("frames", 0) or 0)
    slices = int(ij.get("slices", 1) or 1)
    if ch and frames and slices and data.ndim == 3:
        n = ch * frames * slices
        if data.shape[0] == n:
            vol = data.reshape(frames, ch, slices, data.shape[1], data.shape[2])
            vol = vol[:, :, 0, ...]
            return vol.astype(np.float32), (
                f"ImageJ hyperstack metadata (frames={frames}, channels={ch}, slices={slices})"
            )
    raise ValueError("Could not interpret stack with ImageJ metadata")


def load_tcyx_from_tiff(path: Path, n_time: int, n_ch: int) -> tuple[np.ndarray, str]:
    with tifffile.TiffFile(path) as tf:
        data = tf.asarray()
        ij = tf.imagej_metadata or {}

        if data.ndim == 5:
            if data.shape[2] == n_ch and data.shape[0] == n_time:
                return data[:, 0, ...].astype(np.float32), "5D (T, Z, C, Y, X), Z=0"
            if data.shape[1] == n_time and data.shape[2] == n_ch:
                return data[0, ...].astype(np.float32), "5D, first index dropped"

        if data.ndim == 4:
            t, c, y, x = data.shape
            if t == n_time and c == n_ch:
                return data.astype(np.float32), "4D (T, C, Y, X)"
            if c == n_time and t == n_ch:
                return np.transpose(data, (1, 0, 2, 3)).astype(np.float32), "4D (C, T, Y, X) → (T, C, Y, X)"

        if data.ndim == 3 and ij:
            try:
                return _ij_hyperstack_reshape(data, ij, n_time, n_ch)
            except ValueError:
                pass
            n = data.shape[0]
            if n == n_time * n_ch:
                a = data.reshape(n_time, n_ch, data.shape[1], data.shape[2])
                return a.astype(np.float32), f"3D ({n}, Y, X) → (T, C, Y, X)"

    raise ValueError(
        f"Unsupported TIFF shape {data.shape} for {path}; "
        f"expected {n_time}×{n_ch} (T, C, Y, X). Inspect tifffile metadata."
    )


def _false_color_panel(gray_norm: np.ndarray, rgb_weights: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(rgb_weights, dtype=np.float64), 0.0)
    mx = float(w.max())
    if mx > 0:
        w = w / mx
    return gray_norm[..., None] * w


def _overlay_rgb(channels_norm: np.ndarray, mix: np.ndarray) -> np.ndarray:
    m = np.maximum(np.asarray(mix, dtype=np.float64), 0.0)
    return np.clip(np.einsum("cxy,cj->xyj", channels_norm, m), 0.0, 1.0)


def save_png_lossless_rgb16(path: Path, rgb_float01: np.ndarray) -> None:
    """Write float RGB [0,1] as a lossless 16-bit-per-channel PNG (native H×W, no resampling).

    Pillow down-converts 16-bit RGB to 8-bit on PNG save, so this uses a tiny PNG encoder (zlib).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(np.asarray(rgb_float01, dtype=np.float64), 0.0, 1.0)
    u16 = np.ascontiguousarray(np.round(x * 65535.0).astype(np.uint16))
    if u16.ndim != 3 or u16.shape[2] != 3:
        raise ValueError("expected RGB array with shape (H, W, 3)")
    h, w, _ = u16.shape

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)
    flat = u16.reshape(h, w * 3).astype(np.uint32)
    hi = (flat >> 8).astype(np.uint8)
    lo = (flat & 0xFF).astype(np.uint8)
    be = np.empty((h, w * 6), dtype=np.uint8)
    be[:, 0::2] = hi
    be[:, 1::2] = lo
    raw_rows = [b"\x00" + be[y].tobytes() for y in range(h)]
    compressed = zlib.compress(b"".join(raw_rows), level=6)

    out = bytearray(b"\x89PNG\r\n\x1a\n")
    out.extend(_chunk(b"IHDR", ihdr))
    out.extend(_chunk(b"IDAT", compressed))
    out.extend(_chunk(b"IEND", b""))
    path.write_bytes(out)


def save_collage_pdf(base: np.ndarray, path: Path, dpi: int) -> None:
    """Write a PDF with the collage as a single embedded raster (no labels)."""
    h, w = base.shape[0], base.shape[1]
    fig_w_in = w / float(dpi)
    fig_h_in = h / float(dpi)
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=int(dpi))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(
        np.clip(base, 0.0, 1.0),
        origin="upper",
        interpolation="nearest",
        aspect="equal",
    )
    ax.axis("off")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)

    with mpl.rc_context({"pdf.fonttype": 42}):
        fig.savefig(
            str(path),
            format="pdf",
            dpi=int(dpi),
            bbox_inches=None,
            pad_inches=0,
            facecolor="white",
            edgecolor="none",
        )
    plt.close(fig)


def build_single_row_collage(
    vol: np.ndarray,
    n_time: int,
    n_ch: int,
    single_channel_rgb: np.ndarray,
    overlay_channel_rgb: np.ndarray,
    panel_h: int = 320,
    gap_px: int = 4,
    group_gap_px: int | None = None,
    border_px: int = 2,
    group_norm_p_lo: float = 2.0,
    group_norm_p_hi: float = 99.5,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    t, c, _, _ = vol.shape
    assert t == n_time and c == n_ch

    g_gap = gap_px * 3 if group_gap_px is None else int(group_gap_px)

    scm = np.asarray(single_channel_rgb, dtype=np.float64).reshape(n_ch, 3)
    ocm = np.asarray(overlay_channel_rgb, dtype=np.float64).reshape(n_ch, 3)

    vol_n = np.empty_like(vol, dtype=np.float64)
    for ci in range(n_ch):
        vol_n[:, ci, ...] = _norm_group_over_time(
            vol[:, ci, ...], group_norm_p_lo, group_norm_p_hi
        )

    panels: list[np.ndarray] = []
    named_panels: list[tuple[str, np.ndarray]] = []
    for ci in range(n_ch):
        for ti in range(n_time):
            g = vol_n[ti, ci]
            g = _resize_to_height(g, panel_h)
            p = _false_color_panel(g, scm[ci])
            panels.append(p)
            named_panels.append((f"ch{ci}_t{ti:02d}", np.clip(p, 0.0, 1.0).copy()))

    overlay_rgb: list[np.ndarray] = []
    for ti in range(n_time):
        norms = np.stack([vol_n[ti, ci] for ci in range(n_ch)], axis=0)
        norms = np.stack(
            [_resize_to_height(norms[ci], panel_h) for ci in range(n_ch)], axis=0
        )
        overlay_rgb.append(_overlay_rgb(norms, ocm))
    overlay_stack = np.stack(overlay_rgb, axis=0)
    overlay_stack = _norm_group_over_time_rgb(
        overlay_stack, group_norm_p_lo, group_norm_p_hi
    )
    for ti in range(n_time):
        o = overlay_stack[ti]
        panels.append(o)
        named_panels.append((f"overlay_t{ti:02d}", np.clip(o, 0.0, 1.0).copy()))

    h = panel_h + 2 * border_px
    ws = [p.shape[1] for p in panels]
    n = len(panels)
    between_gaps = [g_gap if (i + 1) % n_time == 0 else gap_px for i in range(n - 1)]
    w = sum(ws) + sum(between_gaps) + 2 * border_px
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    x = border_px
    for i, p in enumerate(panels):
        ph, pw, _ = p.shape
        y0 = border_px + (panel_h - ph) // 2
        canvas[y0 : y0 + ph, x : x + pw] = p
        x += pw
        if i < n - 1:
            x += between_gaps[i]

    return np.clip(canvas, 0.0, 1.0), named_panels


def save_individual_panels(
    parent: Path, stem: str, named_panels: list[tuple[str, np.ndarray]]
) -> Path:
    """Write each RGB panel (float 0–1) as 16-bit lossless PNG under ``<parent>/<stem>_collage_panels/``."""
    out_dir = parent / f"{stem}_collage_panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rgb in named_panels:
        save_png_lossless_rgb16(out_dir / f"{name}.png", rgb)
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input TIFF path (absolute or relative to --repo)",
    )
    p.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Repository root for relative paths (default: auto-detect)",
    )
    p.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="Output PNG (default: <input_stem>_collage.png next to input)",
    )
    p.add_argument(
        "--out-pdf",
        type=Path,
        default=None,
        help="Output PDF (default: <input_stem>_collage.pdf next to input); omit with --no-pdf",
    )
    p.add_argument("--no-pdf", action="store_true", help="Do not write PDF")
    p.add_argument("--n-time", type=int, default=4, help="Number of timepoints")
    p.add_argument("--n-ch", type=int, default=3, help="Number of channels")
    p.add_argument("--panel-h", type=int, default=320, help="Panel height in pixels")
    p.add_argument(
        "--png-dpi",
        type=int,
        default=600,
        help="Unused for main collage PNG (native-resolution 16-bit lossless); kept for compatibility",
    )
    p.add_argument("--pdf-dpi", type=int, default=600, help="PDF embedded raster DPI")
    p.add_argument("--show", action="store_true", help="Show matplotlib preview window")
    args = p.parse_args()

    repo = args.repo.resolve() if args.repo else find_repo_root()
    inp = resolve_input_path(args.input, repo)
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")
    if inp.suffix.lower() not in {".tif", ".tiff"}:
        raise SystemExit(f"TIFF only; got {inp.suffix}")

    stem = inp.stem
    out_png = args.out_png or (inp.parent / f"{stem}_collage.png")
    out_pdf = None if args.no_pdf else (args.out_pdf or (inp.parent / f"{stem}_collage.pdf"))

    single = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64
    )
    overlay = single.copy()

    vol, note = load_tcyx_from_tiff(inp, args.n_time, args.n_ch)
    print(note)
    print("loaded", inp.name, vol.shape, vol.dtype)

    collage, named_panels = build_single_row_collage(
        vol,
        args.n_time,
        args.n_ch,
        single,
        overlay,
        panel_h=args.panel_h,
    )

    panels_dir = save_individual_panels(inp.parent, stem, named_panels)
    print("Wrote", len(named_panels), "panel PNG(s) to", panels_dir)

    save_png_lossless_rgb16(out_png, collage)
    print("Wrote", out_png, f"({collage.shape[1]}×{collage.shape[0]} px, 16-bit RGB)")

    if args.show:
        plt.figure(figsize=(min(32, collage.shape[1] / 80), min(4, collage.shape[0] / 80)))
        plt.imshow(collage, interpolation="nearest")
        plt.axis("off")
        plt.tight_layout()
    if out_pdf is not None:
        save_collage_pdf(collage, out_pdf, args.pdf_dpi)
        print("Wrote", out_pdf)
    if args.show:
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()

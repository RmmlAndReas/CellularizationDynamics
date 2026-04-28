#!/usr/bin/env python3
"""
Combine ``results/Cellularization_trimmed_delta.tif`` movies side by side (one row, multiple columns).

Sample labels are the **keys** from the sample config (e.g. ``hermetia2-lateral``), resolved by
matching each sample folder path to the ``path:`` entry under the given species folder.

Example:

    python scripts/combine_cellularization_trimmed_delta_subsamples.py \\
        --subfolder data/Hermetia \\
        --sample-list 2,6,9,10 \\
        --out-stem Hermetia_trimmed_delta_movies_2_6_9_10

Writes ``<out-stem>.tif`` and, by default, ``<out-stem>.avi`` (H.264 via ffmpeg). Use ``--no-export-avi`` to skip AVI.

Intensity (histogram) alignment across panels:

    Mixed 8-bit vs 12/16-bit TIFFs are first mapped to a **common float [0,1]** space (uint8→/255;
    uint16→/4095 or /65535 via ``--uint16-scale``), then percentiles are computed—so ``global_percentile``
    compares like with like.

    # Same display window for all panels
    --intensity-mode global_percentile --percentile-low 0.5 --percentile-high 99.5

    # Each panel stretched independently (stronger “match”, not comparable in absolute units)
    --intensity-mode per_panel_percentile

    # No extra stretch: only unified 0–1 mapping (uint8/uint16 scaling)
    --intensity-mode full_range

    # Force 12-bit or 16-bit divisor for uint16 stacks (default: auto from max pixel value)
    --uint16-scale 12bit
    --uint16-scale 16bit

    # Save as 8-bit TIFF (smaller; values are still float [0,1] before quantization)
    --output-bit-depth 8
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile
import yaml
from samples_loader import load_samples_config, default_samples_config_path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def _uint16_denominator(arr_f32: np.ndarray, uint16_scale: str) -> float:
    """
    Scale factor for uint16 pixel values → [0,1].

    ``auto``: if max ≤ 4096 treat as 12-bit in 16-bit container (÷4095), else full 16-bit (÷65535).
    """
    if uint16_scale == "12bit":
        return 4095.0
    if uint16_scale == "16bit":
        return 65535.0
    mx = float(np.max(arr_f32)) if arr_f32.size else 0.0
    if mx <= 4096.0:
        return 4095.0
    return 65535.0


def _scale_gray_to_unit01(gray: np.ndarray, uint16_scale: str) -> np.ndarray:
    """Single channel → float32 in [0, 1]."""
    arr = gray.astype(np.float32)
    dtype = gray.dtype
    if np.issubdtype(dtype, np.integer):
        if dtype == np.uint8:
            return np.clip(arr / 255.0, 0.0, 1.0)
        if dtype == np.uint16:
            denom = _uint16_denominator(arr, uint16_scale)
            return np.clip(arr / denom, 0.0, 1.0)
        mx = float(np.iinfo(dtype).max)
        return np.clip(arr / mx, 0.0, 1.0)
    return np.clip(arr, 0.0, 1.0)


def _scale_rgb_to_unit01(rgb: np.ndarray, uint16_scale: str) -> np.ndarray:
    """[..., 3] same dtype → float [0,1]."""
    arr = rgb.astype(np.float32)
    dtype = rgb.dtype
    if np.issubdtype(dtype, np.integer):
        if dtype == np.uint8:
            return np.clip(arr / 255.0, 0.0, 1.0)
        if dtype == np.uint16:
            denom = _uint16_denominator(arr, uint16_scale)
            return np.clip(arr / denom, 0.0, 1.0)
        mx = float(np.iinfo(dtype).max)
        return np.clip(arr / mx, 0.0, 1.0)
    return np.clip(arr, 0.0, 1.0)


def _movie_to_unit01_tyxc(movie: np.ndarray, uint16_scale: str) -> np.ndarray:
    """
    Return float32 [T, Y, X, 3] in [0, 1] with unified scaling across uint8 / uint16 / float.

    Percentile and full_range modes operate on this space so mixed bit depths are comparable.
    """
    if movie.ndim == 3:
        g = _scale_gray_to_unit01(movie, uint16_scale)
        return np.stack([g, g, g], axis=-1)
    if movie.ndim == 4:
        _t, _y, _x, c = movie.shape
        if c == 1:
            g = _scale_gray_to_unit01(movie[..., 0], uint16_scale)
            return np.stack([g, g, g], axis=-1)
        if c >= 3:
            return _scale_rgb_to_unit01(movie[..., :3], uint16_scale)
    raise ValueError(f"Unexpected movie shape: {movie.shape}")


def _flatten_rgb_for_stats(linear_tyxc: np.ndarray) -> np.ndarray:
    """1D float32 of all R,G,B values for percentile estimation."""
    return linear_tyxc.ravel()


def _subsample_1d(arr: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    n = arr.size
    if n <= max_samples:
        return arr
    idx = rng.choice(n, size=max_samples, replace=False)
    return arr[idx]


def _percentile_limits(
    values: np.ndarray,
    p_lo: float,
    p_hi: float,
) -> Tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [p_lo, p_hi]).astype(np.float64)
    lo_f, hi_f = float(lo), float(hi)
    if hi_f <= lo_f:
        hi_f = lo_f + 1.0
    return lo_f, hi_f


def _linear_tyxc_to_unit_range(
    linear: np.ndarray,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Linear stretch to [0, 1]."""
    if hi <= lo:
        return np.zeros_like(linear, dtype=np.float32)
    return np.clip((linear.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def compute_global_percentile_limits(
    paths: Sequence[str],
    p_lo: float,
    p_hi: float,
    max_samples_per_movie: int,
    rng: np.random.Generator,
    uint16_scale: str,
) -> Tuple[float, float]:
    chunks: List[np.ndarray] = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        mov = tifffile.imread(path)
        unit01 = _movie_to_unit01_tyxc(mov, uint16_scale)
        flat = _flatten_rgb_for_stats(unit01)
        flat = _subsample_1d(flat, max_samples_per_movie, rng)
        chunks.append(flat)
    if not chunks:
        return 0.0, 1.0
    pooled = np.concatenate(chunks)
    return _percentile_limits(pooled, p_lo, p_hi)


def compute_per_movie_percentile_limits(
    path: str,
    p_lo: float,
    p_hi: float,
    max_samples: int,
    rng: np.random.Generator,
    uint16_scale: str,
) -> Tuple[float, float]:
    mov = tifffile.imread(path)
    unit01 = _movie_to_unit01_tyxc(mov, uint16_scale)
    flat = _flatten_rgb_for_stats(unit01)
    flat = _subsample_1d(flat, max_samples, rng)
    return _percentile_limits(flat, p_lo, p_hi)


def load_path_to_sample_title(samples_yaml_path: str, species_folder_abs: str) -> Dict[str, str]:
    """
    Map absolute sample directory path -> sample key from the sample config.
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


def load_movie_to_unit_tyxc(
    path: str,
    intensity_mode: str,
    lo_hi: Optional[Tuple[float, float]] = None,
    per_movie_limits: Optional[List[Tuple[float, float]]] = None,
    index: int = 0,
    uint16_scale: str = "auto",
) -> np.ndarray:
    """
    Load TIFF and return float32 [T, Y, X, 3] in [0, 1] using the chosen intensity mapping.

    All modes first apply unified uint8/uint16→[0,1] scaling (``_movie_to_unit01_tyxc``), then
    optional percentile stretch (lo/hi are in **that** [0,1] space).
    """
    mov = tifffile.imread(path)
    unit01 = _movie_to_unit01_tyxc(mov, uint16_scale)
    if intensity_mode == "full_range":
        return unit01
    if intensity_mode == "global_percentile":
        if lo_hi is None:
            raise ValueError("global_percentile requires lo_hi")
        return _linear_tyxc_to_unit_range(unit01, lo_hi[0], lo_hi[1])
    if intensity_mode == "per_panel_percentile":
        if per_movie_limits is None or index < 0 or index >= len(per_movie_limits):
            raise ValueError("per_panel_percentile requires per_movie_limits and valid index")
        lo, hi = per_movie_limits[index]
        return _linear_tyxc_to_unit_range(unit01, lo, hi)
    raise ValueError(f"Unknown intensity_mode: {intensity_mode!r}")


def _render_label_rgba(
    text: str,
    fontsize_pt: float,
    dpi: int = 120,
) -> np.ndarray:
    """Render label to RGBA uint8 (h, w, 4), transparent background."""
    # Figure width scales with text length (rough heuristic).
    fig_w_in = max(1.2, min(12.0, 0.08 * len(text) + 0.4))
    fig_h_in = 0.35
    fig = Figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_facecolor((0, 0, 0, 0))
    fig.patch.set_alpha(0.0)
    ax.text(
        0.02,
        0.92,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize_pt,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", edgecolor="white", alpha=0.65),
    )
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    rgba = buf.reshape((h, w, 4))
    return rgba


def _scale_rgba_to_height(rgba: np.ndarray, target_h: int) -> np.ndarray:
    """Nearest-neighbor scale RGBA to target height (preserve aspect)."""
    if target_h <= 0 or rgba.shape[0] == target_h:
        return rgba

    h0, w0 = rgba.shape[:2]
    scale = target_h / float(h0)
    w1 = max(1, int(round(w0 * scale)))
    # Simple area-based downscale or nearest grid
    y_idx = (np.linspace(0, h0 - 1, target_h)).astype(int)
    x_idx = (np.linspace(0, w0 - 1, w1)).astype(int)
    return rgba[np.ix_(y_idx, x_idx)]


def _scale_rgba_uniform_nearest(rgba: np.ndarray, scale: float) -> np.ndarray:
    """Uniform scale RGBA by nearest-neighbor resampling (scale in (0, 1] typically)."""
    if scale >= 1.0 - 1e-9:
        return rgba
    h0, w0 = rgba.shape[:2]
    new_h = max(1, int(round(h0 * scale)))
    new_w = max(1, int(round(w0 * scale)))
    y_idx = np.linspace(0, h0 - 1, new_h).astype(int)
    x_idx = np.linspace(0, w0 - 1, new_w).astype(int)
    return rgba[np.ix_(y_idx, x_idx)]


def _scale_rgba_to_fit_box(rgba: np.ndarray, max_h: int, max_w: int) -> np.ndarray:
    """
    Uniformly scale RGBA down so it fits inside max_h × max_w (never upscale).
    Fixes long sample names on narrow movie columns (right-hand panels).
    """
    max_h = max(1, int(max_h))
    max_w = max(1, int(max_w))
    h0, w0 = rgba.shape[:2]
    if h0 <= max_h and w0 <= max_w:
        return rgba
    scale = min(max_h / float(h0), max_w / float(w0))
    return _scale_rgba_uniform_nearest(rgba, scale)


def _composite_label_top_left(
    frame_yxc: np.ndarray,
    label_rgba: np.ndarray,
    margin_px: int,
) -> np.ndarray:
    """
    Composite RGBA label onto float32 [Y, X, 3] in [0,1].
    label_rgba: uint8 [h, w, 4].
    """
    Y, X, C = frame_yxc.shape
    if C != 3:
        raise ValueError("Expected 3 channels for compositing")
    max_h = max(1, Y - 2 * margin_px)
    max_w = max(1, X - 2 * margin_px)
    label_rgba = _scale_rgba_to_fit_box(label_rgba, max_h, max_w)
    h, w = label_rgba.shape[0], label_rgba.shape[1]
    y0, x0 = margin_px, margin_px

    rgb = label_rgba[:, :, :3].astype(np.float32) / 255.0
    a = label_rgba[:, :, 3:4].astype(np.float32) / 255.0
    base = frame_yxc[y0 : y0 + h, x0 : x0 + w, :].copy()
    blended = (1.0 - a) * base + a * rgb
    out = frame_yxc.copy()
    out[y0 : y0 + h, x0 : x0 + w, :] = blended
    return out


def _float_tyxc_to_output(tyxc_f: np.ndarray, bit_depth: int) -> np.ndarray:
    """Convert float [T,Y,X,3] in [0,1] to uint8 or uint16."""
    x = np.clip(tyxc_f, 0.0, 1.0)
    if bit_depth == 8:
        return np.rint(x * 255.0).astype(np.uint8)
    if bit_depth == 16:
        return (x * 65535.0).astype(np.uint16)
    raise ValueError(f"Unsupported bit_depth: {bit_depth}")


def _tyxc_to_uint8_rgb(tyxc: np.ndarray) -> np.ndarray:
    """Convert TYXC uint8 or uint16 stack to contiguous uint8 RGB for video."""
    if tyxc.ndim != 4 or tyxc.shape[-1] < 3:
        raise ValueError(f"Expected TYXC with C>=3, got {tyxc.shape}")
    arr = tyxc[..., :3]
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if arr.dtype == np.uint16:
        x = arr.astype(np.float32) / 65535.0
        return np.rint(np.clip(x * 255.0, 0.0, 255.0)).astype(np.uint8)
    raise ValueError(f"Unsupported dtype for video: {arr.dtype}")


def write_h264_avi(
    tyxc: np.ndarray,
    out_avi: str,
    fps: float,
    *,
    ffmpeg_exe: str = "ffmpeg",
    crf: int = 15,
    preset: str = "veryslow",
) -> None:
    """
    Encode an RGB movie stack as H.264 in an AVI container via ffmpeg (raw RGB pipe).

    Uses libx264 with CRF (lower = higher quality; 15 is visually near-lossless for most content)
    and ``veryslow`` preset for best compression efficiency at that quality.
    """
    u8 = _tyxc_to_uint8_rgb(tyxc)
    t, h, w, c = u8.shape
    if c != 3:
        raise ValueError("Expected 3 channels")
    if fps <= 0:
        raise ValueError("fps must be positive")

    cmd = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        out_avi,
    ]
    # Single buffer avoids stdin pipe edge cases on some systems (large but typically OK).
    buf = b"".join(np.ascontiguousarray(u8[i]).tobytes() for i in range(t))
    r = subprocess.run(
        cmd,
        input=buf,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if r.returncode != 0:
        msg = r.stderr.decode("utf-8", errors="replace") if r.stderr else ""
        raise RuntimeError(
            f"ffmpeg failed (exit {r.returncode}) encoding {out_avi}. {msg}"
        )


def combine_movies(
    paths: List[str],
    labels: List[str],
    margin_px: int,
    label_height_px: int,
    fontsize_pt: float,
    intensity_mode: str,
    lo_hi: Optional[Tuple[float, float]],
    per_movie_limits: Optional[List[Tuple[float, float]]],
    uint16_scale: str,
) -> np.ndarray:
    if len(paths) != len(labels):
        raise ValueError("paths and labels length mismatch")
    tiles: List[np.ndarray] = []
    for idx, (p, lab) in enumerate(zip(paths, labels)):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing movie file: {p}")
        tyxc = load_movie_to_unit_tyxc(
            p,
            intensity_mode=intensity_mode,
            lo_hi=lo_hi,
            per_movie_limits=per_movie_limits,
            index=idx,
            uint16_scale=uint16_scale,
        )
        label_rgba = _render_label_rgba(lab, fontsize_pt=fontsize_pt)
        label_rgba = _scale_rgba_to_height(label_rgba, label_height_px)
        t, y, x, c = tyxc.shape
        out_frames = np.empty_like(tyxc)
        for fi in range(t):
            fr = tyxc[fi]
            out_frames[fi] = _composite_label_top_left(fr, label_rgba, margin_px)
        tiles.append(out_frames)

    min_t = min(t.shape[0] for t in tiles)
    max_h = max(t.shape[1] for t in tiles)
    padded: List[np.ndarray] = []
    for t in tiles:
        t = t[:min_t]
        h = t.shape[1]
        if h < max_h:
            pad_b = max_h - h
            t = np.pad(t, ((0, 0), (0, pad_b), (0, 0), (0, 0)), mode="constant", constant_values=0)
        padded.append(t)

    return np.concatenate(padded, axis=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine Cellularization_trimmed_delta.tif movies side by side with sample labels.",
    )
    parser.add_argument(
        "-s",
        "--subfolder",
        required=True,
        help="Species folder containing numeric sample subfolders (e.g. data/Hermetia)",
    )
    parser.add_argument(
        "--sample-list",
        required=True,
        help="Comma-separated sample folder names (e.g. 2,6,9,10)",
    )
    parser.add_argument(
        "--out-stem",
        default="Hermetia_trimmed_delta_movies_combined",
        help="Output filename stem (TIFF) under <subfolder>/results/",
    )
    parser.add_argument(
        "--samples-yaml",
        default=None,
        help=(
            "Path to sample config (single samples.yaml or species files in config/). "
            "Defaults to species-folder mode if present."
        ),
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=8,
        help="Margin in pixels for label placement from top-left.",
    )
    parser.add_argument(
        "--label-height",
        type=int,
        default=36,
        help="Approximate label bar height in pixels (after scaling).",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=11.0,
        help="Font size (pt) for label text in matplotlib.",
    )
    parser.add_argument(
        "--intensity-mode",
        choices=("full_range", "global_percentile", "per_panel_percentile"),
        default="full_range",
        help="How to map to [0,1] before concatenating. "
        "full_range: unified uint8/uint16→[0,1] only (no percentile stretch). "
        "global_percentile: one lo/hi in [0,1] from pooled samples. "
        "per_panel_percentile: separate lo/hi per sample.",
    )
    parser.add_argument(
        "--uint16-scale",
        choices=("auto", "12bit", "16bit"),
        default="auto",
        help="How to scale uint16 pixels to [0,1] before percentiles / full_range. "
        "auto: max≤4096 → ÷4095 (12-bit), else ÷65535. Use 12bit/16bit to force.",
    )
    parser.add_argument(
        "--output-bit-depth",
        type=int,
        choices=(8, 16),
        default=16,
        help="Output TIFF dtype: 8 or 16 bits per channel (default 16).",
    )
    parser.add_argument(
        "--percentile-low",
        type=float,
        default=0.5,
        help="Lower percentile for global_percentile / per_panel_percentile (0-100).",
    )
    parser.add_argument(
        "--percentile-high",
        type=float,
        default=99.5,
        help="Upper percentile for global_percentile / per_panel_percentile (0-100).",
    )
    parser.add_argument(
        "--max-samples-per-movie",
        type=int,
        default=4_000_000,
        help="Max pixels sampled per movie when estimating percentiles (memory/speed).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="RNG seed for subsampling pixels when estimating percentiles.",
    )
    parser.add_argument(
        "--no-export-avi",
        action="store_true",
        help="Do not write an H.264 AVI next to the TIFF (default: export both).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frame rate for the AVI (frames per second).",
    )
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="Path to ffmpeg executable (default: search PATH).",
    )
    parser.add_argument(
        "--h264-crf",
        type=int,
        default=15,
        help="libx264 CRF (0–51, lower = higher quality; 15 is very high quality).",
    )
    parser.add_argument(
        "--h264-preset",
        default="veryslow",
        help="libx264 preset (e.g. veryslow, slower, medium); veryslow = best quality per size.",
    )
    args = parser.parse_args()

    species_folder = os.path.abspath(args.subfolder)
    if not os.path.isdir(species_folder):
        raise SystemExit(f"Subfolder does not exist: {species_folder}")

    samples_yaml = (
        os.path.abspath(args.samples_yaml)
        if args.samples_yaml is not None
        else default_samples_config_path(_repo_root())
    )
    path_to_title = load_path_to_sample_title(samples_yaml, species_folder)

    sample_ids = _parse_sample_list(args.sample_list)
    paths: List[str] = []
    labels: List[str] = []

    for sid in sample_ids:
        try:
            folder = _natural_sample_folder(species_folder, sid)
        except FileNotFoundError as e:
            raise SystemExit(str(e)) from e

        movie_path = os.path.join(folder, "results", "Cellularization_trimmed_delta.tif")
        folder_abs = os.path.abspath(folder)
        title = path_to_title.get(folder_abs)
        if title is None:
            print(
                f"Warning: no sample key in samples config for path {folder_abs}; "
                f"using fallback label 'Sample_{sid}'"
            )
            title = f"Sample_{sid}"

        paths.append(movie_path)
        labels.append(title)

    for p in paths:
        if not os.path.isfile(p):
            raise SystemExit(f"Missing movie file: {p}")

    pl, ph = args.percentile_low, args.percentile_high
    if not (0.0 <= pl < ph <= 100.0):
        raise SystemExit(
            "Require 0 <= --percentile-low < --percentile-high <= 100"
        )

    rng = np.random.default_rng(args.rng_seed)
    lo_hi: Optional[Tuple[float, float]] = None
    per_movie_limits: Optional[List[Tuple[float, float]]] = None

    print(f"uint16 scale: {args.uint16_scale} (uint8 always ÷255)")

    if args.intensity_mode == "global_percentile":
        lo_hi = compute_global_percentile_limits(
            paths,
            pl,
            ph,
            args.max_samples_per_movie,
            rng,
            args.uint16_scale,
        )
        print(
            "Intensity (global_percentile), limits in unified [0,1] space: "
            f"p=[{pl}, {ph}] -> lo={lo_hi[0]:.6f}, hi={lo_hi[1]:.6f}"
        )
    elif args.intensity_mode == "per_panel_percentile":
        per_movie_limits = [
            compute_per_movie_percentile_limits(
                p,
                pl,
                ph,
                args.max_samples_per_movie,
                rng,
                args.uint16_scale,
            )
            for p in paths
        ]
        print(f"Intensity (per_panel_percentile): p=[{pl}, {ph}] (limits in [0,1])")
        for p, lim in zip(paths, per_movie_limits):
            sample_dir = os.path.dirname(os.path.dirname(p))
            print(f"  {sample_dir}: lo={lim[0]:.6f}, hi={lim[1]:.6f}")
    else:
        print(
            "Intensity (full_range): unified uint8/uint16→[0,1] only (no percentile stretch)"
        )

    combined_f = combine_movies(
        paths,
        labels,
        margin_px=args.margin,
        label_height_px=args.label_height,
        fontsize_pt=args.font_size,
        intensity_mode=args.intensity_mode,
        lo_hi=lo_hi,
        per_movie_limits=per_movie_limits,
        uint16_scale=args.uint16_scale,
    )
    combined_out = _float_tyxc_to_output(combined_f, args.output_bit_depth)

    out_dir = os.path.join(species_folder, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.out_stem}.tif")

    metadata = {
        "axes": "TYXC",
        "PhysicalSizeX": 1.0,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": 1.0,
        "PhysicalSizeYUnit": "µm",
    }
    with tifffile.TiffWriter(out_path, bigtiff=False, ome=True) as tif:
        tif.write(combined_out, metadata=metadata)

    print(f"Saved: {out_path}")
    print(f"  samples: {', '.join(sample_ids)}")
    print(f"  dtype: {combined_out.dtype}, output bit depth: {args.output_bit_depth}")
    print(f"  shape (T,Y,X,C): {combined_out.shape}")

    if not args.no_export_avi:
        ffmpeg_exe = args.ffmpeg or shutil.which("ffmpeg")
        if not ffmpeg_exe:
            print(
                "Warning: ffmpeg not found on PATH; skipping H.264 AVI export. "
                "Install ffmpeg or pass --ffmpeg /path/to/ffmpeg"
            )
        else:
            out_avi = os.path.join(out_dir, f"{args.out_stem}.avi")
            try:
                write_h264_avi(
                    combined_out,
                    out_avi,
                    args.fps,
                    ffmpeg_exe=ffmpeg_exe,
                    crf=args.h264_crf,
                    preset=args.h264_preset,
                )
                print(f"Saved (H.264 AVI): {out_avi}")
                print(
                    f"  ffmpeg: {ffmpeg_exe}, fps={args.fps}, crf={args.h264_crf}, preset={args.h264_preset}"
                )
            except Exception as e:
                raise SystemExit(f"AVI export failed: {e}") from e


if __name__ == "__main__":
    main()

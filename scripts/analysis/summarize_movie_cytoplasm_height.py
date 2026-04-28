#!/usr/bin/env python3
"""
Summarize per-movie cytoplasm height from per-x CSV files.

Scans an input root recursively for files named:
    track/cytoplasm_height_vs_x.csv

Writes one output CSV row per movie with:
    - movie_id
    - movie_path (relative to input root)
    - n_x_valid
    - mean_cytoplasm_height_px
    - mean_cytoplasm_height_um

Usage:
    python scripts/analysis/summarize_movie_cytoplasm_height.py \
        --input-root results/movie_cytoplasm_width/Hermetia/wt \
        --out-csv results/movie_cytoplasm_width/Hermetia/wt/cytoplasm_height_movie_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


CSV_NAME = "cytoplasm_height_vs_x.csv"
TRACK_DIR = "track"


@dataclass
class MovieSummary:
    group: str
    movie_id: str
    movie_path: str
    n_x_valid: int
    mean_cytoplasm_height_px: float
    mean_cytoplasm_height_um: Optional[float]


def _parse_float(value: Optional[str]) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _find_profile_csvs(input_root: str) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(input_root):
        if CSV_NAME in files and os.path.basename(root) == TRACK_DIR:
            out.append(os.path.join(root, CSV_NAME))
    return sorted(out)


def _build_group_label(group_prefix: str, rel_movie_path: str) -> str:
    parts = [p for p in rel_movie_path.split("/") if p]
    if not parts:
        return group_prefix
    # Support both:
    # - input root at condition level: dorsal/1 -> wt_dorsal (using group_prefix=wt)
    # - input root at species level: wt/dorsal/1 -> wt_dorsal (from path parts)
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}"
    if len(parts) >= 2:
        return f"{group_prefix}_{parts[0]}"
    return group_prefix


def _summarize_one(input_root: str, csv_path: str, group_prefix: str) -> Optional[MovieSummary]:
    heights_px: List[float] = []
    heights_um: List[float] = []

    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "cytoplasm_height_px" not in reader.fieldnames:
                print(f"Warning: missing column cytoplasm_height_px in {csv_path}")
                return None
            for row in reader:
                px = _parse_float(row.get("cytoplasm_height_px"))
                if np.isfinite(px):
                    heights_px.append(px)
                um = _parse_float(row.get("cytoplasm_height_um"))
                if np.isfinite(um):
                    heights_um.append(um)
    except OSError as exc:
        print(f"Warning: failed reading {csv_path}: {exc}")
        return None

    if not heights_px:
        print(f"Warning: no valid cytoplasm_height_px values in {csv_path}")
        return None

    track_dir = os.path.dirname(csv_path)
    movie_dir = os.path.dirname(track_dir)
    rel_movie_path = os.path.relpath(movie_dir, input_root)
    rel_movie_path = rel_movie_path.replace(os.sep, "/")
    movie_id = os.path.basename(movie_dir)
    group = _build_group_label(group_prefix=group_prefix, rel_movie_path=rel_movie_path)

    mean_um: Optional[float] = None
    if heights_um:
        mean_um = float(np.mean(np.asarray(heights_um, dtype=float)))

    return MovieSummary(
        group=group,
        movie_id=movie_id,
        movie_path=rel_movie_path,
        n_x_valid=len(heights_px),
        mean_cytoplasm_height_px=float(np.mean(np.asarray(heights_px, dtype=float))),
        mean_cytoplasm_height_um=mean_um,
    )


def summarize_movies(input_root: str, out_csv: str, group_prefix: str) -> int:
    csv_paths = _find_profile_csvs(input_root)
    if not csv_paths:
        raise SystemExit(
            f"No {TRACK_DIR}/{CSV_NAME} files found under input root: {input_root}"
        )

    rows: List[MovieSummary] = []
    for csv_path in csv_paths:
        summary = _summarize_one(
            input_root=input_root, csv_path=csv_path, group_prefix=group_prefix
        )
        if summary is not None:
            rows.append(summary)

    rows.sort(key=lambda r: r.movie_path)

    out_dir = os.path.dirname(os.path.abspath(out_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "group",
                "movie_id",
                "movie_path",
                "n_x_valid",
                "mean_cytoplasm_height_px",
                "mean_cytoplasm_height_um",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.group,
                    row.movie_id,
                    row.movie_path,
                    row.n_x_valid,
                    f"{row.mean_cytoplasm_height_px:.6g}",
                    (
                        f"{row.mean_cytoplasm_height_um:.6g}"
                        if row.mean_cytoplasm_height_um is not None
                        else ""
                    ),
                ]
            )

    print(f"Input root: {input_root}")
    print(f"Movies found: {len(csv_paths)}")
    print(f"Movies exported: {len(rows)}")
    print(f"Saved CSV: {out_csv}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize average cytoplasm height per movie."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory to recursively scan for track/cytoplasm_height_vs_x.csv.",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output summary CSV path (one row per movie).",
    )
    parser.add_argument(
        "--group-prefix",
        default=None,
        help=(
            "Fallback prefix for group labels. For paths like '<cond>/<view>/<id>', "
            "group is '<cond>_<view>' (e.g., wt_dorsal). For '<view>/<id>', "
            "group is '<group-prefix>_<view>'. Defaults to basename of --input-root."
        ),
    )
    args = parser.parse_args()

    input_root = os.path.abspath(args.input_root)
    if not os.path.isdir(input_root):
        raise SystemExit(f"Input root is not a directory: {input_root}")

    out_csv = os.path.abspath(args.out_csv)
    group_prefix = args.group_prefix or os.path.basename(input_root.rstrip(os.sep))
    n_movies = summarize_movies(
        input_root=input_root, out_csv=out_csv, group_prefix=group_prefix
    )
    if n_movies == 0:
        raise SystemExit("No valid movie rows were exported.")


if __name__ == "__main__":
    main()

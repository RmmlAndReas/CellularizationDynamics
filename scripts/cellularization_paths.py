#!/usr/bin/env python3
"""Shared path helpers for cellularization dynamics scripts."""

from __future__ import annotations

import os


def resolve_input_movie_path(data_dir: str) -> str:
    """
    Resolve the raw input movie path from a data directory.

    Prefer the canonical name ``Cellularization.tif``. If it does not exist,
    fall back to the best-looking ``*.tif/*.tiff`` in the folder.
    """
    canonical = os.path.join(data_dir, "Cellularization.tif")
    if os.path.exists(canonical):
        return canonical

    candidates = []
    for entry in os.scandir(data_dir):
        if not entry.is_file():
            continue
        lower = entry.name.lower()
        if not (lower.endswith(".tif") or lower.endswith(".tiff")):
            continue
        # Exclude likely derived outputs if they happen to be in the root.
        if lower.startswith("cellularization_trimmed"):
            continue
        if lower.startswith("kymograph"):
            continue
        if lower.endswith("_delta.tif") or lower.endswith("_delta.tiff"):
            continue
        candidates.append(entry.path)

    if not candidates:
        raise FileNotFoundError(
            f"No input movie found in {data_dir}. Expected 'Cellularization.tif' "
            "or any *.tif/*.tiff."
        )

    preferred = [p for p in candidates if "cellularization" in os.path.basename(p).lower()]
    pool = preferred if preferred else candidates
    best = max(pool, key=lambda p: os.path.getsize(p))
    print(f"Using input movie: {best}")
    return best

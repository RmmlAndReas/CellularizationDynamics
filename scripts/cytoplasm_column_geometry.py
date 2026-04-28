"""Shared 1D column geometry for cytoplasm masks (longest valid run along Y)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def longest_cytoplasm_run_along_y(
    col_bool: np.ndarray,
    min_run_length_px: int = 5,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Find apical/basal row indices and thickness for one vertical column.

    True = cytoplasm. Uses the same run-length logic as detect_cytoplasm_region:
    longest contiguous True run with length >= min_run_length_px.

    Parameters
    ----------
    col_bool : (depth,) bool
        One column along the image Y axis.
    min_run_length_px : int
        Ignore shorter runs.

    Returns
    -------
    apical_px, basal_px, height_px
        Float row indices and height (basal - apical), or (None, None, None) if no valid run.
    """
    col_bool = np.asarray(col_bool, dtype=bool)
    if col_bool.ndim != 1:
        raise ValueError(f"Expected 1D column, got shape {col_bool.shape}")

    depth = col_bool.shape[0]
    best_start: Optional[int] = None
    best_end: Optional[int] = None
    best_len = 0

    in_run = False
    run_start = 0

    for y in range(depth):
        if col_bool[y] and not in_run:
            in_run = True
            run_start = y
        elif not col_bool[y] and in_run:
            in_run = False
            run_end = y - 1
            run_len = run_end - run_start + 1
            if run_len >= min_run_length_px and run_len > best_len:
                best_len = run_len
                best_start = run_start
                best_end = run_end
    if in_run:
        run_end = depth - 1
        run_len = run_end - run_start + 1
        if run_len >= min_run_length_px and run_len > best_len:
            best_len = run_len
            best_start = run_start
            best_end = run_end

    if best_start is None or best_end is None:
        return None, None, None

    h = float(best_end - best_start)
    return float(best_start), float(best_end), h

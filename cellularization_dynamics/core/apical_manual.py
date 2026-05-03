"""
Manual apical detection: per-column apical row from a user-drawn polyline.

The user clicks a polyline on the raw (left) kymograph. We fit a smoothing
spline through the deduplicated clicks and evaluate it at every time column
to produce ``apical_px_by_col`` (same shape and semantics as the island path).

Columns outside ``[t_min, t_max]`` of the clicked polyline get NaN so the
existing downstream NaN-apical handling (shift = 0, excluded from ref_row)
applies unchanged.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.interpolate import UnivariateSpline


def _prepare_points(
    time_min_points: Iterable[float],
    depth_px_raw_points: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Sort by time and average duplicates at the same time_min value."""
    t = np.asarray(list(time_min_points), dtype=float).ravel()
    d = np.asarray(list(depth_px_raw_points), dtype=float).ravel()
    if t.size != d.size:
        raise ValueError("time and depth arrays must have the same length")
    mask = np.isfinite(t) & np.isfinite(d)
    t = t[mask]
    d = d[mask]
    if t.size == 0:
        return t, d
    order = np.argsort(t)
    t = t[order]
    d = d[order]
    unique_t, inverse = np.unique(t, return_inverse=True)
    if unique_t.size == t.size:
        return t, d
    # Average duplicates at the same time (same contract as fit_cellu_front_spline).
    avg = np.zeros_like(unique_t)
    counts = np.zeros_like(unique_t)
    np.add.at(avg, inverse, d)
    np.add.at(counts, inverse, 1.0)
    return unique_t, avg / counts


def apical_px_from_manual_polyline(
    time_min_points: Iterable[float],
    depth_px_raw_points: Iterable[float],
    *,
    num_timepoints: int,
    dt_min: float,
    sigma_um: float,
    px2micron: float,
) -> np.ndarray:
    """
    Spline-smooth the polyline and evaluate apical row per kymograph column.

    Parameters
    ----------
    time_min_points, depth_px_raw_points
        Clicked polyline in (minutes, raw depth px). Order and duplicates are
        handled internally (sorted, duplicate-time averaging).
    num_timepoints
        Number of kymograph columns (i.e. output length).
    dt_min
        Minutes per kymograph column (``movie_time_interval_sec / 60``).
    sigma_um
        Expected click noise in microns. Internally ``s = n * (sigma_um / px2micron) ** 2``.
    px2micron
        Sample scale; strictly positive.

    Returns
    -------
    np.ndarray
        Length ``num_timepoints`` array of apical rows in raw px, NaN where
        the spline would extrapolate (outside the clicked time range) or when
        there are fewer than 2 distinct-in-time clicks.
    """
    n_cols = int(num_timepoints)
    out = np.full(n_cols, np.nan, dtype=float)
    if n_cols <= 0:
        return out

    t, d = _prepare_points(time_min_points, depth_px_raw_points)
    if t.size < 2:
        return out

    scale = max(float(px2micron), 1e-12)
    sigma = max(float(sigma_um), 0.0)
    s_value = float(t.size) * (sigma / scale) ** 2
    k = int(min(3, t.size - 1))

    col_t = np.arange(n_cols, dtype=float) * max(float(dt_min), 1e-12)
    in_range = (col_t >= float(t[0])) & (col_t <= float(t[-1]))
    if not np.any(in_range):
        return out

    if k < 1:
        return out

    try:
        spline = UnivariateSpline(t, d, s=s_value, k=k)
        values = spline(col_t[in_range])
    except Exception:
        # Defensive fallback: linear interpolation if SciPy refuses the fit.
        values = np.interp(col_t[in_range], t, d)

    out[in_range] = np.asarray(values, dtype=float)
    return out

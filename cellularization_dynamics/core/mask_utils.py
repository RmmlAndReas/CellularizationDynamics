"""
Shared mask utilities for cytoplasm / apical border detection.

Two detection modes are supported:
  - ``island``: vertical center of selected connected components in
    ``YolkMask.tif`` > 0 (threshold-based).
  - ``manual``: per-column apical row from a user-drawn polyline, spline-smoothed
    (see ``apical_manual.apical_px_from_manual_polyline``).

This module owns the per-mode apical-from-mask computation (island) and the
shared ``apical_px -> alignment`` geometry used by both modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Collection, Literal

import numpy as np
from scipy import ndimage

ApicalMode = Literal["island", "manual"]

STRAIGHTEN_METADATA_VERSION = 1


@dataclass
class AlignmentResult:
    """Canonical per-column apical alignment used for straightening and export."""

    apical_px_by_col: np.ndarray
    shifts_by_col: np.ndarray
    ref_row: int
    crop_top_px: int
    mode: str
    mode_params: dict[str, Any] = field(default_factory=dict)


def alignment_from_apical_px(
    apical_px: np.ndarray,
    *,
    px2micron: float,
    mode: ApicalMode,
    mode_params: dict[str, Any] | None = None,
) -> AlignmentResult:
    """
    Shared ref_row / shifts / crop_top geometry from a per-column apical row.

    Used by both island and manual modes. Raises ``ValueError`` if no column has
    a finite apical (same contract as ``straighten``).
    """
    apical_px = np.asarray(apical_px, dtype=float).ravel()
    valid = ~np.isnan(apical_px)
    if not np.any(valid):
        raise ValueError("No valid apical border detected")

    num_timepoints = int(apical_px.size)
    margin_px = int(round(2.0 / max(float(px2micron), 1e-9)))
    ref_row = int(np.nanmin(apical_px[valid])) + margin_px
    shifts = np.zeros(num_timepoints, dtype=int)
    shifts[valid] = (ref_row - apical_px[valid]).astype(int)
    crop_top = max(0, ref_row - margin_px)

    return AlignmentResult(
        apical_px_by_col=apical_px,
        shifts_by_col=shifts,
        ref_row=int(ref_row),
        crop_top_px=int(crop_top),
        mode=str(mode),
        mode_params=dict(mode_params or {}),
    )


def build_alignment(
    mask_bool: np.ndarray,
    *,
    px2micron: float,
    mode: ApicalMode,
    island_labels: Collection[int] | None = None,
) -> AlignmentResult:
    """
    Island-mode alignment: compute apical_px from the yolk mask, then delegate
    to ``alignment_from_apical_px`` for the shared geometry.
    """
    if mode != "island":
        raise ValueError(
            f"build_alignment only supports mode='island'; got {mode!r}. "
            "Manual mode builds apical_px via apical_manual.apical_px_from_manual_polyline "
            "and calls alignment_from_apical_px directly."
        )
    apical_px, _labels_unused = compute_apical_column_positions(
        mask_bool,
        island_labels=island_labels,
    )
    mode_params: dict[str, Any] = {}
    if island_labels is not None:
        mode_params["island_labels"] = sorted(int(x) for x in island_labels)
    return alignment_from_apical_px(
        apical_px,
        px2micron=px2micron,
        mode=mode,
        mode_params=mode_params,
    )


def _apical_island_center(
    _mask_bool: np.ndarray,
    labels: np.ndarray,
    *,
    island_labels: Collection[int] | None,
) -> np.ndarray:
    _, num_timepoints = _mask_bool.shape
    apical_px = np.full(num_timepoints, np.nan, dtype=float)
    if not island_labels:
        return apical_px
    selected = np.asarray(list(island_labels), dtype=int)
    for t in range(num_timepoints):
        col_rows = np.flatnonzero(np.isin(labels[:, t], selected))
        if col_rows.size > 0:
            apical_px[t] = float(np.mean(col_rows))
    return apical_px


def compute_apical_column_positions(
    mask_bool: np.ndarray,
    *,
    island_labels: Collection[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-timepoint apical reference row and connected-component labels of the mask.

    ``mask_bool`` is True where the yolk/threshold region is masked (same as
    ``YolkMask.tif`` > 0). ``labels`` are the yolk island IDs from ``ndimage.label``.

    Uses the vertical center (mean row) of selected labels within each time column.
    Empty ``island_labels`` yields all-NaN apical.
    """
    labels, _ = ndimage.label(mask_bool.astype(np.uint8))
    apical_px = _apical_island_center(
        mask_bool, labels, island_labels=island_labels
    )
    return apical_px, labels


def serialize_apical_px_for_yaml(apical_px: np.ndarray) -> list[float | None]:
    """JSON/YAML-safe list: NaN becomes None."""
    out: list[float | None] = []
    for v in np.asarray(apical_px, dtype=float).ravel():
        if np.isnan(v):
            out.append(None)
        else:
            out.append(float(v))
    return out

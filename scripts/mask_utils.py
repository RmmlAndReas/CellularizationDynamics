"""
Shared mask utilities for cytoplasm/apical border detection.

Convention:
  YolkMask.tif stores 255 where the threshold was satisfied (blue overlay),
  0 elsewhere.  Cytoplasm is the *non-blue* (mask == 0) region.
  Within a column, y increases downward, so:
    run_start  → apical border  (smaller y, closer to top)
    run_end    → basal border   (larger y,  closer to bottom)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Collection, Literal, Tuple

import numpy as np
from scipy import ndimage

ApicalMode = Literal["longest_run", "island"]

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


def build_alignment(
    mask_bool: np.ndarray,
    *,
    px2micron: float,
    mode: ApicalMode,
    island_labels: Collection[int] | None = None,
    min_run_length_px: int = 5,
) -> AlignmentResult:
    """
    Compute ref_row, integer shifts, and crop_top from mask and apical mode.

    Raises ValueError if no column has a valid apical (same contract as straighten).
    """
    apical_px, _labels_unused = compute_apical_column_positions(
        mask_bool,
        mode=mode,
        island_labels=island_labels,
        min_run_length_px=min_run_length_px,
    )
    valid = ~np.isnan(apical_px)
    if not np.any(valid):
        raise ValueError("No valid apical border detected in YolkMask.tif")

    _depth, num_timepoints = mask_bool.shape
    margin_px = int(round(2.0 / max(float(px2micron), 1e-9)))
    ref_row = int(np.nanmin(apical_px[valid])) + margin_px
    shifts = np.zeros(num_timepoints, dtype=int)
    shifts[valid] = (ref_row - apical_px[valid]).astype(int)
    crop_top = max(0, ref_row - margin_px)

    mode_params: dict[str, Any] = {}
    if mode == "island" and island_labels is not None:
        mode_params["island_labels"] = sorted(int(x) for x in island_labels)

    return AlignmentResult(
        apical_px_by_col=apical_px,
        shifts_by_col=shifts,
        ref_row=int(ref_row),
        crop_top_px=int(crop_top),
        mode=str(mode),
        mode_params=mode_params,
    )


def _apical_longest_run(
    mask_bool: np.ndarray,
    _labels: np.ndarray,
    *,
    min_run_length_px: int,
) -> np.ndarray:
    _, num_timepoints = mask_bool.shape
    apical_px = np.full(num_timepoints, np.nan, dtype=float)
    for t in range(num_timepoints):
        best_start, _ = select_cytoplasm_run(
            mask_bool[:, t], min_run_length_px=min_run_length_px
        )
        if best_start is not None:
            apical_px[t] = float(best_start)
    return apical_px


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
    mode: ApicalMode,
    island_labels: Collection[int] | None = None,
    min_run_length_px: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-timepoint apical reference row and connected-component labels of the mask.

    ``mask_bool`` is True where the yolk/threshold region is masked (same as
    ``YolkMask.tif`` > 0). ``labels`` are the yolk island IDs from ``ndimage.label``.

    For ``mode == "island"``, uses the vertical center (mean row) of selected labels
    within each time column. Empty ``island_labels`` yields all-NaN apical.

    Internal modes dispatch via ``_apical_*`` helpers for extensibility.
    """
    labels, _ = ndimage.label(mask_bool.astype(np.uint8))

    if mode == "island":
        apical_px = _apical_island_center(
            mask_bool, labels, island_labels=island_labels
        )
    else:
        apical_px = _apical_longest_run(
            mask_bool, labels, min_run_length_px=min_run_length_px
        )

    return apical_px, labels


def select_cytoplasm_run(
    col: np.ndarray, min_run_length_px: int = 5
) -> Tuple[int | None, int | None]:
    """
    Find the apical (run_start) and basal (run_end) borders of the longest
    contiguous non-blue (mask == 0) segment in a single mask column.

    Parameters
    ----------
    col : 1-D boolean array
        One column of the binary mask (True = blue/masked, False = cytoplasm).
    min_run_length_px : int
        Minimum run length to consider.

    Returns
    -------
    (run_start, run_end) in pixel coordinates, or (None, None) if nothing found.
    """
    runs = []
    in_run = False
    run_start = 0
    depth = col.shape[0]

    for y in range(depth):
        is_cytoplasm = not bool(col[y])  # non-blue = cytoplasm
        if is_cytoplasm and not in_run:
            in_run = True
            run_start = y
        elif (not is_cytoplasm) and in_run:
            in_run = False
            run_end = y - 1
            run_len = run_end - run_start + 1
            if run_len >= min_run_length_px:
                runs.append((run_start, run_end, run_len))

    if in_run:
        run_end = depth - 1
        run_len = run_end - run_start + 1
        if run_len >= min_run_length_px:
            runs.append((run_start, run_end, run_len))

    if not runs:
        return None, None

    run_start, run_end, _ = max(runs, key=lambda r: r[2])
    return run_start, run_end


def serialize_apical_px_for_yaml(apical_px: np.ndarray) -> list[float | None]:
    """JSON/YAML-safe list: NaN becomes None."""
    out: list[float | None] = []
    for v in np.asarray(apical_px, dtype=float).ravel():
        if np.isnan(v):
            out.append(None)
        else:
            out.append(float(v))
    return out

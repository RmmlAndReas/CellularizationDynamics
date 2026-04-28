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

from typing import Tuple

import numpy as np


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
        is_cytoplasm = not bool(col[y])   # non-blue = cytoplasm
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

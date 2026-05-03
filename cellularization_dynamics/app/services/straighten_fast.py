from __future__ import annotations

import numpy as np


def straighten(kymo: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    depth = kymo.shape[0]
    rows = np.arange(depth)[:, None] - shifts[None, :]
    valid_rows = (rows >= 0) & (rows < depth)
    sampled = np.take_along_axis(kymo, np.clip(rows, 0, depth - 1), axis=0)
    return np.where(valid_rows, sampled, 0)

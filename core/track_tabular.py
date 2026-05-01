"""TSV I/O for per-sample sequential data under ``track/`` (atomic writes)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

APICAL_FRONT_TSV = "apical_front.tsv"
STRAIGHTENING_COLUMNS_TSV = "straightening_columns.tsv"


def _track_dir(work_dir: str | Path) -> Path:
    return Path(work_dir).resolve() / "track"


def apical_front_tsv_path(work_dir: str | Path) -> Path:
    return _track_dir(work_dir) / APICAL_FRONT_TSV


def straightening_columns_tsv_path(work_dir: str | Path) -> Path:
    return _track_dir(work_dir) / STRAIGHTENING_COLUMNS_TSV


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        suffix=".tsv.tmp", prefix=path.name + ".", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_apical_front_tsv(
    work_dir: str | Path,
    time_min: np.ndarray,
    depth_px: np.ndarray,
) -> Path:
    """Write ``time_min`` / ``depth_px`` (straightened kymograph space, minutes)."""
    path = apical_front_tsv_path(work_dir)
    lines = ["time_min\tdepth_px\n"]
    for t, d in zip(np.asarray(time_min, dtype=float).ravel(), np.asarray(depth_px, dtype=float).ravel()):
        lines.append(f"{float(t):.6f}\t{float(d):.6f}\n")
    _atomic_write_bytes(path, "".join(lines).encode("utf-8"))
    return path


def read_apical_front_tsv(work_dir: str | Path) -> tuple[np.ndarray, np.ndarray] | None:
    path = apical_front_tsv_path(work_dir)
    if not path.is_file():
        return None
    try:
        data = np.loadtxt(path, delimiter="\t", skiprows=1)
    except OSError:
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] < 2 or data.shape[1] < 2:
        return None
    time_min = data[:, 0].astype(float)
    depth_px = data[:, 1].astype(float)
    order = np.argsort(time_min)
    return time_min[order].copy(), depth_px[order].copy()


def write_straightening_columns_tsv(
    work_dir: str | Path,
    shifts: np.ndarray,
    apical_px: np.ndarray,
) -> Path:
    """One row per kymograph column: index, integer shift, apical row (NaN if invalid)."""
    path = straightening_columns_tsv_path(work_dir)
    sh = np.asarray(shifts, dtype=np.int64).ravel()
    ap = np.asarray(apical_px, dtype=float).ravel()
    n = max(sh.size, ap.size)
    if sh.size != n or ap.size != n:
        raise ValueError("shifts and apical_px must have the same length")
    col = np.arange(n, dtype=np.int64)
    header = "col_idx\tshift_px\tapical_px_raw\n"
    parts = [header]
    for i in range(n):
        a = ap[i]
        ap_str = "" if np.isnan(a) else f"{float(a):.6f}"
        parts.append(f"{int(col[i])}\t{int(sh[i])}\t{ap_str}\n")
    _atomic_write_bytes(path, "".join(parts).encode("utf-8"))
    return path


def read_straightening_columns_tsv(
    work_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(shifts int64, apical_px float with nan)`` or None if missing/invalid."""
    path = straightening_columns_tsv_path(work_dir)
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    rows: list[tuple[int, int, float]] = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) < 2:
            continue
        col_idx = int(parts[0])
        shift_px = int(parts[1])
        if len(parts) >= 3 and parts[2].strip() != "":
            ap_raw = float(parts[2])
        else:
            ap_raw = float("nan")
        rows.append((col_idx, shift_px, ap_raw))
    if len(rows) < 1:
        return None
    rows.sort(key=lambda r: r[0])
    shifts = np.array([r[1] for r in rows], dtype=np.int64)
    apical = np.array([r[2] for r in rows], dtype=float)
    return shifts, apical


def _front_points_array(raw: Any) -> np.ndarray | None:
    """Parse YAML-style front_points list; None on failure."""
    if not raw:
        return None
    rows: list[tuple[float, float]] = []
    try:
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                rows.append((float(item[0]), float(item[1])))
            elif isinstance(item, dict):
                t = item.get("time_min", item.get("Time"))
                d = item.get("depth_px", item.get("Depth"))
                if t is None or d is None:
                    return None
                rows.append((float(t), float(d)))
            else:
                return None
    except (TypeError, ValueError):
        return None
    if len(rows) < 2:
        return None
    return np.asarray(rows, dtype=float)


def externalize_apical_front_if_inline(work_dir: str | Path, state: dict[str, Any]) -> bool:
    """If ``apical_alignment.front_points`` is present, write TSV and remove key. Returns True if changed."""
    aa = state.get("apical_alignment")
    if not isinstance(aa, dict):
        return False
    fp = aa.get("front_points")
    if not fp:
        return False
    arr = _front_points_array(fp)
    if arr is None:
        return False
    if arr.shape[0] < 2:
        return False
    time_min = arr[:, 0]
    depth_px = arr[:, 1]
    write_apical_front_tsv(work_dir, time_min, depth_px)
    aa2 = dict(aa)
    aa2.pop("front_points", None)
    state["apical_alignment"] = aa2
    return True


def externalize_straightening_if_inline(work_dir: str | Path, state: dict[str, Any]) -> bool:
    """If straightening has inline shifts / apical_px_by_col, write TSV and remove keys."""
    st = state.get("straightening")
    if not isinstance(st, dict):
        return False
    sh_raw = st.get("shifts")
    ap_raw = st.get("apical_px_by_col")
    if sh_raw is None and ap_raw is None:
        return False
    if sh_raw is None or ap_raw is None:
        return False
    shifts = np.asarray(sh_raw, dtype=np.int64).ravel()
    ap_list = list(ap_raw)
    apical = np.array(
        [float("nan") if x is None else float(x) for x in ap_list],
        dtype=float,
    )
    if shifts.size != apical.size:
        return False
    write_straightening_columns_tsv(work_dir, shifts, apical)
    st2 = dict(st)
    st2.pop("shifts", None)
    st2.pop("apical_px_by_col", None)
    state["straightening"] = st2
    return True

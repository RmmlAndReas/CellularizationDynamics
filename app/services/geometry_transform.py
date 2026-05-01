"""
Raw vs straightened kymograph row conversions (shared by UI and exporters).

Convention matches straighten_kymograph / SampleState:
  y_straight = y_raw + shift[col]
  y_raw = y_straight - shift[col]

Depth relative to apical (µm) uses straightened row index and ref_row:
  depth_um = (y_straight - ref_row) * px2micron
"""

from __future__ import annotations


def straight_from_raw(y_raw: float, shift: float) -> float:
    return float(y_raw) + float(shift)


def raw_from_straight(y_straight: float, shift: float) -> float:
    return float(y_straight) - float(shift)


def um_from_straight(y_straight: float, ref_row: float, px2micron: float) -> float:
    return (float(y_straight) - float(ref_row)) * float(px2micron)


def straight_from_um(depth_um: float, ref_row: float, px2micron: float) -> float:
    px = max(float(px2micron), 1e-12)
    return float(ref_row) + float(depth_um) / px

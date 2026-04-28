#!/usr/bin/env python3
"""
Shared ROI utilities for ImageJ-based annotation scripts.

Currently provides a robust helper to extract X/Y coordinates from
polyline / spline / line ROIs, returning NumPy arrays sorted by X.
"""

import numpy as np


def extract_roi_xy(roi):
    """
    Extract ROI coordinates as NumPy arrays (x, y), sorted by x.

    The logic mirrors the robust coordinate extraction used in
    `annotate_cellu_front.py`, but without any assumptions about
    time calibration or downstream processing.

    Parameters
    ----------
    roi : ImageJ ROI object
        ROI containing a line / polyline / spline.

    Returns
    -------
    x_coords : np.ndarray
        X coordinates (float64), sorted ascending.
    y_coords : np.ndarray
        Y coordinates (float64), sorted by the same order as x_coords.
    """
    # Try different methods to get coordinates based on ROI type
    x_coords = None
    y_coords = None

    # Method 1: Try getPolyline() for polyline/segmented line ROIs
    if hasattr(roi, "getPolyline"):
        try:
            polyline = roi.getPolyline()
            if polyline is not None:
                if hasattr(polyline, "xpoints") and hasattr(polyline, "ypoints"):
                    x_pts = polyline.xpoints
                    y_pts = polyline.ypoints
                    if len(x_pts) > 0:
                        x_coords = np.array(x_pts)
                        y_coords = np.array(y_pts)
                        print(f"Extracted {len(x_coords)} points from polyline (getPolyline)")
                elif hasattr(polyline, "getXpoints") and hasattr(polyline, "getYpoints"):
                    x_pts = polyline.getXpoints()
                    y_pts = polyline.getYpoints()
                    if len(x_pts) > 0:
                        x_coords = np.array(x_pts)
                        y_coords = np.array(y_pts)
                        print(
                            f"Extracted {len(x_coords)} points from polyline "
                            "(getPolyline with getters)"
                        )
        except Exception as e:  # pragma: no cover - defensive logging
            print(f"getPolyline failed: {e}")

    # Method 2: Try getFloatPolyline() for float polyline ROIs
    if x_coords is None and hasattr(roi, "getFloatPolyline"):
        try:
            polyline = roi.getFloatPolyline()
            if polyline is not None:
                if hasattr(polyline, "xpoints") and hasattr(polyline, "ypoints"):
                    x_pts = polyline.xpoints
                    y_pts = polyline.ypoints
                    if len(x_pts) > 0:
                        x_coords = np.array(x_pts)
                        y_coords = np.array(y_pts)
                        print(
                            f"Extracted {len(x_coords)} points from float polyline "
                            "(getFloatPolyline)"
                        )
                elif hasattr(polyline, "getXpoints") and hasattr(polyline, "getYpoints"):
                    x_pts = polyline.getXpoints()
                    y_pts = polyline.getYpoints()
                    if len(x_pts) > 0:
                        x_coords = np.array(x_pts)
                        y_coords = np.array(y_pts)
                        print(
                            f"Extracted {len(x_coords)} points from float polyline "
                            "(getFloatPolyline with getters)"
                        )
        except Exception as e:  # pragma: no cover - defensive logging
            print(f"getFloatPolyline failed: {e}")

    # Method 2b: Try getFloatPolygon() which works for polylines and splines
    if x_coords is None and (
        roi.getTypeAsString() == "Polyline" or roi.getTypeAsString() == "Spline"
    ):
        try:
            if hasattr(roi, "getFloatPolygon"):
                polygon = roi.getFloatPolygon()
                if polygon is not None:
                    if hasattr(polygon, "xpoints") and hasattr(polygon, "ypoints"):
                        x_pts = polygon.xpoints
                        y_pts = polygon.ypoints
                        if len(x_pts) > 0:
                            x_coords = np.array(x_pts)
                            y_coords = np.array(y_pts)
                            print(
                                f"Extracted {len(x_coords)} points from polyline/spline "
                                "(getFloatPolygon)"
                            )
        except Exception as e:  # pragma: no cover - defensive logging
            print(f"getFloatPolygon failed: {e}")

    # Method 2c: Try accessing points directly from polyline/spline ROI
    if x_coords is None and (
        roi.getTypeAsString() == "Polyline" or roi.getTypeAsString() == "Spline"
    ):
        try:
            if hasattr(roi, "getXpoints") and hasattr(roi, "getYpoints"):
                x_pts = roi.getXpoints()
                y_pts = roi.getYpoints()
                if x_pts is not None and y_pts is not None and len(x_pts) > 0:
                    x_coords = np.array(x_pts)
                    y_coords = np.array(y_pts)
                    print(
                        f"Extracted {len(x_coords)} points from polyline/spline "
                        "(direct getXpoints/getYpoints)"
                    )
        except Exception as e:  # pragma: no cover - defensive logging
            print(f"Direct polyline point access failed: {e}")

    # Method 2d: Try getNCoordinates and getX/getY methods (works for splines)
    if x_coords is None and (
        roi.getTypeAsString() == "Polyline" or roi.getTypeAsString() == "Spline"
    ):
        try:
            if hasattr(roi, "getNCoordinates"):
                n_coords = roi.getNCoordinates()
                if n_coords > 0:
                    xs = []
                    ys = []
                    for i in range(n_coords):
                        xs.append(float(roi.getX(i)))
                        ys.append(float(roi.getY(i)))
                    x_coords = np.array(xs)
                    y_coords = np.array(ys)
                    print(
                        f"Extracted {len(x_coords)} points from polyline/spline "
                        "(getNCoordinates/getX/getY)"
                    )
        except Exception as e:  # pragma: no cover - defensive logging
            print(f"getNCoordinates method failed: {e}")

    # Method 2e: Try using getInterpolatedPolyline for polylines/splines
    if x_coords is None and (
        roi.getTypeAsString() == "Polyline" or roi.getTypeAsString() == "Spline"
    ):
        try:
            if hasattr(roi, "getInterpolatedPolyline"):
                polyline = roi.getInterpolatedPolyline(1.0)  # 1 pixel spacing
                if polyline is not None:
                    if hasattr(polyline, "xpoints") and hasattr(polyline, "ypoints"):
                        x_pts = polyline.xpoints
                        y_pts = polyline.ypoints
                        if len(x_pts) > 0:
                            x_coords = np.array(x_pts)
                            y_coords = np.array(y_pts)
                            print(
                                f"Extracted {len(x_coords)} points from interpolated "
                                "polyline"
                            )
        except Exception as e:  # pragma: no cover - defensive logging
            print(f"getInterpolatedPolyline failed: {e}")

    # Method 3: For simple line ROI, get start and end points
    if x_coords is None and hasattr(roi, "x1") and hasattr(roi, "x2"):
        try:
            x_coords = np.array([roi.x1, roi.x2])
            y_coords = np.array([roi.y1, roi.y2])
            print("Extracted 2 points from simple line")
        except Exception as e:  # pragma: no cover - defensive logging
            print(f"Simple line extraction failed: {e}")

    # Method 4: Fallback - get bounds
    if x_coords is None:
        try:
            bounds = roi.getBounds()
            x_coords = np.array([bounds.x, bounds.x + bounds.width])
            y_coords = np.array([bounds.y, bounds.y + bounds.height])
            print("Fallback: extracted 2 points from bounds")
        except Exception as e:
            raise ValueError(f"Could not extract coordinates from ROI: {e}")

    # Convert to NumPy arrays and sort by x
    x_coords = np.array(x_coords, dtype=float)
    y_coords = np.array(y_coords, dtype=float)

    sort_idx = np.argsort(x_coords)
    x_sorted = x_coords[sort_idx]
    y_sorted = y_coords[sort_idx]

    print(f"ROI coordinate extraction complete. N={len(x_sorted)}")
    print(f"  X range: {x_sorted.min():.2f} to {x_sorted.max():.2f}")
    print(f"  Y range: {y_sorted.min():.2f} to {y_sorted.max():.2f}")

    return x_sorted, y_sorted


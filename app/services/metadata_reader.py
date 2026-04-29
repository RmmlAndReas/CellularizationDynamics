from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import tifffile


def read_imagej_params(tiff_path: str | Path) -> dict:
    """Extract acquisition params from an ImageJ TIFF when available.

    Returns a partial dict with keys possibly including:
      - px2micron
      - movie_time_interval_sec
    """
    path = Path(tiff_path)
    out: dict = {}

    try:
        with tifffile.TiffFile(str(path)) as tf:
            md = tf.imagej_metadata or {}
            tags = tf.pages[0].tags if tf.pages else {}

            # Time interval: ImageJ finterval is commonly in seconds.
            finterval = md.get("finterval")
            if finterval is not None:
                try:
                    out["movie_time_interval_sec"] = float(finterval)
                except Exception:
                    pass

            # Pixel calibration from TIFF resolution when ImageJ unit is micron.
            unit = str(md.get("unit", "")).strip().lower()
            if unit in {"micron", "microns", "um", "µm", "micrometer", "micrometre"}:
                xres_tag = tags.get("XResolution")
                if xres_tag is not None:
                    val = xres_tag.value
                    try:
                        if isinstance(val, tuple) and len(val) == 2:
                            num, den = float(val[0]), float(val[1])
                            px_per_um = num / den if den != 0 else 0.0
                        else:
                            px_per_um = float(Fraction(val))
                        if px_per_um > 0:
                            out["px2micron"] = 1.0 / px_per_um
                    except Exception:
                        pass
    except Exception:
        return {}

    return out

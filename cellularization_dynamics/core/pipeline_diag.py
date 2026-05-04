"""
Optional diagnostics for ``generate_outputs`` (marker overlay, MP4, figure export).

**Enable** (logs to stderr; avoids broken stdout in some GUI/macOS setups)::

    export CELLULARIZATION_PIPELINE_DIAG=1
    # optional:
    export CELLULARIZATION_LOG_LEVEL=DEBUG

**Disable / revert**

- Unset ``CELLULARIZATION_PIPELINE_DIAG`` — code paths stay as no-ops (tiny overhead:
  one ``os.environ.get`` per call).
- To remove instrumentation entirely: delete this file, then remove the
  ``pipeline_diag`` import and every line that references ``pipeline_diag.`` in
  ``generate_outputs.py`` and ``analyze_worker.py``.
"""

from __future__ import annotations

import logging
import os
import sys

_ENV = "CELLULARIZATION_PIPELINE_DIAG"
_LEVEL_ENV = "CELLULARIZATION_LOG_LEVEL"
_configured = False


def enabled() -> bool:
    v = os.environ.get(_ENV, "").strip().lower()
    return v not in ("", "0", "false", "no", "off")


def configure() -> None:
    """Attach a stderr log handler when diagnostics are enabled (once)."""
    global _configured
    if not enabled() or _configured:
        return
    root = logging.getLogger()
    if root.handlers:
        _configured = True
        return
    level_name = os.environ.get(_LEVEL_ENV, "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = logging.INFO
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)
    _configured = True


def info(name: str, msg: str, *args) -> None:
    if not enabled():
        return
    configure()
    logging.getLogger(name).info(msg, *args)


def debug(name: str, msg: str, *args) -> None:
    if not enabled():
        return
    configure()
    logging.getLogger(name).debug(msg, *args)


def user_line(name: str, line: str) -> None:
    """Same text as ``print(line)`` when diagnostics are off; when on, log to stderr only."""
    if enabled():
        configure()
        logging.getLogger(name).info("%s", line)
    else:
        print(line)


def overlay_frame_tick(name: str, f: int, num_frames: int) -> None:
    """Every 50 frames: ``print`` progress, or stderr log when diagnostics are on."""
    if (f + 1) % 50 != 0:
        return
    if enabled():
        configure()
        logging.getLogger(name).info("Marker overlay frame %s/%s", f + 1, num_frames)
    else:
        print(f"  Frame {f + 1}/{num_frames}")

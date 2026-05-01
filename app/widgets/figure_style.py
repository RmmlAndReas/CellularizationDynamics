"""Figure styling helpers (e.g. match Qt window chrome)."""

from __future__ import annotations

from matplotlib.figure import Figure
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QWidget


def apply_window_background_to_figure(widget: QWidget, fig: Figure) -> None:
    """Set figure facecolor to the same gray (or theme color) as the main window."""
    c = widget.palette().color(QPalette.ColorRole.Window)
    fig.patch.set_facecolor(c.name())

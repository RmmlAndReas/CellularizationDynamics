"""
Import this module first in analysis scripts so the repo ``scripts/`` dir is on ``sys.path``.

Enables ``from samples_loader import ...`` and other helpers next to this package.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/analysis/<this file> -> parent is scripts/
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parent

_s = str(SCRIPTS_DIR)
if _s not in sys.path:
    sys.path.insert(0, _s)

"""Single import boundary between app and core computation modules.

Core modules use sibling imports (e.g. `from mask_utils import ...`) that
assume `core/` is on `sys.path`. We add `core/` here before importing them.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CORE_DIR = os.path.join(_REPO_ROOT, "core")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

from core.create_vertical_kymograph import (  # noqa: E402
    create_single_kymograph,
    horizontal_roi_from_averaging_pct,
)
from core.mask_utils import (  # noqa: E402
    AlignmentResult,
    build_alignment,
    compute_apical_column_positions,
    select_cytoplasm_run,
)
from core.cellu_threshold import (  # noqa: E402
    compute_cytoplasm_size_over_time,
    update_apical_height_in_config,
)
from core.fit_cellu_front_spline import fit_and_save  # noqa: E402
from core.export_geometry_timeseries import export_geometry_timeseries  # noqa: E402
from core.generate_outputs import generate_outputs  # noqa: E402
from core.straighten_kymograph import run as straighten_kymograph_run  # noqa: E402

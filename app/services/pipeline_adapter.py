"""Single import boundary between app and pipeline scripts.

The pipeline scripts use sibling imports (e.g. `from mask_utils import ...`)
that assume `scripts/` is on `sys.path`. We do not modify the pipeline files,
so instead we add `scripts/` to `sys.path` here before importing them.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from scripts.trim_movie import trim_movie  # noqa: E402
from scripts.create_vertical_kymograph import create_single_kymograph  # noqa: E402
from scripts.mask_utils import (  # noqa: E402
    AlignmentResult,
    build_alignment,
    compute_apical_column_positions,
    select_cytoplasm_run,
)
from scripts.cellu_front_annotation import (  # noqa: E402
    points_to_smooth_curve,
    save_tsv,
    save_compat_roi_placeholder,
)
from scripts.cellu_threshold import (  # noqa: E402
    compute_cytoplasm_size_over_time,
    update_apical_height_in_config,
)
from scripts.fit_cellu_front_spline import fit_and_save  # noqa: E402
from scripts.export_geometry_timeseries import export_geometry_timeseries  # noqa: E402
from scripts.generate_outputs import generate_outputs  # noqa: E402
from scripts.straighten_kymograph import run as straighten_kymograph_run  # noqa: E402

#!/usr/bin/env python3
"""
Write per-sample config.yaml for the movie cytoplasm width workflow (single timepoint).

Validates minimal fields: manual.px2micron, preprocessing.keep_every (default 1).
Does not require kymograph, spline, or manual.delta.

Usage:
    python scripts/init_movie_cytoplasm_config.py \\
        --data-dir /abs/path/to/data/sample \\
        --work-dir /abs/path/to/results/movie_cytoplasm_width/... \\
        --samples config/hermetia.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from samples_loader import default_samples_config_path, load_samples_config


def _normalize_data_path(repo_root: str, path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(repo_root, path))


def _find_sample_config(
    data_dir: str, samples_config: Dict[str, Any], repo_root: str
) -> Tuple[str, Dict[str, Any]]:
    data_abs = os.path.normpath(os.path.abspath(data_dir))
    for name, sample_data in samples_config.get("samples", {}).items():
        if not isinstance(sample_data, dict):
            continue
        p = sample_data.get("path")
        if p is None:
            continue
        cand = _normalize_data_path(repo_root, str(p))
        if cand == data_abs:
            cfg = {k: v for k, v in sample_data.items() if k != "path"}
            return name, cfg
    raise ValueError(
        f"Sample path {data_dir!r} not found under samples: in config "
        "(after normalizing relative paths to the repository root)."
    )


def _ensure_defaults(merged: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(merged)
    if "manual" not in out or not isinstance(out["manual"], dict):
        raise ValueError("Sample config must contain a 'manual' mapping")
    manual = dict(out["manual"])
    if "px2micron" not in manual:
        raise ValueError("manual.px2micron is required")
    float(manual["px2micron"])
    out["manual"] = manual

    if "preprocessing" not in out or not isinstance(out["preprocessing"], dict):
        out["preprocessing"] = {}
    pre = dict(out["preprocessing"])
    pre.setdefault("keep_every", 1)
    pre["keep_every"] = int(pre["keep_every"])
    out["preprocessing"] = pre

    return out


def get_config_hash(config_dict: Dict[str, Any]) -> str:
    """Hash of config excluding work-dir-specific noise."""
    payload = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()


def init_movie_cytoplasm_config(
    data_dir: str, work_dir: str, samples_config: Dict[str, Any], repo_root: str
) -> None:
    _, sample_cfg = _find_sample_config(data_dir, samples_config, repo_root)
    merged = _ensure_defaults(sample_cfg)

    config_path = os.path.join(work_dir, "config.yaml")
    hash_path = os.path.join(work_dir, ".config_hash_movie_cytoplasm")

    new_hash = get_config_hash(merged)

    content_changed = True
    if os.path.exists(hash_path):
        with open(hash_path, "r") as f:
            old_hash = f.read().strip()
        if old_hash == new_hash:
            content_changed = False

    if content_changed:
        os.makedirs(work_dir, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(merged, f, default_flow_style=False, sort_keys=False)
        with open(hash_path, "w") as f:
            f.write(new_hash)
        print(f"Wrote {config_path} (movie cytoplasm workflow)")
    else:
        if not os.path.exists(hash_path):
            os.makedirs(work_dir, exist_ok=True)
            with open(hash_path, "w") as f:
                f.write(new_hash)
        else:
            os.utime(hash_path, None)
        print(f"Config unchanged (hash match), touched {hash_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Absolute sample folder under data/")
    parser.add_argument("--work-dir", required=True, help="Pipeline work directory for this sample")
    parser.add_argument(
        "--samples",
        default=None,
        help="Samples YAML or config/ directory (default: config/)",
    )
    args = parser.parse_args()

    samples_path: Optional[str] = (
        os.path.abspath(args.samples) if args.samples else default_samples_config_path()
    )
    samples_config = load_samples_config(samples_path)

    init_movie_cytoplasm_config(
        os.path.abspath(args.data_dir),
        os.path.abspath(args.work_dir),
        samples_config,
        REPO_ROOT,
    )


if __name__ == "__main__":
    main()

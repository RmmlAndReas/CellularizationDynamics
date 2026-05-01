#!/usr/bin/env python3
"""
Initialize or update a sample's config.yaml from sample config.

This is the single source of truth loaded from sample config files.
Per-sample config.yaml files are auto-generated and preserve pipeline updates.
"""

import os
import sys
import yaml
import hashlib
import json
from pathlib import Path
from samples_loader import load_samples_config, default_samples_config_path


def deep_merge(base_dict, override_dict):
    """
    Recursively merge override_dict into base_dict.
    Override values take precedence.
    """
    result = base_dict.copy()
    
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def get_config_hash(config_dict):
    """
    Get a hash of the config dict (excluding pipeline-generated sections).
    
    This hash is used to detect if the source sample config actually changed,
    independent of pipeline-generated values that get added later.
    
    Parameters
    ----------
    config_dict : dict
        Configuration dictionary
    
    Returns
    -------
    str
        MD5 hash of the config (excluding pipeline sections)
    """
    # Create a copy without pipeline-generated sections for hashing
    # These sections are added/updated by pipeline scripts, not from source sample config
    hash_dict = {}
    for key, value in config_dict.items():
        if key not in ['apical_detection', 'cellularization_front']:
            hash_dict[key] = value
    
    # Sort keys for consistent hashing across runs
    config_str = json.dumps(hash_dict, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()


def _validate_required_config(config_dict, sample_path):
    """
    Validate that required sections and fields are present in final merged config.

    This fails fast during init_config so downstream rules don't crash later.
    """
    missing = []

    required_sections = ["manual", "kymograph", "spline"]
    for section in required_sections:
        if section not in config_dict or not isinstance(config_dict.get(section), dict):
            missing.append(section)

    manual = config_dict.get("manual", {})
    if "px2micron" not in manual:
        missing.append("manual.px2micron")
    if "movie_time_interval_sec" not in manual:
        missing.append("manual.movie_time_interval_sec")

    kymograph = config_dict.get("kymograph", {})
    if "time_interval_sec" not in kymograph:
        missing.append("kymograph.time_interval_sec")

    spline = config_dict.get("spline", {})
    if "smoothing" not in spline:
        missing.append("spline.smoothing")
    if "degree" not in spline:
        missing.append("spline.degree")

    if missing:
        raise ValueError(
            "Missing required config fields for sample "
            f"'{sample_path}': {', '.join(missing)}. "
            "Add them to the samples config before rerunning."
        )


def _normalize_minimal_sample_config(sample_config):
    """
    Accept either:
      - legacy nested schema (manual/preprocessing/spline/kymograph), or
      - minimal flat schema and convert to nested runtime config.

    Minimal flat keys:
      px2micron, movie_time_interval_sec (optional: smoothing, degree)
    """
    if not isinstance(sample_config, dict):
        return sample_config

    has_nested = any(
        k in sample_config for k in ("manual", "preprocessing", "spline", "kymograph")
    )
    if has_nested:
        return sample_config

    required_flat = ["px2micron", "movie_time_interval_sec"]
    missing_flat = [k for k in required_flat if k not in sample_config]
    if missing_flat:
        raise ValueError(
            "Missing required minimal config fields: "
            + ", ".join(missing_flat)
            + ". Expected flat keys: px2micron, movie_time_interval_sec "
            "(optional: smoothing, degree)."
        )

    return {
        "manual": {
            "px2micron": float(sample_config["px2micron"]),
            "movie_time_interval_sec": float(sample_config["movie_time_interval_sec"]),
        },
        "spline": {
            "smoothing": float(sample_config.get("smoothing", 0.0)),
            "degree": int(sample_config.get("degree", 3)),
        },
    }


def init_sample_config(data_dir, work_dir, samples_config):
    """
    Initialize or update a sample's config.yaml from sample config.
    
    Parameters:
    -----------
    data_dir : str
        Path to raw sample folder (e.g., "data/Hermetia/wt/dorsal/1")
    work_dir : str
        Path to working folder where config.yaml and hash are stored.
    samples_config : dict
        Loaded sample config content
    """
    # Find sample config from path
    sample_config = None
    sample_name = None
    
    data_dir_abs = os.path.abspath(data_dir)
    for name, sample_data in samples_config.get('samples', {}).items():
        sample_path = sample_data.get("path")
        sample_path_abs = os.path.abspath(sample_path) if sample_path is not None else None
        if sample_path_abs == data_dir_abs:
            sample_name = name
            # Get all config except 'path'
            sample_config = {k: v for k, v in sample_data.items() if k != 'path'}
            break
    
    if sample_config is None:
        raise ValueError(f"Sample path '{data_dir}' not found in sample config")
    
    # Support a minimal flat schema and normalize to nested runtime structure.
    sample_config = _normalize_minimal_sample_config(sample_config)

    # Start with config from source sample config
    merged_config = sample_config.copy()
    
    # Auto-calculate time_interval_sec from movie frame interval (one kymograph column per frame).
    if "manual" in merged_config:
        movie_time_interval_sec = merged_config["manual"].get("movie_time_interval_sec", 10)
        calculated_time_interval_sec = movie_time_interval_sec

        if "kymograph" not in merged_config:
            merged_config["kymograph"] = {}

        if "time_interval_sec" not in merged_config.get("kymograph", {}):
            merged_config["kymograph"]["time_interval_sec"] = calculated_time_interval_sec
            print(
                f"  Auto-calculated time_interval_sec: {calculated_time_interval_sec} sec "
                f"(= movie_time_interval_sec)"
            )
        else:
            manual_value = merged_config["kymograph"]["time_interval_sec"]
            if manual_value != calculated_time_interval_sec:
                print(
                    f"  WARNING: Manual time_interval_sec ({manual_value}) differs from "
                    f"calculated value ({calculated_time_interval_sec} = movie_time_interval_sec)"
                )

    prep = merged_config.get("preprocessing")
    if isinstance(prep, dict):
        prep.pop("keep_every", None)
        if not prep:
            merged_config.pop("preprocessing", None)
    
    # Load existing config if it exists (to preserve pipeline-generated values)
    config_path = os.path.join(work_dir, 'config.yaml')
    existing_config = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            existing_config = yaml.safe_load(f) or {}
        
        # Preserve pipeline-generated sections (these get updated by scripts)
        for section in ['apical_detection', 'cellularization_front']:
            if section in existing_config:
                # Merge: existing (pipeline-generated) takes precedence
                if section in merged_config:
                    merged_config[section] = deep_merge(
                        merged_config[section],
                        existing_config[section]
                    )
                else:
                    merged_config[section] = existing_config[section]
    
    # Validate mandatory fields early to fail fast.
    _validate_required_config(merged_config, data_dir)
    
    # Calculate hash of source config (excluding pipeline-generated sections)
    # This hash tracks whether the actual config content changed, not just file timestamps
    new_hash = get_config_hash(merged_config)
    
    # Check hash file to see if config actually changed
    hash_path = os.path.join(work_dir, '.config_hash')
    content_changed = True
    
    if os.path.exists(hash_path):
        with open(hash_path, 'r') as f:
            old_hash = f.read().strip()
        if old_hash == new_hash:
            content_changed = False
    
    # Write merged config and update hash only if content changed
    if content_changed:
        os.makedirs(work_dir, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)
        # Update hash file to track this version
        with open(hash_path, 'w') as f:
            f.write(new_hash)
        print(f"Updated config.yaml for {work_dir} (from sample config for {data_dir})")
        if 'apical_detection' in merged_config:
            if 'apical_height_px' in merged_config.get('apical_detection', {}):
                print(f"  Preserved pipeline-generated values")
    else:
        # Content unchanged - update hash file timestamp but don't write config.yaml
        # This ensures Snakemake knows the rule completed, but config.yaml timestamp
        # doesn't change, so downstream rules won't rerun
        if not os.path.exists(hash_path):
            os.makedirs(work_dir, exist_ok=True)
            with open(hash_path, 'w') as f:
                f.write(new_hash)
        else:
            # Touch hash file to update timestamp (indicates rule ran successfully)
            os.utime(hash_path, None)
        print(
            f"Config.yaml for {work_dir} unchanged "
            "(hash match, skipping write to prevent unnecessary reruns)"
        )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Initialize sample config.yaml from sample config (single source of truth)'
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Raw sample folder path from config[samples][*].path",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help="Working output folder where config.yaml is written",
    )
    parser.add_argument(
        '--samples',
        type=str,
        default=None,
        help=(
            "Path to sample config source. Can be either a single file with "
            "'samples:' or a directory containing species YAMLs. "
            "Default: config/ (fallback: config/samples.yaml)."
        ),
    )
    
    args = parser.parse_args()
    
    # Load samples config
    samples_config = load_samples_config(
        os.path.abspath(args.samples) if args.samples else default_samples_config_path()
    )
    
    # Initialize config
    init_sample_config(args.data_dir, args.work_dir, samples_config)


if __name__ == '__main__':
    main()

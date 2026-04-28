#!/usr/bin/env python3
"""
Helpers for loading sample configuration from:
  - species YAML files in `config/*.yaml` (default, excluding `samples.yaml`), or
  - a single YAML file with top-level `samples:`.
"""

from __future__ import annotations

import os
import re
from glob import glob
from typing import Dict, List, Optional, Tuple

import yaml


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_yaml(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def default_samples_config_path(repo_root: Optional[str] = None) -> str:
    root = repo_root or _repo_root()
    config_dir = os.path.join(root, "config")
    if os.path.isdir(config_dir):
        return config_dir
    return os.path.join(root, "config", "samples.yaml")


def _deep_merge(base_dict: dict, override_dict: dict) -> dict:
    """Recursively merge override_dict into base_dict."""
    result = dict(base_dict)
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _to_repo_relative_path(path: str) -> str:
    """Convert path to repo-relative POSIX-like path when possible."""
    repo = _repo_root()
    abs_path = os.path.abspath(path)
    rel = os.path.relpath(abs_path, repo)
    return rel.replace(os.sep, "/")


def _sanitize_key_fragment(value: str) -> str:
    value = value.replace(os.sep, "-")
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return value or "sample"


def _normalize_samples_config(cfg: dict, source_path: str) -> Dict[str, dict]:
    """
    Normalize config into a flat {sample_key: sample_cfg} mapping.

    Supported schemas:
      1) Legacy/current:
           samples:
             sample_key:
               path: ...
               ...
      2) Experiment-level defaults:
           experiments:
             experiment_key:
               defaults:
                 ...
               samples:
                 sample_key:
                   path: ...
                   ...
      3) Experiment-level discovery (preferred key: samples):
           experiments:
             experiment_key:
               defaults:
                 ...
               samples: data/Hermetia/centrifugation/259-1min-2min
               # or expanded form:
               samples:
                 root: data/Hermetia/centrifugation/259-1min-2min
                 glob: "*/*"
                 contains_file_glob: "*.tif"
                 key_prefix: hermetia259
                 key_from: relative  # relative | basename
               # legacy alias still supported:
               discover: data/Hermetia/centrifugation/259-1min-2min
    """
    normalized: Dict[str, dict] = {}

    # Legacy/current top-level samples.
    part_samples = cfg.get("samples", {})
    if part_samples:
        if not isinstance(part_samples, dict):
            raise ValueError(f"'samples' must be a mapping in: {source_path}")
        for key, value in part_samples.items():
            if not isinstance(value, dict):
                raise ValueError(f"Sample '{key}' must be a mapping in: {source_path}")
            normalized[key] = value

    # New experiments schema.
    experiments = cfg.get("experiments", {})
    if experiments:
        if not isinstance(experiments, dict):
            raise ValueError(f"'experiments' must be a mapping in: {source_path}")
        for exp_name, exp_cfg in experiments.items():
            if not isinstance(exp_cfg, dict):
                raise ValueError(
                    f"Experiment '{exp_name}' must be a mapping in: {source_path}"
                )
            defaults = exp_cfg.get("defaults", {})
            if not isinstance(defaults, dict):
                raise ValueError(
                    f"Experiment '{exp_name}'.defaults must be a mapping in: {source_path}"
                )

            # Optional auto-discovery of sample folders from a root + glob.
            # Preferred location is experiments.<name>.samples as string/mapping.
            # Backward-compatible alias: experiments.<name>.discover.
            samples_field = exp_cfg.get("samples", {})
            discover_cfg = exp_cfg.get("discover", {})
            explicit_samples_map: Dict[str, dict] = {}

            if isinstance(samples_field, dict):
                # Detect discovery-style dict vs explicit samples map.
                discovery_keys = {
                    "root",
                    "glob",
                    "contains_file_glob",
                    "key_prefix",
                    "key_from",
                }
                if any(k in samples_field for k in discovery_keys):
                    discover_cfg = samples_field
                else:
                    explicit_samples_map = samples_field
            elif isinstance(samples_field, str):
                discover_cfg = samples_field
            elif samples_field not in ({}, None):
                raise ValueError(
                    f"Experiment '{exp_name}'.samples must be a mapping or string in: {source_path}"
                )

            # If both are provided, explicit discover key wins for compatibility.
            if discover_cfg:
                # Short form: discover: "path/to/root"
                if isinstance(discover_cfg, str):
                    discover_cfg = {"root": discover_cfg}
                elif not isinstance(discover_cfg, dict):
                    raise ValueError(
                        f"Experiment '{exp_name}'.discover must be a string or mapping in: {source_path}"
                    )

                root = discover_cfg.get("root")
                if not isinstance(root, str) or not root.strip():
                    raise ValueError(
                        f"Experiment '{exp_name}'.discover.root must be a non-empty string in: {source_path}"
                    )

                # Sensible defaults for folder-only setup:
                # - discover two-level sample dirs (e.g. lateral/1, dorsal/2)
                # - only keep dirs that contain TIFF movies
                search_glob = discover_cfg.get("glob", "*/*")
                if not isinstance(search_glob, str):
                    raise ValueError(
                        f"Experiment '{exp_name}'.discover.glob must be a string in: {source_path}"
                    )

                contains_file_glob = discover_cfg.get("contains_file_glob", "*.tif")
                if contains_file_glob is not None and not isinstance(contains_file_glob, str):
                    raise ValueError(
                        f"Experiment '{exp_name}'.discover.contains_file_glob must be a string in: {source_path}"
                    )

                key_prefix = discover_cfg.get("key_prefix", exp_name)
                if not isinstance(key_prefix, str) or not key_prefix.strip():
                    raise ValueError(
                        f"Experiment '{exp_name}'.discover.key_prefix must be a non-empty string in: {source_path}"
                    )

                key_from = discover_cfg.get("key_from", "relative")
                if key_from not in ("relative", "basename"):
                    raise ValueError(
                        f"Experiment '{exp_name}'.discover.key_from must be 'relative' or 'basename' in: {source_path}"
                    )

                root_abs = root if os.path.isabs(root) else os.path.join(_repo_root(), root)
                search_pattern = os.path.join(root_abs, search_glob)
                discovered_dirs = sorted(
                    [p for p in glob(search_pattern) if os.path.isdir(p)]
                )

                for discovered_path in discovered_dirs:
                    if contains_file_glob:
                        must_have_pattern = os.path.join(discovered_path, contains_file_glob)
                        if not glob(must_have_pattern):
                            continue

                    sample_rel_path = _to_repo_relative_path(discovered_path)
                    if key_from == "basename":
                        key_fragment = _sanitize_key_fragment(os.path.basename(discovered_path))
                    else:
                        key_fragment = _sanitize_key_fragment(
                            os.path.relpath(discovered_path, root_abs)
                        )
                    sample_key = f"{_sanitize_key_fragment(key_prefix)}-{key_fragment}"

                    discovered_cfg = _deep_merge(defaults, {"path": sample_rel_path})
                    if sample_key in normalized:
                        raise ValueError(
                            f"Duplicate sample key '{sample_key}' discovered in {source_path}"
                        )
                    normalized[sample_key] = discovered_cfg

            for sample_key, sample_cfg in explicit_samples_map.items():
                if not isinstance(sample_cfg, dict):
                    raise ValueError(
                        f"Experiment '{exp_name}' sample '{sample_key}' must be a mapping in: {source_path}"
                    )
                merged_sample_cfg = _deep_merge(defaults, sample_cfg)
                if sample_key in normalized:
                    # Allow explicit experiment samples to override discovered samples.
                    normalized[sample_key] = _deep_merge(normalized[sample_key], merged_sample_cfg)
                else:
                    normalized[sample_key] = merged_sample_cfg

    return normalized


def load_samples_config_with_sources(config_path: Optional[str] = None) -> Tuple[Dict, List[str]]:
    """
    Load and return a merged samples config plus source file paths.

    Returns:
        ({"samples": {...}}, [source_file_paths_used_for_loading])
    """
    path = os.path.abspath(config_path or default_samples_config_path())

    # Directory mode: merge all species files from config/*.yaml.
    if os.path.isdir(path):
        species_files = sorted(
            [
                os.path.join(path, name)
                for name in os.listdir(path)
                if (name.endswith(".yaml") or name.endswith(".yml"))
                and name != "samples.yaml"
            ]
        )
        if not species_files:
            raise FileNotFoundError(f"No species YAML files found in: {path}")
        merged: Dict[str, dict] = {}
        for part_path in species_files:
            part_cfg = _read_yaml(part_path)
            part_samples = _normalize_samples_config(part_cfg, part_path)
            for key, value in part_samples.items():
                if key in merged:
                    raise ValueError(f"Duplicate sample key across files: {key}")
                merged[key] = value
        return {"samples": merged}, species_files

    # File mode: allow direct single-file override.
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Samples config not found: {path}")
    cfg = _read_yaml(path)
    samples = _normalize_samples_config(cfg, path)
    return {"samples": samples}, [path]


def load_samples_config(config_path: Optional[str] = None) -> Dict:
    cfg, _ = load_samples_config_with_sources(config_path=config_path)
    return cfg

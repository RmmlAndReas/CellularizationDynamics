#!/usr/bin/env python3
"""
Register the acquisition movie path in unified config.yaml (schema v2).

No longer copies the stack to Cellularization_trimmed.tif — use acquisition.source_movie.

Usage:
    python trim_movie.py --work-dir <work_dir> [--data-dir <data_dir>]
"""

import argparse
import os
import sys

import yaml

from cellularization_paths import resolve_input_movie_path


def assert_config_readable(config_path):
    if not os.path.exists(config_path):
        raise ValueError(f"config.yaml not found at: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError("config.yaml is empty")
    except Exception as e:
        raise ValueError(f"Error reading config.yaml: {e}") from e


def trim_movie(work_dir, data_dir):
    """
    Set acquisition.source_movie to the resolved TIFF under data_dir.
    """
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)
    from work_state import set_source_movie

    movie_path = resolve_input_movie_path(data_dir)
    print(f"\nRegistering acquisition movie: {movie_path}")
    set_source_movie(work_dir, movie_path)
    print(f"Updated {os.path.join(work_dir, 'config.yaml')} (acquisition.source_movie)")


def main():
    parser = argparse.ArgumentParser(
        description="Register source movie path in config.yaml (no TIFF copy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Raw data directory (default: work-dir)",
    )
    args = parser.parse_args()
    if not os.path.isdir(args.work_dir):
        raise ValueError(f"Work directory does not exist: {args.work_dir}")
    data_dir = args.data_dir if args.data_dir else args.work_dir
    if not os.path.isdir(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
    config_path = os.path.join(args.work_dir, "config.yaml")
    assert_config_readable(config_path)
    print("\n" + "=" * 60)
    print("Register movie path")
    print("=" * 60)
    trim_movie(args.work_dir, data_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()

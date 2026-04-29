from __future__ import annotations

from pathlib import Path
import yaml


def build_runtime_config(params: dict) -> dict:
    px2micron = float(params["px2micron"])
    movie_time_interval_sec = float(params["movie_time_interval_sec"])
    keep_every = int(params["keep_every"])
    smoothing = float(params.get("smoothing", 0.0))
    degree = int(params.get("degree", 1))
    time_interval_sec = movie_time_interval_sec * keep_every

    cfg = {
        "manual": {
            "px2micron": px2micron,
            "movie_time_interval_sec": movie_time_interval_sec,
        },
        "preprocessing": {
            "keep_every": keep_every,
        },
        "spline": {
            "smoothing": smoothing,
            "degree": degree,
        },
        "kymograph": {
            "time_interval_sec": float(time_interval_sec),
        },
    }
    return cfg


def load_or_create_config(work_dir: Path, params: dict) -> dict:
    config_path = work_dir / "config.yaml"
    cfg = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    runtime = build_runtime_config(params)
    for section, section_values in runtime.items():
        if isinstance(section_values, dict):
            cfg.setdefault(section, {})
            cfg[section].update(section_values)
        else:
            cfg[section] = section_values

    # Remove deprecated UI options from runtime config to keep this simple.
    cfg.pop("time_window", None)
    cfg.pop("visualization", None)

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    return cfg


def save_config(work_dir: Path, config: dict) -> None:
    with (work_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

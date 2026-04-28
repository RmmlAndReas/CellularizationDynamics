#!/usr/bin/env python3
"""
Export per-movie maximum cellularization-line height by group.

Scans a root folder containing group subfolders (e.g., dorsal/ventral/lateral),
finds ``track/geometry_timeseries.csv`` for each movie, and writes one CSV row
per movie with:
    - group
    - movie_id
    - movie_path
    - n_points
    - max_cellu_height_um
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


DEFAULT_GROUPS: Tuple[str, ...] = ("dorsal", "ventral", "lateral")
def _geometry_csv_candidates(movie_dir: str) -> List[str]:
    """
    Candidate geometry CSV paths for a movie directory.

    Primary location is inside the provided directory (`movie_dir/track`).
    If `movie_dir` is under a `data/` tree, also try the mirrored
    `results/cellularization_dynamics/...` location.
    """
    candidates = [os.path.join(movie_dir, "track", "geometry_timeseries.csv")]

    parts = os.path.abspath(movie_dir).split(os.sep)
    try:
        data_idx = parts.index("data")
    except ValueError:
        return candidates

    repo_root = os.sep.join(parts[:data_idx]) or os.sep
    rel_under_data = os.path.relpath(os.path.abspath(movie_dir), os.path.join(repo_root, "data"))
    mirrored = os.path.join(
        repo_root,
        "results",
        "cellularization_dynamics",
        rel_under_data,
        "track",
        "geometry_timeseries.csv",
    )
    if mirrored not in candidates:
        candidates.append(mirrored)
    return candidates




@dataclass
class MovieSummary:
    group: str
    movie_id: str
    movie_path: str
    experiment: str
    n_points: int
    max_cellu_height_um: float


@dataclass
class DatasetSpec:
    name: str
    root_folder: str
    groups: List[str]
    group_alias: Dict[str, str]


def parse_float(value: Optional[str]) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def list_movie_dirs(group_folder: str) -> List[str]:
    if not os.path.isdir(group_folder):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(group_folder), key=lambda x: (not x.isdigit(), x)):
        path = os.path.join(group_folder, name)
        if os.path.isdir(path):
            out.append(path)
    return out


def infer_experiment_label(movie_dir: str) -> str:
    try:
        names = sorted(os.listdir(movie_dir))
    except OSError:
        return "unknown"
    czi_like = [n for n in names if ".czi" in n.lower()]
    if not czi_like:
        return "unknown"
    source = czi_like[0]

    match = re.search(r"Experiment[-_ ]?(\d+)", source, flags=re.IGNORECASE)
    if match:
        return f"exp{match.group(1)}"

    match = re.search(r"(?<!\d)(20\d{6})(?!\d)", source)
    if match:
        return f"exp{match.group(1)}"

    stem = source.split(".czi", 1)[0].strip()
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return stem[:32] if stem else "unknown"


def summarize_movie(
    group: str,
    movie_dir: str,
    root_folder: str,
    *,
    experiment_label: Optional[str] = None,
) -> Optional[MovieSummary]:
    csv_candidates = _geometry_csv_candidates(movie_dir)
    csv_path = next((p for p in csv_candidates if os.path.isfile(p)), None)
    if csv_path is None:
        print(f"Warning: missing file: {csv_candidates[0]}")
        return None

    values: List[float] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "front_minus_apical_um" not in reader.fieldnames:
                print(f"Warning: missing column front_minus_apical_um in {csv_path}")
                return None
            for row in reader:
                values.append(parse_float(row.get("front_minus_apical_um")))
    except OSError as exc:
        print(f"Warning: failed reading {csv_path}: {exc}")
        return None

    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        print(f"Warning: no finite front_minus_apical_um values in {csv_path}")
        return None

    rel_path = os.path.relpath(movie_dir, root_folder)
    return MovieSummary(
        group=group,
        movie_id=os.path.basename(movie_dir),
        movie_path=rel_path,
        experiment=experiment_label or infer_experiment_label(movie_dir),
        n_points=int(finite.size),
        max_cellu_height_um=float(np.max(finite)),
    )


def export_maxima(
    root_folder: str,
    groups: Sequence[str],
    out_csv: str,
    *,
    experiment_label: Optional[str] = None,
    group_alias: Optional[Dict[str, str]] = None,
) -> Tuple[int, int]:
    rows: List[MovieSummary] = []
    found_movies = 0
    skipped_movies = 0

    alias_map = dict(group_alias or {})
    for group in groups:
        group_folder = os.path.join(root_folder, group)
        movie_dirs = list_movie_dirs(group_folder)
        if not movie_dirs:
            print(f"Warning: no movie folders under group: {group_folder}")
            continue

        for movie_dir in movie_dirs:
            found_movies += 1
            final_group = alias_map.get(group, group)
            summary = summarize_movie(
                group=final_group,
                movie_dir=movie_dir,
                root_folder=root_folder,
                experiment_label=experiment_label,
            )
            if summary is None:
                skipped_movies += 1
                continue
            rows.append(summary)

    rows.sort(key=lambda r: (groups.index(r.group) if r.group in groups else 999, r.movie_id))

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "group",
                "movie_id",
                "movie_path",
                "experiment",
                "n_points",
                "max_cellu_height_um",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.group,
                    row.movie_id,
                    row.movie_path,
                    row.experiment,
                    row.n_points,
                    f"{row.max_cellu_height_um:.6g}",
                ]
            )

    used_movies = len(rows)
    print(f"Root folder: {root_folder}")
    print(f"Groups: {', '.join(groups)}")
    print(f"Movies found: {found_movies}")
    print(f"Movies used: {used_movies}")
    print(f"Movies skipped: {skipped_movies}")
    print(f"Saved CSV: {out_csv}")
    return used_movies, skipped_movies


def _load_datasets_yaml(path: str) -> List[DatasetSpec]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except OSError as exc:
        raise SystemExit(f"Failed reading datasets YAML: {exc}") from exc

    if not isinstance(cfg, dict):
        raise SystemExit(f"Datasets YAML root must be a mapping: {path}")
    datasets = cfg.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise SystemExit("Datasets YAML must contain a non-empty 'datasets' list.")

    base_dir = os.path.dirname(os.path.abspath(path))
    out: List[DatasetSpec] = []
    for idx, item in enumerate(datasets):
        if not isinstance(item, dict):
            raise SystemExit(f"datasets[{idx}] must be a mapping.")
        name = str(item.get("name", "")).strip()
        root_folder = str(item.get("root_folder", "")).strip()
        groups_raw = item.get("groups")
        alias_raw = item.get("group_alias", {})

        if not name:
            raise SystemExit(f"datasets[{idx}].name is required.")
        if not root_folder:
            raise SystemExit(f"datasets[{idx}].root_folder is required.")
        if not isinstance(groups_raw, list) or not groups_raw:
            raise SystemExit(f"datasets[{idx}].groups must be a non-empty list.")
        groups = [str(g).strip() for g in groups_raw if str(g).strip()]
        if not groups:
            raise SystemExit(f"datasets[{idx}].groups must contain non-empty names.")
        if not isinstance(alias_raw, dict):
            raise SystemExit(f"datasets[{idx}].group_alias must be a mapping if provided.")

        alias: Dict[str, str] = {}
        for k, v in alias_raw.items():
            kk = str(k).strip()
            vv = str(v).strip()
            if kk and vv:
                alias[kk] = vv

        root_abs = root_folder if os.path.isabs(root_folder) else os.path.abspath(os.path.join(base_dir, root_folder))
        out.append(DatasetSpec(name=name, root_folder=root_abs, groups=groups, group_alias=alias))
    return out


def export_from_datasets_yaml(datasets_yaml: str, out_csv: str) -> Tuple[int, int]:
    datasets = _load_datasets_yaml(datasets_yaml)
    all_rows: List[MovieSummary] = []
    total_found = 0
    total_skipped = 0

    for ds in datasets:
        print(f"[dataset] {ds.name}: root={ds.root_folder} groups={','.join(ds.groups)}")
        for group in ds.groups:
            group_folder = os.path.join(ds.root_folder, group)
            movie_dirs = list_movie_dirs(group_folder)
            if not movie_dirs:
                print(f"Warning: no movie folders under group: {group_folder}")
                continue
            for movie_dir in movie_dirs:
                total_found += 1
                final_group = ds.group_alias.get(group, group)
                summary = summarize_movie(
                    group=final_group,
                    movie_dir=movie_dir,
                    root_folder=ds.root_folder,
                    experiment_label=ds.name,
                )
                if summary is None:
                    total_skipped += 1
                    continue
                all_rows.append(summary)

    if not all_rows:
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "group",
                    "movie_id",
                    "movie_path",
                    "experiment",
                    "n_points",
                    "max_cellu_height_um",
                ]
            )
        return 0, total_skipped

    dataset_order = {ds.name: i for i, ds in enumerate(datasets)}
    group_order: Dict[str, int] = {}
    for ds in datasets:
        for i, g in enumerate(ds.groups):
            alias = ds.group_alias.get(g, g)
            group_order.setdefault(alias, i)

    all_rows.sort(
        key=lambda r: (
            group_order.get(r.group, 999),
            dataset_order.get(r.experiment, 999),
            r.movie_id,
        )
    )

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "group",
                "movie_id",
                "movie_path",
                "experiment",
                "n_points",
                "max_cellu_height_um",
            ]
        )
        for row in all_rows:
            writer.writerow(
                [
                    row.group,
                    row.movie_id,
                    row.movie_path,
                    row.experiment,
                    row.n_points,
                    f"{row.max_cellu_height_um:.6g}",
                ]
            )

    used_movies = len(all_rows)
    print(f"Datasets file: {datasets_yaml}")
    print(f"Movies found: {total_found}")
    print(f"Movies used: {used_movies}")
    print(f"Movies skipped: {total_skipped}")
    print(f"Saved CSV: {out_csv}")
    return used_movies, total_skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-movie maximum front_minus_apical_um by group to CSV."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root-folder", help="Folder with group subfolders")
    source.add_argument(
        "--datasets-yaml",
        help="YAML file with datasets list (name/root_folder/groups/group_alias)",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path (required).",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        help="Group folder names to scan (default: dorsal ventral lateral)",
    )
    args = parser.parse_args()

    if args.datasets_yaml:
        datasets_yaml = os.path.abspath(args.datasets_yaml)
        if not os.path.isfile(datasets_yaml):
            raise SystemExit(f"Datasets YAML does not exist: {datasets_yaml}")
        out_csv = os.path.abspath(args.out_csv)
        used_movies, _ = export_from_datasets_yaml(datasets_yaml=datasets_yaml, out_csv=out_csv)
    else:
        root_folder = os.path.abspath(args.root_folder)
        if not os.path.isdir(root_folder):
            raise SystemExit(f"Root folder does not exist: {root_folder}")
        groups = list(dict.fromkeys(args.groups))
        if not groups:
            raise SystemExit("At least one group must be provided.")
        out_csv = os.path.abspath(args.out_csv)
        used_movies, _ = export_maxima(root_folder=root_folder, groups=groups, out_csv=out_csv)
    if used_movies == 0:
        raise SystemExit("No valid movies were exported.")


if __name__ == "__main__":
    main()

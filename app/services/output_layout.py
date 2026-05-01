from pathlib import Path


def work_dir_for(input_movie_path: str) -> Path:
    input_path = Path(input_movie_path).expanduser().resolve()
    return input_path.parent / f"CDynamics-{input_path.name}"


def ensure_work_tree(work_dir: Path) -> None:
    (work_dir / "track").mkdir(parents=True, exist_ok=True)
    (work_dir / "results").mkdir(parents=True, exist_ok=True)

#!/usr/bin/env python3
"""
Merge per-sample cellularization PDFs into one comparison PDF.

By default, the script searches recursively for:
    results/Cellularization_combined.pdf

Example:
    python scripts/merge_cellularization_pdfs.py \
        --folder data/Hermetia/wt/dorsal
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

def _natural_key(text: str) -> List[object]:
    parts = re.split(r"(\d+)", text)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _find_pdfs(root: Path, relative_pdf_path: Path) -> List[Path]:
    matches = [p for p in root.rglob(str(relative_pdf_path)) if p.is_file()]
    return sorted(matches, key=lambda p: _natural_key(str(p.relative_to(root))))


def _merge_pdfs(paths: Iterable[Path], output_path: Path) -> int:
    try:
        from pypdf import PdfWriter
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'pypdf'. Install it with:\n"
            "  pip install pypdf"
        ) from exc

    writer = PdfWriter()
    for path in paths:
        writer.append(str(path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        writer.write(f)

    return len(writer.pages)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively collect per-sample PDFs (default: "
            "results/Cellularization_combined.pdf) and merge them."
        )
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Root folder containing sample subfolders (e.g. data/Hermetia/wt/dorsal).",
    )
    parser.add_argument(
        "--pdf-relative-path",
        default="results/Cellularization_combined.pdf",
        help="Relative path to each sample PDF inside sample folders.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output PDF path. Default: <folder>/results/Cellularization_combined_merged.pdf"
        ),
    )
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        raise SystemExit(f"Folder does not exist: {folder}")

    relative_pdf_path = Path(args.pdf_relative_path)
    if relative_pdf_path.is_absolute():
        raise SystemExit("--pdf-relative-path must be a relative path.")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else folder / "results" / "Cellularization_combined_merged.pdf"
    )

    pdf_paths = _find_pdfs(folder, relative_pdf_path)
    if not pdf_paths:
        raise SystemExit(
            f"No files found under {folder} matching: {relative_pdf_path.as_posix()}"
        )

    page_count = _merge_pdfs(pdf_paths, output_path)

    print(f"Merged files: {len(pdf_paths)}")
    print(f"Total pages: {page_count}")
    print(f"Saved: {output_path}")
    print("\nInput order:")
    for p in pdf_paths:
        print(f"  - {p.relative_to(folder)}")


if __name__ == "__main__":
    main()

# Cellularization Dynamics

Desktop app for cellularization annotation and output generation. The repository is **GUI-first**: run the app from the repo root; batch/Snakemake workflows are not used.

## Install

From the repository root:

```bash
conda env create -f environment.yaml
conda activate celludynamics-gui
```

## Start

```bash
python -m app
```

Always run this from the repository root so the `core` package and `app` resolve correctly.

## Layout

- **`app/`** — PyQt6 application (`python -m app`).
- **`core/`** — shared image/geometry code used by the app (kymograph, straightening, spline, exports).

## Config and state

Each sample work directory has a single **`config.yaml`** with `schema_version: 2`. It holds acquisition settings (`acquisition.source_movie` points at the original TIFF; no duplicate trimmed stack), kymograph timing, spline options, apical alignment, straightening metadata, spline-fit metadata, and derived summaries. Legacy layouts with separate `track/*.yaml` files are merged into this file on first load.

## GUI workflow

1. Add one or more `.tif` movies (drag-and-drop or **Open Files**).
2. Select a movie from the list.
3. Set acquisition parameters (`px2micron`, `movie_time_interval_sec`; optional `smoothing`, `degree`) — extracted from movie metadata when available (e.g. ImageJ-saved TIFFs).
4. **Analyze** — records the source movie path in `config.yaml` and writes `track/Kymograph.tif` next to the movie.
5. Adjust the cytoplasm threshold and island (apical) mode as needed; place at least two points on the cellularization front in the straightened view.
6. **Save** — writes mask, alignment metadata, and front annotation under the work folder.
7. **Generate Outputs** — straightens the kymograph, fits the spline, exports `output.csv` (main tabular result in the sample folder), and renders figure/video products (e.g. `Cellularization.png`, `Cellularization_front_markers.mp4`).

Outputs are written next to each input movie in a folder `CDynamics-<movie filename>` (see `app/services/output_layout.py`).

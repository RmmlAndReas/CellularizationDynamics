# Cellularization Dynamics

Single Snakemake pipeline: `cellularization_dynamics`.

This repository supports two workflows:
- CLI workflow via Snakemake (`envs/analysis.yaml`)
- Desktop GUI workflow via PyQt6 (`envs/gui.yaml`)

## CLI Setup

```bash
conda env create -f envs/analysis.yaml
conda activate yolk
```

## CLI Config

Copy `config/example.yaml` and set your sample path(s).
Minimal sample fields are:
- `path`
- `px2micron`
- `movie_time_interval_sec`
- `keep_every`
- optional: `smoothing` (default `0.0`), `degree` (default `3`)

## CLI Run

```bash
snakemake --cores 1 --dry-run --configfile config/example.yaml
snakemake --cores N --configfile config/example.yaml
```

Optional:

```bash
snakemake --cores N --use-conda --configfile config/example.yaml
```

## CLI Manual Steps

The workflow includes two interactive matplotlib steps:
- `scripts/cellu_threshold.py`
- `scripts/cellu_front_annotation.py`

When those jobs start, keep the window open, complete the interaction, then continue.

## CLI Output

Retained pipeline outputs:

- `results/<sample>/results/Cellularization.png`
- `results/<sample>/results/Cellularization_trimmed_delta.tif`
- `results/<sample>/track/Kymograph.tif`
- `results/<sample>/track/YolkMask.tif`
- `results/<sample>/track/VerticalKymoCelluSelection.tsv`
- `results/<sample>/track/VerticalKymoCelluSelection.roi`
- `results/<sample>/track/VerticalKymoCelluSelection_spline.tsv`
- `results/<sample>/track/geometry_timeseries.csv`

## GUI Setup (Desktop App, macOS / Linux)

The PyQt6 desktop app is standalone and writes outputs next to each input movie.

### 1) Create and activate the GUI environment

```bash
conda env create -f envs/gui.yaml
conda activate celludynamics-gui
```

### 2) Launch the app (from repository root)

```bash
python -m app
```

If the window does not open on Linux/WSL, verify that graphical display forwarding is available.

## GUI Usage

### Supported input

- One or more `.tif` / `.tiff` timelapse movies.
- Add files by drag-and-drop into the file list or with **Open Files**.

### Per-movie workflow

1. Drag and drop one or more `.tif` movies.
2. Select a movie in the file list.
3. Set acquisition parameters (`px2micron`, `movie_time_interval_sec`, `keep_every`; optional `smoothing`, `degree`).
4. Click **Analyze** to generate intermediate data for thresholding and annotation.
5. Adjust threshold and add front points (at least two points are required).
6. Click **Save** to write:
   - `track/YolkMask.tif`
   - `track/VerticalKymoCelluSelection.tsv`
   - `track/VerticalKymoCelluSelection.roi`
   - `config.yaml` (with updated `apical_detection`)
7. Click **Generate Figure** to write:
   - `track/VerticalKymoCelluSelection_spline.tsv`
   - `track/geometry_timeseries.csv`
   - `results/Cellularization.png`
   - `results/Cellularization.pdf`
   - `results/Cellularization_trimmed_delta.tif`

### Output location

For an input movie `/path/to/movies/movie01.tif`, outputs are saved under:

`/path/to/movies/movie01/`

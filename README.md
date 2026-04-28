# Cellularization Dynamics

Single Snakemake pipeline: `cellularization_dynamics`.

## Setup

```bash
conda env create -f envs/analysis.yaml
conda activate yolk
```

## Config

Copy `config/example.yaml` and set your sample path(s).
Minimal sample fields are:
- `path`
- `px2micron`
- `movie_time_interval_sec`
- `keep_every`
- optional: `smoothing` (default `0.0`), `degree` (default `3`)

## Run

```bash
snakemake --cores 1 --dry-run --configfile config/example.yaml
snakemake --cores N --configfile config/example.yaml
```

Optional:

```bash
snakemake --cores N --use-conda --configfile config/example.yaml
```

## Manual steps

The workflow includes two interactive matplotlib steps:
- `scripts/cellu_threshold.py`
- `scripts/cellu_front_annotation.py`

When those jobs start, keep the window open, complete the interaction, then continue.

## Output

Retained pipeline outputs:

- `results/<sample>/results/Cellularization.png`
- `results/<sample>/results/Cellularization_trimmed_delta.tif`
- `results/<sample>/track/Kymograph.tif`
- `results/<sample>/track/YolkMask.tif`
- `results/<sample>/track/VerticalKymoCelluSelection.tsv`
- `results/<sample>/track/VerticalKymoCelluSelection.roi`
- `results/<sample>/track/VerticalKymoCelluSelection_spline.tsv`
- `results/<sample>/track/geometry_timeseries.csv`


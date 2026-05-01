# Cellularization Dynamics (GUI)

Desktop app for cellularization annotation and output generation.

## Install

From the repository root:

```bash
conda env create -f envs/gui.yaml
conda activate celludynamics-gui
```

## Start

```bash
python -m app
```

## GUI Usage

1. Add one or more `.tif` movies (drag-and-drop or **Open Files**).
2. Select a movie from the list.
3. Set acquisition parameters (`px2micron`, `movie_time_interval_sec`; optional `smoothing`, `degree`) — these are automatically extracted from the movie metadata when available (e.g. ImageJ-saved TIFFs).
4. Click **Analyze** - this will trim the movie, generate the kymograph
5. Adjust threshold and annotate the front (at least two points) - this will set the apical border and save the cellularization front
6. Click **Save**.
7. Click **Generate Outputs** - This will generate the final outputs for the cellularization pipeline

Outputs are written next to each input movie in its own folder.

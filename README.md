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

1. Add one or more `.tif` / `.tiff` movies (drag-and-drop or **Open Files**).
2. Select a movie from the list.
3. Set acquisition parameters (`px2micron`, `movie_time_interval_sec`, `keep_every`; optional `smoothing`, `degree`).
4. Click **Analyze**.
5. Adjust threshold and annotate the front (at least two points).
6. Click **Save**.
7. Click **Generate Outputs**.

Outputs are written next to each input movie in its own folder.

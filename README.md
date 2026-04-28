# Workflow for Image Analysis

## Next

1. Analyze all the Hermetia movies. Pipeline is ready.
2. Merge the cellularization fronts.

## Manual Steps how

1. Use segmented line. Draw across the membrane - full length? 

2. Cut the time right - last NC until start of gastrulation. I need to see how to define the "start of gastrulation"
   - How to define start of gastrulation? - shared marker across species?. I'll use the yolk behaviour on the kymo - simplest, can be done in code.


   
   - Mark down the time of last NC and start of gastrulation (Two variable in timing)
   
3. Make vertical kymograph for the cellularization dynamics.
   - Shorten the movie to 1min (10s is too much for vertical kymo)
   - Draw a line across in parallel of cellularization and export
   - Draw a line marking the egg shell (Apical side of the cell - this will be used to calculate the cell height)

That's all regards manual steps → python

## Snakemake Pipeline

The analysis workflow has been automated using Snakemake for reproducible processing across multiple samples.

### Setup

1. **Install Snakemake and dependencies:**
   ```bash
   # Install Snakemake (if not already installed)
   conda install -c conda-forge snakemake
   
   # Or use pip
   pip install snakemake
   ```

2. **Configure samples:**
   - Edit species files in `config/` (e.g., `config/hermetia.yaml`) to add your samples
   - Each sample should have:
     - `path`: relative path to the sample folder (e.g., `data/Hermetia/20251106#2`)
     - Optional parameter overrides (see species config files in `config/` for available parameters and structure)

3. **Expected folder structure for each sample:**
   ```
   <sample_path>/
   ├── config.yaml                    # Sample-specific config
   ├── input/
   │   ├── Kymograph.tif              # Vertical kymograph (manual)
   │   ├── VerticalKymoCelluSelection.tsv  # Cell front selection (manual)
   │   └── Cellularization.tif        # Timelapse movie (manual)
   └── output/                        # Created by pipeline
   ```

### Running the Pipeline

The pipeline requires explicit sample config selection. Preferred usage is `--configfile config/<species>.yaml`.

1. **Dry run (check what will be executed):**
   ```bash
   snakemake --cores 1 --dry-run --configfile config/hermetia.yaml
   ```

2. **Run the pipeline:**
   ```bash
   snakemake --cores N --configfile config/hermetia.yaml
   ```
   Where `N` is the number of CPU cores to use.

3. **Run with conda environment management:**
   ```bash
   snakemake --cores N --use-conda --configfile config/hermetia.yaml
   ```
   This will automatically create and use the conda environment defined in `envs/analysis.yaml`.

4. **Run specific sample or rule:**
   ```bash
   # Run only apical detection for a specific sample
   snakemake --cores N --configfile config/hermetia.yaml hermetia_20251106_2/output/Apical_height_output.png
   
   # Run all steps for a specific sample
   snakemake --cores N --configfile config/hermetia.yaml hermetia_20251106_2/output/Kymographs_merged.png
   ```

5. **Alternative inline config key (still supported):**
   ```bash
   snakemake --cores N --config samples=config/hermetia.yaml
   ```

### Pipeline Workflow

The pipeline executes the following steps in order:

1. **Apical Detection** (`apical_detection`): Detects apical side from vertical kymograph
   - Input: `input/Kymograph.tif`, `config.yaml`
   - Output: `output/Apical_height_output.png`, `output/metadata.yaml`

2. **Cellularization Front Spline** (`cell_front_spline`): Fits spline to cellularization front
   - Input: `input/VerticalKymoCelluSelection.tsv`, apical detection results
   - Output: `output/cell_front_spline.tsv`, `output/Cellularization_dynamics.png`, `output/metadata.yaml`

3. **Mark delta on trimmed movie** (`mark_delta_on_trimmed_movie`): Overlays a white arrow on the right edge of each frame of the trimmed movie at the delta position (offset from cellularization front).
   - Input: `Cellularization_trimmed.tif`, spline TSV, `config.yaml`
   - Output: `Cellularization_trimmed_delta.tif`

4. **Create Kymograph** (`create_kymographs`): Creates a single delta kymograph
   - Input: `Cellularization.tif`, spline data
   - Output: `Kymograph_delta.tif`, `Kymograph_delta.png`

5. **Mark Kymographs** (`mark_kymographs`): Marks progress milestones on the delta kymograph
   - Input: `Kymograph_delta.tif`, spline data
   - Output: `Kymograph_delta_marked.png`, `Cellularization_dynamics_marked.png`

6. **Merge Kymographs** (`merge_kymographs`): Combines dynamics and marked delta kymograph into final figure
   - Input: `Cellularization_dynamics_marked.png`, `Kymograph_delta_marked.png`
   - Output: `Kymographs_merged.png`

### Logs

Log files for each step are saved in `{sample}/logs/` for debugging and traceability.

### Configuration

- **Sample configurations**: species files in `config/` (single source of truth)
- **Environment**: `envs/analysis.yaml` (conda environment specification)

## Python Steps (Manual Execution)

For manual execution of individual scripts, see below:

### 1. Calculate cellularization dynamics using 4 variables:

- LastNC
- Start of gastrulation
- Function of Egg shell position over time 
- Function of cellularization front over time

Repeat for 3 embryos. Calculate f(cell height). Then mean and SD across three embryos - Plot with statistics.

### 2. Two kymographs showing membrane dance during cellularization

1. By using the f(cellularization front) make two kymographs:
   - Kymograph from 3px section from **3um** above the cellularization front
      Time is relative to the last NC

   - Kymograph from 3px section from **5um** above the cellularization front
2. Print the markers on kymographs:
   - Last NC
   - Start of gastrulation
   - Cellularization height in % of the max height (This I need to check how to do)

## Species specifics

### Hermetia

- Different sides are quite different
   To capture diversity, I'll use ventral, dorsal and lateral. We'll later decide which one is good.

In 20251106:
   - Dorsal - #4 (single),#5 (double), 
   - Lateral - #2 (single),
   - Ventral - #1 (single)

Vertical kymograph:
   - Best seems to be 51px width multi-kymo


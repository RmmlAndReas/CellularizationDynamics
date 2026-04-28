"""
Snakemake pipeline: single-timepoint cytoplasm mask from a movie (2D), width vs x.

Reads samples from config/*.yaml (same schema as cellularization). Writes under
results/movie_cytoplasm_width/<sample_rel>/.

Usage:
    snakemake -s workflow/movie_cytoplasm_width.smk --cores 5 --configfile config/hermetia.yaml
"""

import os
import sys
from glob import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(workflow.snakefile)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.samples_loader import load_samples_config_with_sources

DATA_ROOT = os.path.join(REPO_ROOT, "data")
MOVIECYTO_ROOT = os.path.join(REPO_ROOT, "results", "movie_cytoplasm_width")


def _relative_sample_path(sample_path: str) -> str:
    sample_abs = os.path.abspath(sample_path)
    data_abs = os.path.abspath(DATA_ROOT)
    rel = os.path.relpath(sample_abs, data_abs)
    if rel == ".." or rel.startswith(".." + os.sep):
        raise ValueError(
            f"Sample path must be under data root. Got sample={sample_abs}, data={data_abs}"
        )
    return rel


samples_config_sources = []
samples_config_arg = config.get("samples")
has_samples_schema = any(k in config for k in ("samples", "experiments"))

if isinstance(samples_config_arg, str) and samples_config_arg.strip():
    samples_config_path = os.path.abspath(samples_config_arg)
    if not (os.path.isfile(samples_config_path) or os.path.isdir(samples_config_path)):
        raise FileNotFoundError(
            f"Samples config path not found: {samples_config_path}. "
            "Expected a YAML file or directory."
        )
    samples_config, samples_config_sources = load_samples_config_with_sources(samples_config_path)
elif has_samples_schema:
    configfiles = list(getattr(workflow, "overwrite_configfiles", []) or [])
    if not configfiles:
        raise ValueError(
            "Could not detect --configfile path. Run with: --configfile config/hermetia.yaml"
        )
    samples_config_path = os.path.abspath(configfiles[0])
    samples_config, samples_config_sources = load_samples_config_with_sources(samples_config_path)
else:
    raise ValueError(
        "Missing samples configuration. "
        "Run with: --configfile config/hermetia.yaml "
        "(preferred) or --config samples=config/hermetia.yaml."
    )


def _resolve_data_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(REPO_ROOT, p))


SAMPLE_DATA_DIRS = [_resolve_data_path(samples_config["samples"][k]["path"]) for k in samples_config["samples"].keys()]
SAMPLE_RELS = [_relative_sample_path(p) for p in SAMPLE_DATA_DIRS]
REL_TO_DATA_DIR = {rel: p for rel, p in zip(SAMPLE_RELS, SAMPLE_DATA_DIRS)}


def _data_dir(rel: str) -> str:
    return REL_TO_DATA_DIR[rel]


def _work_dir(rel: str) -> str:
    return os.path.join(MOVIECYTO_ROOT, rel)


def _prepared_movie_path(rel: str) -> str:
    """
    Select the prepared movie file for this sample.

    The prepared input is expected to end with "-1.tif" / "-1.tiff".
    """
    data_dir = _data_dir(rel)
    patterns = [os.path.join(data_dir, "*-1.tif"), os.path.join(data_dir, "*-1.tiff")]
    matches = []
    for pattern in patterns:
        matches.extend(glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No prepared '-1' TIFF found in sample directory: {data_dir}. "
            "Expected exactly one file matching '*-1.tif' or '*-1.tiff'."
        )
    # If multiple prepared files exist, prefer the largest one.
    return max(matches, key=os.path.getsize)


def _prepared_input_dir(rel: str) -> str:
    return os.path.join(_work_dir(rel), "inputs", "prepared_movie")


localrules: movie_cytoplasm_threshold


rule all:
    input:
        expand(MOVIECYTO_ROOT + "/{sample}/track/cytoplasm_height_vs_x.csv", sample=SAMPLE_RELS),
        expand(MOVIECYTO_ROOT + "/{sample}/track/cytoplasm_height_summary.csv", sample=SAMPLE_RELS),
        expand(
            MOVIECYTO_ROOT + "/{sample}/results/duplicated_slice.tif",
            sample=SAMPLE_RELS,
        ),
        expand(
            MOVIECYTO_ROOT + "/{sample}/results/threshold_outline.tif",
            sample=SAMPLE_RELS,
        ),
        expand(
            MOVIECYTO_ROOT + "/{sample}/results/cytoplasm_height_apical_yolk_overlay.pdf",
            sample=SAMPLE_RELS,
        ),
        expand(
            MOVIECYTO_ROOT + "/{sample}/results/cytoplasm_height_apical_yolk_overlay.png",
            sample=SAMPLE_RELS,
        ),


rule init_movie_cytoplasm_config:
    input:
        samples=ancient(samples_config_sources),
    output:
        config=MOVIECYTO_ROOT + "/{sample}/config.yaml",
        hash_file=MOVIECYTO_ROOT + "/{sample}/.config_hash_movie_cytoplasm",
    params:
        data_dir=lambda wildcards: _data_dir(wildcards.sample),
        work_dir=lambda wildcards: _work_dir(wildcards.sample),
    conda:
        "../envs/analysis.yaml"
    shell:
        """
        python scripts/init_movie_cytoplasm_config.py \
            --data-dir {params.data_dir:q} \
            --work-dir {params.work_dir:q} \
            --samples {samples_config_path}
        """


rule movie_cytoplasm_threshold:
    input:
        config=rules.init_movie_cytoplasm_config.output.config,
        prepared_movie=lambda wildcards: _prepared_movie_path(wildcards.sample),
    output:
        mask=MOVIECYTO_ROOT + "/{sample}/track/CytoplasmMask.tif",
        duplicate_slice=MOVIECYTO_ROOT + "/{sample}/track/MovieSliceDup.tif",
    params:
        prepared_input_dir=lambda wildcards: _prepared_input_dir(wildcards.sample),
        work_dir=lambda wildcards: _work_dir(wildcards.sample),
    conda:
        "../envs/analysis.yaml"
    log:
        MOVIECYTO_ROOT + "/{sample}/logs/movie_cytoplasm_threshold.log",
    shell:
        """
        mkdir -p $(dirname {log:q})
        mkdir -p {params.prepared_input_dir:q}
        ln -sfn {input.prepared_movie:q} {params.prepared_input_dir:q}/Cellularization.tif
        python -u scripts/movie_cytoplasm_threshold.py \
            --data-dir {params.prepared_input_dir:q} \
            --work-dir {params.work_dir:q} \
            > {log:q} 2>&1
        """


rule quantify_movie_cytoplasm:
    input:
        mask=rules.movie_cytoplasm_threshold.output.mask,
    output:
        vs_x=MOVIECYTO_ROOT + "/{sample}/track/cytoplasm_height_vs_x.csv",
        summary=MOVIECYTO_ROOT + "/{sample}/track/cytoplasm_height_summary.csv",
    params:
        work_dir=lambda wildcards: _work_dir(wildcards.sample),
    conda:
        "../envs/analysis.yaml"
    log:
        MOVIECYTO_ROOT + "/{sample}/logs/quantify_movie_cytoplasm.log",
    shell:
        """
        mkdir -p $(dirname {log:q})
        python scripts/quantify_movie_cytoplasm_height.py \
            --work-dir {params.work_dir:q} \
            > {log:q} 2>&1
        """


rule export_movie_cytoplasm_overlay:
    input:
        duplicate_slice=rules.movie_cytoplasm_threshold.output.duplicate_slice,
        mask=rules.movie_cytoplasm_threshold.output.mask,
        vs_x=rules.quantify_movie_cytoplasm.output.vs_x,
        config=rules.init_movie_cytoplasm_config.output.config,
    output:
        duplicated_slice=MOVIECYTO_ROOT + "/{sample}/results/duplicated_slice.tif",
        threshold_outline=MOVIECYTO_ROOT + "/{sample}/results/threshold_outline.tif",
        png=MOVIECYTO_ROOT + "/{sample}/results/cytoplasm_height_apical_yolk_overlay.png",
        pdf=MOVIECYTO_ROOT + "/{sample}/results/cytoplasm_height_apical_yolk_overlay.pdf",
        tif=MOVIECYTO_ROOT + "/{sample}/results/cytoplasm_height_apical_yolk_overlay.tif",
    conda:
        "../envs/analysis.yaml"
    log:
        MOVIECYTO_ROOT + "/{sample}/logs/export_movie_cytoplasm_overlay.log",
    shell:
        """
        mkdir -p $(dirname {log:q})
        python scripts/export_movie_cytoplasm_overlay.py \
            --input-image {input.duplicate_slice:q} \
            --input-mask {input.mask:q} \
            --input-csv {input.vs_x:q} \
            --config {input.config:q} \
            --out-duplicate-slice {output.duplicated_slice:q} \
            --out-threshold-outline {output.threshold_outline:q} \
            --out-png {output.png:q} \
            --out-pdf {output.pdf:q} \
            --out-tif {output.tif:q} \
            > {log:q} 2>&1
        """

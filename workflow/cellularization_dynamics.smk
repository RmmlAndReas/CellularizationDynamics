"""
Snakemake pipeline for Hermetia cellularization dynamics.

This workflow reads raw movies from sample paths configured in `config/*.yaml`
and writes all derived outputs under `results/cellularization_dynamics/`,
mirroring each sample path relative to repository `data/`.

Usage:
    snakemake -s workflow/cellularization_dynamics.smk --cores N --configfile config/<species>.yaml
"""

import os
import sys
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(workflow.snakefile)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.samples_loader import load_samples_config_with_sources

DATA_ROOT = os.path.join(REPO_ROOT, "data")
CELL_DYN_ROOT = os.path.join(REPO_ROOT, "results", "cellularization_dynamics")


def _relative_sample_path(sample_path: str) -> str:
    sample_abs = os.path.abspath(sample_path)
    data_abs = os.path.abspath(DATA_ROOT)
    rel = os.path.relpath(sample_abs, data_abs)
    if rel == ".." or rel.startswith(".."+os.sep):
        raise ValueError(
            f"Sample path must be under data root. Got sample={sample_abs}, data={data_abs}"
        )
    return rel


# Resolve samples config source:
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


SAMPLE_DATA_DIRS = [samples_config["samples"][k]["path"] for k in samples_config["samples"].keys()]
SAMPLE_RELS = [_relative_sample_path(p) for p in SAMPLE_DATA_DIRS]
REL_TO_DATA_DIR = {rel: os.path.abspath(p) for rel, p in zip(SAMPLE_RELS, SAMPLE_DATA_DIRS)}


def _data_dir(rel: str) -> str:
    return REL_TO_DATA_DIR[rel]


def _work_dir(rel: str) -> str:
    return os.path.join(CELL_DYN_ROOT, rel)


localrules: annotate_cellu_front, detect_cytoplasm_region


rule all:
    input:
        expand(CELL_DYN_ROOT + "/{sample}/results/Cellularization.png", sample=SAMPLE_RELS),
        expand(CELL_DYN_ROOT + "/{sample}/results/Kymograph_delta.tif", sample=SAMPLE_RELS),
        expand(CELL_DYN_ROOT + "/{sample}/results/Cellularization_trimmed_delta.tif", sample=SAMPLE_RELS),
        expand(CELL_DYN_ROOT + "/{sample}/results/Kymograph_delta_marked.png", sample=SAMPLE_RELS),
        expand(CELL_DYN_ROOT + "/{sample}/results/Cellularization_combined.png", sample=SAMPLE_RELS)


rule init_config:
    input:
        samples=ancient(samples_config_sources)
    output:
        config=CELL_DYN_ROOT + "/{sample}/config.yaml",
        hash_file=CELL_DYN_ROOT + "/{sample}/.config_hash"
    params:
        data_dir=lambda wildcards: _data_dir(wildcards.sample),
        work_dir=lambda wildcards: _work_dir(wildcards.sample)
    conda:
        "../envs/analysis.yaml"
    shell:
        """
        python scripts/init_config.py \
            --data-dir {params.data_dir:q} \
            --work-dir {params.work_dir:q} \
            --samples {samples_config_path}
        """


rule trim_movie:
    input:
        config=rules.init_config.output.config
    output:
        trimmed=CELL_DYN_ROOT + "/{sample}/Cellularization_trimmed.tif"
    params:
        data_dir=lambda wildcards: _data_dir(wildcards.sample),
        work_dir=lambda wildcards: _work_dir(wildcards.sample)
    conda:
        "../envs/analysis.yaml"
    log:
        CELL_DYN_ROOT + "/{sample}/logs/trim_movie.log"
    shell:
        """
        mkdir -p $(dirname {log:q})
        python scripts/trim_movie.py \
            --work-dir {params.work_dir:q} \
            --data-dir {params.data_dir:q} \
            > {log:q} 2>&1
        """


rule create_vertical_kymograph:
    input:
        trimmed=CELL_DYN_ROOT + "/{sample}/Cellularization_trimmed.tif",
        config=ancient(rules.init_config.output.config)
    output:
        kymograph=CELL_DYN_ROOT + "/{sample}/track/Kymograph.tif"
    params:
        work_dir=lambda wildcards: _work_dir(wildcards.sample)
    conda:
        "../envs/analysis.yaml"
    log:
        CELL_DYN_ROOT + "/{sample}/logs/create_vertical_kymograph.log"
    shell:
        """
        mkdir -p $(dirname {log:q})
        python scripts/create_vertical_kymograph.py \
            --work-dir {params.work_dir:q} \
            > {log:q} 2>&1
        """


rule annotate_cellu_front:
    input:
        kymograph=ancient(CELL_DYN_ROOT + "/{sample}/track/Kymograph.tif"),
        config=ancient(rules.init_config.output.config)
    output:
        tsv=CELL_DYN_ROOT + "/{sample}/track/VerticalKymoCelluSelection.tsv",
        roi=CELL_DYN_ROOT + "/{sample}/track/VerticalKymoCelluSelection.roi"
    params:
        work_dir=lambda wildcards: _work_dir(wildcards.sample)
    threads: 999
    conda:
        "../envs/analysis.yaml"
    log:
        CELL_DYN_ROOT + "/{sample}/logs/annotate_cellu_front.log"
    shell:
        """
        mkdir -p $(dirname {log:q})
        python -u scripts/annotate_cellu_front.py \
            --work-dir {params.work_dir:q} \
            > {log:q} 2>&1
        """


rule detect_cytoplasm_region:
    input:
        kymograph=ancient(CELL_DYN_ROOT + "/{sample}/track/Kymograph.tif"),
        config=ancient(rules.init_config.output.config)
    output:
        mask=CELL_DYN_ROOT + "/{sample}/track/YolkMask.tif",
        cytoplasm_tsv=CELL_DYN_ROOT + "/{sample}/track/cytoplasm_region.tsv"
    params:
        work_dir=lambda wildcards: _work_dir(wildcards.sample)
    threads: 999
    conda:
        "../envs/analysis.yaml"
    log:
        CELL_DYN_ROOT + "/{sample}/logs/detect_cytoplasm_region.log"
    shell:
        """
        mkdir -p $(dirname {log:q})
        python -u scripts/detect_cytoplasm_region.py \
            --work-dir {params.work_dir:q} \
            > {log:q} 2>&1
        """


rule fit_cellu_front_spline:
    input:
        tsv=rules.annotate_cellu_front.output.tsv,
        config=ancient(rules.init_config.output.config)
    output:
        spline_tsv=CELL_DYN_ROOT + "/{sample}/track/VerticalKymoCelluSelection_spline.tsv"
    params:
        work_dir=lambda wildcards: _work_dir(wildcards.sample),
        smoothing=lambda wildcards: (
            yaml.safe_load(open(CELL_DYN_ROOT + f"/{wildcards.sample}/config.yaml"))["spline"]["smoothing"]
        ),
        degree=lambda wildcards: (
            yaml.safe_load(open(CELL_DYN_ROOT + f"/{wildcards.sample}/config.yaml"))["spline"]["degree"]
        ),
        time_interval=lambda wildcards: (
            lambda cfg: cfg["kymograph"].get(
                "time_interval_sec",
                cfg["kymograph"].get("time_interval_min", 60.0) * 60.0,
            )
            / 60.0
        )(yaml.safe_load(open(CELL_DYN_ROOT + f"/{wildcards.sample}/config.yaml")))
    conda:
        "../envs/analysis.yaml"
    log:
        CELL_DYN_ROOT + "/{sample}/logs/fit_cellu_front_spline.log"
    shell:
        """
        mkdir -p $(dirname {log:q})
        python scripts/fit_cellu_front_spline.py \
            --work-dir {params.work_dir:q} \
            --smoothing {params.smoothing} \
            --degree {params.degree} \
            --time-interval-min {params.time_interval} \
            > {log:q} 2>&1
        """


rule export_geometry_timeseries:
    input:
        cytoplasm_tsv=rules.detect_cytoplasm_region.output.cytoplasm_tsv,
        spline_tsv=rules.fit_cellu_front_spline.output.spline_tsv,
        config=ancient(rules.init_config.output.config)
    output:
        geometry_csv=CELL_DYN_ROOT + "/{sample}/track/geometry_timeseries.csv"
    params:
        work_dir=lambda wildcards: _work_dir(wildcards.sample)
    conda:
        "../envs/analysis.yaml"
    log:
        CELL_DYN_ROOT + "/{sample}/logs/export_geometry_timeseries.log"
    shell:
        """
        mkdir -p $(dirname {log:q})
        python scripts/export_geometry_timeseries.py \
            --work-dir {params.work_dir:q} \
            > {log:q} 2>&1
        """


rule generate_outputs:
    input:
        trimmed=CELL_DYN_ROOT + "/{sample}/Cellularization_trimmed.tif",
        kymograph=CELL_DYN_ROOT + "/{sample}/track/Kymograph.tif",
        cytoplasm_tsv=rules.detect_cytoplasm_region.output.cytoplasm_tsv,
        spline_tsv=rules.fit_cellu_front_spline.output.spline_tsv,
        geometry_csv=rules.export_geometry_timeseries.output.geometry_csv,
        config=ancient(rules.init_config.output.config)
    output:
        cellu_png=CELL_DYN_ROOT + "/{sample}/results/Cellularization.png",
        kymo_delta=CELL_DYN_ROOT + "/{sample}/results/Kymograph_delta.tif",
        trimmed_delta=CELL_DYN_ROOT + "/{sample}/results/Cellularization_trimmed_delta.tif",
        kymo_marked_png=CELL_DYN_ROOT + "/{sample}/results/Kymograph_delta_marked.png",
        kymo_marked_pdf=CELL_DYN_ROOT + "/{sample}/results/Kymograph_delta_marked.pdf",
        combined_png=CELL_DYN_ROOT + "/{sample}/results/Cellularization_combined.png",
        combined_pdf=CELL_DYN_ROOT + "/{sample}/results/Cellularization_combined.pdf"
    params:
        data_dir=lambda wildcards: _data_dir(wildcards.sample),
        work_dir=lambda wildcards: _work_dir(wildcards.sample)
    conda:
        "../envs/analysis.yaml"
    log:
        CELL_DYN_ROOT + "/{sample}/logs/generate_outputs.log"
    shell:
        """
        mkdir -p $(dirname {log:q})
        python scripts/generate_outputs.py \
            --work-dir {params.work_dir:q} \
            --data-dir {params.data_dir:q} \
            > {log:q} 2>&1
        """

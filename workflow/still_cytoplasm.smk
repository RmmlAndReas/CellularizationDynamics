"""
Still-image blastoderm cytoplasm pipeline (manual apical/basal polylines + cubic spline CSV).

All ``*.tif`` under ``condition_dir`` / ``control_dir`` are processed. Set ``px2micron`` at
config top level; optional ``files:`` overrides per filename.

Run from repository root (conda env ``yolk`` — same as ``envs/analysis.yaml``):
    conda activate yolk
    snakemake -s workflow/still_cytoplasm.smk --configfile config/hermetia_still_cytoplasm.yaml --cores 1
    # or: conda run -n yolk snakemake ...

See workflow/README_still_cytoplasm.md for details.
"""
import glob
import hashlib
import os
import shlex
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(workflow.snakefile)))


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def _merged_file_meta(fname, files_meta, default_smooth):
    """Merge global config defaults with optional files[fname] overrides."""
    defaults = {"smooth_window_px": default_smooth}
    if config.get("px2micron") is not None:
        defaults["px2micron"] = float(config["px2micron"])
    if config.get("profile_hlines") is not None:
        defaults["profile_hlines"] = config["profile_hlines"]

    override = files_meta.get(fname)
    if override is None:
        override = {}
    elif not isinstance(override, dict):
        raise ValueError(f"files['{fname}'] must be a mapping")

    meta = {**defaults, **override}
    if meta.get("px2micron") is None:
        raise ValueError(
            f"px2micron missing for {fname!r}: set top-level px2micron or "
            f"files['{fname}'] in the config."
        )
    px2 = float(meta["px2micron"])
    smooth = int(meta.get("smooth_window_px", default_smooth))
    return px2, smooth, meta.get("profile_hlines")


def discover_still_entries():
    """All *.tif under condition_dir and control_dir; metadata from defaults + files: overrides."""
    results_root = _abs(config["results_root"])
    cdir = _abs(config["condition_dir"])
    odir = _abs(config["control_dir"])
    files_meta = config.get("files") or {}
    if not isinstance(files_meta, dict):
        raise ValueError("config 'files' must be a mapping")

    default_smooth = int(config.get("smooth_window_px", 1))
    exclude = set(config.get("exclude_files") or [])
    entries = []
    seen_paths = set()

    for group, root in (("condition", cdir), ("control", odir)):
        if not os.path.isdir(root):
            continue
        for pattern in ("*.tif", "*.TIF"):
            for path in sorted(glob.glob(os.path.join(root, pattern))):
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                fname = os.path.basename(path)
                if fname in exclude:
                    continue
                px2, smooth, profile_hlines = _merged_file_meta(
                    fname, files_meta, default_smooth
                )
                uid = hashlib.md5(f"{group}/{fname}".encode()).hexdigest()[:16]
                base = os.path.join(results_root, uid)
                entries.append(
                    {
                        "uid": uid,
                        "group": group,
                        "filename": fname,
                        "tif": path,
                        "px2micron": px2,
                        "smooth_window_px": smooth,
                        "profile_hlines": profile_hlines,
                        "csv_path": os.path.join(base, "track", "cytoplasm_profile.csv"),
                        "json_path": os.path.join(base, "track", "boundary_polylines.json"),
                        "png_path": os.path.join(base, "results", "cytoplasm_profile.png"),
                        "pdf_path": os.path.join(base, "results", "cytoplasm_profile.pdf"),
                        "overlay_png": os.path.join(
                            base, "results", "cytoplasm_apical_basal_overlay.png"
                        ),
                        "overlay_pdf": os.path.join(
                            base, "results", "cytoplasm_apical_basal_overlay.pdf"
                        ),
                        "overlay_tif": os.path.join(
                            base, "results", "cytoplasm_apical_basal_overlay.tif"
                        ),
                    }
                )
    return entries, results_root


STILL_ENTRIES, RESULTS_ROOT = discover_still_entries()
ENTRY_BY_UID = {e["uid"]: e for e in STILL_ENTRIES}


if not STILL_ENTRIES:
    print(
        "WARNING: No *.tif under condition_dir/control_dir (or excluded). "
        "Adjust paths or exclude_files in the config.",
        file=sys.stderr,
    )

CONDITION_CSVS = [e["csv_path"] for e in STILL_ENTRIES if e["group"] == "condition"]
CONTROL_CSVS = [e["csv_path"] for e in STILL_ENTRIES if e["group"] == "control"]
CONDITION_SMOOTHS = [e["smooth_window_px"] for e in STILL_ENTRIES if e["group"] == "condition"]
CONTROL_SMOOTHS = [e["smooth_window_px"] for e in STILL_ENTRIES if e["group"] == "control"]
ALIGN_CONDITION_PEAK = bool(config.get("align_condition_peak", True))
ALIGN_CONTROL_PEAK = bool(config.get("align_control_peak", False))
CENTER_CONTROL_MIDPOINT = bool(config.get("center_control_midpoint", True))


def _group_peak_align_flag(enabled: bool) -> str:
    return "" if enabled else "--no-peak-align"

GROUP_OUT = []
if CONDITION_CSVS:
    GROUP_OUT.append(os.path.join(RESULTS_ROOT, "condition_average.pdf"))
if CONTROL_CSVS:
    GROUP_OUT.append(os.path.join(RESULTS_ROOT, "control_average.pdf"))
COMBINED_OUT = []
if CONDITION_CSVS and CONTROL_CSVS:
    COMBINED_OUT.append(os.path.join(RESULTS_ROOT, "condition_vs_control_average.pdf"))

localrules: still_cytoplasm_draw

rule all:
    input:
        [e["pdf_path"] for e in STILL_ENTRIES]
        + [e["json_path"] for e in STILL_ENTRIES]
        + [e["overlay_pdf"] for e in STILL_ENTRIES]
        + GROUP_OUT
        + COMBINED_OUT,


rule still_cytoplasm_draw:
    input:
        tif=lambda wildcards: ENTRY_BY_UID[wildcards.uid]["tif"],
    output:
        csv=os.path.join(RESULTS_ROOT, "{uid}", "track", "cytoplasm_profile.csv"),
        json=os.path.join(RESULTS_ROOT, "{uid}", "track", "boundary_polylines.json"),
    wildcard_constraints:
        uid="[0-9a-f]{16}",
    params:
        px2=lambda wildcards: ENTRY_BY_UID[wildcards.uid]["px2micron"],
    threads: 999
    conda:
        "../envs/analysis.yaml"
    log:
        os.path.join(RESULTS_ROOT, "{uid}", "logs", "still_cytoplasm_draw.log"),
    shell:
        """
        mkdir -p $(dirname {log:q})
        cd {REPO_ROOT}
        python -u scripts/draw_still_cytoplasm_boundaries.py \
            --input {input.tif:q} \
            --output-csv {output.csv:q} \
            --output-json {output.json:q} \
            --px2micron {params.px2} \
            > {log:q} 2>&1
        """


rule still_cytoplasm_overlay:
    input:
        tif=lambda wildcards: ENTRY_BY_UID[wildcards.uid]["tif"],
        csv=os.path.join(RESULTS_ROOT, "{uid}", "track", "cytoplasm_profile.csv"),
    output:
        png=os.path.join(
            RESULTS_ROOT, "{uid}", "results", "cytoplasm_apical_basal_overlay.png"
        ),
        pdf=os.path.join(
            RESULTS_ROOT, "{uid}", "results", "cytoplasm_apical_basal_overlay.pdf"
        ),
        tif=os.path.join(
            RESULTS_ROOT, "{uid}", "results", "cytoplasm_apical_basal_overlay.tif"
        ),
    wildcard_constraints:
        uid="[0-9a-f]{16}",
    params:
        title_sh=lambda wildcards: shlex.quote(ENTRY_BY_UID[wildcards.uid]["filename"]),
    conda:
        "../envs/analysis.yaml"
    log:
        os.path.join(RESULTS_ROOT, "{uid}", "logs", "still_cytoplasm_overlay.log"),
    shell:
        """
        mkdir -p $(dirname {log:q})
        cd {REPO_ROOT}
        python scripts/export_still_cytoplasm_overlay.py \
            --input-image {input.tif:q} \
            --csv {input.csv:q} \
            --out-png {output.png:q} \
            --out-pdf {output.pdf:q} \
            --out-tif {output.tif:q} \
            --title {params.title_sh} \
            > {log:q} 2>&1
        """


rule still_cytoplasm_plot_per_file:
    input:
        csv=os.path.join(RESULTS_ROOT, "{uid}", "track", "cytoplasm_profile.csv"),
        tif=lambda wildcards: ENTRY_BY_UID[wildcards.uid]["tif"],
    output:
        png=os.path.join(RESULTS_ROOT, "{uid}", "results", "cytoplasm_profile.png"),
        pdf=os.path.join(RESULTS_ROOT, "{uid}", "results", "cytoplasm_profile.pdf"),
    wildcard_constraints:
        uid="[0-9a-f]{16}",
    params:
        title_sh=lambda wildcards: shlex.quote(ENTRY_BY_UID[wildcards.uid]["filename"]),
        smooth=lambda wildcards: int(ENTRY_BY_UID[wildcards.uid]["smooth_window_px"]),
    conda:
        "../envs/analysis.yaml"
    log:
        os.path.join(RESULTS_ROOT, "{uid}", "logs", "still_cytoplasm_plot.log"),
    shell:
        """
        mkdir -p $(dirname {log:q})
        cd {REPO_ROOT}
        python scripts/analysis/plot_still_cytoplasm_profiles.py per-file \
            --csv {input.csv:q} \
            --out-png {output.png:q} \
            --out-pdf {output.pdf:q} \
            --title {params.title_sh} \
            --smooth-window {params.smooth} \
            --input-image {input.tif:q} \
            > {log:q} 2>&1
        """


if CONDITION_CSVS:

    rule still_cytoplasm_group_condition:
        input:
            csvs=CONDITION_CSVS,
        output:
            png=os.path.join(RESULTS_ROOT, "condition_average.png"),
            pdf=os.path.join(RESULTS_ROOT, "condition_average.pdf"),
        conda:
            "../envs/analysis.yaml"
        params:
            csvs_join=" ".join(shlex.quote(p) for p in CONDITION_CSVS),
            smooths_join=" ".join(str(s) for s in CONDITION_SMOOTHS),
            align_flag=lambda w: _group_peak_align_flag(ALIGN_CONDITION_PEAK),
        shell:
            """
            cd {REPO_ROOT}
            python scripts/analysis/plot_still_cytoplasm_profiles.py group \
                --csvs {params.csvs_join} \
                --smooth-windows {params.smooths_join} \
                --out-png {output.png:q} \
                --out-pdf {output.pdf:q} \
                --label condition \
                {params.align_flag} \
            """


if CONTROL_CSVS:

    rule still_cytoplasm_group_control:
        input:
            csvs=CONTROL_CSVS,
        output:
            png=os.path.join(RESULTS_ROOT, "control_average.png"),
            pdf=os.path.join(RESULTS_ROOT, "control_average.pdf"),
        conda:
            "../envs/analysis.yaml"
        params:
            csvs_join=" ".join(shlex.quote(p) for p in CONTROL_CSVS),
            smooths_join=" ".join(str(s) for s in CONTROL_SMOOTHS),
            align_flag=lambda w: _group_peak_align_flag(ALIGN_CONTROL_PEAK),
        shell:
            """
            cd {REPO_ROOT}
            python scripts/analysis/plot_still_cytoplasm_profiles.py group \
                --csvs {params.csvs_join} \
                --smooth-windows {params.smooths_join} \
                --out-png {output.png:q} \
                --out-pdf {output.pdf:q} \
                --label control \
                {params.align_flag}
            """


if CONDITION_CSVS and CONTROL_CSVS:

    rule still_cytoplasm_group_compare:
        input:
            condition_csvs=CONDITION_CSVS,
            control_csvs=CONTROL_CSVS,
        output:
            png=os.path.join(RESULTS_ROOT, "condition_vs_control_average.png"),
            pdf=os.path.join(RESULTS_ROOT, "condition_vs_control_average.pdf"),
        conda:
            "../envs/analysis.yaml"
        params:
            cond_csvs_join=" ".join(shlex.quote(p) for p in CONDITION_CSVS),
            ctrl_csvs_join=" ".join(shlex.quote(p) for p in CONTROL_CSVS),
            cond_smooths_join=" ".join(str(s) for s in CONDITION_SMOOTHS),
            ctrl_smooths_join=" ".join(str(s) for s in CONTROL_SMOOTHS),
            cond_align_flag=lambda w: "" if ALIGN_CONDITION_PEAK else "--condition-no-peak-align",
            ctrl_align_flag=lambda w: "--control-peak-align" if ALIGN_CONTROL_PEAK else "",
            ctrl_mid_flag=lambda w: "" if CENTER_CONTROL_MIDPOINT else "--no-control-midpoint-center",
        shell:
            """
            cd {REPO_ROOT}
            python scripts/analysis/plot_still_cytoplasm_profiles.py group-compare \
                --condition-csvs {params.cond_csvs_join} \
                --control-csvs {params.ctrl_csvs_join} \
                --condition-smooth-windows {params.cond_smooths_join} \
                --control-smooth-windows {params.ctrl_smooths_join} \
                --out-png {output.png:q} \
                --out-pdf {output.pdf:q} \
                {params.cond_align_flag} \
                {params.ctrl_align_flag} \
                {params.ctrl_mid_flag} \
            """

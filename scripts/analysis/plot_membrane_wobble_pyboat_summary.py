#!/usr/bin/env python3
"""
Summarize pyBOAT membrane wobble oscillations from a single results directory.

Input: a sample "results" directory containing one or more:
  MembraneWobble_pyBOAT_*/Position_um_readout.csv

For each pyBOAT folder:
  - onset time: first index where `power` exceeds a threshold
  - post-onset mean period: mean(periods) from onset onward

Output: a tall/narrow 1x2 figure:
  - left: onset time strip plot + mean +/- SD overlay
  - right: post-onset mean period strip plot + mean +/- SD overlay
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

import warnings


FONT_SIZE_LABEL = 30
FONT_SIZE_TICKS = 23
SPINE_LINEWIDTH = 3
TICK_MAJOR_LENGTH = 8
TICK_MAJOR_WIDTH = 3


@dataclass(frozen=True)
class Metric:
    onset_time_min: float
    period_after_onset_mean_min: float
    src_folder: str


def _load_position_um_readout(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pyBOAT Position_um_readout.csv.

    Expected columns:
      time, periods, phase, amplitude, power, frequencies
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing pyBOAT readout: {path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)

    # genfromtxt returns a scalar ndarray if only a single row exists
    if data is None:
        raise ValueError(f"Could not parse: {path}")
    if getattr(data, "ndim", 0) == 0:
        data = np.expand_dims(data, axis=0)

    try:
        time = np.asarray(data["time"], dtype=float)
        periods = np.asarray(data["periods"], dtype=float)
        power = np.asarray(data["power"], dtype=float)
    except Exception as exc:
        raise ValueError(f"Unexpected columns in {path}: {exc}") from exc

    return time, periods, power


def _infer_time_scale_to_minutes(time: np.ndarray) -> float:
    """
    Infer whether `time` is in seconds or minutes using sampling cadence.

    In this repository, config uses movie_time_interval_sec=10 by default.
    If pyBOAT outputs time steps of 10, interpret time as seconds and convert
    to minutes by dividing by 60.
    """
    time = np.asarray(time, dtype=float)
    time = time[np.isfinite(time)]
    if time.size < 2:
        return 1.0

    diffs = np.diff(np.sort(time))
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 1.0

    dt = float(np.median(diffs))
    # Heuristic: dt around 10 indicates seconds; dt around 0.166 indicates minutes.
    if dt > 1.5:
        return 1.0 / 60.0
    return 1.0


def _compute_onset_and_period(
    time: np.ndarray,
    periods: np.ndarray,
    power: np.ndarray,
    power_threshold: float,
) -> Optional[Tuple[float, float]]:
    """
    Compute:
      onset_time_min = time[onset_idx]
      period_after_onset_mean_min = mean(periods[onset_idx:]) (nanmean)
    """
    time = np.asarray(time, dtype=float)
    periods = np.asarray(periods, dtype=float)
    power = np.asarray(power, dtype=float)

    finite = np.isfinite(time) & np.isfinite(periods) & np.isfinite(power)
    if not np.any(finite):
        return None

    time = time[finite]
    periods = periods[finite]
    power = power[finite]

    # Convert time / periods units to minutes (based on time cadence)
    to_min = _infer_time_scale_to_minutes(time)
    time_min = time * to_min
    periods_min = periods * to_min

    max_power = float(np.nanmax(power))
    if not np.isfinite(max_power) or max_power <= 0:
        return None

    # If threshold looks like a fraction (0..1), make it relative to max power.
    if 0.0 <= power_threshold <= 1.0:
        thr = power_threshold * max_power
    else:
        thr = power_threshold

    if not np.isfinite(thr):
        return None

    mask = power >= thr
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return None

    onset_idx = int(idxs[0])

    onset_time_min = float(time_min[onset_idx])
    period_after_onset_mean_min = float(np.nanmean(periods_min[onset_idx:]))
    if not np.isfinite(period_after_onset_mean_min):
        return None

    return onset_time_min, period_after_onset_mean_min


def discover_pyboat_folders(output_dir: Path) -> List[Path]:
    """
    Discover MembraneWobble_pyBOAT_* directories inside output_dir.
    """
    if not output_dir.exists() or not output_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {output_dir}")

    folders = sorted(output_dir.glob("MembraneWobble_pyBOAT_*"))
    folders = [p for p in folders if p.is_dir()]
    return folders


def make_strip_panel(
    ax: plt.Axes,
    values: List[float],
    title: str,
    mean: float,
    sd: float,
    y_bottom: float,
) -> None:
    x_center = 0.0
    if len(values) > 0:
        # Boxplot summary (distribution across pyBOAT folders / ROIs)
        ax.boxplot(
            [values],
            positions=[x_center],
            vert=True,
            # Make boxes wider for improved visibility.
            widths=0.30,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 2.0},
            medianprops={"color": "black", "linewidth": 2.5},
            whiskerprops={"color": "black", "linewidth": 2.0},
            capprops={"color": "black", "linewidth": 2.0},
        )

        rng = np.random.default_rng(0)  # deterministic jitter
        jitter = rng.normal(loc=0.0, scale=0.06, size=len(values))
        ax.scatter(
            np.full(len(values), x_center) + jitter,
            values,
            s=50,
            c="black",
            alpha=0.85,
            zorder=3,
        )

    # Intentionally no mean +/- SD overlay:
    # user request: "Show box with whiskers and individuals as dots. No red".

    ax.set_title(title, fontsize=FONT_SIZE_LABEL, pad=10)
    ax.set_xticks([])
    ax.tick_params(labelsize=FONT_SIZE_TICKS, length=TICK_MAJOR_LENGTH, width=TICK_MAJOR_WIDTH)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINEWIDTH)
    # Match kymograph-style inversion: 0 at top.
    ax.set_ylim(y_bottom, 0.0)
    ax.yaxis.set_major_locator(MultipleLocator(10))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate pyBOAT membrane wobble readouts from one results directory "
            "and plot onset time and post-onset mean period."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Sample results directory containing MembraneWobble_pyBOAT_* folders",
    )
    parser.add_argument(
        "--power-threshold",
        type=float,
        default=0.2,
        help=(
            "Power threshold for onset. If 0..1, treated as a fraction of max(power) "
            "(default 0.2). If >1, treated as absolute power."
        ),
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="MembraneWobble_pyBOAT_onset_period_summary",
        help="Base name for output files (png/pdf) written into output-dir",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pyboat_folders = discover_pyboat_folders(output_dir)
    if not pyboat_folders:
        raise FileNotFoundError(
            f"No MembraneWobble_pyBOAT_* folders found under: {output_dir}"
        )

    power_threshold = float(args.power_threshold)
    metrics: List[Metric] = []
    skipped: List[str] = []
    y_bottom_candidates_min: List[float] = []

    for folder in pyboat_folders:
        readout_path = folder / "Position_um_readout.csv"
        try:
            time, periods, power = _load_position_um_readout(readout_path)

            # Determine full readout time extent (kymograph-like y-range).
            # We intentionally use the readout time range rather than the
            # metric range so the y-axis matches the kymographs.
            to_min = _infer_time_scale_to_minutes(time)
            time_min = np.asarray(time, dtype=float) * float(to_min)
            finite_t = time_min[np.isfinite(time_min)]
            if finite_t.size > 0:
                y_bottom_candidates_min.append(float(np.nanmax(finite_t)))

            res = _compute_onset_and_period(
                time=time,
                periods=periods,
                power=power,
                power_threshold=power_threshold,
            )
            if res is None:
                skipped.append(folder.name)
                continue
            onset_time_min, period_after_onset_mean_min = res
            metrics.append(
                Metric(
                    onset_time_min=onset_time_min,
                    period_after_onset_mean_min=period_after_onset_mean_min,
                    src_folder=folder.name,
                )
            )
        except Exception as exc:
            skipped.append(f"{folder.name} ({exc})")

    if not metrics:
        raise ValueError(
            f"Could not compute metrics from any pyBOAT folders in {output_dir} "
            f"(readout requires Position_um_readout.csv). Skipped: {len(skipped)}"
        )

    onset_vals = [m.onset_time_min for m in metrics]
    period_vals = [m.period_after_onset_mean_min for m in metrics]

    onset_mean = float(np.nanmean(onset_vals))
    period_mean = float(np.nanmean(period_vals))

    ddof = 1 if len(metrics) > 1 else 0
    onset_sd = float(np.nanstd(onset_vals, ddof=ddof))
    period_sd = float(np.nanstd(period_vals, ddof=ddof))

    if y_bottom_candidates_min:
        y_bottom = max(float(np.nanmax(y_bottom_candidates_min)), 1e-9)
    else:
        # Fallback (should be rare): use observed ranges
        y_bottom = max(float(np.nanmax(onset_vals)), float(np.nanmax(period_vals)), 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 10.0), sharey=True, constrained_layout=True)

    make_strip_panel(
        axes[0],
        onset_vals,
        title="Onset",
        mean=onset_mean,
        sd=onset_sd,
        y_bottom=y_bottom,
    )
    axes[0].set_ylabel("Time (min)", fontsize=FONT_SIZE_LABEL)

    make_strip_panel(
        axes[1],
        period_vals,
        title="Period",
        mean=period_mean,
        sd=period_sd,
        y_bottom=y_bottom,
    )
    axes[1].set_ylabel("Time (min)", fontsize=FONT_SIZE_LABEL)

    out_png = output_dir / f"{args.out_name}.png"
    out_pdf = output_dir / f"{args.out_name}.pdf"

    fig.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight", pad_inches=0.05, format="pdf")
    plt.close(fig)

    print("\npyBOAT onset/period summary")
    print("-" * 60)
    print(f"Input results dir    : {output_dir}")
    print(f"pyBOAT folders used   : {len(metrics)} / {len(pyboat_folders)}")
    print(f"Power threshold       : {args.power_threshold} ({'fraction of max' if 0<=power_threshold<=1 else 'absolute'})")
    print(f"Onset time (mean±SD) : {onset_mean:.3f} ± {onset_sd:.3f} min")
    print(f"Post-onset period     : {period_mean:.3f} ± {period_sd:.3f} min")
    if skipped:
        print(f"Skipped ({len(skipped)}) :")
        for s in skipped[:20]:
            print(f"  - {s}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")
    print("-" * 60)
    print(f"Saved figure (png)    : {out_png}")
    print(f"Saved figure (pdf)    : {out_pdf}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from plot_calibration import compute_calibrated_pit_values, compute_rank_values, rank_labels_from_quantiles
from predictor import (
    TimeBlockManager,
    WeatherQuantileArtifact,
    empirical_interval_coverage,
    interval_width,
    load_kmia_training_data,
)


PIT_BIN_COUNT = 10


@dataclass(frozen=True)
class RunSpec:
    """
    One artifact directory and its display label.
    """

    artifact_dir: Path
    label: str


@dataclass
class RunDiagnostics:
    """
    Per-run PIT and rank diagnostics for plotting and summary.
    """

    spec: RunSpec
    pit_values: np.ndarray
    rank_values: np.ndarray
    summary: dict[str, float | int]
    quantiles: Sequence[float]
    uses_distribution_calibration: bool


def parse_run_spec(token: str) -> RunSpec:
    """
    Parse a single ``artifact_dir,label`` token.
    """
    if token.count(",") != 1:
        raise ValueError(f"Run spec must contain exactly one comma: {token!r}")

    artifact_dir_text, label_text = token.split(",", 1)
    artifact_dir = Path(artifact_dir_text.strip())
    label = label_text.strip()
    if not artifact_dir_text.strip() or not label:
        raise ValueError(f"Run spec must contain a non-empty artifact dir and label: {token!r}")

    return RunSpec(artifact_dir=artifact_dir, label=label)


def parse_run_specs(tokens: Sequence[str]) -> list[RunSpec]:
    """
    Parse the full repeated ``--runs`` CLI input.
    """
    specs = [parse_run_spec(token) for token in tokens]
    if not specs:
        raise ValueError("--runs must not be empty.")
    return specs


def compute_figure_size(n_runs: int) -> tuple[float, float]:
    """
    Scale figure size with the number of compared runs.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1.")
    width = 16.0
    height = max(2.4 * n_runs + 2.4, 6.5)
    return width, height


def _reduced_chi_square(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    if counts.size < 2:
        return 0.0

    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0

    expected = total / float(counts.size)
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    dof = float(counts.size - 1)
    return chi2 / dof


def compute_histogram_summary(
    pit_values: np.ndarray,
    rank_values: np.ndarray,
    pit_bin_count: int = PIT_BIN_COUNT,
    rank_bin_count: int | None = None,
) -> dict[str, float | int]:
    """
    Compute central tendency and flatness metrics for PIT and rank histograms.
    """
    pit = np.asarray(pit_values, dtype=float)
    rank = np.asarray(rank_values, dtype=int)
    if pit.ndim != 1 or rank.ndim != 1:
        raise ValueError("pit_values and rank_values must be 1D arrays.")
    if pit.size != rank.size:
        raise ValueError("pit_values and rank_values must have matching lengths.")
    if pit.size == 0:
        raise ValueError("pit_values must not be empty.")
    if pit_bin_count < 2:
        raise ValueError("pit_bin_count must be at least 2.")

    pit_counts, _ = np.histogram(pit, bins=np.linspace(0.0, 1.0, pit_bin_count + 1))
    if rank_bin_count is None:
        rank_bin_count = int(rank.max()) + 1
    if rank_bin_count < 2:
        raise ValueError("rank_bin_count must be at least 2.")
    rank_counts = np.bincount(rank, minlength=rank_bin_count)

    ks_stat = float(stats.kstest(pit, "uniform", args=(0.0, 1.0)).statistic)
    summary: dict[str, float | int] = {
        "n_rows": int(pit.size),
        "mean_pit": float(np.mean(pit)),
        "median_pit": float(np.median(pit)),
        "pit_iqr": float(np.percentile(pit, 75) - np.percentile(pit, 25)),
        "pit_std": float(np.std(pit)),
        "pit_ks_stat": ks_stat,
        "pit_reduced_chi2": float(_reduced_chi_square(pit_counts)),
        "rank_reduced_chi2": float(_reduced_chi_square(rank_counts)),
    }
    return summary


def compute_interval_metrics(
    y_true: np.ndarray,
    pred_df: pd.DataFrame,
    interval_alphas: Sequence[float],
) -> dict[str, float]:
    """
    Compute coverage and width metrics for each supported central interval.
    """
    y = np.asarray(y_true, dtype=float)
    if y.ndim != 1:
        raise ValueError("y_true must be a 1D array.")
    if len(y) == 0:
        raise ValueError("y_true must not be empty.")

    metrics: dict[str, float] = {}
    for alpha in interval_alphas:
        pct = int(round((1.0 - float(alpha)) * 100))
        lower_col = f"interval_{pct}_lower"
        upper_col = f"interval_{pct}_upper"
        if lower_col in pred_df.columns and upper_col in pred_df.columns:
            lower = pred_df[lower_col].to_numpy(dtype=float)
            upper = pred_df[upper_col].to_numpy(dtype=float)
        else:
            q_lo = float(alpha) / 2.0
            q_hi = 1.0 - float(alpha) / 2.0
            lower = pred_df[f"q_{q_lo:.3f}"].to_numpy(dtype=float)
            upper = pred_df[f"q_{q_hi:.3f}"].to_numpy(dtype=float)

        metrics[f"coverage_{pct}"] = empirical_interval_coverage(y, lower, upper)
        metrics[f"width_{pct}"] = interval_width(lower, upper)

    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare multiple KMIA calibration runs.")
    parser.add_argument(
        "--runs",
        nargs="+",
        action="extend",
        required=True,
        help="One or more artifact_dir,label entries, for example artifacts/run_a,base.",
    )
    parser.add_argument("--data", required=True, help="Raw KMIA CSV used to reconstruct the held-out test block.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    return parser


def _load_run_diagnostics(spec: RunSpec, df: pd.DataFrame) -> RunDiagnostics:
    artifact = WeatherQuantileArtifact.load(spec.artifact_dir)
    if artifact.block_cfg is None:
        raise ValueError(f"Artifact {spec.artifact_dir} does not include block_cfg.")

    block_manager = TimeBlockManager(artifact.dataset_cfg, artifact.block_cfg)
    blocks = block_manager.split(df)
    test_df = blocks["test"]
    pred_df = artifact.predict_output_frame(test_df)
    y_true = test_df[artifact.dataset_cfg.target_col].to_numpy()

    pit_values = compute_calibrated_pit_values(y_true, pred_df, artifact.distribution_calibrator)
    rank_values = compute_rank_values(y_true, pred_df)
    summary = compute_histogram_summary(pit_values, rank_values, rank_bin_count=len(artifact.model_cfg.quantiles) + 1)
    summary.update(compute_interval_metrics(y_true, pred_df, artifact.interval_alphas))

    return RunDiagnostics(
        spec=spec,
        pit_values=pit_values,
        rank_values=rank_values,
        summary=summary,
        quantiles=artifact.model_cfg.quantiles,
        uses_distribution_calibration=artifact.distribution_calibrator is not None,
    )


def _format_summary_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def _interval_summary_columns(run_results: Sequence[RunDiagnostics]) -> list[str]:
    pct_values: set[int] = set()
    for run in run_results:
        for key in run.summary:
            if key.startswith("coverage_"):
                pct_values.add(int(key.split("_", 1)[1]))
            elif key.startswith("width_"):
                pct_values.add(int(key.split("_", 1)[1]))

    columns: list[str] = []
    for pct in sorted(pct_values, reverse=True):
        columns.extend([f"coverage_{pct}", f"width_{pct}"])
    return columns


def _render_summary_table(ax: plt.Axes, run_results: Sequence[RunDiagnostics]) -> None:
    ax.axis("off")
    columns = [
        "run",
        "n_rows",
        "mean_pit",
        "median_pit",
        "pit_iqr",
        "pit_std",
        "pit_ks_stat",
        "pit_reduced_chi2",
        "rank_reduced_chi2",
    ]
    columns.extend(_interval_summary_columns(run_results))
    cell_text = [
        [
            run.spec.label,
            _format_summary_value(run.summary["n_rows"]),
            _format_summary_value(run.summary["mean_pit"]),
            _format_summary_value(run.summary["median_pit"]),
            _format_summary_value(run.summary["pit_iqr"]),
            _format_summary_value(run.summary["pit_std"]),
            _format_summary_value(run.summary["pit_ks_stat"]),
            _format_summary_value(run.summary["pit_reduced_chi2"]),
            _format_summary_value(run.summary["rank_reduced_chi2"]),
            *[
                _format_summary_value(run.summary[column])
                for column in _interval_summary_columns(run_results)
            ],
        ]
        for run in run_results
    ]

    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.25)
    ax.set_title("Summary metrics", pad=8)


def build_comparison_figure(run_results: Sequence[RunDiagnostics]) -> plt.Figure:
    n_runs = len(run_results)
    if n_runs < 1:
        raise ValueError("run_results must not be empty.")

    width, height = compute_figure_size(n_runs)
    fig = plt.figure(figsize=(width, height))
    height_ratios = [1.0] * n_runs + [max(1.0, 0.45 * n_runs)]
    grid = fig.add_gridspec(nrows=n_runs + 1, ncols=2, height_ratios=height_ratios, hspace=0.8, wspace=0.3)

    for row_idx, run in enumerate(run_results):
        pit_ax = fig.add_subplot(grid[row_idx, 0])
        rank_ax = fig.add_subplot(grid[row_idx, 1])

        pit_bins = np.linspace(0.0, 1.0, PIT_BIN_COUNT + 1)
        pit_ax.hist(run.pit_values, bins=pit_bins, edgecolor="black")
        pit_ax.set_xlim(0.0, 1.0)
        pit_ax.set_ylabel(run.spec.label, rotation=90, labelpad=20, va="center")
        pit_ax.set_xlabel("PIT value")
        if row_idx == 0:
            pit_ax.set_title(
                "PIT histogram" if not run.uses_distribution_calibration else "PIT histogram (calibrated)"
            )

        rank_bin_count = len(run.quantiles) + 1
        rank_ax.hist(
            run.rank_values,
            bins=np.arange(-0.5, rank_bin_count + 0.5, 1.0),
            edgecolor="black",
            align="mid",
        )
        rank_ax.set_xlim(-0.5, rank_bin_count - 0.5)
        rank_ax.set_xlabel("Quantile interval")
        if row_idx == 0:
            rank_ax.set_title("Quantile rank histogram (raw quantiles)")
        rank_ax.set_xticks(np.arange(rank_bin_count))
        rank_ax.set_xticklabels(rank_labels_from_quantiles(run.quantiles), rotation=45, ha="right")

    summary_ax = fig.add_subplot(grid[n_runs, :])
    _render_summary_table(summary_ax, run_results)
    fig.suptitle("KMIA calibration matrix", y=0.995)
    fig.subplots_adjust(top=0.96, bottom=0.04, left=0.05, right=0.995)
    return fig


def run_comparison(run_specs: Sequence[RunSpec], data_path: str | Path, output_path: str | Path) -> list[RunDiagnostics]:
    df = load_kmia_training_data(data_path)
    run_results = [_load_run_diagnostics(spec, df) for spec in run_specs]
    fig = build_comparison_figure(run_results)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return run_results


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_specs = parse_run_specs(args.runs)
    run_results = run_comparison(run_specs, args.data, args.output)

    print(f"Wrote calibration matrix to {Path(args.output)}")
    for run in run_results:
        print(
            f"{run.spec.label}: n_rows={run.summary['n_rows']}, "
            f"mean_pit={run.summary['mean_pit']:.4f}, "
            f"pit_ks_stat={run.summary['pit_ks_stat']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

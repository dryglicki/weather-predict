from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from market_scoring import build_cdf_from_quantiles
from predictor import DistributionCalibrator, TimeBlockManager, WeatherQuantileArtifact, load_kmia_training_data


def rank_labels_from_quantiles(quantiles: list[float] | tuple[float, ...] | np.ndarray) -> list[str]:
    q_values = [float(q) for q in quantiles]
    if not q_values:
        raise ValueError("quantiles must not be empty.")

    labels = [f"<q_{q_values[0]:.2f}"]
    for left, right in zip(q_values[:-1], q_values[1:]):
        labels.append(f"q_{left:.2f}-q_{right:.2f}")
    labels.append(f">q_{q_values[-1]:.2f}")
    return labels


def compute_pit_values(y_true: np.ndarray | pd.Series, pred_df: pd.DataFrame) -> np.ndarray:
    """
    Compute raw PIT values from repaired quantile predictions.

    Parameters
    ----------
    y_true : np.ndarray or pandas.Series
        Observed target values.
    pred_df : pandas.DataFrame
        Forecast dataframe containing q_* quantile columns.

    Returns
    -------
    np.ndarray
        Raw PIT values in [0, 1].
    """
    y_arr = np.asarray(y_true, dtype=float)
    pit_values = np.empty(len(y_arr), dtype=float)

    for i, (_, row) in enumerate(pred_df.iterrows()):
        pit_values[i] = build_cdf_from_quantiles(row)(float(y_arr[i]))

    return pit_values


def compute_calibrated_pit_values(
    y_true: np.ndarray | pd.Series,
    pred_df: pd.DataFrame,
    distribution_calibrator: DistributionCalibrator | None = None,
) -> np.ndarray:
    """
    Compute PIT values after optional monotone distribution calibration.

    Parameters
    ----------
    y_true : np.ndarray or pandas.Series
        Observed target values.
    pred_df : pandas.DataFrame
        Forecast dataframe containing q_* quantile columns.
    distribution_calibrator : DistributionCalibrator or None
        Optional monotone PIT remap loaded from the artifact bundle.

    Returns
    -------
    np.ndarray
        Calibrated PIT values in [0, 1]. When no calibrator is present, these
        match the raw PIT values.
    """
    pit_values = compute_pit_values(y_true, pred_df)
    if distribution_calibrator is None:
        return pit_values
    return distribution_calibrator.transform(pit_values)


def compute_rank_values(y_true: np.ndarray | pd.Series, pred_df: pd.DataFrame) -> np.ndarray:
    y_arr = np.asarray(y_true, dtype=float)
    quantile_cols = [col for col in pred_df.columns if isinstance(col, str) and col.startswith("q_")]
    quantile_cols.sort()
    rank_values = np.empty(len(y_arr), dtype=int)

    for i, (_, row) in enumerate(pred_df.iterrows()):
        q_values = row[quantile_cols].to_numpy(dtype=float)
        rank_values[i] = int(np.searchsorted(q_values, float(y_arr[i]), side="right"))

    return rank_values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot PIT and rank histograms for KMIA held-out test predictions.")
    parser.add_argument("--artifact-dir", required=True, help="Directory containing model.cbm and artifact.json.")
    parser.add_argument("--data", required=True, help="Raw KMIA CSV used to reconstruct the held-out test block.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    return parser


def _plot_histograms(
    raw_pit_values: np.ndarray,
    calibrated_pit_values: np.ndarray,
    rank_values: np.ndarray,
    quantiles: list[float] | tuple[float, ...] | np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].hist(raw_pit_values, bins=np.linspace(0.0, 1.0, 11), edgecolor="black")
    axes[0].set_title("Raw PIT histogram")
    axes[0].set_xlabel("PIT value")
    axes[0].set_ylabel("Count")
    axes[0].set_xlim(0.0, 1.0)

    axes[1].hist(calibrated_pit_values, bins=np.linspace(0.0, 1.0, 11), edgecolor="black")
    axes[1].set_title("Calibrated PIT histogram")
    axes[1].set_xlabel("PIT value")
    axes[1].set_ylabel("Count")
    axes[1].set_xlim(0.0, 1.0)

    axes[2].hist(
        rank_values,
        bins=np.arange(-0.5, len(quantiles) + 1.5, 1.0),
        edgecolor="black",
        align="mid",
    )
    axes[2].set_title("Quantile rank histogram")
    axes[2].set_xlabel("Quantile interval")
    axes[2].set_ylabel("Count")
    axes[2].set_xlim(-0.5, len(quantiles) + 0.5)
    axes[2].set_xticks(np.arange(len(quantiles) + 1))
    axes[2].set_xticklabels(rank_labels_from_quantiles(quantiles), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()

    artifact = WeatherQuantileArtifact.load(Path(args.artifact_dir))
    if artifact.block_cfg is None:
        raise ValueError("artifact.json does not include block_cfg, so the held-out test block cannot be reconstructed.")

    df = load_kmia_training_data(args.data)
    block_manager = TimeBlockManager(artifact.dataset_cfg, artifact.block_cfg)
    blocks = block_manager.split(df)
    test_df = blocks["test"]
    pred_df = artifact.predict_output_frame(test_df)

    y_true = test_df[artifact.dataset_cfg.target_col].to_numpy()
    raw_pit_values = compute_pit_values(y_true, pred_df)
    calibrated_pit_values = compute_calibrated_pit_values(y_true, pred_df, artifact.distribution_calibrator)
    rank_values = compute_rank_values(y_true, pred_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_histograms(
        raw_pit_values,
        calibrated_pit_values,
        rank_values,
        artifact.model_cfg.quantiles,
        output_path,
    )

    print(f"Wrote calibration plot to {output_path}")
    print(f"Test rows: {len(test_df)}")
    print(f"Mean raw PIT: {float(np.mean(raw_pit_values)):.4f}")
    print(f"Mean calibrated PIT: {float(np.mean(calibrated_pit_values)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from predictor import (
    ExpandingWindowCVConfig,
    DistributionCalibrator,
    QuantileModelConfig,
    WeatherQuantilePipeline,
    build_default_kmia_block_split_config,
    build_kmia_dataset_config,
    load_kmia_training_data,
)
from plot_calibration import (
    compute_calibrated_pit_values,
    compute_pit_values,
    compute_rank_values,
    rank_labels_from_quantiles,
)


DATA_PATH = Path(__file__).resolve().parents[1] / "kalshiTraining_KMIA.dat"


class PlotCalibrationTests(unittest.TestCase):
    def test_compute_pit_values_and_rank_values(self) -> None:
        pred_df = pd.DataFrame(
            {
                "q_0.050": [80.0, 80.0],
                "q_0.500": [90.0, 90.0],
                "q_0.950": [100.0, 100.0],
            }
        )
        y_true = np.array([85.0, 95.0])

        pit_values = compute_pit_values(y_true, pred_df)
        rank_values = compute_rank_values(y_true, pred_df)

        self.assertEqual(len(pit_values), 2)
        self.assertTrue(np.all(pit_values >= 0.0))
        self.assertTrue(np.all(pit_values <= 1.0))
        self.assertListEqual(rank_values.tolist(), [1, 2])

    def test_compute_calibrated_pit_values_uses_distribution_calibrator(self) -> None:
        pred_df = pd.DataFrame(
            {
                "q_0.050": [80.0, 80.0],
                "q_0.500": [90.0, 90.0],
                "q_0.950": [100.0, 100.0],
            }
        )
        y_true = np.array([85.0, 95.0])
        calibrator = DistributionCalibrator(
            pit_knots=np.array([0.0, 0.5, 1.0], dtype=float),
            calibrated_knots=np.array([0.0, 0.75, 1.0], dtype=float),
        )

        raw_pit_values = compute_pit_values(y_true, pred_df)
        calibrated_pit_values = compute_calibrated_pit_values(y_true, pred_df, calibrator)

        self.assertEqual(len(calibrated_pit_values), 2)
        self.assertTrue(np.all(calibrated_pit_values >= 0.0))
        self.assertTrue(np.all(calibrated_pit_values <= 1.0))
        self.assertFalse(np.allclose(raw_pit_values, calibrated_pit_values))

    def test_rank_labels_from_quantiles_uses_actual_interval_bounds(self) -> None:
        labels = rank_labels_from_quantiles([0.05, 0.10, 0.20, 0.25])

        self.assertEqual(
            labels,
            ["<q_0.05", "q_0.05-q_0.10", "q_0.10-q_0.20", "q_0.20-q_0.25", ">q_0.25"],
        )

    def test_plot_calibration_cli_writes_png_from_held_out_test_block(self) -> None:
        df = load_kmia_training_data(DATA_PATH)
        dataset_cfg = build_kmia_dataset_config()
        block_cfg = build_default_kmia_block_split_config(df)
        cv_cfg = ExpandingWindowCVConfig(min_train_days=365, val_days=30, step_days=90, max_folds=1)
        model_cfg = QuantileModelConfig(
            quantiles=[0.05, 0.50, 0.95],
            iterations=5,
            learning_rate=0.1,
            depth=4,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=2,
            task_type="CPU",
        )
        pipeline = WeatherQuantilePipeline(dataset_cfg, block_cfg, cv_cfg, model_cfg, interval_alphas=[0.10])
        pipeline.fit_final(df)
        distribution_calibrator = DistributionCalibrator(
            pit_knots=np.array([0.0, 0.5, 1.0], dtype=float),
            calibrated_knots=np.array([0.0, 0.75, 1.0], dtype=float),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifact"
            artifact = pipeline.build_artifact(distribution_calibrator=distribution_calibrator)
            artifact.save(artifact_dir)

            output_png = Path(tmpdir) / "calibration.png"
            result = subprocess.run(
                [
                    sys.executable,
                    "plot_calibration.py",
                    "--artifact-dir",
                    str(artifact_dir),
                    "--data",
                    str(DATA_PATH),
                    "--output",
                    str(output_png),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output_png.exists())
            self.assertGreater(output_png.stat().st_size, 0)
            self.assertIn("Test rows:", result.stdout)
            self.assertIn("Mean raw PIT:", result.stdout)
            self.assertIn("Mean calibrated PIT:", result.stdout)

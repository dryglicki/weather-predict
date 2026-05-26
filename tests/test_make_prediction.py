from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from predictor import (
    ExpandingWindowCVConfig,
    QuantileModelConfig,
    WeatherQuantilePipeline,
    build_default_kmia_block_split_config,
    build_kmia_dataset_config,
    load_kmia_training_data,
)


DATA_PATH = Path(__file__).resolve().parents[1] / "kalshiTraining_KMIA.dat"


class MakePredictionCLITests(unittest.TestCase):
    def test_make_prediction_cli_handles_single_row_csv(self) -> None:
        df = load_kmia_training_data(DATA_PATH)
        dataset_cfg = build_kmia_dataset_config()
        block_cfg = build_default_kmia_block_split_config(df)
        cv_cfg = ExpandingWindowCVConfig(min_train_days=365, val_days=30, step_days=90, max_folds=1)
        model_cfg = QuantileModelConfig(
            quantiles=[0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95],
            iterations=5,
            learning_rate=0.1,
            depth=4,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=2,
            task_type="CPU",
        )
        pipeline = WeatherQuantilePipeline(dataset_cfg, block_cfg, cv_cfg, model_cfg)
        pipeline.fit_final(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifact"
            pipeline.save_artifact(artifact_dir)

            input_csv = Path(tmpdir) / "single_row.csv"
            pd.DataFrame([pipeline.blocks_["test"].iloc[0].drop(labels=["y"])]).to_csv(input_csv, index=False)
            output_csv = Path(tmpdir) / "single_row_predictions.csv"

            result = subprocess.run(
                [
                    sys.executable,
                    "make_prediction.py",
                    "--artifact-dir",
                    str(artifact_dir),
                    "--input",
                    str(input_csv),
                    "--output",
                    str(output_csv),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            pred_df = pd.read_csv(output_csv)
            self.assertEqual(len(pred_df), 1)
            self.assertIn("interval_90_lower", pred_df.columns)
            self.assertIn("interval_50_upper", pred_df.columns)

    def test_make_prediction_cli_handles_batch_csv(self) -> None:
        df = load_kmia_training_data(DATA_PATH)
        dataset_cfg = build_kmia_dataset_config()
        block_cfg = build_default_kmia_block_split_config(df)
        cv_cfg = ExpandingWindowCVConfig(min_train_days=365, val_days=30, step_days=90, max_folds=1)
        model_cfg = QuantileModelConfig(
            quantiles=[0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95],
            iterations=5,
            learning_rate=0.1,
            depth=4,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=2,
            task_type="CPU",
        )
        pipeline = WeatherQuantilePipeline(dataset_cfg, block_cfg, cv_cfg, model_cfg)
        pipeline.fit_final(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifact"
            pipeline.save_artifact(artifact_dir)

            input_csv = Path(tmpdir) / "batch.csv"
            batch_df = pipeline.blocks_["test"].head(3).drop(columns=["y"])
            batch_df.to_csv(input_csv, index=False)
            output_csv = Path(tmpdir) / "batch_predictions.csv"

            result = subprocess.run(
                [
                    sys.executable,
                    "make_prediction.py",
                    "--artifact-dir",
                    str(artifact_dir),
                    "--input",
                    str(input_csv),
                    "--output",
                    str(output_csv),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            pred_df = pd.read_csv(output_csv)
            self.assertEqual(len(pred_df), 3)
            self.assertIn("interval_90_lower", pred_df.columns)
            self.assertIn("interval_20_upper", pred_df.columns)

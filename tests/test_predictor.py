from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from predictor import (
    DatasetConfig,
    BlockSplitConfig,
    ExpandingWindowCVConfig,
    KMIA_FEATURE_COLUMNS,
    QuantileModelConfig,
    WeatherQuantilePipeline,
    build_default_kmia_block_split_config,
    build_kmia_dataset_config,
    build_phase1_config,
    build_phase2_config_from_phase1,
    load_kmia_training_data,
    run_kmia_smoke_test,
    supported_interval_alphas,
    split_final_train_eval,
    validate_interval_quantiles,
)


DATA_PATH = Path(__file__).resolve().parents[1] / "kalshiTraining_KMIA.dat"


class KMIATrainingDataTests(unittest.TestCase):
    def test_load_kmia_training_data_rejects_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing_kmia_columns.csv"
            path.write_text("validDate,nbmMaxT,observedMaxT\nMAY-20-2020,88,94\n")

            with self.assertRaisesRegex(ValueError, "Missing required KMIA columns"):
                load_kmia_training_data(path)

    def test_load_kmia_training_data_returns_pipeline_ready_frame(self) -> None:
        df = load_kmia_training_data(DATA_PATH)

        self.assertEqual(len(df), 2143)
        self.assertIn("date", df.columns)
        self.assertIn("y", df.columns)
        self.assertNotIn("validDate", df.columns)
        self.assertNotIn("observedMaxT", df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["date"]))
        self.assertTrue(df["date"].is_monotonic_increasing)
        self.assertEqual(df["date"].iloc[0], pd.Timestamp("2020-05-20"))
        self.assertEqual(df["date"].iloc[-1], pd.Timestamp("2026-04-18"))
        self.assertEqual(df["y"].iloc[0], 94)

    def test_build_kmia_dataset_config_maps_features_and_target(self) -> None:
        dataset_cfg = build_kmia_dataset_config()

        self.assertEqual(dataset_cfg.date_col, "date")
        self.assertEqual(dataset_cfg.target_col, "y")
        self.assertEqual(list(dataset_cfg.feature_cols), list(KMIA_FEATURE_COLUMNS))
        self.assertIsNone(dataset_cfg.categorical_cols)

    def test_build_default_kmia_block_split_config_uses_actual_data_range(self) -> None:
        df = load_kmia_training_data(DATA_PATH)

        block_cfg = build_default_kmia_block_split_config(df)

        self.assertEqual(block_cfg.dev_start, "2020-05-20")
        self.assertEqual(block_cfg.dev_end, "2024-04-18")
        self.assertEqual(block_cfg.cal_start, "2024-04-19")
        self.assertEqual(block_cfg.cal_end, "2025-04-18")
        self.assertEqual(block_cfg.test_start, "2025-04-19")
        self.assertEqual(block_cfg.test_end, "2026-04-18")

    def test_main_example_uses_kmia_loader_and_config(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "predictor.py").read_text()

        self.assertIn('load_kmia_training_data("kalshiTraining_KMIA.dat")', source)
        self.assertIn("dataset_cfg = build_kmia_dataset_config()", source)
        self.assertIn("block_cfg = build_default_kmia_block_split_config(df)", source)
        self.assertIn('if "--smoke" in sys.argv:', source)
        self.assertIn("run_kmia_smoke_test(", source)
        self.assertNotIn('pd.read_parquet("station_temperature_features.parquet")', source)


class QuantileIntervalTests(unittest.TestCase):
    def test_supported_interval_alphas_from_phase2_grid(self) -> None:
        quantiles = [0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95]
        self.assertEqual(
            supported_interval_alphas(quantiles),
            [0.10, 0.20, 0.40, 0.50, 0.60, 0.80],
        )

    def test_phase1_quantiles_support_default_ninety_percent_interval(self) -> None:
        model_cfg = build_phase1_config()

        validate_interval_quantiles(model_cfg.quantiles, [0.10])

    def test_phase1_config_uses_cpu(self) -> None:
        model_cfg = build_phase1_config()

        self.assertEqual(model_cfg.task_type, "CPU")

    def test_phase2_config_preserves_cpu_for_multiquantile_training(self) -> None:
        phase1_cfg = build_phase1_config()
        phase2_cfg = build_phase2_config_from_phase1(phase1_cfg)

        self.assertEqual(phase2_cfg.task_type, "CPU")

    def test_validate_interval_quantiles_rejects_missing_endpoints(self) -> None:
        with self.assertRaisesRegex(ValueError, "0.05"):
            validate_interval_quantiles([0.10, 0.50, 0.90], [0.10])

    def test_pipeline_rejects_interval_quantiles_missing_endpoints(self) -> None:
        dataset_cfg = DatasetConfig(date_col="date", target_col="y")
        block_cfg = BlockSplitConfig(
            dev_start="2024-01-01",
            dev_end="2024-01-31",
            cal_start="2024-02-01",
            cal_end="2024-02-15",
            test_start="2024-02-16",
            test_end="2024-02-29",
        )
        cv_cfg = ExpandingWindowCVConfig(min_train_days=7, val_days=3, step_days=3)
        model_cfg = QuantileModelConfig(quantiles=[0.10, 0.50, 0.90])

        with self.assertRaisesRegex(ValueError, "0.05"):
            WeatherQuantilePipeline(
                dataset_cfg=dataset_cfg,
                block_cfg=block_cfg,
                cv_cfg=cv_cfg,
                model_cfg=model_cfg,
                interval_alphas=[0.10],
            )


class FinalTrainingSplitTests(unittest.TestCase):
    def test_split_final_train_eval_uses_tail_of_development_block(self) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        dev_df = pd.DataFrame({"date": dates, "y": range(10)})
        dataset_cfg = DatasetConfig(date_col="date", target_col="y")

        train_df, eval_df = split_final_train_eval(dev_df, dataset_cfg, eval_days=3)

        self.assertEqual(train_df["date"].iloc[0], pd.Timestamp("2024-01-01"))
        self.assertEqual(train_df["date"].iloc[-1], pd.Timestamp("2024-01-07"))
        self.assertEqual(eval_df["date"].iloc[0], pd.Timestamp("2024-01-08"))
        self.assertEqual(eval_df["date"].iloc[-1], pd.Timestamp("2024-01-10"))
        self.assertLess(train_df["date"].max(), eval_df["date"].min())
        self.assertEqual(set(train_df.index).intersection(set(eval_df.index)), set())

    def test_split_final_train_eval_rejects_empty_train_or_eval(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        dev_df = pd.DataFrame({"date": dates, "y": range(3)})
        dataset_cfg = DatasetConfig(date_col="date", target_col="y")

        with self.assertRaisesRegex(ValueError, "final eval"):
            split_final_train_eval(dev_df, dataset_cfg, eval_days=3)


class KMIASmokeTests(unittest.TestCase):
    def test_predictor_smoke_entrypoint_exits_zero_with_metrics(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        result = subprocess.run(
            [sys.executable, "predictor.py", "--smoke"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Smoke metrics:", result.stdout)
        self.assertIn("'cal'", result.stdout)
        self.assertIn("'test'", result.stdout)

    def test_run_kmia_smoke_test_returns_cal_and_test_metrics(self) -> None:
        metrics = run_kmia_smoke_test(DATA_PATH, task_type="CPU")

        self.assertEqual(set(metrics), {"cal", "test"})
        for block_metrics in metrics.values():
            self.assertIn("mean_pinball_loss", block_metrics)
            self.assertIn("coverage_90", block_metrics)
            self.assertIn("width_90", block_metrics)
            self.assertIn("crossing_rate_after_repair", block_metrics)
            self.assertTrue(math.isfinite(block_metrics["mean_pinball_loss"]))
            self.assertGreaterEqual(block_metrics["mean_pinball_loss"], 0.0)
            self.assertGreaterEqual(block_metrics["coverage_90"], 0.0)
            self.assertLessEqual(block_metrics["coverage_90"], 1.0)
            self.assertTrue(math.isfinite(block_metrics["width_90"]))
            self.assertGreaterEqual(block_metrics["width_90"], 0.0)
            self.assertEqual(block_metrics["crossing_rate_after_repair"], 0.0)

    def test_pipeline_artifact_round_trip_saves_cbm_and_json(self) -> None:
        df = load_kmia_training_data(DATA_PATH)
        dataset_cfg = build_kmia_dataset_config()
        block_cfg = build_default_kmia_block_split_config(df)
        cv_cfg = ExpandingWindowCVConfig(
            min_train_days=365,
            val_days=30,
            step_days=90,
            max_folds=1,
        )
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
        pipeline = WeatherQuantilePipeline(
            dataset_cfg=dataset_cfg,
            block_cfg=block_cfg,
            cv_cfg=cv_cfg,
            model_cfg=model_cfg,
            interval_alphas=[0.10],
        )
        pipeline.fit_final(df)

        sample_df = pipeline.blocks_["test"].head(5)
        expected = pipeline.predict_distribution_inputs(sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "kmia_artifact"
            artifact = pipeline.build_artifact(metadata={"run_name": "artifact_round_trip"})
            artifact.save(artifact_dir)

            self.assertTrue((artifact_dir / "model.cbm").exists())
            self.assertTrue((artifact_dir / "artifact.json").exists())

            loaded = type(artifact).load(artifact_dir)
            actual = loaded.predict_distribution_inputs(sample_df)

            np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy())
            self.assertEqual(loaded.metadata["run_name"], "artifact_round_trip")


if __name__ == "__main__":
    unittest.main()

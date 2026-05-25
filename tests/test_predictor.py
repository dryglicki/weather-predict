from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

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
    load_kmia_training_data,
    run_kmia_smoke_test,
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
        self.assertNotIn('pd.read_parquet("station_temperature_features.parquet")', source)


class QuantileIntervalTests(unittest.TestCase):
    def test_phase1_quantiles_support_default_ninety_percent_interval(self) -> None:
        model_cfg = build_phase1_config()

        validate_interval_quantiles(model_cfg.quantiles, [0.10])

    def test_phase1_config_uses_gpu(self) -> None:
        model_cfg = build_phase1_config()

        self.assertEqual(model_cfg.task_type, "GPU")

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
    def test_run_kmia_smoke_test_returns_cal_and_test_metrics(self) -> None:
        metrics = run_kmia_smoke_test(DATA_PATH, task_type="CPU")

        self.assertEqual(set(metrics), {"cal", "test"})
        for block_metrics in metrics.values():
            self.assertIn("mean_pinball_loss", block_metrics)
            self.assertIn("coverage_90", block_metrics)
            self.assertIn("width_90", block_metrics)
            self.assertIn("crossing_rate_after_repair", block_metrics)


if __name__ == "__main__":
    unittest.main()

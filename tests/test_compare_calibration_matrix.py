from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
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


class CompareCalibrationMatrixTests(unittest.TestCase):
    def test_parse_run_specs_accepts_repeated_comma_entries(self) -> None:
        from compare_calibration_matrix import parse_run_specs

        specs = parse_run_specs(["/tmp/run_a,alpha", "/tmp/run_b,beta"])

        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].artifact_dir, Path("/tmp/run_a"))
        self.assertEqual(specs[0].label, "alpha")
        self.assertEqual(specs[1].artifact_dir, Path("/tmp/run_b"))
        self.assertEqual(specs[1].label, "beta")

    def test_parse_run_specs_rejects_malformed_entries(self) -> None:
        from compare_calibration_matrix import parse_run_specs

        with self.assertRaises(ValueError):
            parse_run_specs(["/tmp/run_a"])

    def test_build_parser_accepts_multiple_run_tokens_after_one_flag(self) -> None:
        from compare_calibration_matrix import build_parser

        args = build_parser().parse_args(
            [
                "--runs",
                "artifacts/run_a,base",
                "artifacts/run_b,interval",
                "--data",
                "kalshiTraining_KMIA.dat",
                "--output",
                "comparison.png",
            ]
        )

        self.assertEqual(
            args.runs,
            [
                "artifacts/run_a,base",
                "artifacts/run_b,interval",
            ],
        )
        self.assertEqual(args.data, "kalshiTraining_KMIA.dat")
        self.assertEqual(args.output, "comparison.png")

    def test_compute_histogram_summary_reports_flatness_metrics(self) -> None:
        from compare_calibration_matrix import compute_histogram_summary

        pit_values = np.array([0.0, 0.5, 1.0], dtype=float)
        rank_values = np.array([0, 1, 2], dtype=int)

        summary = compute_histogram_summary(pit_values, rank_values, pit_bin_count=10, rank_bin_count=4)

        self.assertEqual(summary["n_rows"], 3)
        self.assertAlmostEqual(summary["mean_pit"], 0.5, places=6)
        self.assertAlmostEqual(summary["median_pit"], 0.5, places=6)
        self.assertAlmostEqual(summary["pit_iqr"], 0.5, places=6)
        self.assertAlmostEqual(summary["pit_std"], float(np.std(pit_values)), places=6)
        self.assertAlmostEqual(summary["pit_ks_stat"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(summary["pit_reduced_chi2"], 7.0 / 9.0, places=6)
        self.assertAlmostEqual(summary["rank_reduced_chi2"], 1.0 / 3.0, places=6)

    def test_compute_interval_metrics_reports_coverage_and_width(self) -> None:
        from compare_calibration_matrix import compute_interval_metrics

        pred_df = pd.DataFrame(
            {
                "q_0.050": [80.0, 80.0],
                "q_0.500": [90.0, 90.0],
                "q_0.950": [100.0, 100.0],
            }
        )
        y_true = np.array([85.0, 95.0], dtype=float)

        metrics = compute_interval_metrics(y_true, pred_df, [0.10])

        self.assertIn("coverage_90", metrics)
        self.assertIn("width_90", metrics)
        self.assertAlmostEqual(metrics["coverage_90"], 1.0, places=6)
        self.assertAlmostEqual(metrics["width_90"], 20.0, places=6)

    def test_compute_figure_size_scales_with_run_count(self) -> None:
        from compare_calibration_matrix import compute_figure_size

        width_one, height_one = compute_figure_size(1)
        width_four, height_four = compute_figure_size(4)

        self.assertGreater(height_four, height_one)
        self.assertEqual(width_one, width_four)

    def test_compare_calibration_matrix_cli_writes_png(self) -> None:
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

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            artifact_root = tmpdir_path / "artifacts"
            artifact_root.mkdir()
            base_artifact_dir = artifact_root / "run_a"
            pipeline.build_artifact().save(base_artifact_dir)

            run_dirs = []
            for label in ("run_a", "run_b", "run_c", "run_d"):
                run_dir = artifact_root / label
                if run_dir != base_artifact_dir:
                    shutil.copytree(base_artifact_dir, run_dir)
                run_dirs.append(run_dir)

            output_png = tmpdir_path / "comparison.png"
            result = subprocess.run(
                [
                    sys.executable,
                    "compare_calibration_matrix.py",
                    "--runs",
                    f"{run_dirs[0]},alpha",
                    "--runs",
                    f"{run_dirs[1]},beta",
                    "--runs",
                    f"{run_dirs[2]},gamma",
                    "--runs",
                    f"{run_dirs[3]},delta",
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

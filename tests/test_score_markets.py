from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

import pandas as pd


class ScoreMarketsCLITests(unittest.TestCase):
    def _latest_phase2_artifact_dir(self) -> Path:
        artifact_root = Path(__file__).resolve().parents[1] / "artifacts"
        candidates = sorted(artifact_root.glob("phase2_denser_quantiles_*"))
        if not candidates:
            raise AssertionError("No phase2_denser_quantiles_* artifact directories found.")
        return candidates[-1]

    def test_score_markets_cli_ranks_miami_style_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            forecast_csv = Path(tmpdir) / "forecast.csv"
            markets_csv = Path(tmpdir) / "markets.csv"
            output_csv = Path(tmpdir) / "scored.csv"

            pd.DataFrame(
                [
                    {
                        "date": "2026-05-26",
                        "q_0.050": 80.0,
                        "q_0.100": 82.0,
                        "q_0.200": 85.0,
                        "q_0.250": 86.0,
                        "q_0.300": 87.0,
                        "q_0.400": 89.0,
                        "q_0.500": 90.0,
                        "q_0.600": 91.0,
                        "q_0.700": 92.0,
                        "q_0.750": 93.0,
                        "q_0.800": 94.0,
                        "q_0.900": 96.0,
                        "q_0.950": 98.0,
                    }
                ]
            ).to_csv(forecast_csv, index=False)

            pd.DataFrame(
                [
                    {
                        "market_id": "mia_1",
                        "market_name": "Miami <=84",
                        "comparison": "le",
                        "lower_bound": "",
                        "upper_bound": 84,
                        "yes_price": 1,
                        "no_price": 99,
                    },
                    {
                        "market_id": "mia_2",
                        "market_name": "Miami 85-86",
                        "comparison": "between",
                        "lower_bound": 85,
                        "upper_bound": 86,
                        "yes_price": 6,
                        "no_price": 98,
                    },
                    {
                        "market_id": "mia_3",
                        "market_name": "Miami 87-88",
                        "comparison": "between",
                        "lower_bound": 87,
                        "upper_bound": 88,
                        "yes_price": 31,
                        "no_price": 72,
                    },
                    {
                        "market_id": "mia_4",
                        "market_name": "Miami 89-90",
                        "comparison": "between",
                        "lower_bound": 89,
                        "upper_bound": 90,
                        "yes_price": 59,
                        "no_price": 47,
                    },
                    {
                        "market_id": "mia_5",
                        "market_name": "Miami 91-92",
                        "comparison": "between",
                        "lower_bound": 91,
                        "upper_bound": 92,
                        "yes_price": 12,
                        "no_price": 93,
                    },
                    {
                        "market_id": "mia_6",
                        "market_name": "Miami >=93",
                        "comparison": "ge",
                        "lower_bound": 93,
                        "upper_bound": "",
                        "yes_price": 2,
                        "no_price": 98,
                    },
                ]
            ).to_csv(markets_csv, index=False)

            result = subprocess.run(
                [
                    sys.executable,
                    "score_markets.py",
                    "--forecast",
                    str(forecast_csv),
                    "--markets",
                    str(markets_csv),
                    "--output",
                    str(output_csv),
                    "--fee-rate",
                    "0.02",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            scored_df = pd.read_csv(output_csv)
            self.assertEqual(len(scored_df), 6)
            self.assertIn("p_yes", scored_df.columns)
            self.assertIn("yes_ev_cents", scored_df.columns)
            self.assertIn("recommended_side", scored_df.columns)
            self.assertEqual(scored_df.iloc[0]["market_id"], "mia_4")

    def test_score_markets_cli_can_enable_distribution_calibration_from_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            forecast_csv = tmpdir_path / "forecast.csv"
            markets_csv = tmpdir_path / "markets.csv"
            output_csv = tmpdir_path / "scored.csv"
            artifact_dir = tmpdir_path / "artifact"

            shutil.copytree(self._latest_phase2_artifact_dir(), artifact_dir)
            payload_path = artifact_dir / "artifact.json"
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            payload["distribution_calibration"] = {
                "pit_knots": [0.0, 0.5, 1.0],
                "calibrated_knots": [0.0, 0.75, 1.0],
            }
            payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            pd.DataFrame(
                [
                    {
                        "date": "2026-05-26",
                        "q_0.050": 80.0,
                        "q_0.500": 90.0,
                        "q_0.950": 100.0,
                    }
                ]
            ).to_csv(forecast_csv, index=False)

            pd.DataFrame(
                [
                    {
                        "market_id": "mia_1",
                        "market_name": "Miami 88-89",
                        "comparison": "between",
                        "lower_bound": 88,
                        "upper_bound": 89,
                        "yes_price": 20,
                        "no_price": 80,
                    }
                ]
            ).to_csv(markets_csv, index=False)

            result = subprocess.run(
                [
                    sys.executable,
                    "score_markets.py",
                    "--forecast",
                    str(forecast_csv),
                    "--markets",
                    str(markets_csv),
                    "--output",
                    str(output_csv),
                    "--artifact-dir",
                    str(artifact_dir),
                    "--distribution-calibration",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            scored_df = pd.read_csv(output_csv)
            self.assertIn("calibrated_p_yes", scored_df.columns)
            self.assertIn("calibrated_yes_ev_cents", scored_df.columns)
            self.assertIn("calibrated_no_ev_cents", scored_df.columns)
            self.assertIn("raw_p_yes", scored_df.columns)
            self.assertNotEqual(scored_df.iloc[0]["raw_p_yes"], scored_df.iloc[0]["p_yes"])

    def test_score_markets_cli_rejects_multiple_forecast_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            forecast_csv = Path(tmpdir) / "forecast.csv"
            markets_csv = Path(tmpdir) / "markets.csv"
            output_csv = Path(tmpdir) / "scored.csv"

            pd.DataFrame(
                [
                    {
                        "date": "2026-05-26",
                        "q_0.050": 80.0,
                        "q_0.500": 90.0,
                        "q_0.950": 100.0,
                    },
                    {
                        "date": "2026-05-27",
                        "q_0.050": 81.0,
                        "q_0.500": 91.0,
                        "q_0.950": 101.0,
                    },
                ]
            ).to_csv(forecast_csv, index=False)

            pd.DataFrame(
                [
                    {
                        "market_id": "mia_1",
                        "market_name": "Miami <=84",
                        "comparison": "le",
                        "lower_bound": "",
                        "upper_bound": 84,
                        "yes_price": 1,
                        "no_price": 99,
                    }
                ]
            ).to_csv(markets_csv, index=False)

            result = subprocess.run(
                [
                    sys.executable,
                    "score_markets.py",
                    "--forecast",
                    str(forecast_csv),
                    "--markets",
                    str(markets_csv),
                    "--output",
                    str(output_csv),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exactly one row", result.stderr)

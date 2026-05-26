from __future__ import annotations

import math
import unittest

import pandas as pd

from predictor import DistributionCalibrator
from market_scoring import build_cdf_from_quantiles, contract_yes_probability, expected_value_cents, score_contract_row


class MarketScoringTests(unittest.TestCase):
    def test_between_probability_uses_integer_continuity_correction(self) -> None:
        forecast_row = pd.Series(
            {
                "q_0.050": 80.0,
                "q_0.500": 90.0,
                "q_0.950": 100.0,
            }
        )

        prob = contract_yes_probability(
            forecast_row,
            comparison="between",
            lower_bound=85.0,
            upper_bound=86.0,
        )

        self.assertAlmostEqual(prob, 0.09, places=6)

    def test_build_cdf_from_quantiles_can_apply_distribution_calibration(self) -> None:
        forecast_row = pd.Series(
            {
                "q_0.050": 80.0,
                "q_0.500": 90.0,
                "q_0.950": 100.0,
            }
        )
        calibrator = DistributionCalibrator(
            pit_knots=pd.Series([0.0, 0.5, 1.0], dtype=float).to_numpy(),
            calibrated_knots=pd.Series([0.0, 0.75, 1.0], dtype=float).to_numpy(),
        )

        raw_cdf = build_cdf_from_quantiles(forecast_row)
        calibrated_cdf = build_cdf_from_quantiles(forecast_row, distribution_calibrator=calibrator)

        self.assertAlmostEqual(raw_cdf(90.0), 0.5, places=6)
        self.assertAlmostEqual(calibrated_cdf(90.0), 0.75, places=6)

    def test_expected_value_cents_applies_two_percent_fee(self) -> None:
        yes_ev = expected_value_cents(p_yes=0.45, price_cents=40.0, fee_rate=0.02)
        no_ev = expected_value_cents(p_yes=0.45, price_cents=60.0, fee_rate=0.02, side="no")

        self.assertAlmostEqual(yes_ev, 4.2, places=6)
        self.assertAlmostEqual(no_ev, -6.2, places=6)

    def test_score_contract_row_uses_inclusive_integer_bounds(self) -> None:
        forecast_row = pd.Series(
            {
                "q_0.050": 80.0,
                "q_0.500": 90.0,
                "q_0.950": 100.0,
            }
        )
        market_row = pd.Series(
            {
                "market_id": "mia_88_89",
                "market_name": "Miami 88 to 89",
                "comparison": "between",
                "lower_bound": 88.0,
                "upper_bound": 89.0,
                "yes_price": 94.0,
                "no_price": 12.0,
            }
        )

        scored = score_contract_row(forecast_row, market_row, fee_rate=0.02)

        self.assertIn(scored["recommended_side"], {"yes", "no", "pass"})
        self.assertTrue(math.isfinite(scored["yes_ev_cents"]))
        self.assertTrue(math.isfinite(scored["no_ev_cents"]))
        self.assertGreaterEqual(scored["p_yes"], 0.0)
        self.assertLessEqual(scored["p_yes"], 1.0)
        self.assertNotIn("calibrated_p_yes", scored)

    def test_score_contract_row_returns_calibrated_columns_when_requested(self) -> None:
        forecast_row = pd.Series(
            {
                "q_0.050": 80.0,
                "q_0.500": 90.0,
                "q_0.950": 100.0,
            }
        )
        calibrator = DistributionCalibrator(
            pit_knots=pd.Series([0.0, 0.5, 1.0], dtype=float).to_numpy(),
            calibrated_knots=pd.Series([0.0, 0.75, 1.0], dtype=float).to_numpy(),
        )
        market_row = pd.Series(
            {
                "market_id": "mia_88_89",
                "market_name": "Miami 88 to 89",
                "comparison": "between",
                "lower_bound": 88.0,
                "upper_bound": 89.0,
                "yes_price": 94.0,
                "no_price": 12.0,
            }
        )

        scored = score_contract_row(
            forecast_row,
            market_row,
            fee_rate=0.02,
            distribution_calibrator=calibrator,
        )

        self.assertIn("calibrated_p_yes", scored)
        self.assertIn("calibrated_yes_ev_cents", scored)
        self.assertIn("calibrated_no_ev_cents", scored)
        self.assertIn("raw_p_yes", scored)
        self.assertIn("raw_yes_ev_cents", scored)
        self.assertNotEqual(scored["raw_p_yes"], scored["p_yes"])

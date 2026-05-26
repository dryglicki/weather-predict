from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from predictor import DistributionCalibrator


def _quantile_columns(forecast_row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    quantile_pairs: list[tuple[float, float]] = []
    for col, value in forecast_row.items():
        if isinstance(col, str) and col.startswith("q_"):
            quantile_pairs.append((float(col[2:]), float(value)))

    if not quantile_pairs:
        raise ValueError("forecast_row must contain q_* quantile columns.")

    quantile_pairs.sort(key=lambda item: item[0])
    probs = np.array([p for p, _ in quantile_pairs], dtype=float)
    values = np.array([v for _, v in quantile_pairs], dtype=float)
    values = np.maximum.accumulate(values)

    unique_values: list[float] = []
    unique_probs: list[float] = []
    for prob, value in zip(probs, values, strict=False):
        if unique_values and np.isclose(value, unique_values[-1]):
            unique_probs[-1] = max(unique_probs[-1], float(prob))
        else:
            unique_values.append(float(value))
            unique_probs.append(float(prob))

    return np.asarray(unique_probs, dtype=float), np.asarray(unique_values, dtype=float)


def build_cdf_from_quantiles(
    forecast_row: pd.Series,
    distribution_calibrator: DistributionCalibrator | None = None,
) -> Callable[[float], float]:
    probs, values = _quantile_columns(forecast_row)

    def raw_cdf(x: float) -> float:
        return float(np.interp(x, values, probs, left=0.0, right=1.0))

    if distribution_calibrator is None:
        return raw_cdf

    def cdf(x: float) -> float:
        raw_value = raw_cdf(x)
        calibrated_value = distribution_calibrator.transform(np.array([raw_value], dtype=float))[0]
        return float(calibrated_value)

    return cdf


def contract_yes_probability(
    forecast_row: pd.Series,
    comparison: str,
    lower_bound: float | None,
    upper_bound: float | None,
    distribution_calibrator: DistributionCalibrator | None = None,
) -> float:
    cdf = build_cdf_from_quantiles(forecast_row, distribution_calibrator=distribution_calibrator)

    if comparison == "le":
        if upper_bound is None:
            raise ValueError("upper_bound is required for comparison='le'.")
        return cdf(float(upper_bound) + 0.5)

    if comparison == "ge":
        if lower_bound is None:
            raise ValueError("lower_bound is required for comparison='ge'.")
        return 1.0 - cdf(float(lower_bound) - 0.5)

    if comparison == "between":
        if lower_bound is None or upper_bound is None:
            raise ValueError("lower_bound and upper_bound are required for comparison='between'.")
        return max(0.0, cdf(float(upper_bound) + 0.5) - cdf(float(lower_bound) - 0.5))

    raise ValueError(f"Unsupported comparison: {comparison}")


def expected_value_cents(
    p_yes: float,
    price_cents: float,
    fee_rate: float,
    side: str = "yes",
) -> float:
    if side not in {"yes", "no"}:
        raise ValueError("side must be 'yes' or 'no'.")

    payout_cents = 100.0 * p_yes if side == "yes" else 100.0 * (1.0 - p_yes)
    return float(payout_cents - (1.0 + fee_rate) * price_cents)


def score_contract_row(
    forecast_row: pd.Series,
    market_row: pd.Series,
    fee_rate: float,
    distribution_calibrator: DistributionCalibrator | None = None,
) -> dict[str, float | str]:
    def _maybe_float(value: object) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)

    p_yes = contract_yes_probability(
        forecast_row,
        comparison=str(market_row["comparison"]),
        lower_bound=_maybe_float(market_row.get("lower_bound")),
        upper_bound=_maybe_float(market_row.get("upper_bound")),
    )
    yes_ev = expected_value_cents(p_yes, float(market_row["yes_price"]), fee_rate, side="yes")
    no_ev = expected_value_cents(p_yes, float(market_row["no_price"]), fee_rate, side="no")

    scored_row: dict[str, float | str] = {
        "market_id": market_row["market_id"],
        "market_name": market_row["market_name"],
        "p_yes": p_yes,
        "p_no": 1.0 - p_yes,
        "yes_ev_cents": yes_ev,
        "no_ev_cents": no_ev,
    }

    if distribution_calibrator is not None:
        calibrated_p_yes = contract_yes_probability(
            forecast_row,
            comparison=str(market_row["comparison"]),
            lower_bound=_maybe_float(market_row.get("lower_bound")),
            upper_bound=_maybe_float(market_row.get("upper_bound")),
            distribution_calibrator=distribution_calibrator,
        )
        calibrated_yes_ev = expected_value_cents(
            calibrated_p_yes,
            float(market_row["yes_price"]),
            fee_rate,
            side="yes",
        )
        calibrated_no_ev = expected_value_cents(
            calibrated_p_yes,
            float(market_row["no_price"]),
            fee_rate,
            side="no",
        )
        scored_row.update(
            {
                "raw_p_yes": p_yes,
                "raw_p_no": 1.0 - p_yes,
                "raw_yes_ev_cents": yes_ev,
                "raw_no_ev_cents": no_ev,
                "p_yes": calibrated_p_yes,
                "p_no": 1.0 - calibrated_p_yes,
                "yes_ev_cents": calibrated_yes_ev,
                "no_ev_cents": calibrated_no_ev,
                "calibrated_p_yes": calibrated_p_yes,
                "calibrated_p_no": 1.0 - calibrated_p_yes,
                "calibrated_yes_ev_cents": calibrated_yes_ev,
                "calibrated_no_ev_cents": calibrated_no_ev,
            }
        )
        if calibrated_yes_ev <= 0.0 and calibrated_no_ev <= 0.0:
            recommended_side = "pass"
            recommended_edge = 0.0
        elif calibrated_yes_ev >= calibrated_no_ev:
            recommended_side = "yes"
            recommended_edge = calibrated_yes_ev
        else:
            recommended_side = "no"
            recommended_edge = calibrated_no_ev
    else:
        if yes_ev <= 0.0 and no_ev <= 0.0:
            recommended_side = "pass"
            recommended_edge = 0.0
        elif yes_ev >= no_ev:
            recommended_side = "yes"
            recommended_edge = yes_ev
        else:
            recommended_side = "no"
            recommended_edge = no_ev

    scored_row["recommended_side"] = recommended_side
    scored_row["recommended_edge_cents"] = recommended_edge
    return scored_row


def score_market_table(
    forecast_row: pd.Series,
    market_df: pd.DataFrame,
    fee_rate: float,
    distribution_calibrator: DistributionCalibrator | None = None,
) -> pd.DataFrame:
    scored_rows = [
        score_contract_row(
            forecast_row,
            row,
            fee_rate,
            distribution_calibrator=distribution_calibrator,
        )
        for _, row in market_df.iterrows()
    ]
    scored_df = pd.DataFrame(scored_rows)
    return scored_df.sort_values(
        ["recommended_edge_cents", "yes_ev_cents", "no_ev_cents"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_scoring import score_market_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score Miami Kalshi weather contracts from a KMIA forecast CSV.")
    parser.add_argument("--forecast", required=True, help="Forecast CSV from make_prediction.py.")
    parser.add_argument("--markets", required=True, help="CSV of contract rows with yes/no prices and bounds.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--fee-rate", type=float, default=0.02, help="Fee rate applied to each wager.")
    parser.add_argument(
        "--artifact-dir",
        help="Artifact directory containing optional distribution calibration state.",
    )
    parser.add_argument(
        "--distribution-calibration",
        action="store_true",
        help="Apply distribution calibration from the loaded artifact when available.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    forecast_df = pd.read_csv(args.forecast)
    if len(forecast_df) != 1:
        raise ValueError("forecast CSV must contain exactly one row for the initial scorer.")

    market_df = pd.read_csv(args.markets)
    required = {"market_id", "market_name", "comparison", "lower_bound", "upper_bound", "yes_price", "no_price"}
    missing = sorted(required.difference(market_df.columns))
    if missing:
        raise ValueError(f"Missing required market columns: {', '.join(missing)}")

    forecast_row = forecast_df.iloc[0]
    distribution_calibrator = None
    if args.distribution_calibration:
        if args.artifact_dir is None:
            raise ValueError("--artifact-dir is required when --distribution-calibration is set.")
        from predictor import WeatherQuantileArtifact

        artifact = WeatherQuantileArtifact.load(Path(args.artifact_dir))
        if artifact.distribution_calibrator is None:
            raise ValueError("Artifact does not contain distribution calibration state.")
        distribution_calibrator = artifact.distribution_calibrator

    scored_df = score_market_table(
        forecast_row,
        market_df,
        fee_rate=args.fee_rate,
        distribution_calibrator=distribution_calibrator,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

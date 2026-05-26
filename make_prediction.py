from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from predictor import WeatherQuantileArtifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run KMIA predictions from a saved artifact bundle.")
    parser.add_argument("--artifact-dir", required=True, help="Directory containing model.cbm and artifact.json.")
    parser.add_argument("--input", required=True, help="Input CSV with one row or many rows of features.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = WeatherQuantileArtifact.load(Path(args.artifact_dir))
    df = pd.read_csv(args.input)

    missing = [col for col in artifact.feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")

    out = df.copy()
    pred_df = artifact.predict_output_frame(df)
    out = pd.concat([out, pred_df], axis=1)
    out.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

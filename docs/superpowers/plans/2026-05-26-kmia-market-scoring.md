# KMIA Market Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the KMIA forecast distribution into a scoring tool for separate Kalshi yes/no temperature contracts, compute yes/no expected value after a 2% fee, and rank the best Miami trades for a 10 AM market open using the 7 AM forecast.

**Architecture:** Keep `make_prediction.py` as the forecast generator and add a separate, pure scoring layer. The scorer should consume one forecast CSV row with repaired quantiles plus a contract CSV with Miami bin definitions and yes/no prices. A new `market_scoring.py` module will own the CDF interpolation and expected-value math; a new `score_markets.py` CLI will load the forecast, score each contract independently, and emit a ranked table. The scorer should support three comparison types: `le` (yes if temperature is at or below the upper bound), `between` (yes if the temperature falls inside the band), and `ge` (yes if the temperature is at or above the lower bound).

**Tech Stack:** Python 3.14, pandas, NumPy, argparse, pytest.

---

## File Structure

- `market_scoring.py`: Pure scoring helpers that convert a forecast row of `q_*` columns into a monotone CDF, compute yes probabilities for a contract row, and calculate EV after fees.
- `score_markets.py`: New CLI that reads one forecast CSV row plus a contract CSV, ranks the market edges, and writes a scored CSV.
- `tests/test_market_scoring.py`: Unit tests for the probability math, CDF interpolation, fee handling, and recommendation logic.
- `tests/test_score_markets.py`: CLI integration tests using a Miami-style contract table and a synthetic forecast row.
- `README.md`: Usage notes for the scoring CLI and the expected CSV schemas.
- `docs/superpowers/plans/2026-05-26-kmia-market-scoring.md`: This execution plan.

## Current Baseline

- Branch: `station-kmia-training`.
- `make_prediction.py` already exists and produces a CSV with repaired quantiles (`q_*`) and calibrated interval columns.
- The KMIA artifact bundle already exists and is the input to `make_prediction.py`.
- The scorer is a separate layer: it should not re-train the model and should not read the CatBoost artifact directly.
- Operational rule: forecast cutoff is the 7 AM NBM run; market pricing is taken at the 10 AM open; the fee model is 2% of the wager.

---

### Task 1: Add Pure Market-Scoring Helpers

**Files:**
- Create: `market_scoring.py`
- Create: `tests/test_market_scoring.py`

- [ ] **Step 1: Write the failing unit tests for CDF interpolation and EV math**

Create `tests/test_market_scoring.py` with these tests:

```python
from __future__ import annotations

import math
import unittest

import pandas as pd

from market_scoring import contract_yes_probability, expected_value_cents, score_contract_row


class MarketScoringTests(unittest.TestCase):
    def test_between_probability_uses_piecewise_linear_cdf(self) -> None:
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
            lower_bound=80.0,
            upper_bound=90.0,
        )

        self.assertAlmostEqual(prob, 0.45, places=6)

    def test_expected_value_cents_applies_two_percent_fee(self) -> None:
        yes_ev = expected_value_cents(p_yes=0.45, price_cents=40.0, fee_rate=0.02)
        no_ev = expected_value_cents(p_yes=0.45, price_cents=60.0, fee_rate=0.02, side="no")

        self.assertAlmostEqual(yes_ev, 4.2, places=6)
        self.assertAlmostEqual(no_ev, -6.2, places=6)

    def test_score_contract_row_prefers_positive_edge_and_can_pass(self) -> None:
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
```

- [ ] **Step 2: Run the targeted tests and confirm they fail**

Run:

```bash
python -m pytest tests/test_market_scoring.py -q
```

Expected: FAIL because `market_scoring.py` does not exist yet.

- [ ] **Step 3: Implement the pure scoring module**

Create `market_scoring.py` with the following shape:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd


def _quantile_columns(forecast_row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    quantile_pairs = []
    for col, value in forecast_row.items():
        if col.startswith("q_"):
            quantile_pairs.append((float(col[2:]), float(value)))
    quantile_pairs.sort(key=lambda item: item[0])
    probs = np.array([p for p, _ in quantile_pairs], dtype=float)
    values = np.array([v for _, v in quantile_pairs], dtype=float)
    values = np.maximum.accumulate(values)
    return probs, values


def build_cdf_from_quantiles(forecast_row: pd.Series) -> Callable[[float], float]:
    probs, values = _quantile_columns(forecast_row)

    def cdf(x: float) -> float:
        return float(np.interp(x, values, probs, left=0.0, right=1.0))

    return cdf


def contract_yes_probability(
    forecast_row: pd.Series,
    comparison: str,
    lower_bound: float | None,
    upper_bound: float | None,
) -> float:
    cdf = build_cdf_from_quantiles(forecast_row)

    if comparison == "le":
        if upper_bound is None:
            raise ValueError("upper_bound is required for comparison='le'.")
        return cdf(float(upper_bound))

    if comparison == "ge":
        if lower_bound is None:
            raise ValueError("lower_bound is required for comparison='ge'.")
        return 1.0 - cdf(float(lower_bound))

    if comparison == "between":
        if lower_bound is None or upper_bound is None:
            raise ValueError("lower_bound and upper_bound are required for comparison='between'.")
        return max(0.0, cdf(float(upper_bound)) - cdf(float(lower_bound)))

    raise ValueError(f"Unsupported comparison: {comparison}")


def expected_value_cents(
    p_yes: float,
    price_cents: float,
    fee_rate: float,
    side: str = "yes",
) -> float:
    payout_cents = 100.0 * p_yes if side == "yes" else 100.0 * (1.0 - p_yes)
    return float(payout_cents - (1.0 + fee_rate) * price_cents)


def score_contract_row(
    forecast_row: pd.Series,
    market_row: pd.Series,
    fee_rate: float,
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

    if yes_ev <= 0.0 and no_ev <= 0.0:
        recommended_side = "pass"
        recommended_edge = 0.0
    elif yes_ev >= no_ev:
        recommended_side = "yes"
        recommended_edge = yes_ev
    else:
        recommended_side = "no"
        recommended_edge = no_ev

    return {
        "market_id": market_row["market_id"],
        "market_name": market_row["market_name"],
        "p_yes": p_yes,
        "p_no": 1.0 - p_yes,
        "yes_ev_cents": yes_ev,
        "no_ev_cents": no_ev,
        "recommended_side": recommended_side,
        "recommended_edge_cents": recommended_edge,
    }


def score_market_table(
    forecast_row: pd.Series,
    market_df: pd.DataFrame,
    fee_rate: float,
) -> pd.DataFrame:
    scored_rows = [score_contract_row(forecast_row, row, fee_rate) for _, row in market_df.iterrows()]
    scored_df = pd.DataFrame(scored_rows)
    return scored_df.sort_values(
        ["recommended_edge_cents", "yes_ev_cents", "no_ev_cents"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
```

- [ ] **Step 4: Run the targeted tests again**

Run:

```bash
python -m pytest tests/test_market_scoring.py -q
```

Expected: PASS.

- [ ] **Step 5: Run the full test suite**

Run:

```bash
python -m pytest -q
```

Expected: all existing tests and the new scoring tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add market_scoring.py tests/test_market_scoring.py
git commit -m "feat: add KMIA market scoring primitives"
```

Expected: commit succeeds on `station-kmia-training`.

---

### Task 2: Add the Market Scoring CLI

**Files:**
- Create: `score_markets.py`
- Create: `tests/test_score_markets.py`
- Modify: `market_scoring.py`

- [ ] **Step 1: Write the failing CLI integration tests**

Create `tests/test_score_markets.py` with these tests:

```python
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class ScoreMarketsCLITests(unittest.TestCase):
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
```

Use a real assertion for the second test that the CLI fails when the forecast CSV contains more than one row.

- [ ] **Step 2: Run the tests and confirm they fail**

Run:

```bash
python -m pytest tests/test_score_markets.py -q
```

Expected: FAIL because `score_markets.py` does not exist yet.

- [ ] **Step 3: Implement the CLI**

Create `score_markets.py` with:

```python
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
    scored_df = score_market_table(forecast_row, market_df, fee_rate=args.fee_rate)
    scored_df.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

If `score_market_table(...)` does not exist yet or needs a small adjustment to preserve extra market columns, add that in `market_scoring.py` as part of this task.

- [ ] **Step 4: Run the tests again**

Run:

```bash
python -m pytest tests/test_score_markets.py -q
```

Expected: PASS.

- [ ] **Step 5: Run a manual scorer smoke check**

Run:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

tmp = Path("tmp_market_smoke")
tmp.mkdir(exist_ok=True)

pd.DataFrame([
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
]).to_csv(tmp / "forecast.csv", index=False)

pd.DataFrame([
    {"market_id": "mia_1", "market_name": "Miami <=84", "comparison": "le", "lower_bound": "", "upper_bound": 84, "yes_price": 1, "no_price": 99},
    {"market_id": "mia_2", "market_name": "Miami 85-86", "comparison": "between", "lower_bound": 85, "upper_bound": 86, "yes_price": 6, "no_price": 98},
    {"market_id": "mia_3", "market_name": "Miami 87-88", "comparison": "between", "lower_bound": 87, "upper_bound": 88, "yes_price": 31, "no_price": 72},
    {"market_id": "mia_4", "market_name": "Miami 89-90", "comparison": "between", "lower_bound": 89, "upper_bound": 90, "yes_price": 59, "no_price": 47},
    {"market_id": "mia_5", "market_name": "Miami 91-92", "comparison": "between", "lower_bound": 91, "upper_bound": 92, "yes_price": 12, "no_price": 93},
    {"market_id": "mia_6", "market_name": "Miami >=93", "comparison": "ge", "lower_bound": 93, "upper_bound": "", "yes_price": 2, "no_price": 98},
]).to_csv(tmp / "markets.csv", index=False)
PY
python score_markets.py --forecast tmp_market_smoke/forecast.csv --markets tmp_market_smoke/markets.csv --output tmp_market_smoke/scored.csv
python - <<'PY'
import pandas as pd
print(pd.read_csv("tmp_market_smoke/scored.csv")[["market_id", "recommended_side", "recommended_edge_cents"]])
PY
```

Expected: writes a ranked CSV with `p_yes`, `yes_ev_cents`, `no_ev_cents`, `recommended_side`, and `recommended_edge_cents`.

- [ ] **Step 6: Commit**

Run:

```bash
git add market_scoring.py score_markets.py tests/test_market_scoring.py tests/test_score_markets.py
git commit -m "feat: add KMIA market scoring cli"
```

Expected: commit succeeds on `station-kmia-training`.

---

### Task 3: Document the Scorer Workflow and Verify the Full Stack

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the README with the scoring workflow**

Replace or extend the KMIA section with:

```markdown
## KMIA Scoring Workflow

The KMIA pipeline has three steps:

1. Train and save the model:

```bash
python predictor.py
```

2. Generate a forecast CSV:

```bash
python make_prediction.py --artifact-dir artifacts/phase2_denser_quantiles --input input.csv --output forecast.csv
```

3. Score Miami weather contracts:

```bash
python score_markets.py --forecast forecast.csv --markets markets.csv --output scored.csv
```

The scorer expects one forecast row and a CSV of contract rows. The market CSV should contain `market_id`, `market_name`, `comparison`, `lower_bound`, `upper_bound`, `yes_price`, and `no_price`. Supported comparisons are `le`, `between`, and `ge`.

The scorer computes a yes probability from the calibrated quantile distribution, converts that to expected value after a 2% wager fee, and ranks contracts by the best positive edge.
```

- [ ] **Step 2: Run the full test suite**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass, including the new scoring unit and CLI tests.

- [ ] **Step 3: Run the syntax check**

Run:

```bash
python -m compileall -q predictor.py make_prediction.py market_scoring.py score_markets.py tests/test_predictor.py tests/test_make_prediction.py tests/test_market_scoring.py tests/test_score_markets.py
```

Expected: exit code 0 and no output.

- [ ] **Step 4: Run the full end-to-end path**

Run:

```bash
python predictor.py
python make_prediction.py --artifact-dir artifacts/phase2_denser_quantiles --input input.csv --output forecast.csv
python score_markets.py --forecast forecast.csv --markets markets.csv --output scored.csv
```

Expected: the model training completes, the forecast file contains repaired quantiles and interval columns, and the scorer writes a ranked list of contract edges.

- [ ] **Step 5: Review recent history**

Run:

```bash
git log --oneline --decorate --max-count=10
```

Expected: the recent commits are on `station-kmia-training`, not `main`.

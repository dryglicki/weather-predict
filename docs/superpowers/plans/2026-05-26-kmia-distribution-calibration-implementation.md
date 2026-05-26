# KMIA Distribution Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional monotone PIT calibration layer for KMIA so we can improve spread calibration without removing the existing interval conformal step.

**Architecture:** Keep the current CatBoost quantile model, interval conformal calibration, and versioned artifact bundle intact. Add a separate distribution-calibration object that learns a simple monotone remap of PIT values on the held-out calibration block, persists that state in `artifact.json`, and can be toggled on for CDF-based scoring and PIT plots. Existing behavior must remain unchanged when the toggle is off.

**Tech Stack:** Python 3.14, pandas, numpy, CatBoost, matplotlib, pytest.

---

### Task 1: Add distribution-calibration state to the artifact bundle

**Files:**
- Modify: `predictor.py:778-930`
- Modify: `tests/test_predictor.py`

- [ ] **Step 1: Write the failing test**

```python
def test_distribution_calibrator_round_trip(self) -> None:
    pit_values = np.array([0.02, 0.05, 0.11, 0.18, 0.24, 0.63, 0.79, 0.92, 0.97], dtype=float)
    calibrator = DistributionCalibrator.fit(pit_values, n_knots=5)

    self.assertTrue(np.all(np.diff(calibrator.pit_knots) > 0.0))
    self.assertTrue(np.all(np.diff(calibrator.calibrated_knots) >= 0.0))

    mapped = calibrator.transform(np.array([0.02, 0.50, 0.98], dtype=float))
    self.assertTrue(np.all(mapped >= 0.0))
    self.assertTrue(np.all(mapped <= 1.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_predictor.py -k distribution_calibrator_round_trip`
Expected: FAIL because `DistributionCalibrator` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass
class DistributionCalibrator:
    pit_knots: ArrayLike
    calibrated_knots: ArrayLike

    @classmethod
    def fit(cls, pit_values: ArrayLike, n_knots: int = 101) -> "DistributionCalibrator":
        pit = np.asarray(pit_values, dtype=float)
        if pit.size == 0:
            raise ValueError("pit_values must not be empty.")
        knot_probs = np.linspace(0.0, 1.0, n_knots)
        pit_knots = np.quantile(pit, knot_probs)
        calibrated_knots = knot_probs
        return cls(pit_knots=pit_knots, calibrated_knots=calibrated_knots)

    def transform(self, u: ArrayLike) -> ArrayLike:
        values = np.asarray(u, dtype=float)
        return np.interp(values, self.pit_knots, self.calibrated_knots, left=0.0, right=1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pit_knots": self.pit_knots.tolist(),
            "calibrated_knots": self.calibrated_knots.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DistributionCalibrator":
        return cls(
            pit_knots=np.asarray(payload["pit_knots"], dtype=float),
            calibrated_knots=np.asarray(payload["calibrated_knots"], dtype=float),
        )
```

Extend `WeatherQuantileArtifact` so `artifact.json` can optionally store:

```python
{
    "distribution_calibration_enabled": True,
    "distribution_calibration": {
        "pit_knots": [0.0, 0.2, 0.5, 0.8, 1.0],
        "calibrated_knots": [0.0, 0.25, 0.5, 0.75, 1.0]
    }
}
```

When the payload is absent, `WeatherQuantileArtifact.load()` must keep the current raw behavior.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/test_predictor.py -k distribution_calibrator_round_trip`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add predictor.py tests/test_predictor.py
git commit -m "feat: add kmia distribution calibrator state"
```

### Task 2: Thread optional calibrated CDF into market scoring

**Files:**
- Modify: `market_scoring.py`
- Modify: `score_markets.py`
- Modify: `tests/test_market_scoring.py`
- Modify: `tests/test_score_markets.py`

- [ ] **Step 1: Write the failing test**

```python
def test_score_contract_row_uses_distribution_calibration(self) -> None:
    forecast_row = pd.Series({
        "q_0.05": 80.0,
        "q_0.10": 81.0,
        "q_0.20": 82.0,
        "q_0.25": 83.0,
        "q_0.30": 84.0,
        "q_0.40": 85.0,
        "q_0.50": 86.0,
        "q_0.60": 87.0,
        "q_0.70": 88.0,
        "q_0.75": 89.0,
        "q_0.80": 90.0,
        "q_0.90": 91.0,
        "q_0.95": 92.0,
    })
    calibrator = DistributionCalibrator(
        pit_knots=np.array([0.0, 0.5, 1.0], dtype=float),
        calibrated_knots=np.array([0.0, 0.75, 1.0], dtype=float),
    )
    market_row = pd.Series({
        "market_id": "mia-1",
        "market_name": "Tomorrow 88-89",
        "comparison": "between",
        "lower_bound": 88,
        "upper_bound": 89,
        "yes_price": 31,
        "no_price": 72,
    })

    scored = score_contract_row(forecast_row, market_row, fee_rate=0.02, distribution_calibrator=calibrator)
    self.assertIn("p_yes", scored)
    self.assertIn("calibrated_p_yes", scored)
    self.assertNotEqual(scored["p_yes"], scored["calibrated_p_yes"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_market_scoring.py -k distribution_calibration`
Expected: FAIL because scoring does not yet accept a distribution calibrator.

- [ ] **Step 3: Write minimal implementation**

```python
def build_cdf_from_quantiles(
    forecast_row: pd.Series,
    distribution_calibrator: DistributionCalibrator | None = None,
) -> Callable[[float], float]:
    probs, values = _quantile_columns(forecast_row)

    def cdf(x: float) -> float:
        raw_u = float(np.interp(x, values, probs, left=0.0, right=1.0))
        if distribution_calibrator is None:
            return raw_u
        return float(distribution_calibrator.transform(np.array([raw_u], dtype=float))[0])

    return cdf

def score_contract_row(
    forecast_row: pd.Series,
    market_row: pd.Series,
    fee_rate: float,
    distribution_calibrator: DistributionCalibrator | None = None,
) -> dict[str, float | str]:
    p_yes = contract_yes_probability(
        forecast_row,
        comparison=str(market_row["comparison"]),
        lower_bound=_maybe_float(market_row.get("lower_bound")),
        upper_bound=_maybe_float(market_row.get("upper_bound")),
        distribution_calibrator=None,
    )
    calibrated_p_yes = contract_yes_probability(
        forecast_row,
        comparison=str(market_row["comparison"]),
        lower_bound=_maybe_float(market_row.get("lower_bound")),
        upper_bound=_maybe_float(market_row.get("upper_bound")),
        distribution_calibrator=distribution_calibrator,
    )
    yes_ev = expected_value_cents(p_yes, float(market_row["yes_price"]), fee_rate, side="yes")
    calibrated_yes_ev = expected_value_cents(
        calibrated_p_yes,
        float(market_row["yes_price"]),
        fee_rate,
        side="yes",
    )
    no_ev = expected_value_cents(p_yes, float(market_row["no_price"]), fee_rate, side="no")
    calibrated_no_ev = expected_value_cents(
        calibrated_p_yes,
        float(market_row["no_price"]),
        fee_rate,
        side="no",
    )
    return {
        "market_id": market_row["market_id"],
        "market_name": market_row["market_name"],
        "p_yes": p_yes,
        "calibrated_p_yes": calibrated_p_yes,
        "yes_ev_cents": yes_ev,
        "calibrated_yes_ev_cents": calibrated_yes_ev,
        "no_ev_cents": no_ev,
        "calibrated_no_ev_cents": calibrated_no_ev,
        "recommended_side": "yes" if calibrated_yes_ev >= calibrated_no_ev else "no",
        "recommended_edge_cents": max(calibrated_yes_ev, calibrated_no_ev),
    }
```

Update `score_markets.py` so the CLI loads `artifact.json`, respects a `--distribution-calibration` flag, and writes both raw and calibrated probabilities/EV columns when the layer is enabled.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/test_market_scoring.py tests/test_score_markets.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add market_scoring.py score_markets.py tests/test_market_scoring.py tests/test_score_markets.py
git commit -m "feat: score kmia markets with optional cdf calibration"
```

### Task 3: Show raw and calibrated PIT in calibration plots

**Files:**
- Modify: `plot_calibration.py`
- Modify: `tests/test_plot_calibration.py`

- [ ] **Step 1: Write the failing test**

```python
def test_plot_calibration_writes_both_pit_views(self) -> None:
    output_path = tmp_path / "calibration.png"
    result = main([
        "--artifact-dir",
        str(artifact_dir),
        "--data",
        str(data_path),
        "--output",
        str(output_path),
    ])
    self.assertEqual(result, 0)
    self.assertTrue(output_path.exists())
```

Add a helper-level assertion that the script computes two PIT series:

```python
self.assertIn("Mean raw PIT", stdout)
self.assertIn("Mean calibrated PIT", stdout)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_plot_calibration.py -k both_pit_views`
Expected: FAIL because the script currently plots only one PIT view.

- [ ] **Step 3: Write minimal implementation**

```python
def compute_raw_pit_values(
    y_true: np.ndarray | pd.Series,
    pred_df: pd.DataFrame,
) -> np.ndarray:
    y_arr = np.asarray(y_true, dtype=float)
    pit_values = np.empty(len(y_arr), dtype=float)
    for i, (_, row) in enumerate(pred_df.iterrows()):
        pit_values[i] = build_cdf_from_quantiles(row)(float(y_arr[i]))
    return pit_values

def compute_calibrated_pit_values(
    y_true: np.ndarray | pd.Series,
    pred_df: pd.DataFrame,
    distribution_calibrator: DistributionCalibrator | None,
) -> np.ndarray:
    raw_pit = compute_raw_pit_values(y_true, pred_df)
    if distribution_calibrator is None:
        return raw_pit
    return distribution_calibrator.transform(raw_pit)

def _plot_histograms(
    raw_pit_values: np.ndarray,
    calibrated_pit_values: np.ndarray,
    rank_values: np.ndarray,
    quantiles: list[float] | tuple[float, ...] | np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].hist(raw_pit_values, bins=np.linspace(0.0, 1.0, 11), edgecolor="black")
    axes[0].set_title("Raw PIT histogram")
    axes[1].hist(calibrated_pit_values, bins=np.linspace(0.0, 1.0, 11), edgecolor="black")
    axes[1].set_title("Calibrated PIT histogram")
    axes[2].hist(rank_values, bins=np.arange(-0.5, len(quantiles) + 1.5, 1.0), edgecolor="black", align="mid")
    axes[2].set_title("Quantile rank histogram")
```

Render three panels:

- raw PIT histogram
- distribution-calibrated PIT histogram
- rank histogram with the existing quantile labels

Keep the rank histogram logic intact so the existing quantile interval labels remain visible.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/test_plot_calibration.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plot_calibration.py tests/test_plot_calibration.py
git commit -m "feat: add kmia calibrated pit plotting"
```

### Task 4: Document the toggle and verify the full workflow

**Files:**
- Modify: `README.md`
- Create: `tests/test_readme.py`

- [ ] **Step 1: Write the failing test**

```python
def test_readme_mentions_distribution_calibration_toggle(self) -> None:
    source = Path("README.md").read_text()
    self.assertIn("distribution calibration", source.lower())
    self.assertIn("plot_calibration.py", source)
    self.assertIn("score_markets.py", source)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_readme.py`
Expected: FAIL because the README does not yet mention the distribution-calibration toggle.

- [ ] **Step 3: Write minimal implementation**

Update the README sections for:

- artifact contents
- the distribution-calibration toggle
- the raw vs calibrated PIT plot
- the market scoring workflow

- [ ] **Step 4: Run the full verification suite**

Run:

```bash
python -m pytest -q
python -m compileall -q predictor.py make_prediction.py market_scoring.py score_markets.py plot_calibration.py tests/test_predictor.py tests/test_market_scoring.py tests/test_score_markets.py tests/test_plot_calibration.py
```

Expected: all tests pass, compileall exits 0.

- [ ] **Step 5: Commit**
Run: `git status --short --branch`
Expected: only source/doc changes are staged or committed; no generated artifacts should be added here.

### Task 5: Final artifact smoke check

**Files:**
- None

- [ ] **Step 1: Run the full training entrypoint**

Run: `python predictor.py`
Expected: it should write a versioned artifact directory and include the distribution-calibration payload when the toggle is enabled.

- [ ] **Step 2: Inspect the artifact payload**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
artifact_dirs = sorted(Path("artifacts").glob("*"))
latest = artifact_dirs[-1]
payload = json.loads((latest / "artifact.json").read_text())
print(payload.get("distribution_calibration_enabled"))
print("distribution_calibration" in payload)
PY
```

Expected: the payload reflects the toggle state and the new calibration state is present when enabled.

- [ ] **Step 3: Commit**
Run: `git status --short --branch`
Expected: the generated artifact directory remains untracked unless the repo already ignores it.

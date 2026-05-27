# KMIA Calibration Toggles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independent training toggles for interval conformal calibration and PIT distribution calibration so KMIA artifacts can carry either calibration layer, both layers, or neither layer.

**Architecture:** Keep the current CatBoost quantile model as the base forecaster. Extend the pipeline so interval conformal calibration and PIT calibration are two separate optional post-processing stages controlled by explicit CLI flags. The artifact bundle should persist whichever states were actually fitted, and downstream plotting/scoring should continue to work when either calibration layer is absent.

**Tech Stack:** Python 3.14, pandas, NumPy, CatBoost, matplotlib, pytest.

---

## File Structure

- `predictor.py`: Owns the KMIA pipeline, interval calibration, PIT calibration fitting, artifact serialization, and the training CLI entrypoint.
- `tests/test_predictor.py`: Covers pipeline toggles, artifact round trips, and CLI parsing / smoke-path behavior.
- `README.md`: Documents the new training flags and the artifact states they control.

## Current Baseline

- Interval conformal calibration already exists and is stored in `artifact.json`.
- A `DistributionCalibrator` already exists, and downstream scoring / PIT plotting can already consume it when present.
- No training-time path currently fits the PIT calibrator, so the saved artifacts still have `distribution_calibration: null`.
- The entrypoint still uses a hardcoded `--smoke` branch plus the full training path with no calibration toggles.

The missing piece is to thread two booleans through the pipeline and CLI so each calibration stage can be switched on or off independently.

---

### Task 1: Make interval calibration optional end-to-end

**Files:**
- Modify: `predictor.py`
- Modify: `tests/test_predictor.py`

- [ ] **Step 1: Write the failing test**

Add a pipeline test that disables interval calibration and proves the output frame contains only repaired quantile columns.

```python
def test_pipeline_without_interval_calibration_omits_interval_columns(self) -> None:
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
    pipeline = WeatherQuantilePipeline(
        dataset_cfg=dataset_cfg,
        block_cfg=block_cfg,
        cv_cfg=cv_cfg,
        model_cfg=model_cfg,
        interval_alphas=[0.10],
        enable_interval_calibration=False,
    )
    pipeline.fit_final(df)

    artifact = pipeline.build_artifact(metadata={"run_name": "no_interval"})
    output_df = artifact.predict_output_frame(pipeline.blocks_["test"].head(5))
    self.assertTrue(all(col.startswith("q_") for col in output_df.columns))
    self.assertFalse(any(col.startswith("interval_") for col in output_df.columns))
```

Add a round-trip test that saves an artifact with interval calibration disabled and confirms the JSON stays loadable.

```python
def test_artifact_round_trip_without_interval_calibration(self) -> None:
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
    pipeline = WeatherQuantilePipeline(
        dataset_cfg=dataset_cfg,
        block_cfg=block_cfg,
        cv_cfg=cv_cfg,
        model_cfg=model_cfg,
        interval_alphas=[0.10],
        enable_interval_calibration=False,
    )
    pipeline.fit_final(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir) / "artifact"
        artifact = pipeline.build_artifact(metadata={"run_name": "no_interval"})
        artifact.save(artifact_dir)

        payload = json.loads((artifact_dir / "artifact.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["calibration_corrections"], {})
        loaded = type(artifact).load(artifact_dir)
        self.assertIsNone(loaded.calibrator)
```

- [ ] **Step 2: Run the targeted tests and confirm they fail**

Run:

```bash
python -m pytest -q tests/test_predictor.py -k "without_interval_calibration or artifact_round_trip_without_interval_calibration"
```

Expected: FAIL because `WeatherQuantilePipeline` still always fits interval calibration and `predict_output_frame()` still always tries to emit interval columns.

- [ ] **Step 3: Write the minimal implementation**

Update the artifact and pipeline paths so the interval calibrator can be absent.

```python
@dataclass
class WeatherQuantileArtifact:
    calibrator: Optional[ConformalCalibrator]
    ...

    def predict_interval(self, df: pd.DataFrame, alpha: float) -> Tuple[ArrayLike, ArrayLike]:
        if self.calibrator is None:
            raise RuntimeError("Interval calibration is disabled for this artifact.")
        pred_matrix = self.predict_quantiles(df, repair_crossing=True)
        return self.calibrator.predict_interval(pred_matrix, alpha)

    def predict_output_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        pred_matrix = self.predict_quantiles(df, repair_crossing=True)
        out = pd.DataFrame(index=df.index)
        for j, q in enumerate(self.model_cfg.quantiles):
            out[f"q_{q:.3f}"] = pred_matrix[:, j]
        if self.calibrator is not None:
            for alpha in self.interval_alphas:
                lower, upper = self.calibrator.predict_interval(pred_matrix, alpha)
                pct = int(round((1.0 - alpha) * 100))
                out[f"interval_{pct}_lower"] = lower
                out[f"interval_{pct}_upper"] = upper
        return out
```

Teach `fit_final()` to skip interval fitting when the toggle is off.

```python
if self.enable_interval_calibration:
    self.calibrator_ = ConformalCalibrator(final_cfg.quantiles)
    y_cal = cal_df[self.dataset_cfg.target_col].to_numpy()
    for alpha in self.interval_alphas:
        self.calibrator_.fit_interval(y_cal, pred_cal, alpha)
else:
    self.calibrator_ = None
```

Make `evaluate_block()` omit coverage and width metrics when no interval calibrator exists.

```python
metrics["mean_pinball_loss"] = mean_pinball_loss(...)
metrics["crossing_rate_after_repair"] = 0.0
if self.calibrator_ is not None:
    for alpha in self.interval_alphas:
        lower, upper = self.calibrator_.predict_interval(pred_matrix, alpha)
        pct = int(round((1.0 - alpha) * 100))
        metrics[f"coverage_{pct}"] = empirical_interval_coverage(y_true, lower, upper)
        metrics[f"width_{pct}"] = interval_width(lower, upper)
```

Keep the JSON payload explicit by storing an interval-enabled flag alongside the existing correction map.

```python
"interval_calibration_enabled": self.calibrator is not None,
"calibration_corrections": (
    {str(alpha): float(correction) for alpha, correction in self.calibrator.result_.alpha_to_correction.items()}
    if self.calibrator is not None
    else {}
),
```

`WeatherQuantileArtifact.load()` should accept the disabled case by skipping correction validation when the flag is false.

- [ ] **Step 4: Run the targeted tests and confirm they pass**

Run:

```bash
python -m pytest -q tests/test_predictor.py -k "without_interval_calibration or artifact_round_trip_without_interval_calibration"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add predictor.py tests/test_predictor.py
git commit -m "feat: make kmia interval calibration optional"
```

### Task 2: Fit the PIT calibrator during training and persist it by default when enabled

**Files:**
- Modify: `predictor.py`
- Modify: `tests/test_predictor.py`

- [ ] **Step 1: Write the failing test**

Add a pipeline test that enables PIT calibration and checks the fitted calibrator is persisted by default.

```python
def test_pipeline_fits_distribution_calibrator_when_enabled(self) -> None:
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
    pipeline = WeatherQuantilePipeline(
        dataset_cfg=dataset_cfg,
        block_cfg=block_cfg,
        cv_cfg=cv_cfg,
        model_cfg=model_cfg,
        interval_alphas=[0.10],
        enable_pit_calibration=True,
    )
    pipeline.fit_final(df)

    self.assertIsNotNone(pipeline.distribution_calibrator_)

    artifact = pipeline.build_artifact(metadata={"run_name": "pit_enabled"})
    self.assertIsNotNone(artifact.distribution_calibrator)
```

Add a test that confirms `build_artifact()` uses the fitted pipeline calibrator when no override is passed.

```python
def test_build_artifact_defaults_to_pipeline_distribution_calibrator(self) -> None:
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
    pipeline = WeatherQuantilePipeline(
        dataset_cfg=dataset_cfg,
        block_cfg=block_cfg,
        cv_cfg=cv_cfg,
        model_cfg=model_cfg,
        interval_alphas=[0.10],
        enable_pit_calibration=True,
    )
    pipeline.fit_final(df)

    artifact = pipeline.build_artifact(metadata={"run_name": "pit_enabled"})
    self.assertIsNotNone(artifact.distribution_calibrator)
```

- [ ] **Step 2: Run the targeted tests and confirm they fail**

Run:

```bash
python -m pytest -q tests/test_predictor.py -k "pit_calibrator or pipeline_fits_distribution_calibrator"
```

Expected: FAIL because `fit_final()` still never fits the PIT calibrator.

- [ ] **Step 3: Write the minimal implementation**

Add a pipeline flag and a fitted attribute.

```python
class WeatherQuantilePipeline:
    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        block_cfg: BlockSplitConfig,
        cv_cfg: ExpandingWindowCVConfig,
        model_cfg: QuantileModelConfig,
        interval_alphas: Optional[Sequence[float]] = None,
        enable_interval_calibration: bool = True,
        enable_pit_calibration: bool = False,
    ) -> None:
        ...
        self.enable_interval_calibration = enable_interval_calibration
        self.enable_pit_calibration = enable_pit_calibration
        self.distribution_calibrator_: Optional[DistributionCalibrator] = None
```

In `fit_final()`, fit the PIT calibrator after the calibration block predictions exist.

```python
pred_cal = model.predict_quantiles(cal_df, repair_crossing=True)
pred_test = model.predict_quantiles(test_df, repair_crossing=True)

self.predictions_["cal"] = pred_cal
self.predictions_["test"] = pred_test

if self.enable_pit_calibration:
    y_cal = cal_df[self.dataset_cfg.target_col].to_numpy()
    pit_values = compute_raw_pit_values(y_cal, pred_cal, self.model_.model_cfg.quantiles)
    self.distribution_calibrator_ = DistributionCalibrator.fit(pit_values)
else:
    self.distribution_calibrator_ = None
```

Add the small helper that computes raw PIT values from repaired quantiles and the matching quantile grid.

```python
def compute_raw_pit_values(
    y_true: ArrayLike,
    pred_matrix: ArrayLike,
    quantiles: Sequence[float],
) -> ArrayLike:
    y_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(pred_matrix, dtype=float)
    q_values = np.asarray(quantiles, dtype=float)
    pit_values = np.empty(len(y_arr), dtype=float)

    for i, row in enumerate(pred_arr):
        pit_values[i] = float(np.interp(float(y_arr[i]), row, q_values, left=0.0, right=1.0))

    return pit_values
```

Make `build_artifact()` default to the fitted pipeline calibrator when no override is supplied.

```python
distribution_calibrator = (
    self.distribution_calibrator_ if distribution_calibrator is None else distribution_calibrator
)
```

Persist the PIT calibration flag and payload in the artifact JSON.

```python
"distribution_calibration_enabled": self.distribution_calibrator is not None,
"distribution_calibration": (
    self.distribution_calibrator.to_dict()
    if self.distribution_calibrator is not None
    else None
),
```

`WeatherQuantileArtifact.load()` should reconstruct the calibrator only when the payload is present and the flag is true.

- [ ] **Step 4: Run the targeted tests and confirm they pass**

Run:

```bash
python -m pytest -q tests/test_predictor.py -k "pit_calibrator or pipeline_fits_distribution_calibrator"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add predictor.py tests/test_predictor.py
git commit -m "feat: fit kmia pit calibrator during training"
```

### Task 3: Add CLI toggles and thread them through smoke / full training runs

**Files:**
- Modify: `predictor.py`
- Modify: `tests/test_predictor.py`

- [ ] **Step 1: Write the failing test**

Add a parser test that proves the new flags default correctly and can be flipped from the command line.

```python
def test_predictor_parser_exposes_calibration_toggles(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["--no-interval-calibration", "--pit-calibration", "--smoke"])

    self.assertFalse(args.interval_calibration)
    self.assertTrue(args.pit_calibration)
    self.assertTrue(args.smoke)
```

Add a smoke-path test that proves the CLI toggles reach the training code.

```python
def test_predictor_smoke_entrypoint_can_disable_interval_and_enable_pit_calibration(self) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "predictor.py",
            "--smoke",
            "--no-interval-calibration",
            "--pit-calibration",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("Smoke metrics:", result.stdout)
    self.assertNotIn("coverage_90", result.stdout)
```

- [ ] **Step 2: Run the targeted tests and confirm they fail**

Run:

```bash
python -m pytest -q tests/test_predictor.py -k "predictor_parser_exposes_calibration_toggles or predictor_smoke_entrypoint_can_disable_interval_and_enable_pit_calibration"
```

Expected: FAIL because the entrypoint still has no parser and the smoke branch ignores calibration toggles.

- [ ] **Step 3: Write the minimal implementation**

Add a parser and a real `main()` helper.

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the KMIA quantile model.")
    parser.add_argument("--smoke", action="store_true", help="Run the small smoke test instead of full training.")
    parser.add_argument(
        "--interval-calibration",
        dest="interval_calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable split-conformal interval calibration.",
    )
    parser.add_argument(
        "--pit-calibration",
        dest="pit_calibration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable PIT distribution calibration.",
    )
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        smoke_metrics = run_kmia_smoke_test(
            "kalshiTraining_KMIA.dat",
            task_type="CPU",
            enable_interval_calibration=args.interval_calibration,
            enable_pit_calibration=args.pit_calibration,
        )
        print("Smoke metrics:", smoke_metrics)
        return 0
    ...
```

Thread the two booleans through the full training helpers.

```python
def run_kmia_smoke_test(
    data_path: str | Path = "kalshiTraining_KMIA.dat",
    task_type: str = "CPU",
    enable_interval_calibration: bool = True,
    enable_pit_calibration: bool = False,
) -> Dict[str, Dict[str, float]]:
    ...
    pipeline = WeatherQuantilePipeline(
        dataset_cfg=dataset_cfg,
        block_cfg=block_cfg,
        cv_cfg=cv_cfg,
        model_cfg=model_cfg,
        interval_alphas=[0.10],
        enable_interval_calibration=enable_interval_calibration,
        enable_pit_calibration=enable_pit_calibration,
    )
```

Use the same pattern in `run_phase(...)` so the full entrypoint can turn each calibration layer on or off while saving artifacts.

```python
def run_phase(
    df: pd.DataFrame,
    dataset_cfg: DatasetConfig,
    block_cfg: BlockSplitConfig,
    cv_cfg: ExpandingWindowCVConfig,
    model_cfg: QuantileModelConfig,
    n_trials: int,
    study_name: str,
    interval_alphas: Optional[Sequence[float]] = None,
    enable_interval_calibration: bool = True,
    enable_pit_calibration: bool = False,
) -> WeatherQuantilePipeline:
    pipeline = WeatherQuantilePipeline(
        dataset_cfg=dataset_cfg,
        block_cfg=block_cfg,
        cv_cfg=cv_cfg,
        model_cfg=model_cfg,
        interval_alphas=interval_alphas,
        enable_interval_calibration=enable_interval_calibration,
        enable_pit_calibration=enable_pit_calibration,
    )
    ...
```

- [ ] **Step 4: Run the targeted tests and confirm they pass**

Run:

```bash
python -m pytest -q tests/test_predictor.py -k "predictor_parser_exposes_calibration_toggles or predictor_smoke_entrypoint_can_disable_interval_and_enable_pit_calibration"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add predictor.py tests/test_predictor.py
git commit -m "feat: add kmia calibration cli toggles"
```

### Task 4: Document the toggle workflow

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the README with the training flags and artifact states**

Add a small section that shows the new CLI flags and explains the artifact states they control.

```markdown
## KMIA Calibration Toggles

The training entrypoint accepts two independent flags:

- `--interval-calibration` / `--no-interval-calibration`
- `--pit-calibration` / `--no-pit-calibration`

Interval calibration controls the `calibration_corrections` payload in `artifact.json`.
PIT calibration controls the `distribution_calibration` payload in `artifact.json`.
```

Also note that the smoke path accepts the same flags, so a quick CPU run can validate the behavior without doing a full tuning pass.

- [ ] **Step 2: Run the full regression suite**

Run:

```bash
python -m pytest -q
python -m compileall -q predictor.py tests/test_predictor.py
```

Expected: all tests pass and the compile check is clean.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: describe kmia calibration toggles"
```

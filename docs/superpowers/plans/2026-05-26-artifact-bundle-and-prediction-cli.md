# KMIA Artifact Bundle and Prediction CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save the fitted KMIA model as `model.cbm` plus `artifact.json`, calibrate every interval supported by the current quantile grid, and add a `make_prediction.py` CLI that loads the artifact and emits repaired quantiles and calibrated intervals for both batch and single-row CSV input.

**Architecture:** Keep the existing `predictor.py` training pipeline as the source of truth for fitting, calibration, and artifact creation. Add one small helper that derives all supported central interval alphas from the quantile grid so the pipeline cannot silently under-calibrate. Add a thin inference script that loads the persisted artifact bundle, validates input columns against the saved feature list, and writes prediction outputs without re-training.

**Tech Stack:** Python 3.14, pandas, NumPy, CatBoost, Optuna, pytest, argparse.

---

## File Structure

- `predictor.py`: Owns the KMIA training pipeline, quantile model wrapper, conformal calibration, artifact save/load, and the helper that derives supported interval alphas from the quantile grid.
- `make_prediction.py`: New CLI entrypoint that loads `model.cbm` plus `artifact.json`, reads CSV input, and writes repaired quantiles plus calibrated interval bounds.
- `tests/test_predictor.py`: Existing training tests, extended to cover automatic interval derivation and artifact bundles that contain every supported correction.
- `tests/test_make_prediction.py`: New CLI tests for one-row and batch CSV prediction output.
- `README.md`: Usage note for the new prediction CLI and the corrected calibration workflow.
- `docs/superpowers/plans/2026-05-26-artifact-bundle-and-prediction-cli.md`: This execution plan.

## Current Baseline

- Branch: `station-kmia-training`.
- The current pipeline already saves an artifact bundle with `model.cbm` and `artifact.json`.
- The current artifact bundle only has one interval correction because the pipeline is still being asked to calibrate one interval explicitly.
- The current phase-2 quantile grid supports 90%, 80%, 60%, and 40% central intervals. Adding `0.25` and `0.75` to the grid will also support 50%.
- CPU verification works in Codex; GPU training is not available in the sandbox, but the local environment has already confirmed CatBoost GPU visibility.

---

### Task 1: Derive All Supported Interval Alphas From the Quantile Grid

**Files:**
- Modify: `predictor.py`
- Modify: `tests/test_predictor.py`

- [ ] **Step 1: Write the failing test for automatic interval discovery**

Add this test method to `QuantileIntervalTests` in `tests/test_predictor.py`:

```python
    def test_supported_interval_alphas_from_phase2_grid(self) -> None:
        quantiles = [0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95]
        self.assertEqual(
            supported_interval_alphas(quantiles),
            [0.10, 0.20, 0.40, 0.50, 0.60, 0.80],
        )
```

Also add `supported_interval_alphas` to the import list at the top of `tests/test_predictor.py`:

```python
    supported_interval_alphas,
```

- [ ] **Step 2: Run the targeted test and confirm it fails**

Run:

```bash
python -m pytest tests/test_predictor.py::QuantileIntervalTests::test_supported_interval_alphas_from_phase2_grid -q
```

Expected: FAIL because `supported_interval_alphas` does not exist yet.

- [ ] **Step 3: Implement the helper and make the pipeline derive all supported intervals by default**

Add this helper near the quantile validation functions in `predictor.py`:

```python
def supported_interval_alphas(quantiles: Sequence[float]) -> List[float]:
    """
    Return all central-interval miscoverage levels supported by the quantile grid.

    For example, a grid containing 0.05 and 0.95 supports alpha=0.10.
    """
    validate_quantiles(quantiles)
    q_values = np.asarray(quantiles, dtype=float)

    alphas: List[float] = []
    for q_lo in q_values[:-1]:
        q_hi = 1.0 - q_lo
        alpha = float(2.0 * q_lo)
        if alpha >= 1.0:
            continue
        if np.any(np.isclose(q_values, q_hi)) and alpha not in alphas:
            alphas.append(alpha)

    return alphas
```

Update `WeatherQuantilePipeline.__init__` so `interval_alphas=None` means:

```python
        self.interval_alphas = list(interval_alphas) if interval_alphas is not None else supported_interval_alphas(self.model_cfg.quantiles)
```

Update the phase-1 and phase-2 quantile grids so both include `0.25` and `0.75`:

```python
        quantiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
```

for phase 1, and:

```python
        quantiles=[0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95]
```

for phase 2.

Update the full KMIA entrypoint so `run_phase(...)` uses the pipeline default interval set instead of hard-coding a single alpha.

Change the `run_phase` signature to:

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
) -> WeatherQuantilePipeline:
```

and pass `interval_alphas=interval_alphas` into `WeatherQuantilePipeline(...)`. Update the two calls in `__main__` so they omit the old single-alpha list and use the pipeline default.

- [ ] **Step 4: Run the targeted test again**

Run:

```bash
python -m pytest tests/test_predictor.py::QuantileIntervalTests::test_supported_interval_alphas_from_phase2_grid -q
```

Expected: PASS.

- [ ] **Step 5: Run the full test suite**

Run:

```bash
python -m pytest -q
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add predictor.py tests/test_predictor.py
git commit -m "feat: derive KMIA interval calibration grid"
```

Expected: commit succeeds on `station-kmia-training`.

---

### Task 2: Persist Every Supported Calibration Correction in the Artifact Bundle

**Files:**
- Modify: `predictor.py`
- Modify: `tests/test_predictor.py`

- [ ] **Step 1: Write the failing artifact test**

Add this test method to `KMIASmokeTests` in `tests/test_predictor.py`:

```python
    def test_pipeline_artifact_round_trip_saves_all_interval_corrections(self) -> None:
        df = load_kmia_training_data(DATA_PATH)
        dataset_cfg = build_kmia_dataset_config()
        block_cfg = build_default_kmia_block_split_config(df)
        cv_cfg = ExpandingWindowCVConfig(
            min_train_days=365,
            val_days=30,
            step_days=90,
            max_folds=1,
        )
        model_cfg = QuantileModelConfig(
            quantiles=[0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95],
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
        )
        pipeline.fit_final(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "kmia_artifact"
            artifact = pipeline.build_artifact(metadata={"run_name": "artifact_round_trip"})
            artifact.save(artifact_dir)

            payload = json.loads((artifact_dir / "artifact.json").read_text(encoding="utf-8"))
            self.assertEqual(
                sorted(payload["calibration_corrections"]),
                ["0.1", "0.2", "0.4", "0.5", "0.6", "0.8"],
            )

            loaded = type(artifact).load(artifact_dir)
            self.assertEqual(sorted(loaded.calibrator.result_.alpha_to_correction), [0.1, 0.2, 0.4, 0.5, 0.6, 0.8])
```

Also add `json` to the import list at the top of `tests/test_predictor.py`:

```python
import json
```

- [ ] **Step 2: Run the targeted test and confirm it fails**

Run:

```bash
python -m pytest tests/test_predictor.py::KMIASmokeTests::test_pipeline_artifact_round_trip_saves_all_interval_corrections -q
```

Expected: FAIL because the pipeline still only calibrates one interval until Task 1 is complete.

- [ ] **Step 3: Implement the artifact-facing output helper**

Add a method to `WeatherQuantileArtifact` in `predictor.py` that appends all calibrated interval columns to the repaired quantile dataframe:

```python
    def predict_output_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.predict_distribution_inputs(df)
        for alpha in self.interval_alphas:
            lower, upper = self.predict_interval(df, alpha)
            pct = int(round((1.0 - alpha) * 100))
            out[f"interval_{pct}_lower"] = lower
            out[f"interval_{pct}_upper"] = upper
        return out
```

Update `WeatherQuantilePipeline.build_artifact(...)` to carry the derived `interval_alphas`, and keep `WeatherQuantileArtifact.save()` / `load()` writing and reading `artifact.json` with every correction in `calibration_corrections`.

- [ ] **Step 4: Run the targeted test again**

Run:

```bash
python -m pytest tests/test_predictor.py::KMIASmokeTests::test_pipeline_artifact_round_trip_saves_all_interval_corrections -q
```

Expected: PASS.

- [ ] **Step 5: Run the full test suite**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass, including the prior smoke tests.

- [ ] **Step 6: Commit**

Run:

```bash
git add predictor.py tests/test_predictor.py
git commit -m "feat: save full KMIA calibration artifact"
```

Expected: commit succeeds on `station-kmia-training`.

---

### Task 3: Add `make_prediction.py` for Batch and Single-Row CSV Prediction

**Files:**
- Create: `make_prediction.py`
- Modify: `tests/test_make_prediction.py`
- Modify: `predictor.py`

- [ ] **Step 1: Write the failing CLI tests**

Create `tests/test_make_prediction.py` with these tests:

```python
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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


class MakePredictionCLITests(unittest.TestCase):
    def test_make_prediction_cli_handles_single_row_csv(self) -> None:
        df = load_kmia_training_data(DATA_PATH)
        dataset_cfg = build_kmia_dataset_config()
        block_cfg = build_default_kmia_block_split_config(df)
        cv_cfg = ExpandingWindowCVConfig(min_train_days=365, val_days=30, step_days=90, max_folds=1)
        model_cfg = QuantileModelConfig(
            quantiles=[0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95],
            iterations=5,
            learning_rate=0.1,
            depth=4,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=2,
            task_type="CPU",
        )
        pipeline = WeatherQuantilePipeline(dataset_cfg, block_cfg, cv_cfg, model_cfg)
        pipeline.fit_final(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifact"
            pipeline.save_artifact(artifact_dir)

            input_csv = Path(tmpdir) / "single_row.csv"
            pd.DataFrame([pipeline.blocks_["test"].iloc[0].drop(labels=["y"])]).to_csv(input_csv, index=False)
            output_csv = Path(tmpdir) / "single_row_predictions.csv"

            result = subprocess.run(
                [
                    sys.executable,
                    "make_prediction.py",
                    "--artifact-dir",
                    str(artifact_dir),
                    "--input",
                    str(input_csv),
                    "--output",
                    str(output_csv),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            pred_df = pd.read_csv(output_csv)
            self.assertEqual(len(pred_df), 1)
            self.assertIn("interval_90_lower", pred_df.columns)
            self.assertIn("interval_50_upper", pred_df.columns)

    def test_make_prediction_cli_handles_batch_csv(self) -> None:
        df = load_kmia_training_data(DATA_PATH)
        dataset_cfg = build_kmia_dataset_config()
        block_cfg = build_default_kmia_block_split_config(df)
        cv_cfg = ExpandingWindowCVConfig(min_train_days=365, val_days=30, step_days=90, max_folds=1)
        model_cfg = QuantileModelConfig(
            quantiles=[0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95],
            iterations=5,
            learning_rate=0.1,
            depth=4,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=2,
            task_type="CPU",
        )
        pipeline = WeatherQuantilePipeline(dataset_cfg, block_cfg, cv_cfg, model_cfg)
        pipeline.fit_final(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifact"
            pipeline.save_artifact(artifact_dir)

            input_csv = Path(tmpdir) / "batch.csv"
            batch_df = pipeline.blocks_["test"].head(3).drop(columns=["y"])
            batch_df.to_csv(input_csv, index=False)
            output_csv = Path(tmpdir) / "batch_predictions.csv"

            result = subprocess.run(
                [
                    sys.executable,
                    "make_prediction.py",
                    "--artifact-dir",
                    str(artifact_dir),
                    "--input",
                    str(input_csv),
                    "--output",
                    str(output_csv),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            pred_df = pd.read_csv(output_csv)
            self.assertEqual(len(pred_df), 3)
            self.assertIn("interval_90_lower", pred_df.columns)
            self.assertIn("interval_20_upper", pred_df.columns)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run:

```bash
python -m pytest tests/test_make_prediction.py -q
```

Expected: FAIL because `make_prediction.py` does not exist yet.

- [ ] **Step 3: Implement the CLI script**

Create `make_prediction.py` with:

```python
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
```

If `WeatherQuantileArtifact.predict_output_frame(...)` does not exist yet, add it in `predictor.py` as part of this task.

- [ ] **Step 4: Run the tests again**

Run:

```bash
python -m pytest tests/test_make_prediction.py -q
```

Expected: PASS for both the single-row and batch cases.

- [ ] **Step 5: Run a manual CLI smoke check**

Run:

```bash
python make_prediction.py --artifact-dir artifacts/phase2_denser_quantiles --input some_input.csv --output some_output.csv
```

Expected: exits 0 and writes a CSV containing the preserved input columns, `q_*` columns, and `interval_*_lower` / `interval_*_upper` columns.

- [ ] **Step 6: Commit**

Run:

```bash
git add make_prediction.py predictor.py tests/test_make_prediction.py
git commit -m "feat: add KMIA prediction CLI"
```

Expected: commit succeeds on `station-kmia-training`.

---

### Task 4: Update Usage Notes and Final Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the README to match the new workflow**

Replace the current KMIA section with:

```markdown
# weather-predict

Kalshi weather trader experiments.

## KMIA Daily Maximum Temperature

The KMIA proof of concept trains a single-station daily maximum temperature
quantile model from `kalshiTraining_KMIA.dat`, saves the final artifact bundle
as `model.cbm` plus `artifact.json`, and exposes a small prediction CLI.

Run the test suite:

```bash
python -m pytest -q
```

Run a local CPU smoke test:

```bash
python predictor.py --smoke
```

Run the full KMIA training entrypoint:

```bash
python predictor.py
```

Run the prediction CLI on an input CSV:

```bash
python make_prediction.py --artifact-dir artifacts/phase2_denser_quantiles --input input.csv --output predictions.csv
```

The full quantile training path uses CPU because CatBoost does not support
`MultiQuantile` on GPU. The sandbox verifies smoke behavior on CPU, and David's
local environment can still use GPU for other CatBoost experiments where the
objective supports it.
```

- [ ] **Step 2: Run the full test suite**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass, including the new CLI tests.

- [ ] **Step 3: Run the syntax check**

Run:

```bash
python -m compileall -q predictor.py make_prediction.py tests/test_predictor.py tests/test_make_prediction.py
```

Expected: exit code 0 and no output.

- [ ] **Step 4: Run the full training entrypoint**

Run:

```bash
python predictor.py
```

Expected: both phases complete, `artifacts/phase2_denser_quantiles/model.cbm` and `artifacts/phase2_denser_quantiles/artifact.json` are written, and the printed artifact metrics include every supported calibrated interval.

- [ ] **Step 5: Review the recent history**

Run:

```bash
git log --oneline --decorate --max-count=10
```

Expected: the recent commits are on `station-kmia-training`, not `main`.

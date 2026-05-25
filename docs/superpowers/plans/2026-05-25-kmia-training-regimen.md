# KMIA Training Regimen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the KMIA proof of concept into a tested single-station daily maximum temperature training pipeline using the KMIA Kalshi training file and CatBoost quantile regression.

**Architecture:** Keep the current single-file POC structure for this pass because the repository is still small and the priority is to stabilize behavior before reorganizing files. Add narrow helpers around the existing pipeline: KMIA data loading, deterministic time splits, interval-grid validation, leakage-safe final fitting, and repeatable smoke/verification commands.

**Tech Stack:** Python 3.14, pandas, NumPy, CatBoost, Optuna, pytest.

---

## File Structure

- `predictor.py`: Current POC module and executable script. Owns configs, KMIA data loading, split helpers, CatBoost wrapper, conformal calibration, tuning pipeline, and the KMIA entrypoint for this pass.
- `kalshiTraining_KMIA.dat`: KMIA daily maximum temperature training data. The file is CSV despite the `.dat` extension.
- `tests/test_predictor.py`: Focused tests for data loading, split behavior, quantile interval validation, and entrypoint wiring.
- `.gitignore`: Ignores generated Python, pytest, and CatBoost artifacts.
- `docs/superpowers/plans/2026-05-25-kmia-training-regimen.md`: This execution plan.

## Current Verified Baseline

- Branch: `station-kmia-training`.
- Existing tests pass with `python -m pytest -q`: 10 tests.
- CPU smoke test trains and evaluates end to end with tiny iterations.
- GPU CatBoost works in David's shell, but Codex's execution sandbox cannot see a CUDA device. Codex should use CPU for smoke tests and leave production/default model config on GPU.

---

### Task 1: Commit the Stabilized Baseline

**Files:**
- Add: `.gitignore`
- Add: `kalshiTraining_KMIA.dat`
- Add: `predictor.py`
- Add: `tests/test_predictor.py`
- Add: `docs/superpowers/plans/2026-05-25-kmia-training-regimen.md`

- [ ] **Step 1: Verify branch**

Run:

```bash
git status --short --branch
```

Expected: branch starts with `## station-kmia-training...origin/main`.

- [ ] **Step 2: Run tests**

Run:

```bash
python -m pytest -q
```

Expected: `10 passed`.

- [ ] **Step 3: Run compile check**

Run:

```bash
python -m compileall -q predictor.py tests/test_predictor.py
```

Expected: exit code 0 and no output.

- [ ] **Step 4: Stage baseline files**

Run:

```bash
git add .gitignore kalshiTraining_KMIA.dat predictor.py tests/test_predictor.py docs/superpowers/plans/2026-05-25-kmia-training-regimen.md
```

Expected: no output.

- [ ] **Step 5: Commit baseline**

Run:

```bash
git commit -m "feat: add KMIA training pipeline baseline"
```

Expected: commit succeeds on `station-kmia-training`.

---

### Task 2: Add Loader Validation for KMIA Schema

**Files:**
- Modify: `tests/test_predictor.py`
- Modify: `predictor.py`

- [ ] **Step 1: Write failing test for missing required columns**

Add this test method to `KMIATrainingDataTests` in `tests/test_predictor.py`:

```python
    def test_load_kmia_training_data_rejects_missing_columns(self) -> None:
        path = Path("missing_kmia_columns.csv")
        path.write_text("validDate,nbmMaxT,observedMaxT\nMAY-20-2020,88,94\n")

        try:
            with self.assertRaisesRegex(ValueError, "Missing required KMIA columns"):
                load_kmia_training_data(path)
        finally:
            path.unlink()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_predictor.py::KMIATrainingDataTests::test_load_kmia_training_data_rejects_missing_columns -q
```

Expected: FAIL because `load_kmia_training_data` raises `KeyError` or does not raise the expected `ValueError`.

- [ ] **Step 3: Implement minimal validation**

Update `load_kmia_training_data` in `predictor.py` so it validates the raw CSV columns before renaming:

```python
    df = pd.read_csv(path)
    required_columns = {"validDate", "observedMaxT", *KMIA_FEATURE_COLUMNS}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required KMIA columns: {missing}")

    df = df.rename(columns={"validDate": "date", "observedMaxT": "y"})
```

- [ ] **Step 4: Run targeted test**

Run:

```bash
python -m pytest tests/test_predictor.py::KMIATrainingDataTests::test_load_kmia_training_data_rejects_missing_columns -q
```

Expected: PASS.

- [ ] **Step 5: Run full tests**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add predictor.py tests/test_predictor.py
git commit -m "test: validate KMIA training schema"
```

Expected: commit succeeds on `station-kmia-training`.

---

### Task 3: Add a Reusable CPU Smoke Function

**Files:**
- Modify: `tests/test_predictor.py`
- Modify: `predictor.py`

- [ ] **Step 1: Write failing test for smoke helper**

Add imports in `tests/test_predictor.py`:

```python
    run_kmia_smoke_test,
```

Add this test class near the bottom of `tests/test_predictor.py`:

```python
class KMIASmokeTests(unittest.TestCase):
    def test_run_kmia_smoke_test_returns_cal_and_test_metrics(self) -> None:
        metrics = run_kmia_smoke_test(DATA_PATH, task_type="CPU")

        self.assertEqual(set(metrics), {"cal", "test"})
        for block_metrics in metrics.values():
            self.assertIn("mean_pinball_loss", block_metrics)
            self.assertIn("coverage_90", block_metrics)
            self.assertIn("width_90", block_metrics)
            self.assertIn("crossing_rate_after_repair", block_metrics)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_predictor.py::KMIASmokeTests::test_run_kmia_smoke_test_returns_cal_and_test_metrics -q
```

Expected: FAIL because `run_kmia_smoke_test` is not defined.

- [ ] **Step 3: Implement minimal smoke helper**

Add this helper near `run_phase` in `predictor.py`:

```python
def run_kmia_smoke_test(
    data_path: str | Path = "kalshiTraining_KMIA.dat",
    task_type: str = "CPU",
) -> Dict[str, Dict[str, float]]:
    """
    Run a small KMIA training smoke test.

    Parameters
    ----------
    data_path : str or pathlib.Path, default="kalshiTraining_KMIA.dat"
        Path to the KMIA training CSV file.
    task_type : str, default="CPU"
        CatBoost task type used for the smoke model.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Metrics keyed by block name: ``cal`` and ``test``.
    """
    df = load_kmia_training_data(data_path)
    dataset_cfg = build_kmia_dataset_config()
    block_cfg = build_default_kmia_block_split_config(df)
    cv_cfg = ExpandingWindowCVConfig(
        min_train_days=365,
        val_days=30,
        step_days=90,
        max_folds=2,
    )
    model_cfg = QuantileModelConfig(
        quantiles=[0.05, 0.50, 0.95],
        iterations=20,
        learning_rate=0.05,
        depth=4,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=5,
        task_type=task_type,
    )
    pipeline = WeatherQuantilePipeline(
        dataset_cfg=dataset_cfg,
        block_cfg=block_cfg,
        cv_cfg=cv_cfg,
        model_cfg=model_cfg,
        interval_alphas=[0.10],
    )
    pipeline.fit_final(df)
    return {
        "cal": pipeline.evaluate_block("cal"),
        "test": pipeline.evaluate_block("test"),
    }
```

- [ ] **Step 4: Run targeted smoke test**

Run:

```bash
python -m pytest tests/test_predictor.py::KMIASmokeTests::test_run_kmia_smoke_test_returns_cal_and_test_metrics -q
```

Expected: PASS. This is a real CatBoost CPU smoke test and may take a few seconds.

- [ ] **Step 5: Run full tests**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add predictor.py tests/test_predictor.py
git commit -m "test: add KMIA CPU smoke helper"
```

Expected: commit succeeds on `station-kmia-training`.

---

### Task 4: Add Entrypoint Mode Selection

**Files:**
- Modify: `tests/test_predictor.py`
- Modify: `predictor.py`

- [ ] **Step 1: Write failing test for smoke entrypoint wiring**

Add this assertion to `test_main_example_uses_kmia_loader_and_config`:

```python
        self.assertIn('if "--smoke" in sys.argv:', source)
        self.assertIn("run_kmia_smoke_test(", source)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_predictor.py::KMIATrainingDataTests::test_main_example_uses_kmia_loader_and_config -q
```

Expected: FAIL because `--smoke` wiring is not present.

- [ ] **Step 3: Add top-level import**

Modify the imports at the top of `predictor.py`:

```python
import sys
from dataclasses import dataclass, field
from pathlib import Path
```

- [ ] **Step 4: Wire smoke mode in `__main__`**

At the start of the `if __name__ == "__main__":` block, add:

```python
    if "--smoke" in sys.argv:
        smoke_metrics = run_kmia_smoke_test("kalshiTraining_KMIA.dat", task_type="CPU")
        print("Smoke metrics:", smoke_metrics)
        raise SystemExit(0)
```

Keep the existing real tuning run below this block unchanged.

- [ ] **Step 5: Run targeted test**

Run:

```bash
python -m pytest tests/test_predictor.py::KMIATrainingDataTests::test_main_example_uses_kmia_loader_and_config -q
```

Expected: PASS.

- [ ] **Step 6: Run smoke entrypoint manually**

Run:

```bash
python predictor.py --smoke
```

Expected: prints `Smoke metrics:` with `cal` and `test` metrics, exits 0.

- [ ] **Step 7: Run full tests**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

Run:

```bash
git add predictor.py tests/test_predictor.py
git commit -m "feat: add KMIA smoke entrypoint"
```

Expected: commit succeeds on `station-kmia-training`.

---

### Task 5: Document Local Verification and GPU Caveat

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Replace `README.md` content with:

```markdown
# weather-predict

Kalshi weather trader experiments.

## KMIA Daily Maximum Temperature

The current proof of concept trains a single-station KMIA daily maximum
temperature quantile model from `kalshiTraining_KMIA.dat`.

Run the test suite:

```bash
python -m pytest -q
```

Run a local CPU smoke test:

```bash
python predictor.py --smoke
```

Run the full KMIA tuning entrypoint:

```bash
python predictor.py
```

The default phase-1 CatBoost config uses `task_type="GPU"`. Codex's sandbox
does not expose a CUDA device, so Codex verifies smoke behavior on CPU. David's
local `meteorology_py3d14` environment has verified that CatBoost can train with
`task_type="GPU"`.
```

- [ ] **Step 2: Run tests**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run smoke entrypoint**

Run:

```bash
python predictor.py --smoke
```

Expected: prints `Smoke metrics:` with `cal` and `test` metrics, exits 0.

- [ ] **Step 4: Commit**

Run:

```bash
git add README.md
git commit -m "docs: document KMIA training workflow"
```

Expected: commit succeeds on `station-kmia-training`.

---

### Task 6: Final Verification Before Push or PR

**Files:**
- No code changes.

- [ ] **Step 1: Confirm branch**

Run:

```bash
git status --short --branch
```

Expected: branch starts with `## station-kmia-training`.

- [ ] **Step 2: Run tests**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run compile check**

Run:

```bash
python -m compileall -q predictor.py tests/test_predictor.py
```

Expected: exit code 0 and no output.

- [ ] **Step 4: Run smoke entrypoint**

Run:

```bash
python predictor.py --smoke
```

Expected: prints `Smoke metrics:` with `cal` and `test` metrics, exits 0.

- [ ] **Step 5: Review commit history**

Run:

```bash
git log --oneline --decorate --max-count=8
```

Expected: recent commits are on `station-kmia-training`, not `main`.

- [ ] **Step 6: Decide integration**

Ask David whether to push the branch, open a PR, or keep the branch local.

---

## Self-Review

- Spec coverage: The plan covers KMIA data loading, deterministic time splitting, interval validation, leakage-safe final fitting, CPU smoke verification, GPU caveat documentation, and branch-safe commits.
- Placeholder scan: No `TBD`, `TODO`, or unspecified implementation steps remain.
- Type consistency: Function names match current code and planned tests: `load_kmia_training_data`, `build_kmia_dataset_config`, `build_default_kmia_block_split_config`, `split_final_train_eval`, `run_kmia_smoke_test`, and `WeatherQuantilePipeline`.

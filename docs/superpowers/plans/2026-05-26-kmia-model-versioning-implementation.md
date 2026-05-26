# KMIA Model Versioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save each KMIA training run into its own timestamped artifact directory with `model.cbm`, `artifact.json`, and `metrics.json`, so earlier runs are never overwritten and run-to-run comparison is straightforward.

**Architecture:** Keep the existing CatBoost plus JSON artifact bundle as the model payload, but wrap it in a versioned run directory derived automatically from the phase name, a timestamp, and the short git SHA. `predictor.py` remains the single source of truth for training, artifact creation, and metrics capture. The training entrypoint writes one versioned directory per saved phase, while inference code continues to load a single artifact directory unchanged.

**Tech Stack:** Python 3.14, pandas, NumPy, pathlib, json, pytest.

---

## File Structure

- `predictor.py`: Owns version-name helpers, artifact save-path logic, metrics JSON writing, and the training entrypoint that saves phase artifacts into versioned directories.
- `tests/test_predictor.py`: Adds failing and passing tests for versioned artifact directory naming, `metrics.json` content, and save/load round trips from a versioned directory.
- `README.md`: Documents the new artifact directory layout and the compare workflow.
- `docs/superpowers/plans/2026-05-26-kmia-model-versioning-implementation.md`: This execution plan.

## Current Baseline

- The KMIA pipeline already saves `model.cbm` plus `artifact.json`.
- The current entrypoint still writes to a fixed path like `artifacts/phase2_denser_quantiles/`.
- `make_prediction.py`, `plot_calibration.py`, and `score_markets.py` already load a single artifact directory.
- The training pipeline already computes calibration and test metrics through `WeatherQuantilePipeline.evaluate_block(...)`.

The missing piece is a versioned artifact root that prevents overwrites and preserves a metrics summary alongside each saved run.

---

### Task 1: Add Version-Name and Save-Path Tests

**Files:**
- Modify: `tests/test_predictor.py`
- Modify: `predictor.py`

- [ ] **Step 1: Write the failing tests for version naming and versioned save output**

Add tests that prove the version name format and the new save layout.

```python
from predictor import build_artifact_version_name


def test_build_artifact_version_name_uses_phase_timestamp_and_sha(self) -> None:
    self.assertEqual(
        build_artifact_version_name("phase2", "20260526_143012", "9106e55"),
        "phase2_20260526_143012_9106e55",
    )


def test_build_artifact_version_name_falls_back_to_nogit(self) -> None:
    self.assertEqual(
        build_artifact_version_name("phase2", "20260526_143012", None),
        "phase2_20260526_143012_nogit",
    )
```

Add a pipeline-level test that saves into a versioned root and checks the directory contents.

```python
with tempfile.TemporaryDirectory() as tmpdir:
    artifact_root = Path(tmpdir) / "artifacts"
    pipeline.fit_final(df)

    run_dir = pipeline.save_artifact(
        artifact_root,
        phase_name="phase2_denser_quantiles",
        metadata={"study_name": "phase2_denser_quantiles"},
        metrics={
            "phase_name": "phase2_denser_quantiles",
            "study_name": "phase2_denser_quantiles",
            "cal": {"mean_pinball_loss": 0.1},
            "test": {"mean_pinball_loss": 0.2},
        },
    )

    self.assertRegex(
        run_dir.name,
        r"^phase2_denser_quantiles_\d{8}_\d{6}_(?:[0-9a-f]{7}|nogit)$",
    )
    self.assertTrue((run_dir / "model.cbm").exists())
    self.assertTrue((run_dir / "artifact.json").exists())
    self.assertTrue((run_dir / "metrics.json").exists())
```

Also assert that `metrics.json` contains the summary fields needed for comparison:

```python
metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
self.assertEqual(metrics["phase_name"], "phase2_denser_quantiles")
self.assertEqual(metrics["study_name"], "phase2_denser_quantiles")
self.assertEqual(metrics["artifact_version"], run_dir.name)
self.assertIn("timestamp", metrics)
self.assertIn("git_sha", metrics)
self.assertIn("cal", metrics)
self.assertIn("test", metrics)
```

- [ ] **Step 2: Run the targeted tests and confirm they fail**

Run:

```bash
python -m pytest tests/test_predictor.py -q
```

Expected: the new versioning assertions fail because the code still writes to a fixed artifact path and does not emit `metrics.json`.

---

### Task 2: Implement Versioned Artifact Saving in `predictor.py`

**Files:**
- Modify: `predictor.py`
- Modify: `tests/test_predictor.py`

- [ ] **Step 1: Add the version-name helper and git SHA helper**

Add a pure helper that formats the directory name, plus a small helper that returns the short git SHA when available.

```python
def build_artifact_version_name(
    phase_name: str,
    timestamp: str,
    git_sha: str | None,
) -> str:
    short_sha = git_sha or "nogit"
    return f"{phase_name}_{timestamp}_{short_sha}"
```

The runtime helper can call `git rev-parse --short HEAD` and fall back to `None` if git is unavailable.

- [ ] **Step 2: Change `WeatherQuantilePipeline.save_artifact(...)` to create a versioned subdirectory**

Update the method so the caller passes an artifact root and a phase name, not a final artifact directory.

```python
def save_artifact(
    self,
    output_root: str | Path,
    phase_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    artifact = self.build_artifact(metadata=metadata)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    git_sha = get_git_short_sha()
    version_name = build_artifact_version_name(phase_name, timestamp, git_sha)
    artifact_path = Path(output_root) / version_name
    artifact.save(artifact_path)

    metrics_payload = {
        "artifact_version": version_name,
        "phase_name": phase_name,
        "timestamp": timestamp,
        "git_sha": git_sha or "nogit",
    }
    if metrics is not None:
        metrics_payload.update(metrics)

    (artifact_path / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact_path
```

Keep `WeatherQuantileArtifact.save()` and `WeatherQuantileArtifact.load()` unchanged. The versioning wrapper belongs in the pipeline layer, not in the CatBoost bundle itself.

- [ ] **Step 3: Update the training entrypoint to save both phase artifacts**

In the `__main__` block, save the phase 1 and phase 2 pipelines into `artifacts/` as separate versioned directories.

```python
phase1_metrics = {
    "study_name": phase1_pipeline.study_.study_name,
    "best_optuna_score": phase1_pipeline.study_.best_trial.value,
    "best_params": phase1_pipeline.study_.best_trial.params,
    "cal": phase1_pipeline.evaluate_block("cal"),
    "test": phase1_pipeline.evaluate_block("test"),
}
phase1_path = phase1_pipeline.save_artifact(
    Path("artifacts"),
    phase_name="phase1_coarse_quantiles",
    metadata={"study_name": "phase1_coarse_quantiles"},
    metrics=phase1_metrics,
)

phase2_metrics = {
    "study_name": phase2_pipeline.study_.study_name,
    "best_optuna_score": phase2_pipeline.study_.best_trial.value,
    "best_params": phase2_pipeline.study_.best_trial.params,
    "cal": phase2_pipeline.evaluate_block("cal"),
    "test": phase2_pipeline.evaluate_block("test"),
}
phase2_path = phase2_pipeline.save_artifact(
    Path("artifacts"),
    phase_name="phase2_denser_quantiles",
    metadata={"study_name": "phase2_denser_quantiles"},
    metrics=phase2_metrics,
)
print(f"\nSaved artifact bundle to {phase1_path}")
print(f"\nSaved artifact bundle to {phase2_path}")
```

This keeps the comparison story simple: each phase gets its own versioned directory, and the metrics summary is written next to the model.

- [ ] **Step 4: Run the targeted tests and confirm they pass**

Run:

```bash
python -m pytest tests/test_predictor.py -q
```

Expected: the version-name and versioned-save assertions pass.

- [ ] **Step 5: Run the full test suite and compile check**

Run:

```bash
python -m pytest -q
python -m compileall -q predictor.py make_prediction.py plot_calibration.py market_scoring.py score_markets.py tests/test_predictor.py tests/test_make_prediction.py tests/test_plot_calibration.py tests/test_market_scoring.py tests/test_score_markets.py
```

Expected: all tests pass and compilation is clean.

---

### Task 3: Document the Versioned Artifact Workflow

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the README with the new artifact layout**

Add a short section that documents the versioned directory pattern and the metrics file.

```markdown
## KMIA Artifact Versions

Each training run writes a versioned artifact directory under `artifacts/`:

```text
artifacts/<phase_name>_<YYYYMMDD_HHMMSS>_<shortsha>/
```

Each run directory contains:

- `model.cbm`
- `artifact.json`
- `metrics.json`

The `metrics.json` file is the quick comparison surface for different runs. Compare runs by inspecting their directory names and reading the metrics files side by side.
```

Keep the existing inference examples, but point them at a specific versioned directory instead of a fixed `phase2_denser_quantiles` path.

- [ ] **Step 2: Run the final end-to-end verification**

Run:

```bash
python predictor.py
find artifacts -maxdepth 2 -name metrics.json -print
python -m pytest -q
```

Expected: the full training entrypoint writes versioned phase artifact directories, each with a `metrics.json`, and the test suite remains green.

- [ ] **Step 3: Commit**

Run:

```bash
git add predictor.py tests/test_predictor.py README.md
git commit -m "feat: version kmia artifact outputs"
```

Expected: commit succeeds on `station-kmia-training`.

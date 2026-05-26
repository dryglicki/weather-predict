# KMIA Model Versioning Design

**Goal:** Save each KMIA model run into its own timestamped artifact directory so forecast artifacts, metrics, and provenance stay together and can be compared across runs.

**Architecture:** Keep the existing CatBoost + JSON artifact bundle as the model payload, but place it inside a uniquely named run directory derived from the phase name, a timestamp, and the short git SHA. Each run directory becomes the unit of comparison: it contains the serialized model, the artifact metadata, and a small metrics file with the evaluation summary for that run. No central registry is required for the first version; the filesystem is the source of truth.

**Scope:** KMIA single-station temperature forecasting only.

---

## Current State

The training pipeline already saves a fitted artifact bundle with:

- `model.cbm`
- `artifact.json`

The saved JSON already contains enough information to reconstruct:

- model configuration
- feature columns
- interval calibration corrections
- time-block split metadata
- git and run metadata when provided

What is missing is a consistent versioned directory structure that prevents later runs from overwriting earlier artifacts and makes comparison across runs straightforward.

## Design Summary

Each completed training run writes its outputs into a dedicated directory under `artifacts/`:

```text
artifacts/<phase_name>_<YYYYMMDD>_<HHMMSS>_<shortsha>/
```

Example:

```text
artifacts/phase2_20260526_143012_9106e55/
```

That directory contains:

- `model.cbm`
- `artifact.json`
- `metrics.json`

The directory name is derived automatically from the training run, so users do not need to invent version names by hand.

## Artifact Contents

### `model.cbm`

The CatBoost model file saved by the existing artifact bundle.

### `artifact.json`

The existing metadata payload, including:

- dataset configuration
- block split configuration
- CV configuration
- model configuration
- feature columns
- interval alphas
- calibration corrections
- run metadata

### `metrics.json`

A compact summary of the fitted run, including:

- phase name
- timestamp
- short git SHA
- artifact directory name
- mean pinball loss on calibration
- mean pinball loss on test
- coverage values for supported intervals
- interval widths for supported intervals
- crossing-rate diagnostics
- mean PIT when calibration plots are generated

This file is intentionally summary-level. It should not duplicate the full artifact payload.

## Naming Rules

The versioned directory name must be derived automatically from:

- phase name, such as `phase1` or `phase2`
- current timestamp in `YYYYMMDD_HHMMSS` format
- short git SHA

If git metadata is unavailable, the directory name should still be created by using a fixed fallback token such as `nogit` in the SHA slot.

If the timestamp and SHA are both available, directory names are stable enough for day-to-day comparison and easy to sort chronologically.

The model code should not require a manually supplied version name for the common path.

## Comparison Workflow

The primary comparison workflow is filesystem-based:

1. train a run
2. inspect the run directory
3. compare `metrics.json` files across runs
4. load any `artifact.json` / `model.cbm` pair for inference

This keeps each run self-contained and avoids introducing a registry or database before we need one.

## Constraints

- The versioning scheme must not change the actual model serialization format.
- Existing prediction code should continue to load a single artifact directory.
- The training path must still be able to run without a git SHA if the repository is unavailable, but the SHA should be recorded when present.
- Do not introduce a separate model registry in this phase.

## Success Criteria

The design is successful when:

- every full training run writes to a unique artifact directory
- earlier runs are not overwritten by later runs
- each run directory contains enough metadata to reproduce and compare the run
- downstream inference can load any versioned directory without special handling
- the comparison story stays simple: inspect directories, compare metrics files

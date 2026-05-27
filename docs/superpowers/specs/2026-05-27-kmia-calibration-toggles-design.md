# KMIA Calibration Toggles Design

**Goal:** Make the KMIA training pipeline support two independent calibration toggles: interval conformal calibration and PIT distribution calibration. The interval layer controls coverage on quoted intervals. The PIT layer controls CDF shape for calibration plots and market scoring. Both layers must be optional, persisted in the artifact bundle only when enabled, and loadable downstream without changing the existing forecast CSV contract.

**Architecture:** Keep the current CatBoost quantile model as the base forecaster. Add two independent post-processing stages:

- interval conformal calibration, fit on the held-out calibration block
- distribution calibration, fit on raw PIT values from the same calibration block

The training CLI should expose explicit switches for both layers. The artifact bundle should record which layers were enabled, store the corresponding calibration state only when present, and keep downstream consumers tolerant of either layer being absent.

**Tech Stack:** Python 3.14, pandas, numpy, CatBoost, matplotlib, pytest.

---

## Current State

The repository already has:

- a CatBoost quantile model for KMIA
- split-conformal interval calibration
- an optional PIT remap object in code
- artifact save/load support for both calibration state types
- downstream plotting and scoring code that can consume either raw or calibrated CDFs

What is missing is the training-time wiring that actually fits the PIT remap when requested, and the explicit CLI controls for turning each calibration stage on or off.

## Design Summary

The training entrypoint will support two independent booleans:

- interval calibration: enable or disable split-conformal interval fitting
- PIT calibration: enable or disable distribution calibration fitting

Recommended defaults:

- interval calibration enabled by default, to preserve current coverage behavior
- PIT calibration disabled by default, until we validate that it improves PIT shape and downstream scoring

The CLI should accept independent toggles in a way that makes the training intent explicit. The underlying pipeline should receive those booleans and act accordingly.

## Pipeline Behavior

### Interval calibration

When interval calibration is enabled:

1. fit the CatBoost quantile model
2. predict on the calibration block
3. fit conformal interval corrections for every requested `alpha`
4. persist the interval calibrator into the artifact bundle

When interval calibration is disabled:

- do not fit conformal interval corrections
- do not write interval correction state into the artifact
- downstream consumers should still be able to load the model and use repaired quantiles

### PIT calibration

When PIT calibration is enabled:

1. build repaired quantile predictions on the calibration block
2. compute raw PIT values from the calibration block
3. fit the monotone PIT remap
4. persist the distribution calibrator into the artifact bundle

When PIT calibration is disabled:

- do not fit the PIT remap
- do not write distribution-calibration state into the artifact
- plotting and market scoring should fall back to raw CDF behavior

## Artifact Contract

The artifact bundle should remain a self-contained directory with:

- `model.cbm`
- `artifact.json`
- `metrics.json` for versioned training runs

`artifact.json` should record:

- model metadata
- feature columns
- interval alpha list
- block split metadata
- a flag for interval calibration enabled/disabled
- a flag for PIT calibration enabled/disabled
- interval calibration state when present
- distribution calibration state when present

If a calibration layer is disabled, its payload should be omitted or set to `null`, but the artifact must still load cleanly.

## Downstream Behavior

`make_prediction.py` should continue to emit the repaired quantile columns and, when interval calibration is enabled, the interval endpoint columns. It does not need to force distribution calibration into the CSV output.

`plot_calibration.py` should render:

- raw PIT histogram
- calibrated PIT histogram when the PIT remap is present
- quantile rank histogram

`score_markets.py` should use the calibrated CDF only when the loaded artifact actually contains PIT calibration state and the user requests calibrated scoring.

## Error Handling

- If interval calibration is requested but the quantile grid cannot support the requested intervals, fail fast.
- If PIT calibration is requested but the calibration block cannot produce a valid PIT sample, fail fast.
- If a downstream command requests calibrated scoring but the artifact does not contain the needed state, raise a clear error instead of silently falling back.
- Loading must reject malformed calibration payloads rather than guessing.

## Testing

Add tests for:

- interval calibration enabled and PIT calibration disabled
- PIT calibration enabled and interval calibration disabled
- both calibrations enabled
- both calibrations disabled
- artifact save/load round-trips for each combination
- plotting and scoring behavior when PIT calibration is absent
- plotting and scoring behavior when PIT calibration is present

The tests should use the real KMIA training table for pipeline coverage and small deterministic synthetic inputs for calibrator-specific behavior.

## Success Criteria

This design is successful when:

- the training CLI can turn interval calibration and PIT calibration on or off independently
- the artifact records which calibration layers were actually fitted
- the downstream pipeline behaves correctly whether either calibration layer is present or absent
- the PIT remap changes the calibrated PIT histogram only when it is enabled
- the existing interval coverage behavior remains available when interval calibration is enabled

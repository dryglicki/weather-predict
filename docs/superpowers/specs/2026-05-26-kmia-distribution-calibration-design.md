# KMIA Distribution Calibration Design

**Goal:** Add an optional monotone PIT calibration layer for the KMIA forecast CDF so we can improve spread calibration without removing the existing interval conformal step.

**Architecture:** Keep the current CatBoost quantile model and interval conformal calibration unchanged. Add a separate distribution-calibration object that fits on the held-out calibration block, learns a simple monotone remap of PIT values, and applies that remap to the raw CDF during inference when enabled. The new layer is toggleable, persisted in the artifact bundle, and surfaced in plotting and market scoring so we can compare raw versus distribution-calibrated behavior.

**Tech Stack:** Python 3.14, pandas, numpy, CatBoost, matplotlib, pytest.

---

## Current State

The KMIA pipeline already does the following:

- trains a CatBoost quantile regressor
- repairs quantile crossings
- fits split-conformal interval corrections on the calibration block
- saves and reloads a versioned `model.cbm` + `artifact.json` bundle
- builds a smooth CDF from repaired quantiles for PIT plots and market scoring

The current calibration layer only widens requested intervals. It does not change the shape of the full CDF, so it does not directly address a U-shaped PIT histogram or underdispersed forecast spread.

## Design Summary

The new layer is a monotone PIT remap learned from the calibration block:

1. Predict repaired quantiles for each calibration row.
2. Build the raw CDF from those repaired quantiles.
3. Compute raw PIT values `u = F_raw(y_true)` on the calibration block.
4. Fit a monotone empirical remap `G(u)` that maps raw PIT values toward a calibrated uniform target.
5. At inference time, compute `F_cal(x) = G(F_raw(x))` when distribution calibration is enabled.

This layer is optional and must sit alongside interval conformal calibration, not replace it. Interval calibration remains responsible for coverage on quoted bins; distribution calibration is responsible for CDF shape and PIT flattening.

## Components

### DistributionCalibrator

Add a small calibrator object responsible for:

- fitting on raw PIT values from the calibration block
- storing a monotone lookup table or equivalent empirical mapping
- applying the remap to a raw CDF value at prediction time
- serializing and deserializing its state inside `artifact.json`

The first implementation should be deliberately simple:

- use an empirical monotone mapping
- interpolate linearly between knots
- keep the state small and explicit

### Artifact Support

Extend the existing artifact payload so it can optionally store distribution-calibration state alongside:

- model configuration
- feature columns
- interval alphas
- interval conformal corrections
- block metadata

When distribution calibration is disabled, the artifact must continue to load and behave exactly as it does now.

### Prediction and Scoring

The prediction scripts should support both modes:

- raw repaired quantiles only
- repaired quantiles plus distribution calibration

`make_prediction.py` should keep emitting quantiles and conformal intervals. The calibrated CDF path should be available to downstream consumers without forcing the existing forecast CSV format to change.

`market_scoring.py` and `score_markets.py` should use the calibrated CDF when the toggle is enabled and fall back to the raw CDF otherwise.

### Calibration Plotting

`plot_calibration.py` should show both:

- raw PIT histogram
- distribution-calibrated PIT histogram

That gives a direct before/after check for spread calibration while leaving the interval coverage diagnostics intact.

## Data Flow

### Training / Artifact Build

1. Fit the CatBoost quantile model.
2. Fit interval conformal corrections on the calibration block.
3. If distribution calibration is enabled, compute raw PIT on the calibration block and fit the monotone PIT remap.
4. Save the model, interval calibration metadata, and optional distribution calibration state into the versioned artifact directory.

### Inference / Scoring

1. Load the artifact bundle.
2. Predict repaired quantiles.
3. Build the raw CDF from the quantiles.
4. If distribution calibration is enabled, pass the raw CDF through the PIT remap.
5. Use the resulting CDF for PIT plots, contract probabilities, and market EV scoring.

## Error Handling

- If the artifact is missing distribution-calibration state and the toggle is on, the loader should raise a clear error rather than silently falling back.
- If the calibration block is too small to support the remap meaningfully, the code should fail fast with a descriptive message.
- The remap must remain monotone. If the fitted state is not monotone, the artifact should be rejected during load.
- The current interval conformal path must continue to work even when distribution calibration is disabled or unavailable.

## Testing

Add tests for:

- fitting and serializing the distribution calibrator on a small PIT sample
- loading an artifact with distribution calibration enabled
- applying the calibrated CDF in `market_scoring.py`
- plotting raw versus calibrated PIT in `plot_calibration.py`
- preserving current behavior when the toggle is off

The tests should use real model outputs or small deterministic synthetic data where the logic under test is the calibrator itself.

## Success Criteria

The design is successful when:

- distribution calibration can be enabled or disabled without changing the interval conformal behavior
- the calibrated CDF is monotone and serializable
- `plot_calibration.py` shows improved PIT shape when the layer helps
- market scoring can use either raw or distribution-calibrated CDFs
- existing prediction and artifact loading continue to work when the new layer is off


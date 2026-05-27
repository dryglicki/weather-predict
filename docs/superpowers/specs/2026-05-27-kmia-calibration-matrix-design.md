# KMIA Calibration Matrix Design

**Goal:** Add a comparison script that plots and summarizes multiple KMIA calibration runs in one figure so the four calibration modes can be compared side by side.

**Architecture:** Keep the existing single-run calibration plot intact. Add a new flexible comparison CLI that accepts multiple `artifact_dir,label` entries from the command line, reconstructs the held-out test block from the saved artifact metadata, computes PIT and quantile-rank diagnostics for each run, and renders one row per run with PIT on the left and quantile rank histogram on the right. The figure height should scale with the number of runs, and a summary table should appear at the bottom with both central tendency and flatness diagnostics.

**Tech Stack:** Python 3.14, pandas, numpy, matplotlib, scipy, pytest.

---

## Current State

The repository already has:

- `plot_calibration.py`, which renders raw PIT, calibrated PIT, and quantile rank histograms for a single artifact
- `WeatherQuantileArtifact.load()`, which provides the model, saved quantiles, interval settings, and calibration state
- `TimeBlockManager`, which reconstructs the held-out test block from artifact metadata and raw KMIA CSV input

What is missing is a multi-run comparison tool that can ingest several artifact directories at once and render the comparisons in one figure with a concise summary table.

## Design Summary

The new script will:

1. accept a comma-separated list of `artifact_dir,label` arguments from the CLI
2. load each artifact and reconstruct its test block from the saved block metadata and the raw KMIA CSV
3. compute raw PIT values and quantile-rank values for each run
4. render one row per run, with PIT on the left and rank histogram on the right
5. scale the figure height by the number of runs
6. write a summary table at the bottom containing:
   - row count
   - mean PIT
   - median PIT
   - PIT IQR
   - PIT standard deviation
   - KS statistic vs Uniform(0, 1)
   - reduced chi-square for PIT histogram bins
   - reduced chi-square for rank histogram bins

The script should be flexible enough to handle any reasonable number of runs, not just four, but the initial use case is the four calibration combinations.

## CLI Contract

The CLI should expose:

- `--runs`: one or more comma-separated `artifact_dir,label` entries
- `--data`: raw KMIA CSV used to rebuild the held-out test block
- `--output`: output PNG path

Example:

```bash
python compare_calibration_matrix.py \
  --runs artifacts/run_a,label_a artifacts/run_b,label_b artifacts/run_c,label_c artifacts/run_d,label_d \
  --data kalshiTraining_KMIA.dat \
  --output calibration_matrix.png
```

The parser should accept multiple `--runs` tokens. Each token contains exactly one comma and splits into artifact path plus display label.

## Plot Layout

For `N` runs, the figure should render `N` rows and 2 columns:

- left column: PIT histogram for the run
- right column: quantile rank histogram for the run

The total figure height should scale with `N`, so the output remains readable whether the user compares 2 runs or 8 runs.

Each row should include the run label visibly, and the summary table should sit below the plot grid.

## Summary Metrics

For each run, compute:

- `n_rows`
- `mean_pit`
- `median_pit`
- `pit_iqr`
- `pit_std`
- `pit_ks_stat`
- `pit_reduced_chi2`
- `rank_reduced_chi2`

These are the summary values the user asked for. The PIT summary is centered on Uniform(0, 1), and the histogram flatness metrics should be interpreted relative to that null distribution.

## Flatness Interpretation

The metric family should reflect the fact that PIT flatness means approximate uniformity:

- `mean_pit` should be near `0.5`
- `median_pit` should be near `0.5`
- `pit_iqr` should be near `0.5`
- `pit_std` should be near `sqrt(1/12) ≈ 0.2887`
- `pit_ks_stat` should be near `0`
- `pit_reduced_chi2` should be near `1`
- `rank_reduced_chi2` should be near `1`

The script does not need to invent a new flatness metric. It should report the standard summary values plus the KS / chi-square diagnostics so the user can judge whether the histogram is genuinely uniform or only looks centered.

## Error Handling

- Fail if `--runs` is empty.
- Fail if any `artifact_dir,label` token does not contain exactly one comma.
- Fail if the artifact cannot be loaded or does not contain block metadata.
- Fail if the raw KMIA CSV cannot reconstruct the test block.
- Fail if the output path cannot be written.

## Testing

Add tests for:

- parsing multiple `artifact_dir,label` tokens
- rejecting malformed `--runs` values
- computing PIT summary statistics and flatness metrics
- rendering a multi-row figure with the expected number of subplots
- output scaling behavior as the number of runs changes
- end-to-end CLI execution against small synthetic or real KMIA-backed fixtures

The tests should avoid mocking the core histogram math. They should exercise the real plotting and metric functions with small deterministic inputs where possible.

## Success Criteria

This design is successful when:

- the CLI accepts a comma-separated run specification list through repeated `--runs` arguments
- the script scales to an arbitrary number of runs
- each run gets one PIT panel and one rank panel
- the summary table reports both central tendency and flatness diagnostics
- the output file path is user-controlled
- the four calibration modes can be compared in one generated figure

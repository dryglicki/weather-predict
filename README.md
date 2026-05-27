# weather-predict

Kalshi weather trader experiments.

## KMIA Daily Maximum Temperature

The current proof of concept trains a single-station KMIA daily maximum
temperature quantile model from `kalshiTraining_KMIA.dat`, saves a CatBoost
`model.cbm` plus `artifact.json`, and exposes a prediction CLI for downstream
scoring.

Run the test suite:

```bash
python -m pytest -q
```

Run a local CPU smoke test:

```bash
python predictor.py --smoke --no-pit-calibration
```

Run the full KMIA tuning entrypoint:

```bash
python predictor.py --interval-calibration --no-pit-calibration
```

Run the prediction CLI on a CSV:

```bash
python make_prediction.py --artifact-dir artifacts/phase2_denser_quantiles --input input.csv --output predictions.csv
```

Plot raw vs calibrated PIT on the held-out test block:

```bash
python plot_calibration.py --artifact-dir artifacts/phase2_denser_quantiles --data kalshiTraining_KMIA.dat --output calibration.png
```

Score Miami weather contracts from a forecast CSV:

```bash
python score_markets.py --forecast predictions.csv --markets markets.csv --output scored.csv
```

To use the optional distribution-calibration scoring path, pass the saved artifact bundle and enable the toggle:

```bash
python score_markets.py --artifact-dir artifacts/phase2_denser_quantiles --distribution-calibration --forecast predictions.csv --markets markets.csv --output scored.csv
```

### KMIA Calibration Toggles

The training entrypoint accepts two independent calibration toggles:

- `--interval-calibration` / `--no-interval-calibration`
- `--pit-calibration` / `--no-pit-calibration`

`interval_calibration_enabled` and `calibration_corrections` in `artifact.json` control the interval-coverage layer.
`distribution_calibration_enabled` and `distribution_calibration` in `artifact.json` control the PIT remap layer.

Both layers are optional. Interval calibration widens the quoted intervals. PIT calibration remaps the calibrated CDF used by `plot_calibration.py` and the optional `--distribution-calibration` scoring path.

The market CSV should contain `market_id`, `market_name`, `comparison`, `lower_bound`, `upper_bound`, `yes_price`, and `no_price`. Supported comparisons are `le`, `between`, and `ge`. Contract bounds are inclusive, and settlement temperatures are rounded to integers.

The CatBoost `MultiQuantile` objective used by this pipeline runs on CPU. The
sandbox verifies smoke behavior on CPU, and the saved artifact bundle is the
interface used by downstream prediction code.

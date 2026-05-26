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
python predictor.py --smoke
```

Run the full KMIA tuning entrypoint:

```bash
python predictor.py
```

Run the prediction CLI on a CSV:

```bash
python make_prediction.py --artifact-dir artifacts/phase2_denser_quantiles --input input.csv --output predictions.csv
```

Score Miami weather contracts from a forecast CSV:

```bash
python score_markets.py --forecast predictions.csv --markets markets.csv --output scored.csv
```

The market CSV should contain `market_id`, `market_name`, `comparison`, `lower_bound`, `upper_bound`, `yes_price`, and `no_price`. Supported comparisons are `le`, `between`, and `ge`. Contract bounds are inclusive, and settlement temperatures are rounded to integers.

The CatBoost `MultiQuantile` objective used by this pipeline runs on CPU. The
sandbox verifies smoke behavior on CPU, and the saved artifact bundle is the
interface used by downstream prediction code.

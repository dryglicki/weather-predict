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

The CatBoost `MultiQuantile` objective used by this pipeline runs on CPU. The
sandbox verifies smoke behavior on CPU, and the saved artifact bundle is the
interface used by downstream prediction code.

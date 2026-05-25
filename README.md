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

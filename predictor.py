# weather_quantile_cv_optuna.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import optuna

from catboost import CatBoostRegressor, Pool
from optuna.samplers import TPESampler

try:
    from optuna.integration import CatBoostPruningCallback
    HAS_CATBOOST_PRUNER = True
except Exception:
    HAS_CATBOOST_PRUNER = False


ArrayLike = np.ndarray

KMIA_FEATURE_COLUMNS: Tuple[str, ...] = (
    "nbmMaxT",
    "nbmDeltaT",
    "skyCover",
    "windSpeed",
    "sin(windDir)",
    "cos(windDir)",
    "6hrQPF",
)


# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass
class DatasetConfig:
    """
    Configuration for dataframe columns.
    """
    date_col: str = "date"
    target_col: str = "y"
    feature_cols: Optional[Sequence[str]] = None
    categorical_cols: Optional[Sequence[str]] = None


@dataclass
class BlockSplitConfig:
    """
    Top-level time blocks.

    Development block is used for CV + Optuna only.
    Calibration block is held out for conformal calibration only.
    Test block is held out for final evaluation only.
    """
    dev_start: str
    dev_end: str
    cal_start: str
    cal_end: str
    test_start: str
    test_end: str


@dataclass
class ExpandingWindowCVConfig:
    """
    Expanding-window CV configuration for thin per-station data.

    Parameters are in days.
    """
    min_train_days: int = 365
    val_days: int = 90
    step_days: int = 90
    max_folds: Optional[int] = 5


@dataclass
class QuantileModelConfig:
    """
    CatBoost multi-quantile model configuration.
    """
    quantiles: Sequence[float]

    iterations: int = 1500
    learning_rate: float = 0.03
    depth: int = 6
    l2_leaf_reg: float = 3.0
    min_data_in_leaf: int = 20
    random_strength: float = 1.0
    bagging_temperature: float = 0.0
    border_count: int = 254
    loss_function: Optional[str] = None
    eval_metric: Optional[str] = None
    random_seed: int = 42
    verbose: int = 0
    early_stopping_rounds: int = 100
    task_type: str = "CPU"

    def build_loss_function(self) -> str:
        """
        Build CatBoost MultiQuantile loss string.

        Example:
            MultiQuantile:alpha=0.1,0.25,0.5,0.75,0.9
        """
        q_str = ",".join(f"{q:.6f}".rstrip("0").rstrip(".") for q in self.quantiles)
        return f"MultiQuantile:alpha={q_str}"


@dataclass
class ConformalIntervalResult:
    """
    Stores fitted conformal correction constants for central intervals.
    """
    alpha_to_correction: Dict[float, float] = field(default_factory=dict)


# =============================================================================
# Utility functions
# =============================================================================

def validate_quantiles(quantiles: Sequence[float]) -> None:
    """
    Validate quantile list.
    """
    q = np.asarray(quantiles, dtype=float)
    if q.ndim != 1:
        raise ValueError("Quantiles must be a 1D sequence.")
    if np.any(q <= 0.0) or np.any(q >= 1.0):
        raise ValueError("All quantiles must lie strictly between 0 and 1.")
    if np.any(np.diff(q) <= 0.0):
        raise ValueError("Quantiles must be strictly increasing.")


def validate_interval_quantiles(
    quantiles: Sequence[float],
    interval_alphas: Sequence[float],
) -> None:
    """
    Validate that requested central intervals have matching quantile endpoints.

    Parameters
    ----------
    quantiles : Sequence[float]
        Quantile levels available from the model.
    interval_alphas : Sequence[float]
        Central-interval miscoverage levels. For example, 0.10 requires
        quantiles 0.05 and 0.95.

    Raises
    ------
    ValueError
        If any required interval endpoint is absent from ``quantiles``.
    """
    validate_quantiles(quantiles)
    q_values = np.asarray(quantiles, dtype=float)

    for alpha in interval_alphas:
        q_lo = alpha / 2.0
        q_hi = 1.0 - alpha / 2.0
        for q_required in (q_lo, q_hi):
            if not np.any(np.isclose(q_values, q_required)):
                raise ValueError(
                    f"Interval alpha {alpha} requires quantile {q_required:.6g}."
                )


def load_kmia_training_data(path: str | Path) -> pd.DataFrame:
    """
    Load KMIA daily maximum temperature training data.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file with KMIA NBM predictor columns and observed maximum
        temperature.

    Returns
    -------
    pandas.DataFrame
        Pipeline-ready dataframe sorted by date. The output has ``date`` as the
        timestamp column, ``y`` as the target column, and the KMIA NBM feature
        columns unchanged.

    Raises
    ------
    ValueError
        If the file is missing one or more required KMIA columns.
    """
    df = pd.read_csv(path)
    required_columns = {"validDate", "observedMaxT", *KMIA_FEATURE_COLUMNS}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required KMIA columns: {missing}")

    df = df.rename(columns={"validDate": "date", "observedMaxT": "y"})
    df["date"] = pd.to_datetime(df["date"], format="%B-%d-%Y")

    columns = ["date", *KMIA_FEATURE_COLUMNS, "y"]
    df = df.loc[:, columns].sort_values("date").reset_index(drop=True)
    return df


def build_kmia_dataset_config() -> DatasetConfig:
    """
    Build the dataset configuration for the KMIA training file.

    Returns
    -------
    DatasetConfig
        Dataset configuration mapping the normalized KMIA dataframe to the
        generic quantile pipeline.
    """
    return DatasetConfig(
        date_col="date",
        target_col="y",
        feature_cols=KMIA_FEATURE_COLUMNS,
        categorical_cols=None,
    )


def build_default_kmia_block_split_config(
    df: pd.DataFrame,
    *,
    cal_days: int = 365,
    test_days: int = 365,
) -> BlockSplitConfig:
    """
    Build a deterministic KMIA block split from observed date availability.

    Parameters
    ----------
    df : pandas.DataFrame
        KMIA training dataframe containing a ``date`` column.
    cal_days : int, default=365
        Number of trailing calendar days reserved for the calibration block.
    test_days : int, default=365
        Number of trailing calendar days reserved for the final test block.

    Returns
    -------
    BlockSplitConfig
        Date-bounded development, calibration, and test blocks that are
        contiguous and non-overlapping.

    Raises
    ------
    ValueError
        If the dataframe is empty or there is insufficient history to allocate
        all three blocks.
    """
    if df.empty:
        raise ValueError("KMIA dataframe is empty.")
    if cal_days <= 0 or test_days <= 0:
        raise ValueError("cal_days and test_days must be positive.")

    dates = pd.to_datetime(df["date"]).sort_values().dt.normalize()
    start_date = dates.iloc[0]
    end_date = dates.iloc[-1]

    test_start = end_date - pd.Timedelta(days=test_days - 1)
    cal_end = test_start - pd.Timedelta(days=1)
    cal_start = cal_end - pd.Timedelta(days=cal_days - 1)
    dev_end = cal_start - pd.Timedelta(days=1)

    if dev_end < start_date:
        raise ValueError("Insufficient KMIA history for dev/cal/test split.")

    return BlockSplitConfig(
        dev_start=start_date.strftime("%Y-%m-%d"),
        dev_end=dev_end.strftime("%Y-%m-%d"),
        cal_start=cal_start.strftime("%Y-%m-%d"),
        cal_end=cal_end.strftime("%Y-%m-%d"),
        test_start=test_start.strftime("%Y-%m-%d"),
        test_end=end_date.strftime("%Y-%m-%d"),
    )


def pinball_loss(y_true: ArrayLike, y_pred: ArrayLike, q: float) -> float:
    """
    Compute pinball loss for a single quantile.

    Parameters
    ----------
    y_true : np.ndarray
        Observed values, shape (n_samples,).
    y_pred : np.ndarray
        Predicted quantile values, shape (n_samples,).
    q : float
        Quantile level in (0, 1).

    Returns
    -------
    float
        Mean pinball loss.
    """
    err = y_true - y_pred
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))


def mean_pinball_loss(
    y_true: ArrayLike,
    y_pred_matrix: ArrayLike,
    quantiles: Sequence[float],
) -> float:
    """
    Compute mean pinball loss across a quantile grid.

    Parameters
    ----------
    y_true : np.ndarray
        Observed values, shape (n_samples,).
    y_pred_matrix : np.ndarray
        Predicted quantiles, shape (n_samples, n_quantiles).
    quantiles : Sequence[float]
        Quantile levels aligned to columns of y_pred_matrix.

    Returns
    -------
    float
        Average pinball loss across quantiles.
    """
    losses = [
        pinball_loss(y_true, y_pred_matrix[:, j], q)
        for j, q in enumerate(quantiles)
    ]
    return float(np.mean(losses))


def empirical_interval_coverage(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> float:
    """
    Compute empirical coverage for interval [lower, upper].
    """
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def interval_width(lower: ArrayLike, upper: ArrayLike) -> float:
    """
    Mean interval width.
    """
    return float(np.mean(upper - lower))


def enforce_non_crossing_cummax(y_pred_matrix: ArrayLike) -> ArrayLike:
    """
    Enforce non-crossing quantiles using cumulative maximum.

    Parameters
    ----------
    y_pred_matrix : np.ndarray
        Shape (n_samples, n_quantiles).

    Returns
    -------
    np.ndarray
        Non-crossing quantiles, same shape.
    """
    repaired = np.asarray(y_pred_matrix).copy()
    repaired = np.maximum.accumulate(repaired, axis=1)
    return repaired


def count_crossings(y_pred_matrix: ArrayLike) -> int:
    """
    Count number of crossing instances across all adjacent quantile pairs.

    Returns
    -------
    int
        Number of rows with at least one crossing.
    """
    preds = np.asarray(y_pred_matrix)
    if preds.shape[1] < 2:
        return 0
    return int(np.sum(np.any(np.diff(preds, axis=1) < 0.0, axis=1)))


def slice_time_block(
    df: pd.DataFrame,
    date_col: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Slice dataframe by inclusive time bounds.
    """
    mask = (df[date_col] >= pd.Timestamp(start)) & (df[date_col] <= pd.Timestamp(end))
    out = df.loc[mask].copy()
    if out.empty:
        raise ValueError(f"No rows found between {start} and {end}.")
    return out


def split_final_train_eval(
    dev_df: pd.DataFrame,
    dataset_cfg: DatasetConfig,
    eval_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a development block into final training and early-stopping eval sets.

    Parameters
    ----------
    dev_df : pandas.DataFrame
        Development-period dataframe.
    dataset_cfg : DatasetConfig
        Dataset column configuration.
    eval_days : int
        Number of trailing calendar days to reserve for final-fit evaluation.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(train_df, eval_df)`` where both frames come from the development
        block and the eval frame is strictly later than the train frame.

    Raises
    ------
    ValueError
        If the split would leave an empty final training or final eval frame.
    """
    if eval_days <= 0:
        raise ValueError("final eval days must be positive.")

    date_col = dataset_cfg.date_col
    df = dev_df.sort_values(date_col).copy()
    eval_start = df[date_col].max().normalize() - pd.Timedelta(days=eval_days - 1)

    train_df = df.loc[df[date_col] < eval_start].copy()
    eval_df = df.loc[df[date_col] >= eval_start].copy()

    if train_df.empty or eval_df.empty:
        raise ValueError("final eval split produced an empty train or eval block.")

    return train_df, eval_df


# =============================================================================
# Split management
# =============================================================================

class TimeBlockManager:
    """
    Split dataframe into development, calibration, and test blocks.
    """

    def __init__(self, dataset_cfg: DatasetConfig, block_cfg: BlockSplitConfig) -> None:
        self.dataset_cfg = dataset_cfg
        self.block_cfg = block_cfg

    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split into dev/cal/test.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Keys: dev, cal, test
        """
        date_col = self.dataset_cfg.date_col
        return {
            "dev": slice_time_block(df, date_col, self.block_cfg.dev_start, self.block_cfg.dev_end),
            "cal": slice_time_block(df, date_col, self.block_cfg.cal_start, self.block_cfg.cal_end),
            "test": slice_time_block(df, date_col, self.block_cfg.test_start, self.block_cfg.test_end),
        }


class ExpandingWindowSplitter:
    """
    Create expanding-window train/validation folds within a development block.

    Example:
        train: first 365 days
        val: next 90 days
        step: 90 days
    """

    def __init__(self, dataset_cfg: DatasetConfig, cv_cfg: ExpandingWindowCVConfig) -> None:
        self.dataset_cfg = dataset_cfg
        self.cv_cfg = cv_cfg

    def split(self, dev_df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Build expanding-window folds.

        Parameters
        ----------
        dev_df : pd.DataFrame
            Development-period dataframe only.

        Returns
        -------
        List[Tuple[pd.DataFrame, pd.DataFrame]]
            List of (train_df, val_df) folds.
        """
        date_col = self.dataset_cfg.date_col
        df = dev_df.sort_values(date_col).reset_index(drop=True).copy()

        start_date = df[date_col].min().normalize()
        end_date = df[date_col].max().normalize()

        min_train = pd.Timedelta(days=self.cv_cfg.min_train_days)
        val_len = pd.Timedelta(days=self.cv_cfg.val_days)
        step = pd.Timedelta(days=self.cv_cfg.step_days)

        folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []

        train_end = start_date + min_train - pd.Timedelta(days=1)
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + val_len - pd.Timedelta(days=1)

        while val_end <= end_date:
            train_mask = (df[date_col] >= start_date) & (df[date_col] <= train_end)
            val_mask = (df[date_col] >= val_start) & (df[date_col] <= val_end)

            train_df = df.loc[train_mask].copy()
            val_df = df.loc[val_mask].copy()

            if not train_df.empty and not val_df.empty:
                folds.append((train_df, val_df))

            train_end = train_end + step
            val_start = train_end + pd.Timedelta(days=1)
            val_end = val_start + val_len - pd.Timedelta(days=1)

        if self.cv_cfg.max_folds is not None and len(folds) > self.cv_cfg.max_folds:
            folds = folds[-self.cv_cfg.max_folds:]

        if len(folds) == 0:
            raise ValueError(
                "No CV folds were created. "
                "Check min_train_days, val_days, step_days, and development block length."
            )

        return folds


# =============================================================================
# Model wrapper
# =============================================================================

class WeatherQuantileModel:
    """
    CatBoost multi-quantile regression model wrapper.
    """

    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        model_cfg: QuantileModelConfig,
    ) -> None:
        validate_quantiles(model_cfg.quantiles)
        self.dataset_cfg = dataset_cfg
        self.model_cfg = model_cfg

        self.model: Optional[CatBoostRegressor] = None
        self.feature_cols_: Optional[List[str]] = None
        self.cat_feature_indices_: Optional[List[int]] = None

    def _resolve_feature_cols(self, df: pd.DataFrame) -> List[str]:
        if self.dataset_cfg.feature_cols is not None:
            return list(self.dataset_cfg.feature_cols)

        excluded = {self.dataset_cfg.date_col, self.dataset_cfg.target_col}
        return [c for c in df.columns if c not in excluded]

    def _resolve_cat_indices(self, feature_cols: Sequence[str]) -> List[int]:
        if not self.dataset_cfg.categorical_cols:
            return []
        cat_set = set(self.dataset_cfg.categorical_cols)
        return [i for i, c in enumerate(feature_cols) if c in cat_set]

    def _make_pool(self, df: pd.DataFrame) -> Pool:
        if self.feature_cols_ is None:
            raise RuntimeError("Feature columns are not initialized.")

        return Pool(
            data=df[self.feature_cols_],
            label=df[self.dataset_cfg.target_col],
            cat_features=self.cat_feature_indices_,
        )

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        trial: Optional[optuna.trial.Trial] = None,
    ) -> None:
        """
        Fit CatBoost multi-quantile model.
        """
        self.feature_cols_ = self._resolve_feature_cols(train_df)
        self.cat_feature_indices_ = self._resolve_cat_indices(self.feature_cols_)

        loss_function = self.model_cfg.loss_function or self.model_cfg.build_loss_function()
        eval_metric = self.model_cfg.eval_metric or loss_function

        train_pool = self._make_pool(train_df)
        val_pool = self._make_pool(val_df)

        params = {
            "loss_function": loss_function,
            "eval_metric": eval_metric,
            "iterations": self.model_cfg.iterations,
            "learning_rate": self.model_cfg.learning_rate,
            "depth": self.model_cfg.depth,
            "l2_leaf_reg": self.model_cfg.l2_leaf_reg,
            "min_data_in_leaf": self.model_cfg.min_data_in_leaf,
            "random_strength": self.model_cfg.random_strength,
            "bagging_temperature": self.model_cfg.bagging_temperature,
            "border_count": self.model_cfg.border_count,
            "random_seed": self.model_cfg.random_seed,
            "verbose": self.model_cfg.verbose,
            "task_type": self.model_cfg.task_type,
        }

        self.model = CatBoostRegressor(**params)

        callbacks = []
        if trial is not None and HAS_CATBOOST_PRUNER:
            callbacks.append(CatBoostPruningCallback(trial, metric=eval_metric))

        self.model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            early_stopping_rounds=self.model_cfg.early_stopping_rounds,
            callbacks=callbacks if callbacks else None,
        )

        if trial is not None and HAS_CATBOOST_PRUNER and callbacks:
            callbacks[0].check_pruned()

    def predict_quantiles(
        self,
        df: pd.DataFrame,
        repair_crossing: bool = True,
    ) -> ArrayLike:
        """
        Predict quantiles.

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_quantiles)
        """
        if self.model is None or self.feature_cols_ is None:
            raise RuntimeError("Model has not been fit.")

        preds = np.asarray(self.model.predict(df[self.feature_cols_]))
        if preds.ndim == 1:
            preds = preds[:, None]

        if repair_crossing:
            preds = enforce_non_crossing_cummax(preds)

        return preds

    def predict_quantile_dataframe(
        self,
        df: pd.DataFrame,
        repair_crossing: bool = True,
        include_target: bool = False,
    ) -> pd.DataFrame:
        """
        Return quantiles as a dataframe.
        """
        preds = self.predict_quantiles(df, repair_crossing=repair_crossing)

        out = pd.DataFrame(index=df.index)
        for j, q in enumerate(self.model_cfg.quantiles):
            out[f"q_{q:.3f}"] = preds[:, j]

        if include_target and self.dataset_cfg.target_col in df.columns:
            out[self.dataset_cfg.target_col] = df[self.dataset_cfg.target_col].values

        return out


# =============================================================================
# Conformal calibration
# =============================================================================

class ConformalCalibrator:
    """
    Split-conformal calibrator for central intervals.

    First-pass version:
    - fit on calibration block only
    - additive interval widening
    """

    def __init__(self, quantiles: Sequence[float]) -> None:
        validate_quantiles(quantiles)
        self.quantiles = np.asarray(quantiles, dtype=float)
        self.result_ = ConformalIntervalResult()

    def _find_quantile_index(self, q: float) -> int:
        idx = np.where(np.isclose(self.quantiles, q))[0]
        if len(idx) == 0:
            raise ValueError(f"Required quantile {q} not found in quantile grid.")
        return int(idx[0])

    def fit_interval(
        self,
        y_true: ArrayLike,
        pred_matrix: ArrayLike,
        alpha: float,
    ) -> None:
        """
        Fit a central interval conformal correction.

        Parameters
        ----------
        y_true : np.ndarray
            Shape (n_samples,)
        pred_matrix : np.ndarray
            Shape (n_samples, n_quantiles)
        alpha : float
            Miscoverage, e.g. 0.10 for a 90% interval.
        """
        q_lo = alpha / 2.0
        q_hi = 1.0 - alpha / 2.0

        i_lo = self._find_quantile_index(q_lo)
        i_hi = self._find_quantile_index(q_hi)

        lower = pred_matrix[:, i_lo]
        upper = pred_matrix[:, i_hi]

        scores = np.maximum(lower - y_true, y_true - upper)

        n = len(scores)
        rank = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
        rank = min(max(rank, 0), n - 1)

        correction = float(np.sort(scores)[rank])
        self.result_.alpha_to_correction[alpha] = correction

    def predict_interval(
        self,
        pred_matrix: ArrayLike,
        alpha: float,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Apply fitted conformal correction to interval endpoints.
        """
        if alpha not in self.result_.alpha_to_correction:
            raise RuntimeError(f"No conformal correction fit for alpha={alpha}.")

        q_lo = alpha / 2.0
        q_hi = 1.0 - alpha / 2.0

        i_lo = self._find_quantile_index(q_lo)
        i_hi = self._find_quantile_index(q_hi)

        c = self.result_.alpha_to_correction[alpha]

        lower = pred_matrix[:, i_lo] - c
        upper = pred_matrix[:, i_hi] + c
        return lower, upper


# =============================================================================
# CV scoring
# =============================================================================

def evaluate_quantile_predictions(
    y_true: ArrayLike,
    pred_matrix_raw: ArrayLike,
    quantiles: Sequence[float],
) -> Dict[str, float]:
    """
    Evaluate raw and repaired quantile predictions for one fold or block.
    """
    pred_matrix_raw = np.asarray(pred_matrix_raw)
    pred_matrix_repaired = enforce_non_crossing_cummax(pred_matrix_raw)

    metrics: Dict[str, float] = {}
    metrics["mean_pinball_loss_raw"] = mean_pinball_loss(y_true, pred_matrix_raw, quantiles)
    metrics["mean_pinball_loss_repaired"] = mean_pinball_loss(y_true, pred_matrix_repaired, quantiles)
    metrics["n_crossing_rows_raw"] = float(count_crossings(pred_matrix_raw))
    metrics["crossing_rate_raw"] = float(count_crossings(pred_matrix_raw) / len(y_true))
    return metrics


def summarize_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate fold metrics into means and standard deviations.
    """
    if len(fold_metrics) == 0:
        raise ValueError("fold_metrics is empty.")

    keys = fold_metrics[0].keys()
    out: Dict[str, float] = {}

    for key in keys:
        vals = np.array([m[key] for m in fold_metrics], dtype=float)
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals, ddof=0))

    return out


# =============================================================================
# Optuna objective
# =============================================================================

def suggest_catboost_params(
    trial: optuna.trial.Trial,
    base_model_cfg: QuantileModelConfig,
) -> QuantileModelConfig:
    """
    Suggest a narrow but useful CatBoost hyperparameter search space
    for thin per-station samples.
    """
    return QuantileModelConfig(
        quantiles=base_model_cfg.quantiles,
        iterations=trial.suggest_int("iterations", 500, 2500, step=250),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        depth=trial.suggest_int("depth", 4, 8),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 5, 60),
        random_strength=trial.suggest_float("random_strength", 1e-3, 5.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 3.0),
        border_count=trial.suggest_int("border_count", 64, 255),
        loss_function=base_model_cfg.loss_function,
        eval_metric=base_model_cfg.eval_metric,
        random_seed=base_model_cfg.random_seed,
        verbose=0,
        early_stopping_rounds=base_model_cfg.early_stopping_rounds,
        task_type=base_model_cfg.task_type,
    )


class WalkForwardCVObjective:
    """
    Optuna objective for expanding-window CV on the development block only.

    Objective score:
        mean repaired pinball loss across folds
        + lambda * std repaired pinball loss across folds

    This slightly penalizes fragile hyperparameter settings.
    """

    def __init__(
        self,
        dev_df: pd.DataFrame,
        dataset_cfg: DatasetConfig,
        cv_cfg: ExpandingWindowCVConfig,
        base_model_cfg: QuantileModelConfig,
        stability_penalty: float = 0.25,
    ) -> None:
        self.dev_df = dev_df
        self.dataset_cfg = dataset_cfg
        self.cv_cfg = cv_cfg
        self.base_model_cfg = base_model_cfg
        self.stability_penalty = stability_penalty

        self.cv_splitter = ExpandingWindowSplitter(dataset_cfg, cv_cfg)
        self.folds = self.cv_splitter.split(dev_df)

    def __call__(self, trial: optuna.trial.Trial) -> float:
        model_cfg = suggest_catboost_params(trial, self.base_model_cfg)

        fold_metrics: List[Dict[str, float]] = []

        for fold_idx, (train_df, val_df) in enumerate(self.folds):
            model = WeatherQuantileModel(self.dataset_cfg, model_cfg)
            model.fit(train_df, val_df, trial=trial)

            y_val = val_df[self.dataset_cfg.target_col].to_numpy()
            pred_raw = model.predict_quantiles(val_df, repair_crossing=False)

            metrics = evaluate_quantile_predictions(
                y_true=y_val,
                pred_matrix_raw=pred_raw,
                quantiles=model_cfg.quantiles,
            )
            fold_metrics.append(metrics)

            trial.set_user_attr(f"fold_{fold_idx+1}_pinball_repaired", metrics["mean_pinball_loss_repaired"])
            trial.set_user_attr(f"fold_{fold_idx+1}_crossing_rate_raw", metrics["crossing_rate_raw"])

        summary = summarize_fold_metrics(fold_metrics)

        mean_loss = summary["mean_pinball_loss_repaired_mean"]
        std_loss = summary["mean_pinball_loss_repaired_std"]

        score = mean_loss + self.stability_penalty * std_loss

        for k, v in summary.items():
            trial.set_user_attr(k, v)

        return float(score)


# =============================================================================
# Final pipeline
# =============================================================================

class WeatherQuantilePipeline:
    """
    End-to-end pipeline.

    Flow:
        1. split full data into dev / cal / test
        2. tune hyperparameters with expanding-window CV on dev only
        3. refit final model on full development block
        4. predict on cal and test
        5. fit conformal calibration on cal only
        6. evaluate on cal and test
    """

    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        block_cfg: BlockSplitConfig,
        cv_cfg: ExpandingWindowCVConfig,
        model_cfg: QuantileModelConfig,
        interval_alphas: Optional[Sequence[float]] = None,
    ) -> None:
        self.dataset_cfg = dataset_cfg
        self.block_cfg = block_cfg
        self.cv_cfg = cv_cfg
        self.model_cfg = model_cfg
        self.interval_alphas = list(interval_alphas or [0.10])
        validate_interval_quantiles(self.model_cfg.quantiles, self.interval_alphas)

        self.block_manager = TimeBlockManager(dataset_cfg, block_cfg)

        self.blocks_: Optional[Dict[str, pd.DataFrame]] = None
        self.best_model_cfg_: Optional[QuantileModelConfig] = None
        self.model_: Optional[WeatherQuantileModel] = None
        self.calibrator_: Optional[ConformalCalibrator] = None
        self.predictions_: Dict[str, ArrayLike] = {}
        self.study_: Optional[optuna.study.Study] = None

    def tune(
        self,
        df: pd.DataFrame,
        n_trials: int = 40,
        stability_penalty: float = 0.25,
        study_name: str = "weather_quantile_station",
        storage: Optional[str] = None,
    ) -> optuna.study.Study:
        """
        Run Optuna tuning on development block only.
        """
        self.blocks_ = self.block_manager.split(df)
        dev_df = self.blocks_["dev"]

        objective = WalkForwardCVObjective(
            dev_df=dev_df,
            dataset_cfg=self.dataset_cfg,
            cv_cfg=self.cv_cfg,
            base_model_cfg=self.model_cfg,
            stability_penalty=stability_penalty,
        )

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=storage is not None,
            direction="minimize",
            sampler=TPESampler(seed=self.model_cfg.random_seed),
        )
        study.optimize(objective, n_trials=n_trials)
        self.study_ = study

        best_params = study.best_trial.params
        self.best_model_cfg_ = QuantileModelConfig(
            quantiles=self.model_cfg.quantiles,
            iterations=int(best_params["iterations"]),
            learning_rate=float(best_params["learning_rate"]),
            depth=int(best_params["depth"]),
            l2_leaf_reg=float(best_params["l2_leaf_reg"]),
            min_data_in_leaf=int(best_params["min_data_in_leaf"]),
            random_strength=float(best_params["random_strength"]),
            bagging_temperature=float(best_params["bagging_temperature"]),
            border_count=int(best_params["border_count"]),
            loss_function=self.model_cfg.loss_function,
            eval_metric=self.model_cfg.eval_metric,
            random_seed=self.model_cfg.random_seed,
            verbose=200,
            early_stopping_rounds=self.model_cfg.early_stopping_rounds,
            task_type=self.model_cfg.task_type,
        )

        return study

    def fit_final(self, df: pd.DataFrame) -> None:
        """
        Fit final model on full development block, using the best Optuna config
        if available, otherwise the base config.
        """
        if self.blocks_ is None:
            self.blocks_ = self.block_manager.split(df)

        dev_df = self.blocks_["dev"]
        cal_df = self.blocks_["cal"]
        test_df = self.blocks_["test"]

        final_cfg = self.best_model_cfg_ or self.model_cfg

        model = WeatherQuantileModel(self.dataset_cfg, final_cfg)

        train_df, eval_df = split_final_train_eval(
            dev_df=dev_df,
            dataset_cfg=self.dataset_cfg,
            eval_days=self.cv_cfg.val_days,
        )
        model.fit(train_df, eval_df, trial=None)

        self.model_ = model

        pred_cal = model.predict_quantiles(cal_df, repair_crossing=True)
        pred_test = model.predict_quantiles(test_df, repair_crossing=True)

        self.predictions_["cal"] = pred_cal
        self.predictions_["test"] = pred_test

        self.calibrator_ = ConformalCalibrator(final_cfg.quantiles)

        y_cal = cal_df[self.dataset_cfg.target_col].to_numpy()
        for alpha in self.interval_alphas:
            self.calibrator_.fit_interval(y_cal, pred_cal, alpha)

    def evaluate_block(self, block_name: str) -> Dict[str, float]:
        """
        Evaluate a block after final fit.

        Parameters
        ----------
        block_name : str
            'cal' or 'test'
        """
        if self.blocks_ is None or self.model_ is None:
            raise RuntimeError("Pipeline has not been fit.")
        if block_name not in {"cal", "test"}:
            raise ValueError("block_name must be 'cal' or 'test'.")
        if block_name not in self.predictions_:
            raise RuntimeError(f"No predictions stored for block '{block_name}'.")

        df = self.blocks_[block_name]
        y_true = df[self.dataset_cfg.target_col].to_numpy()
        pred_matrix = self.predictions_[block_name]

        metrics: Dict[str, float] = {}
        metrics["mean_pinball_loss"] = mean_pinball_loss(
            y_true=y_true,
            y_pred_matrix=pred_matrix,
            quantiles=self.model_.model_cfg.quantiles,
        )
        metrics["crossing_rate_after_repair"] = 0.0

        for alpha in self.interval_alphas:
            lower, upper = self.calibrator_.predict_interval(pred_matrix, alpha)
            pct = int(round((1.0 - alpha) * 100))
            metrics[f"coverage_{pct}"] = empirical_interval_coverage(y_true, lower, upper)
            metrics[f"width_{pct}"] = interval_width(lower, upper)

        return metrics

    def predict_distribution_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return repaired quantiles for downstream CDF construction.
        """
        if self.model_ is None:
            raise RuntimeError("Final model has not been fit.")

        return self.model_.predict_quantile_dataframe(
            df=df,
            repair_crossing=True,
            include_target=False,
        )


# =============================================================================
# Convenience helpers for Phase 1 and Phase 2
# =============================================================================

def build_phase1_config() -> QuantileModelConfig:
    """
    Coarse quantile grid for Phase 1.
    """
    return QuantileModelConfig(
        quantiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        min_data_in_leaf=20,
        random_strength=1.0,
        bagging_temperature=0.0,
        border_count=254,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100,
        task_type="GPU",
    )


def build_phase2_config_from_phase1(phase1_cfg: QuantileModelConfig) -> QuantileModelConfig:
    """
    Moderate densification for Phase 2.
    Reuse Phase 1 hyperparameters, but change quantile grid.
    """
    return QuantileModelConfig(
        quantiles=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
        iterations=phase1_cfg.iterations,
        learning_rate=phase1_cfg.learning_rate,
        depth=phase1_cfg.depth,
        l2_leaf_reg=phase1_cfg.l2_leaf_reg,
        min_data_in_leaf=phase1_cfg.min_data_in_leaf,
        random_strength=phase1_cfg.random_strength,
        bagging_temperature=phase1_cfg.bagging_temperature,
        border_count=phase1_cfg.border_count,
        loss_function=phase1_cfg.loss_function,
        eval_metric=phase1_cfg.eval_metric,
        random_seed=phase1_cfg.random_seed,
        verbose=0,
        early_stopping_rounds=phase1_cfg.early_stopping_rounds,
        task_type=phase1_cfg.task_type,
    )


# =============================================================================
# Example usage
# =============================================================================

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


def run_phase(
    df: pd.DataFrame,
    dataset_cfg: DatasetConfig,
    block_cfg: BlockSplitConfig,
    cv_cfg: ExpandingWindowCVConfig,
    model_cfg: QuantileModelConfig,
    interval_alphas: Sequence[float],
    n_trials: int,
    study_name: str,
) -> WeatherQuantilePipeline:
    """
    Run one full phase: tune -> fit final -> evaluate.
    """
    pipeline = WeatherQuantilePipeline(
        dataset_cfg=dataset_cfg,
        block_cfg=block_cfg,
        cv_cfg=cv_cfg,
        model_cfg=model_cfg,
        interval_alphas=interval_alphas,
    )

    study = pipeline.tune(
        df=df,
        n_trials=n_trials,
        stability_penalty=0.25,
        study_name=study_name,
        storage=None,
    )

    pipeline.fit_final(df)

    print(f"\n=== {study_name} ===")
    print("Best Optuna score:", study.best_trial.value)
    print("Best params:", study.best_trial.params)
    print("Calibration metrics:", pipeline.evaluate_block("cal"))
    print("Test metrics:", pipeline.evaluate_block("test"))

    return pipeline


if __name__ == "__main__":
    df = load_kmia_training_data("kalshiTraining_KMIA.dat")
    dataset_cfg = build_kmia_dataset_config()
    block_cfg = build_default_kmia_block_split_config(df)

    # Conservative CV for thin per-station sample.
    cv_cfg = ExpandingWindowCVConfig(
        min_train_days=365,
        val_days=90,
        step_days=90,
        max_folds=5,
    )

    # -----------------------------
    # Phase 1: coarse quantiles
    # -----------------------------
    phase1_cfg = build_phase1_config()

    # This entrypoint is a real tuning run (not a smoke test).
    phase1_pipeline = run_phase(
        df=df,
        dataset_cfg=dataset_cfg,
        block_cfg=block_cfg,
        cv_cfg=cv_cfg,
        model_cfg=phase1_cfg,
        interval_alphas=[0.10],   # 90% interval
        n_trials=25,
        study_name="phase1_coarse_quantiles",
    )

    # -----------------------------
    # Phase 2: denser quantiles
    # Option A = retrain from scratch
    # -----------------------------
    phase1_best_cfg = phase1_pipeline.best_model_cfg_ or phase1_cfg
    phase2_cfg = build_phase2_config_from_phase1(phase1_best_cfg)

    phase2_pipeline = run_phase(
        df=df,
        dataset_cfg=dataset_cfg,
        block_cfg=block_cfg,
        cv_cfg=cv_cfg,
        model_cfg=phase2_cfg,
        interval_alphas=[0.10],
        n_trials=12,  # smaller retune is usually enough for phase 2
        study_name="phase2_denser_quantiles",
    )

    # Quantiles available for later CDF work
    test_quantiles = phase2_pipeline.predict_distribution_inputs(
        phase2_pipeline.blocks_["test"]
    )
    print("\nTest quantile head:")
    print(test_quantiles.head())

"""Walk-forward training for XGBoost and LightGBM ML strategies.

Trains on each walk-forward window's train period, predicts on test period.
Uses modest hyperparameters to reduce overfit risk. Models are persisted
via joblib for production inference.
"""
from __future__ import annotations

import logging
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.windows import Window
from .features import build_training_set

logger = logging.getLogger(__name__)

# Default model directory relative to repo root
_MODELS_DIR = Path(__file__).resolve().parents[4] / "models"

# Modest hyperparameters — intentionally conservative to resist overfit
_XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    objective="reg:squarederror",
    random_state=42,
    verbosity=0,
)

_LGB_PARAMS = dict(
    n_estimators=200,
    num_leaves=15,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    objective="regression",
    random_state=42,
    verbose=-1,
)

_CATBOOST_PARAMS = dict(
    iterations=200,
    depth=4,
    learning_rate=0.05,
    rsm=0.7,  # column subsample
    loss_function="RMSE",
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
)

# DoubleEnsemble: K LightGBM submodels, sample weights updated each round to
# focus on hard-but-consistent examples (per Zhang et al., AAAI 2021).
_DENSEMBLE_NUM_MODELS = 3
_DENSEMBLE_SHRINK = 0.5  # downweight low-loss samples (consistency)
_DENSEMBLE_DIVERSITY = 0.3  # upweight high-variance samples (diversity)

# Ridge regression for the linear bot — qlib's LinearModel uses OLS with
# optional L2; sklearn.linear_model.Ridge matches that interface exactly.
_RIDGE_ALPHA = 1.0


def _sharpe(returns: list[float]) -> float:
    """Annualised Sharpe ratio from daily return series."""
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(252)


def _top_decile_returns(
    predictions: np.ndarray,
    realized_returns: np.ndarray,
    n_bins: int = 10,
) -> list[float]:
    """Simulate long-only top-decile daily returns from ranked predictions.

    Returns a list of daily returns — one per test sample where we hold the
    top-decile predicted symbols.
    """
    if len(predictions) == 0:
        return []
    top_n = max(1, len(predictions) // n_bins)
    ranked_idx = np.argsort(predictions)[::-1][:top_n]
    return [float(realized_returns[i]) for i in ranked_idx]


def _build_fold_oos_returns(
    model,
    histories: dict,
    target_symbols: list[str],
    window: Window,
    horizon: int,
) -> list[float]:
    """Compute OOS daily-level returns by running the model over the test window.

    Important: realized returns must be RAW (not rank-transformed) so the
    downstream Sharpe calculation reflects actual P&L from holding top-decile
    predicted symbols. The model can be trained on rank labels but evaluated
    against raw forward returns — that's the whole point.
    """
    X_test, y_test = build_training_set(
        histories=histories,
        target_symbols=target_symbols,
        train_start=window.test_start,
        train_end=window.test_end,
        horizon=horizon,
        sample_every_days=5,
        use_rank_labels=False,  # raw returns for honest Sharpe calc
    )
    if X_test.empty or y_test.empty:
        return []

    X_clean = X_test.fillna(0.0)
    preds = model.predict(X_clean.values)
    realized = y_test.values
    return _top_decile_returns(preds, realized)


def _fit_xgboost(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Fit an XGBoost regressor."""
    from xgboost import XGBRegressor

    params = {**_XGB_PARAMS, "random_state": seed}
    model = XGBRegressor(**params)
    model.fit(X, y)
    return model


def _fit_lightgbm(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Fit a LightGBM regressor."""
    import lightgbm as lgb

    params = {**_LGB_PARAMS, "random_state": seed}
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    return model


def _fit_catboost(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Fit a CatBoost regressor — gradient boosting with ordered boosting
    that resists target leakage better than XGB/LGBM."""
    from catboost import CatBoostRegressor

    params = {**_CATBOOST_PARAMS, "random_seed": seed}
    model = CatBoostRegressor(**params)
    model.fit(X, y)
    return model


class _DoubleEnsemble:
    """K-submodel boosted ensemble with sample-weight feedback.

    Each round trains a LightGBM on weighted samples, computes per-sample
    residuals, and updates weights for the next round: low-residual
    (consistent) samples are downweighted, high-residual (informative)
    samples are upweighted. Final prediction averages all submodels.

    Simpler than qlib's full DoubleEnsemble (no feature selection round),
    but captures the core sample-reweighting mechanism.
    """

    def __init__(self, num_models: int = _DENSEMBLE_NUM_MODELS, seed: int = 42) -> None:
        self.num_models = num_models
        self.seed = seed
        self.submodels: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_DoubleEnsemble":
        import lightgbm as lgb

        n = len(y)
        weights = np.ones(n, dtype=float)
        for k in range(self.num_models):
            params = {**_LGB_PARAMS, "random_state": self.seed + k}
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y, sample_weight=weights)
            self.submodels.append(model)
            if k + 1 < self.num_models:
                preds = model.predict(X)
                residuals = np.abs(y - preds)
                # Normalize so weight updates are scale-free.
                if residuals.std() > 0:
                    z = (residuals - residuals.mean()) / residuals.std()
                else:
                    z = np.zeros_like(residuals)
                # Downweight easy samples, upweight hard ones — clamp to [0.1, 5].
                weights = np.clip(
                    1.0 + _DENSEMBLE_DIVERSITY * z - _DENSEMBLE_SHRINK * np.exp(-z),
                    0.1,
                    5.0,
                )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.submodels:
            return np.zeros(len(X))
        preds = np.stack([m.predict(X) for m in self.submodels], axis=0)
        return preds.mean(axis=0)


def _fit_double_ensemble(X: np.ndarray, y: np.ndarray, seed: int = 42) -> "_DoubleEnsemble":
    """Fit a DoubleEnsemble (3 LightGBM submodels with sample reweighting)."""
    return _DoubleEnsemble(num_models=_DENSEMBLE_NUM_MODELS, seed=seed).fit(X, y)


def _fit_ridge(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Fit a ridge regression — linear model with L2 regularization.

    Used as a deliberately simple baseline against the gradient-boosting
    and neural-net bots: if a tree model can't beat ridge, the signal
    is too thin to extract with extra capacity.
    """
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=_RIDGE_ALPHA, random_state=seed)
    model.fit(X, y)
    return model


def _walkforward_core(
    fit_fn,
    model_name: str,
    histories: dict,
    target_symbols: list[str],
    windows: list[Window],
    horizon: int = 5,
    seed: int = 42,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Shared walk-forward training loop used by both XGBoost and LightGBM."""
    fold_results = []
    fold_sharpes = []

    for window in windows:
        X_train, y_train = build_training_set(
            histories=histories,
            target_symbols=target_symbols,
            train_start=window.train_start,
            train_end=window.train_end,
            horizon=horizon,
            sample_every_days=5,
        )
        if X_train.empty or len(y_train) < 10:
            logger.warning("[%s] Insufficient training data for window %s, skipping.", model_name, window.label)
            continue

        X_clean = X_train.fillna(0.0).values
        y_arr = y_train.values

        model = fit_fn(X_clean, y_arr, seed=seed)

        # OOS evaluation on test period
        oos_returns = _build_fold_oos_returns(model, histories, target_symbols, window, horizon)
        fold_sharpe = _sharpe(oos_returns)
        fold_sharpes.append(fold_sharpe)

        fold_results.append(
            {
                "window": window.label,
                "train_start": window.train_start.isoformat(),
                "train_end": window.train_end.isoformat(),
                "test_end": window.test_end.isoformat(),
                "n_train_samples": len(y_train),
                "fold_sharpe": fold_sharpe,
                "model": model,
            }
        )
        logger.info("[%s] Window %s: n_train=%d fold_sharpe=%.3f", model_name, window.label, len(y_train), fold_sharpe)

    # Train final model on ALL data across all windows
    X_all_parts: list[pd.DataFrame] = []
    y_all_parts: list[pd.Series] = []
    if windows:
        X_full, y_full = build_training_set(
            histories=histories,
            target_symbols=target_symbols,
            train_start=windows[0].train_start,
            train_end=windows[-1].train_end,
            horizon=horizon,
            sample_every_days=5,
        )
        if not X_full.empty:
            X_all_parts.append(X_full)
            y_all_parts.append(y_full)

    final_model = None
    if X_all_parts:
        X_final = pd.concat(X_all_parts).fillna(0.0).values
        y_final = pd.concat(y_all_parts).values
        if len(y_final) >= 10:
            final_model = fit_fn(X_final, y_final, seed=seed)

    # Persist final model
    persisted_path: str | None = None
    if final_model is not None:
        import joblib

        mdir = models_dir or _MODELS_DIR
        mdir.mkdir(parents=True, exist_ok=True)
        today_str = date.today().isoformat()
        model_path = mdir / f"{model_name}-{today_str}.joblib"
        joblib.dump(final_model, model_path)
        persisted_path = str(model_path)
        logger.info("[%s] Persisted final model to %s", model_name, model_path)

    return {
        "model_name": model_name,
        "fold_results": fold_results,
        "fold_sharpes": fold_sharpes,
        "final_model": final_model,
        "persisted_path": persisted_path,
        "median_fold_sharpe": float(np.median(fold_sharpes)) if fold_sharpes else 0.0,
    }


def train_xgboost_walkforward(
    histories: dict,
    target_symbols: list[str],
    windows: list[Window],
    horizon: int = 5,
    seed: int = 42,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Train XGBoost on each window's train period, evaluate on test.

    Returns dict with per-window OOS predictions, fold Sharpes, final model.
    Persists final model to models/gradboost-<YYYY-MM-DD>.joblib.
    """
    return _walkforward_core(
        fit_fn=_fit_xgboost,
        model_name="gradboost",
        histories=histories,
        target_symbols=target_symbols,
        windows=windows,
        horizon=horizon,
        seed=seed,
        models_dir=models_dir,
    )


def train_lightgbm_walkforward(
    histories: dict,
    target_symbols: list[str],
    windows: list[Window],
    horizon: int = 5,
    seed: int = 42,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Train LightGBM on each window's train period, evaluate on test.

    Returns dict with per-window OOS predictions, fold Sharpes, final model.
    Persists final model to models/lightforest-<YYYY-MM-DD>.joblib.
    """
    return _walkforward_core(
        fit_fn=_fit_lightgbm,
        model_name="lightforest",
        histories=histories,
        target_symbols=target_symbols,
        windows=windows,
        horizon=horizon,
        seed=seed,
        models_dir=models_dir,
    )


def train_catboost_walkforward(
    histories: dict,
    target_symbols: list[str],
    windows: list[Window],
    horizon: int = 5,
    seed: int = 42,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Train CatBoost on each window's train period. Persists to
    models/catboost-<YYYY-MM-DD>.joblib."""
    return _walkforward_core(
        fit_fn=_fit_catboost,
        model_name="catboost",
        histories=histories,
        target_symbols=target_symbols,
        windows=windows,
        horizon=horizon,
        seed=seed,
        models_dir=models_dir,
    )


def train_double_ensemble_walkforward(
    histories: dict,
    target_symbols: list[str],
    windows: list[Window],
    horizon: int = 5,
    seed: int = 42,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Train DoubleEnsemble (K LightGBM submodels with sample reweighting).
    Persists to models/double-ensemble-<YYYY-MM-DD>.joblib."""
    return _walkforward_core(
        fit_fn=_fit_double_ensemble,
        model_name="double-ensemble",
        histories=histories,
        target_symbols=target_symbols,
        windows=windows,
        horizon=horizon,
        seed=seed,
        models_dir=models_dir,
    )


def train_ridge_walkforward(
    histories: dict,
    target_symbols: list[str],
    windows: list[Window],
    horizon: int = 5,
    seed: int = 42,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Train ridge regression. Persists to models/qlib-linear-<YYYY-MM-DD>.joblib.

    Named 'qlib-linear' to mark its lineage from qlib's LinearModel even though
    we use sklearn directly to avoid the qlib framework dependency.
    """
    return _walkforward_core(
        fit_fn=_fit_ridge,
        model_name="qlib-linear",
        histories=histories,
        target_symbols=target_symbols,
        windows=windows,
        horizon=horizon,
        seed=seed,
        models_dir=models_dir,
    )

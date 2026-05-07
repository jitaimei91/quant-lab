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
from .features import build_training_set, compute_features

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
    """Compute OOS daily-level returns by running the model over the test window."""
    X_test, y_test = build_training_set(
        histories=histories,
        target_symbols=target_symbols,
        train_start=window.test_start,
        train_end=window.test_end,
        horizon=horizon,
        sample_every_days=5,
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

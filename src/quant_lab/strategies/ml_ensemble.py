"""MLEnsemble: equal-weight ensemble of GradBoost + LightForest predictions.

Always registers. Holds 100% SPY when neither component model is available
or when feature computation fails (see gradboost.py for rationale).
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np

from ..types import Bar
from .base import Strategy, register
from ..ml.features import compute_features
from .gradboost import (
    _FALLBACK_WEIGHTS,
    _INDEX_PROXIES,
    _VALIDATION_FILE,
    _gates_passed,
    _load_latest_model,
    _record_failure,
)

# Back-compat alias for tests.
_load_validation_state = _gates_passed

logger = logging.getLogger(__name__)

_COMPONENT_BOT_IDS = ("gradboost", "lightforest")


class MLEnsemble(Strategy):
    """Equal-weight ensemble of XGBoost + LightGBM → top-decile = buy.

    Falls back to 100% SPY whenever neither component is trustworthy.
    """

    bot_id = "ml-ensemble"
    description = "Equal-weight ensemble of XGBoost + LightGBM (SPY fallback)"
    cash_buffer: float = 0.05

    def __init__(self) -> None:
        self._gates_passed = all(_gates_passed(bid) for bid in _COMPONENT_BOT_IDS)
        if self._gates_passed:
            self._xgb_model = _load_latest_model("gradboost")
            self._lgb_model = _load_latest_model("lightforest")
        else:
            self._xgb_model = None
            self._lgb_model = None
            _record_failure(self.bot_id)

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        models = [m for m in [self._xgb_model, self._lgb_model] if m is not None]
        if not self._gates_passed or not models:
            return dict(_FALLBACK_WEIGHTS)

        symbols = [s for s in histories if s not in _INDEX_PROXIES]
        if not symbols:
            return dict(_FALLBACK_WEIGHTS)

        X, _ = compute_features(histories, symbols, as_of)
        if X is None or X.empty:
            return dict(_FALLBACK_WEIGHTS)

        X_clean = X.fillna(0.0).values
        all_preds = np.stack([m.predict(X_clean) for m in models], axis=0)
        scores = np.mean(all_preds, axis=0)

        top_n = max(1, len(scores) // 10)
        ranked_idx = np.argsort(scores)[::-1][:top_n]
        selected = [X.index[i] for i in ranked_idx]
        if not selected:
            return dict(_FALLBACK_WEIGHTS)

        weight = (1.0 - self.cash_buffer) / len(selected)
        return {sym: weight for sym in selected}


register(MLEnsemble)

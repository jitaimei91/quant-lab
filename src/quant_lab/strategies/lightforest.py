"""LightForest strategy: LightGBM ranking on ~30 technical features.

Always registers. When the ML pipeline can't produce a signal, the bot
holds 100% SPY rather than cash (see gradboost.py for rationale).
"""
from __future__ import annotations

import logging
from datetime import date

from ..types import Bar
from .base import Strategy, register
from ..ml.features import compute_features
from .gradboost import (
    _FALLBACK_WEIGHTS,
    _INDEX_PROXIES,
    _gates_passed,
    _load_latest_model,
    _record_failure,
)

# Back-compat alias for tests that imported the prior name on this module.
_load_validation_state = _gates_passed

logger = logging.getLogger(__name__)


class LightForest(Strategy):
    """LightGBM ranking ~30 technical features → top-decile = buy.

    Falls back to 100% SPY whenever the signal isn't trustworthy.
    """

    bot_id = "lightforest"
    description = "LightGBM ranking ~30 technical features → top-decile = buy (SPY fallback)"
    cash_buffer: float = 0.05

    def __init__(self) -> None:
        self._gates_passed = _gates_passed("lightforest")
        self._model = _load_latest_model("lightforest") if self._gates_passed else None
        if not self._gates_passed:
            _record_failure("lightforest")

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        if not self._gates_passed or self._model is None:
            return dict(_FALLBACK_WEIGHTS)

        symbols = [s for s in histories if s not in _INDEX_PROXIES]
        if not symbols:
            return dict(_FALLBACK_WEIGHTS)

        X, _ = compute_features(histories, symbols, as_of)
        if X is None or X.empty:
            return dict(_FALLBACK_WEIGHTS)

        scores = self._model.predict(X.fillna(0.0).values)
        import numpy as np

        top_n = max(1, len(scores) // 10)
        ranked_idx = np.argsort(scores)[::-1][:top_n]
        selected = [X.index[i] for i in ranked_idx]
        if not selected:
            return dict(_FALLBACK_WEIGHTS)

        weight = (1.0 - self.cash_buffer) / len(selected)
        return {sym: weight for sym in selected}


register(LightForest)

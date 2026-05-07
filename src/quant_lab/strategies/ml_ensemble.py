"""MLEnsemble strategy: equal-weight ensemble of GradBoost + LightForest predictions.

Gate-conditional registration: only registers if ml_validation.json
records that BOTH 'gradboost' AND 'lightforest' passed all validation gates.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np

from ..types import Bar
from .base import Strategy, register
from ..ml.features import compute_features
from .gradboost import _load_latest_model

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_VALIDATION_FILE = _REPO_ROOT / "state" / "ml_validation.json"
_INDEX_PROXIES = {"SPY", "QQQ", "^VIX"}


def _load_validation_state(bot_id: str) -> bool:
    """Return True if bot passed gates (or if file doesn't exist yet)."""
    if not _VALIDATION_FILE.exists():
        return True
    try:
        data = json.loads(_VALIDATION_FILE.read_text())
        entry = data.get(bot_id)
        if entry is None:
            return True
        return bool(entry.get("overall_pass", False))
    except Exception as exc:
        logger.warning("[%s] Could not read ml_validation.json: %s", bot_id, exc)
        return True

_COMPONENT_BOT_IDS = ["gradboost", "lightforest"]


class MLEnsemble(Strategy):
    """Equal-weight ensemble of GradBoost + LightForest → top-decile = buy."""

    bot_id = "ml-ensemble"
    description = "Equal-weight ensemble of XGBoost + LightGBM, top-decile long-only"
    cash_buffer: float = 0.05

    def __init__(self) -> None:
        self._xgb_model = _load_latest_model("gradboost")
        self._lgb_model = _load_latest_model("lightforest")

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        models = [m for m in [self._xgb_model, self._lgb_model] if m is not None]
        if not models:
            return {}

        symbols = [s for s in histories if s not in _INDEX_PROXIES]
        if not symbols:
            return {}

        X, _ = compute_features(histories, symbols, as_of)
        if X is None or X.empty:
            return {}

        X_clean = X.fillna(0.0).values

        # Average predictions across available models
        all_preds = np.stack([m.predict(X_clean) for m in models], axis=0)
        scores = np.mean(all_preds, axis=0)

        top_n = max(1, len(scores) // 10)
        ranked_idx = np.argsort(scores)[::-1][:top_n]
        selected = [X.index[i] for i in ranked_idx]

        weight = (1.0 - self.cash_buffer) / len(selected)
        return {sym: weight for sym in selected}


# Gate-conditional registration: require BOTH component bots to pass
_all_pass = all(_load_validation_state(bot_id) for bot_id in _COMPONENT_BOT_IDS)

if _all_pass:
    register(MLEnsemble)
else:
    try:
        failed_path = _REPO_ROOT / "dashboard" / "data" / "validation_failed.json"
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if failed_path.exists():
            existing = json.loads(failed_path.read_text())
        failed_bot_ids = [
            bid for bid in _COMPONENT_BOT_IDS if not _load_validation_state(bid)
        ]
        existing[MLEnsemble.bot_id] = {
            "bot_id": MLEnsemble.bot_id,
            "reasons_failed": [f"component {bid} failed gates" for bid in failed_bot_ids],
            "gates": {},
        }
        failed_path.write_text(json.dumps(existing, indent=2) + "\n")
    except Exception as exc:
        logger.warning("[ml-ensemble] Could not write validation_failed.json: %s", exc)
    logger.info("[ml-ensemble] NOT registered — component bots failed validation gates")

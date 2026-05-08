"""GradBoost strategy: XGBoost ranking on ~30 technical features.

Always registers. When the ML pipeline can't produce a signal (gates failed,
no model file, no features), the bot benchmarks to 100% SPY rather than
sitting in cash — cash drag is structural (~10%/yr in a bull) and worse
than the honest "we have no edge, hold the index" admission.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from ..types import Bar
from .base import Strategy, register
from ..ml.features import compute_features

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_STATE_DIR = _REPO_ROOT / "state"
_VALIDATION_FILE = _STATE_DIR / "ml_validation.json"
_MODELS_DIR = _REPO_ROOT / "models"
_INDEX_PROXIES = {"SPY", "QQQ", "^VIX", "SSO", "TMF", "UGL", "SVXY", "SHY"}  # also exclude apex-only sleeves

# When the ML signal isn't trustworthy we hold SPY 100%, NOT cash.
_FALLBACK_WEIGHTS: dict[str, float] = {"SPY": 1.0}


def _gates_passed(bot_id: str) -> bool:
    """True if the bot's gates passed in ml_validation.json. Defaults True
    in dev mode (file absent or entry missing) so the bot trades its model."""
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


# Back-compat alias for tests that imported the prior name.
_load_validation_state = _gates_passed


def _record_failure(bot_id: str) -> None:
    """Persist failure detail for the dashboard so users see WHY a bot is on fallback."""
    try:
        failed_path = _REPO_ROOT / "dashboard" / "data" / "validation_failed.json"
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if failed_path.exists():
            existing = json.loads(failed_path.read_text())
        if _VALIDATION_FILE.exists():
            data = json.loads(_VALIDATION_FILE.read_text())
            entry = data.get(bot_id, {})
            existing[bot_id] = {
                "bot_id": bot_id,
                "reasons_failed": entry.get("reasons_failed", []),
                "gates": entry.get("gates", {}),
                "fallback": "SPY 100%",
            }
        failed_path.write_text(json.dumps(existing, indent=2) + "\n")
    except Exception as exc:
        logger.warning("[%s] Could not write validation_failed.json: %s", bot_id, exc)


def _load_latest_model(prefix: str):
    """Load the most-recently-persisted model file matching models/<prefix>-*.joblib."""
    try:
        import joblib

        candidates = sorted(_MODELS_DIR.glob(f"{prefix}-*.joblib"), reverse=True)
        if not candidates:
            return None
        return joblib.load(candidates[0])
    except Exception as exc:
        logger.warning("[%s] Could not load model: %s", prefix, exc)
        return None


class GradBoost(Strategy):
    """XGBoost ranking ~30 technical features → top-decile = buy.

    Falls back to 100% SPY whenever the signal isn't trustworthy.
    """

    bot_id = "gradboost"
    description = "XGBoost ranking ~30 technical features → top-decile = buy (SPY fallback)"
    cash_buffer: float = 0.05

    def __init__(self) -> None:
        self._gates_passed = _gates_passed("gradboost")
        self._model = _load_latest_model("gradboost") if self._gates_passed else None
        if not self._gates_passed:
            _record_failure("gradboost")

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

        top_n = max(1, len(scores) // 10)  # top decile
        ranked_idx = np.argsort(scores)[::-1][:top_n]
        selected = [X.index[i] for i in ranked_idx]
        if not selected:
            return dict(_FALLBACK_WEIGHTS)

        weight = (1.0 - self.cash_buffer) / len(selected)
        return {sym: weight for sym in selected}


register(GradBoost)

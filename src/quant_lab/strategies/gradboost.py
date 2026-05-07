"""GradBoost strategy: XGBoost ranking on ~30 technical features.

Gate-conditional registration: only registers if ml_validation.json
records that the 'gradboost' bot passed all validation gates.
If the JSON is absent, registers anyway (dev/test mode) but target_weights
returns {} when no model file is found.
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
_INDEX_PROXIES = {"SPY", "QQQ", "^VIX"}


def _load_validation_state(bot_id: str) -> bool:
    """Return True if bot passed gates (or if file doesn't exist yet)."""
    if not _VALIDATION_FILE.exists():
        return True  # dev mode: register all
    try:
        data = json.loads(_VALIDATION_FILE.read_text())
        entry = data.get(bot_id)
        if entry is None:
            return True  # not yet evaluated → register (dev mode)
        return bool(entry.get("overall_pass", False))
    except Exception as exc:
        logger.warning("[%s] Could not read ml_validation.json: %s", bot_id, exc)
        return True  # fail-open for dev


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
    """XGBoost ranking ~30 technical features → top-decile = buy."""

    bot_id = "gradboost"
    description = "XGBoost ranking ~30 technical features → top-decile = buy"
    cash_buffer: float = 0.05

    def __init__(self) -> None:
        self._model = _load_latest_model("gradboost")

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        if self._model is None:
            return {}

        symbols = [s for s in histories if s not in _INDEX_PROXIES]
        if not symbols:
            return {}

        X, _ = compute_features(histories, symbols, as_of)
        if X is None or X.empty:
            return {}

        scores = self._model.predict(X.fillna(0.0).values)
        import numpy as np

        top_n = max(1, len(scores) // 10)  # top decile
        ranked_idx = np.argsort(scores)[::-1][:top_n]
        selected = [X.index[i] for i in ranked_idx]

        weight = (1.0 - self.cash_buffer) / len(selected)
        return {sym: weight for sym in selected}


# Gate-conditional registration
if _load_validation_state(GradBoost.bot_id):
    register(GradBoost)
else:
    # Write failure notice to dashboard/data/validation_failed.json
    try:
        failed_path = _REPO_ROOT / "dashboard" / "data" / "validation_failed.json"
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if failed_path.exists():
            existing = json.loads(failed_path.read_text())
        if _VALIDATION_FILE.exists():
            data = json.loads(_VALIDATION_FILE.read_text())
            entry = data.get(GradBoost.bot_id, {})
            existing[GradBoost.bot_id] = {
                "bot_id": GradBoost.bot_id,
                "reasons_failed": entry.get("reasons_failed", []),
                "gates": entry.get("gates", {}),
            }
        failed_path.write_text(json.dumps(existing, indent=2) + "\n")
    except Exception as exc:
        logger.warning("[gradboost] Could not write validation_failed.json: %s", exc)
    logger.info("[gradboost] NOT registered — failed validation gates")

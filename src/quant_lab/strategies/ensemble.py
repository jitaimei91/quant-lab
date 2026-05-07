"""MetaEnsemble: calibrated-evidence-weighted blend of all other strategies.

Loads strategy weights from live_weights.json (if present) or falls back to
backtest_results.json. Calls each component strategy's target_weights() directly,
then blends by ensemble weight. Excludes itself to avoid recursion.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ..types import Bar
from ..ensemble.weights import compute_strategy_weights
from .base import Strategy, register, get_all

_REPO_ROOT = Path(__file__).resolve().parents[3]

_BACKTEST_RESULTS = (
    _REPO_ROOT / "dashboard" / "data" / "backtest" / "backtest_results.json"
)
_LIVE_WEIGHTS = (
    _REPO_ROOT / "dashboard" / "data" / "backtest" / "live_weights.json"
)

_PER_TICKER_CAP = 0.10
_TOTAL_CAP = 0.95


def _load_weights_from_file(path: Path) -> dict[str, float]:
    """Parse live_weights.json (dict format) or backtest_results.json (strategies list)."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    # live_weights.json is {bot_id: weight, ...}
    if isinstance(data, dict) and "strategies" not in data:
        return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
    # backtest_results.json has {"strategies": [...]}
    strategies = data.get("strategies", [])
    if not strategies:
        return {}
    return compute_strategy_weights(strategies)


@register
class MetaEnsemble(Strategy):
    """Calibrated-evidence-weighted blend of all other strategies."""

    bot_id = "meta-ensemble"
    description = "Calibrated-evidence-weighted blend of all other strategies"

    def __init__(
        self,
        weights_path: Path | None = None,
        *,
        weights_override: dict[str, float] | None = None,
    ) -> None:
        """Load calibration weights.

        Priority order:
        1. weights_override (for testing)
        2. live_weights.json (if present and non-empty)
        3. backtest_results.json
        4. Equal weight across all other registered strategies

        Args:
            weights_path: Override path for backtest results (testing/custom paths).
            weights_override: Direct weight dict, bypasses file loading entirely.
        """
        if weights_override is not None:
            self._weights: dict[str, float] = weights_override
            return

        # Try live weights first
        live_w = _load_weights_from_file(_LIVE_WEIGHTS)
        if live_w:
            self._weights = live_w
            return

        # Fall back to backtest results
        bt_path = weights_path or _BACKTEST_RESULTS
        bt_w = _load_weights_from_file(bt_path)
        if bt_w:
            self._weights = bt_w
            return

        # No calibration data: use equal weight (populated lazily in target_weights)
        self._weights = {}

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        """Blend component strategies by ensemble weight.

        For each component strategy with weight > 0:
          1. Call its target_weights(histories, as_of)
          2. Multiply each ticker weight by the ensemble weight for that strategy
        Aggregate by ticker, then apply per-ticker cap (0.10) and total cap (0.95).
        """
        all_strategies = [s for s in get_all() if s.bot_id != "meta-ensemble"]

        # Determine effective weights
        if self._weights:
            ens_weights = self._weights
        else:
            # Equal weight fallback
            n = len(all_strategies)
            if n == 0:
                return {}
            ens_weights = {s.bot_id: 1.0 / n for s in all_strategies}

        # Aggregate weighted ticker exposures
        ticker_weights: dict[str, float] = {}
        for strat in all_strategies:
            ew = ens_weights.get(strat.bot_id, 0.0)
            if ew <= 0.0:
                continue
            try:
                component_weights = strat.target_weights(histories, as_of)
            except Exception:
                continue
            for ticker, w in component_weights.items():
                ticker_weights[ticker] = ticker_weights.get(ticker, 0.0) + ew * w

        if not ticker_weights:
            return {}

        # Apply per-ticker cap
        ticker_weights = {
            t: min(w, _PER_TICKER_CAP) for t, w in ticker_weights.items()
        }

        # Apply total cap
        total = sum(ticker_weights.values())
        if total > _TOTAL_CAP:
            scale = _TOTAL_CAP / total
            ticker_weights = {t: w * scale for t, w in ticker_weights.items()}

        return ticker_weights

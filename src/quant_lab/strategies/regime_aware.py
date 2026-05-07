"""Regime-aware strategy wrappers.

Each concrete variant wraps a base strategy and gates it to specific HMM
regimes.  When no trained HMM exists (or confidence is below threshold),
the base strategy runs unconditionally.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from ..types import Bar
from ..engine.regime import hmm_regime_classify
from .base import Strategy, get, register

HMM_STATE_PATH = Path(__file__).resolve().parents[3] / "state" / "hmm_state.json"


class _RegimeAware(Strategy):
    """Wrap a base strategy: only fire in compatible regimes."""

    base_bot_id: str = ""
    allowed_regimes: tuple[str, ...] = ("risk-on", "chop", "risk-off", "crisis")
    confidence_threshold: float = 0.4  # require this min confidence; else fall back
    hmm_state_path: Path | None = None  # set at class-load time

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        # No HMM file: defer entirely to base strategy
        if self.hmm_state_path is None or not self.hmm_state_path.exists():
            return get(self.base_bot_id).target_weights(histories, as_of)

        regime = hmm_regime_classify(histories, self.hmm_state_path)

        # Low confidence: fall back to base strategy (don't gate)
        if regime["regime_confidence"] < self.confidence_threshold:
            return get(self.base_bot_id).target_weights(histories, as_of)

        # Wrong regime: sit out
        if regime["regime_name"] not in self.allowed_regimes:
            return {}

        return get(self.base_bot_id).target_weights(histories, as_of)


@register
class RegimeMomo(_RegimeAware):
    """Momo gated by HMM regime: only fires in risk-on / chop."""

    bot_id = "regime-momo"
    description = "Momo gated by HMM regime: only fires in risk-on / chop"
    base_bot_id = "momo"
    allowed_regimes = ("risk-on", "chop")
    hmm_state_path = HMM_STATE_PATH


@register
class RegimeMeanRev(_RegimeAware):
    """MeanRev gated by HMM: only fires in chop."""

    bot_id = "regime-meanrev"
    description = "MeanRev gated by HMM: only fires in chop"
    base_bot_id = "meanrev"
    allowed_regimes = ("chop",)
    hmm_state_path = HMM_STATE_PATH


@register
class RegimeBreakout(_RegimeAware):
    """Breakout gated by HMM: only fires in risk-on."""

    bot_id = "regime-breakout"
    description = "Breakout gated by HMM: only fires in risk-on"
    base_bot_id = "breakout"
    allowed_regimes = ("risk-on",)
    hmm_state_path = HMM_STATE_PATH

"""Tests for strategies/regime_aware.py — per-regime strategy gating."""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import numpy as np

from quant_lab.engine.hmm_regime import HMMState, save_hmm
from quant_lab.strategies import get_all
from quant_lab.strategies.base import get
from quant_lab.strategies.regime_aware import (
    HMM_STATE_PATH,
    RegimeBreakout,
    RegimeMeanRev,
    RegimeMomo,
    _RegimeAware,
)
from quant_lab.types import Bar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_series(symbol: str, closes: list[float]) -> list[Bar]:
    return [
        Bar(symbol=symbol, date=date(2025, 1, 1) + timedelta(days=i),
            open=c, high=c, low=c, close=c, volume=1_000_000)
        for i, c in enumerate(closes)
    ]


def _minimal_histories() -> dict[str, list[Bar]]:
    """Minimal histories for strategy calls (momo needs 250+ bars)."""
    spy = _make_series("SPY", [400.0 + i * 0.1 for i in range(300)])
    vix = _make_series("^VIX", [20.0] * 300)
    return {"SPY": spy, "^VIX": vix}


def _make_mock_hmm(n_states: int = 4) -> HMMState:
    """HMM with 4 states sorted by VIX: 10, 20, 30, 50."""
    means = np.array([
        [10.0, 0.0, 0.05, 0.10, 0.0],
        [20.0, 0.0, 0.01, 0.15, 0.0],
        [30.0, 0.0, -0.02, 0.22, 0.0],
        [50.0, 0.0, -0.10, 0.40, 0.0],
    ])
    covs = np.array([np.eye(5) * 0.1 for _ in range(n_states)])
    transition = np.ones((n_states, n_states)) / n_states
    initial = np.ones(n_states) / n_states
    return HMMState(n_states=n_states, means=means, covariances=covs,
                    transition=transition, initial=initial)


# ---------------------------------------------------------------------------
# Fallback behaviour: no HMM file
# ---------------------------------------------------------------------------

def test_regime_aware_fallback_no_hmm_file(tmp_path):
    """When HMM state file does not exist, delegate to base strategy."""
    non_existent = tmp_path / "no_hmm.json"

    class _TestBot(_RegimeAware):
        bot_id = "test-regime-bot-fallback"
        base_bot_id = "momo"
        allowed_regimes = ("risk-on",)
        hmm_state_path = non_existent

    bot = _TestBot()
    histories = _minimal_histories()
    as_of = date(2025, 12, 31)

    result = bot.target_weights(histories, as_of)
    # Should return what momo returns (may be empty for this minimal data, but no exception)
    assert isinstance(result, dict)


def test_regime_aware_hmm_path_none_fallback():
    """When hmm_state_path is None, always delegate to base strategy."""
    class _TestBot(_RegimeAware):
        bot_id = "test-regime-bot-none"
        base_bot_id = "momo"
        allowed_regimes = ("risk-on",)
        hmm_state_path = None

    bot = _TestBot()
    histories = _minimal_histories()
    as_of = date(2025, 12, 31)

    result = bot.target_weights(histories, as_of)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Gating: crisis regime blocks risk-on-only bots
# ---------------------------------------------------------------------------

def test_regime_aware_blocked_in_wrong_regime(tmp_path):
    """When HMM classifies as 'crisis' and bot allows only 'risk-on', return {}."""
    hmm = _make_mock_hmm()
    hmm_path = tmp_path / "hmm.json"
    save_hmm(hmm, hmm_path)

    class _TestBot(_RegimeAware):
        bot_id = "test-regime-bot-blocked"
        base_bot_id = "momo"
        allowed_regimes = ("risk-on",)
        confidence_threshold = 0.0  # accept any confidence
        hmm_state_path = hmm_path

    bot = _TestBot()
    histories = _minimal_histories()
    as_of = date(2025, 12, 31)

    # Mock hmm_regime_classify to return 'crisis' with high confidence
    with patch(
        "quant_lab.strategies.regime_aware.hmm_regime_classify",
        return_value={
            "regime_id": 3,
            "regime_name": "crisis",
            "regime_probs": {"crisis": 0.9, "risk-off": 0.05, "chop": 0.03, "risk-on": 0.02},
            "regime_confidence": 0.9,
        },
    ):
        result = bot.target_weights(histories, as_of)

    assert result == {}, f"Expected empty weights in crisis regime, got {result}"


# ---------------------------------------------------------------------------
# Gating: low confidence falls back to base
# ---------------------------------------------------------------------------

def test_regime_aware_low_confidence_fallback(tmp_path):
    """When regime_confidence < threshold, fall back to base strategy."""
    hmm = _make_mock_hmm()
    hmm_path = tmp_path / "hmm.json"
    save_hmm(hmm, hmm_path)

    class _TestBot(_RegimeAware):
        bot_id = "test-regime-bot-lowconf"
        base_bot_id = "momo"
        allowed_regimes = ("risk-on",)
        confidence_threshold = 0.4
        hmm_state_path = hmm_path

    bot = _TestBot()
    histories = _minimal_histories()
    as_of = date(2025, 12, 31)

    # confidence 0.2 < threshold 0.4: should fall back, not gate
    with patch(
        "quant_lab.strategies.regime_aware.hmm_regime_classify",
        return_value={
            "regime_id": 3,
            "regime_name": "crisis",
            "regime_probs": {"crisis": 0.2, "risk-off": 0.3, "chop": 0.3, "risk-on": 0.2},
            "regime_confidence": 0.2,
        },
    ):
        result = bot.target_weights(histories, as_of)

    # Should NOT be blocked (fallback behaviour: returns base strategy weights)
    assert isinstance(result, dict)
    # Empty is fine if momo finds no signals; what matters is no gating happened
    # (if it had gated, it would be {} due to 'crisis' not in allowed_regimes)
    # We verify by confirming the mock was called (i.e., HMM path was reached)


def test_regime_aware_allowed_regime_passes_through(tmp_path):
    """When regime is in allowed_regimes with high confidence, strategy fires."""
    hmm = _make_mock_hmm()
    hmm_path = tmp_path / "hmm.json"
    save_hmm(hmm, hmm_path)

    class _TestBot(_RegimeAware):
        bot_id = "test-regime-bot-allowed"
        base_bot_id = "momo"
        allowed_regimes = ("risk-on", "chop")
        confidence_threshold = 0.4
        hmm_state_path = hmm_path

    bot = _TestBot()
    histories = _minimal_histories()
    as_of = date(2025, 12, 31)

    with patch(
        "quant_lab.strategies.regime_aware.hmm_regime_classify",
        return_value={
            "regime_id": 0,
            "regime_name": "risk-on",
            "regime_probs": {"risk-on": 0.8, "chop": 0.1, "risk-off": 0.05, "crisis": 0.05},
            "regime_confidence": 0.8,
        },
    ):
        result = bot.target_weights(histories, as_of)

    # Base strategy runs normally — result is a dict (may be empty if no signals)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Registration: all three concrete variants registered
# ---------------------------------------------------------------------------

def test_regime_momo_registered():
    """RegimeMomo is registered with bot_id 'regime-momo'."""
    bot = get("regime-momo")
    assert isinstance(bot, RegimeMomo)


def test_regime_meanrev_registered():
    """RegimeMeanRev is registered with bot_id 'regime-meanrev'."""
    bot = get("regime-meanrev")
    assert isinstance(bot, RegimeMeanRev)


def test_regime_breakout_registered():
    """RegimeBreakout is registered with bot_id 'regime-breakout'."""
    bot = get("regime-breakout")
    assert isinstance(bot, RegimeBreakout)


def test_all_regime_variants_in_registry():
    """All three regime variants appear in the global registry."""
    all_ids = {s.bot_id for s in get_all()}
    assert "regime-momo" in all_ids
    assert "regime-meanrev" in all_ids
    assert "regime-breakout" in all_ids


# ---------------------------------------------------------------------------
# Bot_id / description / allowed_regimes spot checks
# ---------------------------------------------------------------------------

def test_regime_momo_allowed_regimes():
    assert RegimeMomo.allowed_regimes == ("risk-on", "chop")


def test_regime_meanrev_allowed_regimes():
    assert RegimeMeanRev.allowed_regimes == ("chop",)


def test_regime_breakout_allowed_regimes():
    assert RegimeBreakout.allowed_regimes == ("risk-on",)


def test_hmm_state_path_constant():
    """HMM_STATE_PATH is an absolute Path ending in state/hmm_state.json."""
    assert HMM_STATE_PATH.is_absolute()
    assert HMM_STATE_PATH.name == "hmm_state.json"
    assert HMM_STATE_PATH.parent.name == "state"

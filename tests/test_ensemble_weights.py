"""Tests for ensemble weight computation."""
from __future__ import annotations

import pytest

from quant_lab.ensemble.weights import compute_strategy_weights, regime_stability_factor


# ---------------------------------------------------------------------------
# regime_stability_factor tests
# ---------------------------------------------------------------------------

def test_regime_stability_all_positive():
    """All windows positive → stability = 1.0."""
    windows = [{"sharpe": 1.0}, {"sharpe": 0.5}, {"sharpe": 0.8}]
    assert regime_stability_factor(windows) == pytest.approx(1.0)


def test_regime_stability_half_flip():
    """Half positive, half negative → stability = 0.5."""
    windows = [{"sharpe": 1.0}, {"sharpe": -1.0}]
    result = regime_stability_factor(windows)
    assert result == pytest.approx(0.5)


def test_regime_stability_empty():
    """Empty window list → stability = 1.0 (no evidence of instability)."""
    assert regime_stability_factor([]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_strategy_weights tests
# ---------------------------------------------------------------------------

def _make_strat(bot_id, sharpe_ci_lo, sharpe_point, sig_weight, per_window=None):
    """Helper to build a calibration result dict."""
    return {
        "bot_id": bot_id,
        "aggregate": {
            "sharpe": sharpe_point,
            "sharpe_ci_lo": sharpe_ci_lo,
            "significance_weight": sig_weight,
        },
        "per_window": per_window or [{"sharpe": sharpe_point}],
    }


def test_high_sharpe_strategy_gets_high_weight():
    """Strategy with Sharpe CI lo=1.0, sig=0.8, all positive windows → high weight.
    Use cap=0.90 so cap is not the binding constraint; the raw weights determine ranking.
    """
    strats = [
        _make_strat("alpha", 1.0, 1.5, 0.8, [{"sharpe": 1.0}, {"sharpe": 0.9}]),
        _make_strat("beta",  0.1, 0.3, 0.1, [{"sharpe": 0.1}]),
    ]
    weights = compute_strategy_weights(strats, cap=0.90)
    assert "alpha" in weights
    assert "beta" in weights
    assert weights["alpha"] > weights["beta"]


def test_zero_sharpe_strategy_gets_zero_weight():
    """Strategy with sharpe_ci_lo=0, sig_weight=0 → excluded from weights."""
    strats = [
        _make_strat("zero", 0.0, 0.0, 0.0),
        _make_strat("good", 0.5, 1.0, 0.5, [{"sharpe": 0.5}]),
    ]
    weights = compute_strategy_weights(strats, cap=0.30)
    assert weights.get("zero", 0.0) == pytest.approx(0.0)
    assert "good" in weights


def test_all_negative_sharpe_falls_back_to_equal_weight():
    """All strategies with negative/zero raw weight → equal-weight across positive Sharpe ones."""
    strats = [
        _make_strat("neg_a", -0.5, 0.1, 0.0),   # sharpe_ci_lo negative
        _make_strat("neg_b", -0.3, 0.05, 0.0),  # sharpe_ci_lo negative
    ]
    weights = compute_strategy_weights(strats, cap=0.30)
    # Both have positive Sharpe (0.1, 0.05), should get equal weight = 0.5
    assert "neg_a" in weights
    assert "neg_b" in weights
    assert weights["neg_a"] == pytest.approx(weights["neg_b"], abs=1e-9)


def test_per_strategy_cap_enforced():
    """No single strategy exceeds the cap."""
    strats = [
        _make_strat("dominant", 5.0, 5.0, 1.0, [{"sharpe": 5.0}] * 4),
        _make_strat("minor_a",  0.1, 0.2, 0.1, [{"sharpe": 0.1}]),
        _make_strat("minor_b",  0.1, 0.2, 0.1, [{"sharpe": 0.1}]),
        _make_strat("minor_c",  0.1, 0.2, 0.1, [{"sharpe": 0.1}]),
    ]
    cap = 0.30
    weights = compute_strategy_weights(strats, cap=cap)
    for bid, w in weights.items():
        assert w <= cap + 1e-9, f"{bid} weight {w} exceeds cap {cap}"


def test_weights_sum_to_one():
    """Returned weights must sum to 1.0 (or 0.0 if empty)."""
    strats = [
        _make_strat("a", 0.3, 0.8, 0.5, [{"sharpe": 0.8}]),
        _make_strat("b", 0.2, 0.5, 0.3, [{"sharpe": 0.5}]),
        _make_strat("c", 0.0, 0.0, 0.0),
    ]
    weights = compute_strategy_weights(strats, cap=0.30)
    total = sum(weights.values())
    assert total == pytest.approx(1.0, abs=1e-9) or total == pytest.approx(0.0)


def test_all_negative_sharpe_no_positive_returns_empty():
    """If every strategy has negative Sharpe, return {} (all cash)."""
    strats = [
        _make_strat("bad_a", -1.0, -0.5, 0.0),
        _make_strat("bad_b", -0.8, -0.3, 0.0),
    ]
    weights = compute_strategy_weights(strats, cap=0.30)
    assert weights == {}

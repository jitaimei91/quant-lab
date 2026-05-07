"""Tests for confidence-weighted live↔backtest blend."""
from __future__ import annotations

import pytest

from quant_lab.ensemble.blend import confidence_weight, blend_weights


class TestConfidenceWeight:
    def test_zero_days_returns_zero(self):
        assert confidence_weight(0) == 0.0

    def test_full_confidence_days_returns_one(self):
        assert confidence_weight(365) == pytest.approx(1.0)

    def test_half_confidence_days_returns_half(self):
        assert confidence_weight(182) == pytest.approx(182 / 365)

    def test_exceeding_full_confidence_clamps_to_one(self):
        assert confidence_weight(500) == 1.0

    def test_negative_days_returns_zero(self):
        assert confidence_weight(-10) == 0.0

    def test_custom_full_confidence_days(self):
        assert confidence_weight(100, full_confidence_days=200) == pytest.approx(0.5)


class TestBlendWeights:
    def test_zero_days_uses_backtest_weight(self):
        backtest = {"bot-a": 1.0}
        live = {"bot-a": 1.0}
        days = {"bot-a": 0}
        result = blend_weights(backtest, live, days)
        # alpha=0 → pure backtest; since only one bot, normalizes to 1.0
        assert result["bot-a"] == pytest.approx(1.0)

    def test_full_days_uses_live_weight(self):
        backtest = {"bot-a": 0.3, "bot-b": 0.7}
        live = {"bot-a": 0.8, "bot-b": 0.2}
        days = {"bot-a": 365, "bot-b": 365}
        result = blend_weights(backtest, live, days)
        # alpha=1 → pure live; live weights already sum to 1
        assert result["bot-a"] == pytest.approx(0.8)
        assert result["bot-b"] == pytest.approx(0.2)

    def test_half_days_produces_blend(self):
        backtest = {"bot-a": 0.6, "bot-b": 0.4}
        live = {"bot-a": 0.4, "bot-b": 0.6}
        days = {"bot-a": 182, "bot-b": 182}
        result = blend_weights(backtest, live, days)
        # alpha ≈ 0.5 → average of backtest and live (both sum to 1)
        total = sum(result.values())
        assert total == pytest.approx(1.0)
        # bot-a blended raw ≈ 0.5*0.6 + 0.5*0.4 = 0.5
        # bot-b blended raw ≈ 0.5*0.4 + 0.5*0.6 = 0.5
        assert result["bot-a"] == pytest.approx(result["bot-b"], abs=0.05)

    def test_result_sums_to_one(self):
        backtest = {"bot-a": 0.5, "bot-b": 0.3, "bot-c": 0.2}
        live = {"bot-a": 0.7, "bot-b": 0.2, "bot-c": 0.1}
        days = {"bot-a": 100, "bot-b": 200, "bot-c": 50}
        result = blend_weights(backtest, live, days)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)

    def test_bot_absent_from_live_uses_backtest_only_for_zero_days(self):
        backtest = {"bot-a": 0.6, "bot-b": 0.4}
        live = {"bot-a": 1.0}  # bot-b not in live
        days = {"bot-a": 0, "bot-b": 0}
        result = blend_weights(backtest, live, days)
        # alpha=0 for both → all from backtest
        # bot-b backtest=0.4, bot-a backtest=0.6; renorm to 1.0
        assert result["bot-a"] == pytest.approx(0.6)
        assert result["bot-b"] == pytest.approx(0.4)

    def test_empty_dicts_return_empty(self):
        result = blend_weights({}, {}, {})
        assert result == {}

from datetime import date

from quant_lab.strategies.base import Strategy, register, get_all
from quant_lab.strategies.qqq_vol import QQQVol
from quant_lab.strategies.spy_vol import SPYVol
from quant_lab.types import Bar


class _FakeStrat(Strategy):
    bot_id = "fake-strat"
    description = "Test strategy"

    def target_weights(self, histories, as_of):
        return {"SPY": 0.5}


def test_strategy_subclass_required_fields():
    s = _FakeStrat()
    assert s.bot_id == "fake-strat"
    assert s.description == "Test strategy"


def test_strategy_target_weights_returns_dict():
    s = _FakeStrat()
    weights = s.target_weights({}, date(2026, 5, 6))
    assert weights == {"SPY": 0.5}


def test_register_and_get_all():
    register(_FakeStrat)
    instances = get_all()
    assert any(s.bot_id == "fake-strat" for s in instances)


def _synth_bars(symbol, n_days=300, vol=0.01):
    import random
    random.seed(42)
    base_date = date(2025, 1, 1)
    price = 500.0
    bars = []
    for i in range(n_days):
        ret = random.gauss(0.0003, vol)
        price *= (1 + ret)
        bars.append(
            Bar(
                symbol=symbol,
                date=base_date.replace(day=1) if i == 0 else base_date.fromordinal(base_date.toordinal() + i),
                open=price * 0.999,
                high=price * 1.005,
                low=price * 0.995,
                close=price,
                volume=50_000_000,
            )
        )
    return bars


def test_spy_vol_target_low_when_vol_high():
    """When realized vol exceeds 15% target, weight should be < 1.0."""
    bars = _synth_bars("SPY", n_days=300, vol=0.025)  # ~40% annualized
    strat = SPYVol(target_vol=0.15)
    weights = strat.target_weights({"SPY": bars}, bars[-1].date)
    assert "SPY" in weights
    assert 0 < weights["SPY"] < 1.0


def test_spy_vol_caps_leverage_at_one():
    """When realized vol is below target, weight is capped at 1.0 (no leverage)."""
    bars = _synth_bars("SPY", n_days=300, vol=0.001)  # very low
    strat = SPYVol(target_vol=0.15)
    weights = strat.target_weights({"SPY": bars}, bars[-1].date)
    assert weights["SPY"] == 1.0


def test_spy_vol_returns_zero_with_insufficient_history():
    bars = _synth_bars("SPY", n_days=10)
    strat = SPYVol(target_vol=0.15)
    weights = strat.target_weights({"SPY": bars}, bars[-1].date)
    assert weights == {} or weights.get("SPY", 0.0) == 0.0


def test_qqq_vol_targets_qqq():
    bars = _synth_bars("QQQ", n_days=300, vol=0.018)
    strat = QQQVol(target_vol=0.15)
    weights = strat.target_weights({"QQQ": bars}, bars[-1].date)
    assert "QQQ" in weights
    assert 0 < weights["QQQ"] <= 1.0

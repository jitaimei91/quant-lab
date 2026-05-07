from datetime import date
from quant_lab.strategies.base import Strategy, register, get_all
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

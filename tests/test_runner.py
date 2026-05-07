from datetime import date

from quant_lab.tournament.runner import run_morning_for_strategies
from quant_lab.types import Bar, Portfolio
from quant_lab.strategies.base import Strategy, register


class _AlwaysHoldSPY(Strategy):
    bot_id = "test-hold-spy"
    description = "Test"

    def target_weights(self, histories, as_of):
        return {"SPY": 1.0}


def _bars(symbol, n=100, start_price=400.0):
    base = date(2026, 1, 2)
    bars = []
    price = start_price
    for i in range(n):
        d = base.fromordinal(base.toordinal() + i)
        price = price * (1.0 + 0.001)
        bars.append(Bar(symbol=symbol, date=d, open=price, high=price, low=price, close=price, volume=10_000_000))
    return bars


def test_run_morning_initializes_portfolios():
    histories = {"SPY": _bars("SPY", 100)}
    advs = {"SPY": 1e10}
    register(_AlwaysHoldSPY)
    portfolios, trades, navs = run_morning_for_strategies(
        strategies=[_AlwaysHoldSPY()],
        histories=histories,
        advs=advs,
        prior_portfolios={},
        prior_navs={},
        as_of=histories["SPY"][-1].date,
        starting_cash=100_000,
    )
    assert "test-hold-spy" in portfolios
    p = portfolios["test-hold-spy"]
    # Should have bought SPY toward 100% weight
    assert "SPY" in p.positions
    assert p.weight("SPY", {"SPY": histories["SPY"][-1].close}) > 0.95


def test_run_morning_records_nav():
    histories = {"SPY": _bars("SPY", 100)}
    advs = {"SPY": 1e10}
    portfolios, trades, navs = run_morning_for_strategies(
        strategies=[_AlwaysHoldSPY()],
        histories=histories,
        advs=advs,
        prior_portfolios={},
        prior_navs={},
        as_of=histories["SPY"][-1].date,
        starting_cash=100_000,
    )
    assert "test-hold-spy" in navs
    assert len(navs["test-hold-spy"]) >= 1
    assert navs["test-hold-spy"][-1][0] == histories["SPY"][-1].date

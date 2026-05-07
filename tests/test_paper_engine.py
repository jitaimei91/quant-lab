from datetime import date

from quant_lab.engine import rebalance, PaperResult
from quant_lab.types import Bar, Portfolio, Position


def _bar(symbol, d, close):
    return Bar(symbol=symbol, date=d, open=close, high=close, low=close, close=close, volume=10_000_000)


def test_rebalance_buys_to_target_weight():
    portfolio = Portfolio(bot_id="t", cash=10_000.0, positions={})
    weights = {"SPY": 0.5}
    prices = {"SPY": 500.0}
    advs = {"SPY": 1e10}
    today = date(2026, 5, 6)

    result = rebalance(portfolio, weights, prices, advs, as_of=today)

    assert isinstance(result, PaperResult)
    assert result.portfolio.positions["SPY"].shares > 0
    # Should have spent close to 50% of equity (minus slippage)
    new_weight = result.portfolio.weight("SPY", prices)
    assert 0.45 < new_weight < 0.55
    assert any(t.side == "BUY" for t in result.trades)


def test_rebalance_sells_when_target_lower():
    portfolio = Portfolio(
        bot_id="t",
        cash=5_000.0,
        positions={"SPY": Position(symbol="SPY", shares=10, avg_cost=500.0)},
    )
    weights = {"SPY": 0.0}  # exit
    prices = {"SPY": 500.0}
    advs = {"SPY": 1e10}
    today = date(2026, 5, 6)

    result = rebalance(portfolio, weights, prices, advs, as_of=today)

    assert "SPY" not in result.portfolio.positions or result.portfolio.positions["SPY"].shares == 0
    assert any(t.side == "SELL" for t in result.trades)


def test_rebalance_skips_tiny_drift():
    """When at-target within tolerance, no trade should fire."""
    portfolio = Portfolio(
        bot_id="t",
        cash=5_000.0,
        positions={"SPY": Position(symbol="SPY", shares=10, avg_cost=500.0)},
    )
    weights = {"SPY": 0.5}  # already ~50%
    prices = {"SPY": 500.0}
    advs = {"SPY": 1e10}
    result = rebalance(portfolio, weights, prices, advs, as_of=date(2026, 5, 6), drift_threshold=0.01)
    assert result.trades == []

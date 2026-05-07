from datetime import date
from quant_lab.types import Bar, Position, Trade, Portfolio


def test_bar_construction():
    bar = Bar(symbol="SPY", date=date(2026, 5, 6), open=500.0, high=502.0, low=498.0, close=501.0, volume=50_000_000)
    assert bar.symbol == "SPY"
    assert bar.close == 501.0
    assert bar.volume == 50_000_000


def test_position_market_value():
    pos = Position(symbol="AAPL", shares=10, avg_cost=180.0)
    assert pos.market_value(200.0) == 2000.0


def test_trade_construction():
    trade = Trade(
        bot_id="spy-vol",
        symbol="SPY",
        side="BUY",
        shares=2.0,
        price=500.0,
        slippage_bps=5.0,
        timestamp=date(2026, 5, 6),
    )
    assert trade.side == "BUY"
    assert trade.bot_id == "spy-vol"


def test_portfolio_equity_with_positions():
    portfolio = Portfolio(
        bot_id="spy-vol",
        cash=2_000.0,
        positions={
            "SPY": Position(symbol="SPY", shares=10, avg_cost=480.0),
        },
    )
    prices = {"SPY": 500.0}
    assert portfolio.equity(prices) == 2_000.0 + 10 * 500.0


def test_portfolio_weight():
    portfolio = Portfolio(
        bot_id="qqq-vol",
        cash=1_000.0,
        positions={"QQQ": Position(symbol="QQQ", shares=2, avg_cost=400.0)},
    )
    prices = {"QQQ": 500.0}
    weight = portfolio.weight("QQQ", prices)
    assert abs(weight - (1000.0 / 2000.0)) < 1e-9

# tests/test_backtest_harness.py
from datetime import date

from quant_lab.backtest.harness import run_walk_forward
from quant_lab.backtest.windows import Window
from quant_lab.strategies.base import Strategy
from quant_lab.types import Bar


class _AlwaysSPY(Strategy):
    bot_id = "test-always-spy"
    description = "Test strategy"

    def target_weights(self, histories, as_of):
        return {"SPY": 1.0}


def _bars(symbol, start, end, drift=0.0005):
    base = start
    bars, price = [], 400.0
    n_days = (end - start).days
    for i in range(n_days):
        d = base.fromordinal(base.toordinal() + i)
        # Skip weekends roughly
        if d.weekday() >= 5:
            continue
        price *= (1 + drift)
        bars.append(Bar(symbol=symbol, date=d, open=price, high=price, low=price, close=price, volume=10_000_000))
    return bars


def test_run_walk_forward_produces_nav_per_strategy():
    histories = {
        "SPY": _bars("SPY", date(2018, 1, 1), date(2024, 1, 1)),
        "QQQ": _bars("QQQ", date(2018, 1, 1), date(2024, 1, 1), drift=0.0007),
    }
    window = Window(
        train_start=date(2018, 1, 1),
        train_end=date(2020, 1, 1),
        test_end=date(2021, 1, 1),
        label="test",
    )
    result = run_walk_forward(
        strategies=[_AlwaysSPY()],
        histories=histories,
        windows=[window],
        starting_cash=100_000,
    )
    assert "test-always-spy" in result.nav_by_window["test"]
    nav = result.nav_by_window["test"]["test-always-spy"]
    assert len(nav) > 100  # ~year of trading days
    assert nav[-1] > nav[0]  # positive drift => positive NAV change


def test_run_walk_forward_multiple_windows_isolated():
    histories = {"SPY": _bars("SPY", date(2018, 1, 1), date(2024, 1, 1))}
    w1 = Window(date(2018, 1, 1), date(2020, 1, 1), date(2021, 1, 1), "w1")
    w2 = Window(date(2019, 1, 1), date(2021, 1, 1), date(2022, 1, 1), "w2")
    result = run_walk_forward(
        strategies=[_AlwaysSPY()],
        histories=histories,
        windows=[w1, w2],
        starting_cash=100_000,
    )
    # Each window has its own NAV series starting from the same starting_cash
    assert result.nav_by_window["w1"]["test-always-spy"][0] != result.nav_by_window["w2"]["test-always-spy"][-1]


from quant_lab.backtest.slippage_sweep import run_slippage_sweep


def test_slippage_sweep_returns_one_result_per_multiplier():
    histories = {"SPY": _bars("SPY", date(2018, 1, 1), date(2021, 1, 1))}
    window = Window(date(2018, 1, 1), date(2020, 1, 1), date(2020, 12, 31), "test")
    sweep = run_slippage_sweep(
        strategies=[_AlwaysSPY()],
        histories=histories,
        windows=[window],
        multipliers=(1.0, 2.0, 5.0),
    )
    assert set(sweep.results.keys()) == {1.0, 2.0, 5.0}
    # Higher slippage should produce same-or-lower NAV (cumulative cost drag)
    nav_1x = sweep.results[1.0].nav_by_window["test"]["test-always-spy"][-1]
    nav_5x = sweep.results[5.0].nav_by_window["test"]["test-always-spy"][-1]
    assert nav_5x <= nav_1x

from __future__ import annotations

from datetime import date, timedelta
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from morning_quant_bot.backtest import run_backtest
from morning_quant_bot.features import max_drawdown, sharpe_ratio
from morning_quant_bot.models import Bar
from morning_quant_bot.strategy import DEFAULT_STRATEGY, target_weights
from morning_quant_bot.backtest import prepare_date_indexes


def synthetic_bars(symbol: str, start: date, days: int, drift: float) -> list[Bar]:
    bars: list[Bar] = []
    price = 100.0
    current = start
    for index in range(days):
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue
        cycle = 0.0015 if index % 5 in (0, 1, 2) else -0.0025
        price *= 1.0 + drift + cycle
        bars.append(
            Bar(
                symbol=symbol,
                date=current,
                open=price * 0.99,
                high=price * 1.01,
                low=price * 0.98,
                close=price,
                volume=1_000_000,
            )
        )
        current += timedelta(days=1)
    return bars


class QuantBotTests(unittest.TestCase):
    def test_metrics_are_finite(self) -> None:
        curve = [100.0, 101.0, 99.0, 103.0, 104.0]
        self.assertLessEqual(max_drawdown(curve), 0.0)
        self.assertIsInstance(sharpe_ratio(curve), float)

    def test_strategy_generates_weight_for_uptrend(self) -> None:
        bars = {
            "AAA": synthetic_bars("AAA", date(2020, 1, 1), 500, 0.0015),
            "BBB": synthetic_bars("BBB", date(2020, 1, 1), 500, -0.0003),
        }
        date_indexes = prepare_date_indexes(bars)
        latest = bars["AAA"][-1].date
        weights, reasons = target_weights(bars, date_indexes, latest, DEFAULT_STRATEGY)
        self.assertIn("AAA", weights)
        self.assertGreater(weights["AAA"], 0.0)
        self.assertIn("AAA", reasons)

    def test_backtest_runs(self) -> None:
        bars = {
            "AAA": synthetic_bars("AAA", date(2020, 1, 1), 900, 0.0012),
            "BBB": synthetic_bars("BBB", date(2020, 1, 1), 900, -0.0001),
            "CCC": synthetic_bars("CCC", date(2020, 1, 1), 900, 0.0002),
        }
        result = run_backtest(bars, DEFAULT_STRATEGY)
        self.assertGreater(result.metrics.final_equity, 0.0)
        self.assertGreater(len(result.equity_curve), 0)


if __name__ == "__main__":
    unittest.main()

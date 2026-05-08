"""Tests for the Calendar (turn-of-month) bot."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant_lab.types import Bar
from quant_lab.strategies.calendar import Calendar, _TARGET_WEIGHT


def _make_bars(year: int, months: list[int], extra_per_month: int = 0) -> list[Bar]:
    """Generate weekday-only SPY bars for the given months in `year`."""
    out: list[Bar] = []
    price = 100.0
    for month in months:
        d = date(year, month, 1)
        while d.month == month:
            if d.weekday() < 5:  # weekday only
                price *= 1.0005
                out.append(
                    Bar(symbol="SPY", date=d, open=price, high=price, low=price,
                        close=price, volume=1_000_000)
                )
            d = d + timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# In/out window detection
# ---------------------------------------------------------------------------


def test_first_day_of_month_is_in_window():
    bars = _make_bars(2024, [1])
    first = next(b for b in bars if b.date.day <= 7)
    weights = Calendar().target_weights({"SPY": bars}, first.date)
    assert weights == {"SPY": _TARGET_WEIGHT}


def test_third_day_of_month_is_in_window():
    bars = _make_bars(2024, [1])
    # Third trading day of January
    third = bars[2]
    weights = Calendar().target_weights({"SPY": bars}, third.date)
    assert weights == {"SPY": _TARGET_WEIGHT}


def test_fourth_day_of_month_is_out_of_window_when_more_days_remain():
    """The 4th trading day of a month with more days ahead → out of window."""
    bars = _make_bars(2024, [1])  # full January, ~22 trading days
    fourth = bars[3]
    weights = Calendar().target_weights({"SPY": bars}, fourth.date)
    assert weights == {}


def test_last_day_of_month_is_in_window():
    bars = _make_bars(2024, [1])
    last_jan = bars[-1]  # last trading day of January
    weights = Calendar().target_weights({"SPY": bars}, last_jan.date)
    assert weights == {"SPY": _TARGET_WEIGHT}


def test_third_to_last_day_of_month_is_in_window():
    bars = _make_bars(2024, [1])
    third_to_last = bars[-3]
    weights = Calendar().target_weights({"SPY": bars}, third_to_last.date)
    assert weights == {"SPY": _TARGET_WEIGHT}


def test_mid_month_is_out_of_window():
    bars = _make_bars(2024, [1])
    n = len(bars)
    mid = bars[n // 2]
    weights = Calendar().target_weights({"SPY": bars}, mid.date)
    assert weights == {}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_no_spy_bars_returns_empty():
    weights = Calendar().target_weights({}, date(2024, 1, 5))
    assert weights == {}


def test_non_trading_day_returns_empty():
    """as_of on a weekend (no SPY bar that date) → empty."""
    bars = _make_bars(2024, [1])
    saturday = date(2024, 1, 6)  # Saturday
    weights = Calendar().target_weights({"SPY": bars}, saturday.date if hasattr(saturday, "date") else saturday)
    assert weights == {}


def test_truncated_month_does_not_falsely_flag_mid_month():
    """If we only have bars up to mid-month (truncated dataset), an
    arbitrary mid-month bar must NOT be treated as in-window just because
    no later same-month bars exist in the data. The last-N rule must
    consult the calendar, not 'last bar I've seen'."""
    bars = _make_bars(2024, [1])[:13]  # only the first ~13 weekdays of Jan
    last = bars[-1]  # somewhere mid-Jan
    weights = Calendar().target_weights({"SPY": bars}, last.date)
    assert weights == {}

"""Calendar bot: harvest the turn-of-month effect in equities.

The empirical regularity: from the late 1980s onward, US equities post the
bulk of their monthly return in a narrow window around the month boundary
(last 3 trading days + first 3 trading days). Several proposed mechanisms —
month-end pension flows, retail 401k contributions on the 1st/15th, fund
window-dressing — but the effect persists out-of-sample even after costs.

Documented Sharpe contribution in academic studies: 0.3–0.5 standalone over
a long sample, lower (0.1–0.3) in shorter windows or with realistic costs.
A small, **uncorrelated** sleeve to add to the meta-ensemble.

Implementation choices:
- Universe: SPY only. Keep this simple and orthogonal — it's a calendar
  effect, not asset selection. Use SPY because it's the most liquid and
  the documented effect is strongest in large-cap US equities.
- "First N" is identified by walking the bars we've seen so far this month
  (point-in-time safe).
- "Last N" is identified by computing the last N weekdays of `as_of`'s
  calendar month — cheap, no future data needed, and accurate enough since
  market holidays affecting this window are rare.
- Cash (empty position) outside the window — do NOT default to SPY here,
  because the whole point is to NOT hold equities outside the window.
"""
from __future__ import annotations

from calendar import monthrange
from datetime import date

from ..types import Bar
from .base import Strategy, register


_SYMBOL = "SPY"
_TARGET_WEIGHT = 0.95  # full long during the window, with cash buffer
_LAST_DAYS = 3   # last N weekdays of month
_FIRST_DAYS = 3  # first N trading days seen this month


def _last_n_weekdays_of_month(year: int, month: int, n: int) -> list[date]:
    """Return the last n weekdays (Mon-Fri) of the given calendar month.

    Doesn't account for market holidays — for a 3-day window the typical
    edge case (Christmas Eve / day after Thanksgiving) doesn't bite us
    because they always sit inside the last 3 weekdays anyway.
    """
    last_day = monthrange(year, month)[1]
    out: list[date] = []
    for day_of_month in range(last_day, 0, -1):
        candidate = date(year, month, day_of_month)
        if candidate.weekday() < 5:
            out.append(candidate)
        if len(out) >= n:
            break
    return out


def _is_in_tom_window(bars: list[Bar], as_of: date) -> bool:
    """True iff `as_of` is in the first N or last N trading-window days of
    its calendar month.

    Point-in-time safe: only reads bars with date <= as_of, plus the calendar.
    """
    # Treat as_of as a no-signal day if it isn't a trading bar in the universe
    same_month_bars = sorted(
        b.date for b in bars
        if b.date <= as_of and b.date.year == as_of.year and b.date.month == as_of.month
    )
    if as_of not in same_month_bars:
        return False

    # First-N rule (point-in-time safe — only uses bars we've already seen)
    if as_of in same_month_bars[:_FIRST_DAYS]:
        return True

    # Last-N rule — computed from the calendar (no future bars required)
    return as_of in _last_n_weekdays_of_month(as_of.year, as_of.month, _LAST_DAYS)


@register
class Calendar(Strategy):
    """Long SPY during turn-of-month window, cash otherwise."""

    bot_id = "calendar"
    description = "Turn-of-month SPY long sleeve (last 3 + first 3 trading days)"

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        bars = histories.get(_SYMBOL, [])
        if not bars:
            return {}
        if _is_in_tom_window(bars, as_of):
            return {_SYMBOL: _TARGET_WEIGHT}
        return {}

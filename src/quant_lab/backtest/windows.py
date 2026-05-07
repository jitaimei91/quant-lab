"""Walk-forward windows + regime-stress windows.

Walk-forward: rolling fixed-length train window, fixed-length test immediately
following, stepping forward in time. Each test period is fully out-of-sample.

Regime-stress: hand-picked windows over historical crises so calibration shows
how each strategy behaves outside benign markets.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from dateutil.relativedelta import relativedelta


@dataclass(frozen=True, slots=True)
class Window:
    train_start: date
    train_end: date
    test_end: date
    label: str = ""

    @property
    def test_start(self) -> date:
        return self.train_end


def walk_forward_windows(
    start: date,
    end: date,
    train_years: int = 5,
    step_months: int = 12,
    test_months: int = 12,
) -> list[Window]:
    """Generate rolling train→test windows from `start` to `end`.

    Each window: train on [train_start, train_end), test on [train_end, test_end).
    """
    windows: list[Window] = []
    cursor = start + relativedelta(years=train_years)
    while True:
        train_end = cursor
        test_end = train_end + relativedelta(months=test_months)
        if test_end > end:
            break
        train_start = train_end - relativedelta(years=train_years)
        windows.append(
            Window(
                train_start=train_start,
                train_end=train_end,
                test_end=test_end,
                label=f"wf-{train_end.isoformat()}",
            )
        )
        cursor = cursor + relativedelta(months=step_months)
    return windows


def regime_stress_windows() -> list[Window]:
    """Hand-picked stress windows covering known regime breaks."""
    return [
        Window(
            train_start=date(2003, 1, 1),
            train_end=date(2007, 1, 1),
            test_end=date(2009, 12, 31),
            label="2008-financial-crisis",
        ),
        Window(
            train_start=date(2015, 1, 1),
            train_end=date(2020, 1, 1),
            test_end=date(2020, 12, 31),
            label="2020-covid",
        ),
        Window(
            train_start=date(2017, 1, 1),
            train_end=date(2022, 1, 1),
            test_end=date(2023, 12, 31),
            label="2022-rate-hikes",
        ),
    ]

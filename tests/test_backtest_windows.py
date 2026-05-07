# tests/test_backtest_windows.py
from datetime import date

from quant_lab.backtest.windows import (
    walk_forward_windows,
    regime_stress_windows,
    Window,
)


def test_walk_forward_windows_basic():
    windows = walk_forward_windows(
        start=date(2015, 1, 1),
        end=date(2025, 1, 1),
        train_years=5,
        step_months=12,
    )
    assert all(isinstance(w, Window) for w in windows)
    # First window: train 2015-2019, test 2020
    assert windows[0].train_start == date(2015, 1, 1)
    assert windows[0].train_end == date(2020, 1, 1)
    assert windows[0].test_end == date(2021, 1, 1)
    # Each subsequent window steps by 12 months
    assert (windows[1].train_end - windows[0].train_end).days >= 350
    # Last test_end <= end
    assert windows[-1].test_end <= date(2025, 1, 1)


def test_walk_forward_windows_monthly_step():
    windows = walk_forward_windows(
        start=date(2018, 1, 1),
        end=date(2025, 1, 1),
        train_years=3,
        step_months=1,
    )
    # 3-year train + monthly stepping = many windows
    assert len(windows) > 30


def test_regime_stress_windows_includes_crisis_periods():
    windows = regime_stress_windows()
    labels = {w.label for w in windows}
    assert "2008-financial-crisis" in labels
    assert "2020-covid" in labels
    assert "2022-rate-hikes" in labels

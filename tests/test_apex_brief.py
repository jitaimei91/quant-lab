"""Tests for the Discord apex brief — diffing, confidence tiers, format."""
from __future__ import annotations

from datetime import date


from quant_lab.reporting.apex_brief import (
    build_brief,
    diff_portfolios,
    _confidence_for,
)


# ---------------------------------------------------------------------------
# Confidence tiers
# ---------------------------------------------------------------------------


def test_confidence_paper_only_under_30_days():
    tier = _confidence_for(5)
    assert tier.label == "PAPER ONLY"
    assert "NOISE" in tier.note


def test_confidence_low_at_30_to_89():
    tier = _confidence_for(45)
    assert tier.label == "LOW CONFIDENCE"


def test_confidence_medium_at_90_to_179():
    tier = _confidence_for(120)
    assert tier.label == "MEDIUM CONFIDENCE"


def test_confidence_live_calibrated_at_180_plus():
    tier = _confidence_for(365)
    assert tier.label == "LIVE-CALIBRATED"


# ---------------------------------------------------------------------------
# diff_portfolios
# ---------------------------------------------------------------------------


def test_diff_no_change_yields_holds_only():
    target = {"SPY": 0.10, "TLT": 0.08}
    current = {"SPY": 0.10, "TLT": 0.08}
    holds, buys, sells = diff_portfolios(target, current)
    assert holds == {"SPY": 0.10, "TLT": 0.08}
    assert buys == {}
    assert sells == {}


def test_diff_new_position_is_a_buy():
    target = {"SPY": 0.10, "QQQ": 0.05}
    current = {"SPY": 0.10}
    holds, buys, sells = diff_portfolios(target, current)
    assert "QQQ" in buys
    assert buys["QQQ"] == 0.05
    assert "SPY" in holds


def test_diff_position_dropped_is_a_sell():
    target = {"SPY": 0.10}
    current = {"SPY": 0.10, "USO": 0.04}
    holds, buys, sells = diff_portfolios(target, current)
    assert "USO" in sells
    assert sells["USO"] == 0.04


def test_diff_increase_above_threshold_is_a_buy():
    target = {"QQQ": 0.10}
    current = {"QQQ": 0.05}
    holds, buys, sells = diff_portfolios(target, current)
    assert "QQQ" in buys


def test_diff_small_change_below_threshold_stays_in_holds():
    target = {"SPY": 0.105}
    current = {"SPY": 0.10}
    holds, buys, sells = diff_portfolios(target, current)
    assert "SPY" in holds
    assert "SPY" not in buys


def test_diff_decrease_above_threshold_is_a_sell():
    """Trim from 10% to 8.5% (1.5pp delta) crosses the 1pp threshold."""
    target = {"SPY": 0.085}
    current = {"SPY": 0.10}
    holds, buys, sells = diff_portfolios(target, current)
    assert "SPY" in sells


# ---------------------------------------------------------------------------
# build_brief integration
# ---------------------------------------------------------------------------


def test_brief_under_30_days_shows_paper_only_warning():
    msg = build_brief(
        today=date(2026, 5, 9),
        target_weights={"SPY": 0.10},
        current_weights={"SPY": 0.10},
        days_of_data=2,
    )
    assert "PAPER ONLY" in msg
    assert "NOISE" in msg


def test_brief_no_trades_message_when_unchanged():
    msg = build_brief(
        today=date(2026, 5, 9),
        target_weights={"SPY": 0.10, "TLT": 0.08},
        current_weights={"SPY": 0.10, "TLT": 0.08},
        days_of_data=200,
    )
    assert "No trades today" in msg
    assert "LIVE-CALIBRATED" in msg


def test_brief_renders_buy_and_sell_blocks():
    msg = build_brief(
        today=date(2026, 5, 9),
        target_weights={"SPY": 0.12, "QQQ": 0.08},
        current_weights={"SPY": 0.05, "USO": 0.06},
        days_of_data=120,
    )
    # SPY went up → BUY, QQQ is new → BUY, USO removed → SELL
    assert "BUY" in msg
    assert "SELL" in msg
    assert "QQQ" in msg
    assert "USO" in msg
    assert "MEDIUM CONFIDENCE" in msg


def test_brief_includes_pnl_when_provided():
    msg = build_brief(
        today=date(2026, 5, 9),
        target_weights={"SPY": 0.10},
        current_weights={"SPY": 0.10},
        days_of_data=200,
        portfolio_return=0.012,
        spy_benchmark_return=0.008,
    )
    assert "beat SPY" in msg


def test_brief_includes_market_snapshot_and_dashboard_url():
    msg = build_brief(
        today=date(2026, 5, 9),
        target_weights={"SPY": 0.10},
        current_weights={"SPY": 0.10},
        days_of_data=2,
        market_snapshot={"SPY": {"change_pct": 1.2, "ytd_pct": 8.0}, "QQQ": {"change_pct": 1.5, "ytd_pct": 12.0}},
        dashboard_url="https://jitaimei91.github.io/quant-lab/",
    )
    assert "SPY +1.20%" in msg
    assert "QQQ +1.50%" in msg
    assert "jitaimei91.github.io" in msg


def test_brief_disclaimer_always_present():
    msg = build_brief(
        today=date(2026, 5, 9),
        target_weights={"SPY": 0.10},
        current_weights={},
        days_of_data=200,
    )
    assert "Not financial advice" in msg


# ---------------------------------------------------------------------------
# account_size_usd: dollar amounts + dust filter
# ---------------------------------------------------------------------------


def test_brief_with_account_size_renders_dollar_amounts():
    """When account_size_usd is set, BUY lines include ($X.XX)."""
    msg = build_brief(
        today=date(2026, 5, 9),
        target_weights={"SPY": 0.10},
        current_weights={},
        days_of_data=200,
        account_size_usd=224.32,
    )
    # 10% of $224.32 = $22.43
    assert "$22.43" in msg
    assert "Account: $224.32" in msg


def test_brief_filters_trades_below_min_dollar_threshold():
    """At a $200 account, a 0.3% allocation = $0.60 → below $1 threshold → suppressed."""
    msg = build_brief(
        today=date(2026, 5, 9),
        target_weights={"SPY": 0.20, "USO": 0.003},  # USO would be $0.60
        current_weights={"SPY": 0.20},
        days_of_data=200,
        account_size_usd=200.0,
        min_trade_dollars=1.0,
    )
    assert "USO" not in msg  # dust filter killed the tiny rebalance
    assert "No trades today" in msg


def test_brief_keeps_trades_above_min_dollar_threshold():
    """A 5% allocation on $1000 = $50 → above $1 threshold → kept."""
    msg = build_brief(
        today=date(2026, 5, 9),
        target_weights={"SPY": 0.20, "QQQ": 0.05},
        current_weights={"SPY": 0.20},
        days_of_data=200,
        account_size_usd=1000.0,
        min_trade_dollars=1.0,
    )
    assert "QQQ" in msg
    assert "$50" in msg


def test_brief_without_account_size_shows_percentages_only():
    """Default behavior (no account_size) does not show dollar amounts."""
    msg = build_brief(
        today=date(2026, 5, 9),
        target_weights={"SPY": 0.10},
        current_weights={},
        days_of_data=200,
    )
    assert "10.0%" in msg
    # No dollar markers
    assert "$" not in msg or "Account" not in msg
    # Specifically: should not have a "($x.xx)" parenthetical next to the BUY
    assert "($" not in msg

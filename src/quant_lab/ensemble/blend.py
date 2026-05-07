"""Confidence-weighted live↔backtest weight blend.

As live NAV history accumulates, trust shifts smoothly from backtest-calibrated
weights toward live-evidence weights. The confidence factor is simply:

    alpha = min(1.0, days_of_live / full_confidence_days)

At 0 days: pure backtest. At 365+ days: pure live. In between: smooth ramp.
"""
from __future__ import annotations


def confidence_weight(days_of_live: int, full_confidence_days: int = 365) -> float:
    """Map days of live history to [0, 1].

    0 days → 0.0 (full backtest).
    full_confidence_days+ days → 1.0 (full live).
    Linear ramp in between.
    """
    if days_of_live <= 0:
        return 0.0
    return min(1.0, days_of_live / full_confidence_days)


def blend_weights(
    backtest_weights: dict[str, float],
    live_weights: dict[str, float],
    days_of_live_per_bot: dict[str, int],
    full_confidence_days: int = 365,
) -> dict[str, float]:
    """Blend backtest and live weights per-bot using confidence-weighted interpolation.

    For each bot present in either dict:
        alpha = confidence_weight(days_of_live_per_bot.get(bot, 0))
        weight = (1 - alpha) * backtest_weight + alpha * live_weight

    Bots absent from a source dict are treated as having weight 0 in that dict.
    Result is re-normalized to sum=1.0. Bots not in either dict are dropped.

    Args:
        backtest_weights: {bot_id: weight} from backtest calibration.
        live_weights: {bot_id: weight} from live NAV calibration.
        days_of_live_per_bot: {bot_id: int} days of live history per bot.
        full_confidence_days: Days until live weight has full confidence.

    Returns:
        dict[str, float] of blended, re-normalized weights.
    """
    all_bots = set(backtest_weights) | set(live_weights)
    blended: dict[str, float] = {}

    for bot_id in all_bots:
        days = days_of_live_per_bot.get(bot_id, 0)
        alpha = confidence_weight(days, full_confidence_days)
        bt_w = backtest_weights.get(bot_id, 0.0)
        live_w = live_weights.get(bot_id, 0.0)
        blended[bot_id] = (1.0 - alpha) * bt_w + alpha * live_w

    # Re-normalize to sum=1.0
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}

    return blended

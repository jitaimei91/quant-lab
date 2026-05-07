"""Calibrated-Sharpe-weighted strategy weight computation.

Weight formula per strategy:
    raw_weight = max(0, sharpe_ci_lo) * significance_weight * regime_stability_factor

Weights are then normalized to sum = 1.0, with a per-strategy cap of `cap`.
If all strategies have raw_weight = 0, falls back to equal-weight across
positive-Sharpe strategies. If none have positive Sharpe, returns {} (all cash).
"""
from __future__ import annotations


def regime_stability_factor(per_window_results: list[dict]) -> float:
    """Reward strategies consistent across walk-forward windows.

    Returns 1.0 if all per-window Sharpes match the sign of the aggregate
    (i.e., all positive or all negative). Decreases linearly toward 0 as
    sign-flipping increases.

    Args:
        per_window_results: List of dicts with at least a 'sharpe' key.

    Returns:
        float in [0.0, 1.0]. 1.0 = fully consistent; 0.0 = always flipping.
    """
    if not per_window_results:
        return 1.0

    sharpes = [w.get("sharpe", 0.0) for w in per_window_results]
    if not sharpes:
        return 1.0

    # Determine aggregate sign from median
    positive_count = sum(1 for s in sharpes if s > 0)
    dominant_sign_positive = positive_count >= len(sharpes) / 2

    # Count windows consistent with dominant sign
    consistent = sum(
        1 for s in sharpes if (s > 0) == dominant_sign_positive
    )
    return consistent / len(sharpes)


def compute_strategy_weights(
    calibration_results: list[dict],
    floor: float = 0.0,
    cap: float = 0.30,
) -> dict[str, float]:
    """Compute normalized ensemble weights from calibration results.

    For each strategy:
        raw_weight = max(0, sharpe_ci_lo) * significance_weight * regime_stability

    Then normalize so weights sum to 1.0. Clip each weight at `cap`.
    Falls back to equal-weight across positive-Sharpe strategies if all
    raw weights are zero. Returns {} if no strategies qualify.

    Args:
        calibration_results: List of strategy dicts from backtest_results.json["strategies"].
        floor: Minimum raw weight threshold (currently unused, reserved for future).
        cap: Maximum per-strategy weight after normalization.

    Returns:
        dict mapping bot_id -> normalized weight.
    """
    raw: dict[str, float] = {}
    sharpes: dict[str, float] = {}

    for strat in calibration_results:
        bot_id = strat.get("bot_id", "")
        if not bot_id:
            continue
        agg = strat.get("aggregate", {})
        sharpe_ci_lo = agg.get("sharpe_ci_lo", 0.0)
        sig_weight = agg.get("significance_weight", 0.0)
        per_window = strat.get("per_window", [])
        stability = regime_stability_factor(per_window)

        sharpe_point = agg.get("sharpe", 0.0)
        sharpes[bot_id] = sharpe_point

        raw_w = max(0.0, sharpe_ci_lo) * sig_weight * stability
        raw[bot_id] = raw_w

    total_raw = sum(raw.values())

    if total_raw <= 0.0:
        # Fall back: equal-weight across positive-Sharpe strategies
        positive_ids = [bid for bid, sh in sharpes.items() if sh > 0]
        if not positive_ids:
            return {}
        eq = 1.0 / len(positive_ids)
        return {bid: min(eq, cap) for bid in positive_ids}

    # Normalize
    normalized: dict[str, float] = {bid: w / total_raw for bid, w in raw.items() if w > 0}

    # Apply cap iteratively (redistribute excess to uncapped strategies)
    for _ in range(len(normalized) + 1):
        over_cap = {bid: w for bid, w in normalized.items() if w > cap}
        if not over_cap:
            break
        excess = sum(w - cap for w in over_cap.values())
        for bid in over_cap:
            normalized[bid] = cap
        under_cap = {bid: w for bid, w in normalized.items() if w < cap}
        if not under_cap:
            break
        total_under = sum(under_cap.values())
        if total_under <= 0:
            break
        for bid in under_cap:
            normalized[bid] += excess * (normalized[bid] / total_under)

    # Final re-normalize to ensure sum = 1.0
    final_total = sum(normalized.values())
    if final_total > 0:
        normalized = {bid: w / final_total for bid, w in normalized.items()}

    return normalized

"""Evidence-weighted strategy weight computation.

Weight formula per strategy:
    evidence = max(0, sharpe_ci_lo) * (1 + max(0, median_alpha_t))
    raw_weight = evidence * stability_factor * (1 + significance_weight)

Rationale:
- `sharpe_ci_lo` is the bootstrap lower bound on Sharpe. Above zero ≈ persistent
  edge; below zero ≈ noise. Strict floor at 0 filters out negative-edge bots.
- `median_alpha_t` adds a multiplicative boost when alpha vs SPY is positive
  (rare but real evidence of true alpha).
- `significance_weight` (0–1, gates on alpha t-stat ≥ 1.96) becomes a 1–2x
  amplifier rather than a hard multiplier — previously 0 for every bot in
  practice, which collapsed all raw weights to 0 and triggered noisy
  equal-weight fallback. Now significance amplifies but doesn't gate.
- `stability_factor` rewards bots whose per-window Sharpes don't sign-flip.

Normalized to sum = 1.0 with per-strategy cap. If no bot has positive
sharpe_ci_lo (no edge anywhere), returns {} so the caller can fall back to
the index — equal-weighting noise is worse than admitting we have no edge.
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

    for strat in calibration_results:
        bot_id = strat.get("bot_id", "")
        if not bot_id:
            continue
        agg = strat.get("aggregate", {})
        sharpe_ci_lo = agg.get("sharpe_ci_lo", 0.0)
        median_alpha_t = agg.get("median_alpha_t", 0.0)
        sig_weight = agg.get("significance_weight", 0.0)
        per_window = strat.get("per_window", [])
        stability = regime_stability_factor(per_window)

        # Evidence: positive lower-CI Sharpe, boosted multiplicatively by
        # positive median alpha t-stat (cap the alpha boost at +3 to prevent
        # outliers from dominating).
        alpha_boost = 1.0 + min(3.0, max(0.0, median_alpha_t))
        evidence = max(0.0, sharpe_ci_lo) * alpha_boost
        # Significance amplifier: 1.0 (no significance) → 2.0 (full significance)
        sig_amplifier = 1.0 + max(0.0, min(1.0, sig_weight))

        raw[bot_id] = evidence * stability * sig_amplifier

    total_raw = sum(raw.values())

    if total_raw <= 0.0:
        # No bot has a positive lower-CI Sharpe → no demonstrated edge.
        # Return {} so the caller (MetaEnsemble) falls back to SPY rather
        # than spreading capital across negative-evidence bots.
        return {}

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

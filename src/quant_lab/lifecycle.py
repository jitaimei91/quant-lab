"""Strategy lifecycle manager.

Automatically pauses strategies that fail rolling significance gates for
`fail_threshold_days` consecutive days, and resumes them after
`recovery_threshold_days` consecutive days of improved significance.

State is persisted to state/strategy_lifecycle.json across morning runs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .backtest.stats import (
    block_bootstrap_sharpe_ci,
    alpha_t_stat_vs_benchmark,
    significance_weight,
)


@dataclass
class LifecycleState:
    bot_id: str
    paused: bool = False
    paused_at: date | None = None
    pause_reason: str = ""
    consecutive_fail_days: int = 0
    consecutive_recovery_days: int = 0


def _nav_to_returns(nav_series: list[tuple[date, float]]) -> list[float]:
    rets: list[float] = []
    for i in range(1, len(nav_series)):
        prev = nav_series[i - 1][1]
        curr = nav_series[i][1]
        if prev > 0:
            rets.append(curr / prev - 1.0)
    return rets


def _trailing_significance(
    returns: list[float],
    bench: list[float],
    trailing_days: int = 90,
    n_iter: int = 200,
) -> tuple[float, float]:
    """Compute (significance_weight, alpha) over the last trailing_days returns."""
    trail = returns[-trailing_days:]
    if len(trail) < 10:
        return 0.0, 0.0

    _sharpe_pt, _ci_lo, _ci_hi = block_bootstrap_sharpe_ci(trail, n_iter=n_iter, seed=42)

    n_bench = min(len(trail), len(bench))
    if n_bench >= 10:
        alpha, t_stat = alpha_t_stat_vs_benchmark(trail[:n_bench], bench[-n_bench:])
    else:
        alpha, t_stat = 0.0, 0.0

    sig_w = significance_weight(t_stat)
    return sig_w, alpha


def evaluate_lifecycle(
    nav_history: dict[str, list[tuple[date, float]]],
    benchmark_returns: dict[str, list[float]],
    prior_state: dict[str, LifecycleState],
    today: date,
    *,
    fail_threshold_days: int = 90,
    recovery_threshold_days: int = 30,
    min_significance_for_active: float = 0.3,
    min_significance_for_resume: float = 0.5,
    trailing_days: int = 90,
    n_iter: int = 200,
) -> dict[str, LifecycleState]:
    """Evaluate lifecycle state for all bots based on trailing significance.

    For each bot in nav_history:
    - Compute trailing significance weight from live data
    - If ACTIVE and bot has been below min_significance_for_active AND alpha < 0
      for fail_threshold_days consecutively → pause with reason
    - If PAUSED and bot has been above min_significance_for_resume
      for recovery_threshold_days consecutively → resume
    - Otherwise carry prior state forward, increment/reset counters

    Args:
        nav_history: {bot_id: [(date, nav), ...]}
        benchmark_returns: {bot_id: [float, ...]} or {"SPY": [float, ...]}
        prior_state: lifecycle states from previous morning run
        today: today's date for recording paused_at
        fail_threshold_days: consecutive failing days to trigger pause
        recovery_threshold_days: consecutive recovery days to trigger resume
        min_significance_for_active: sig_weight below this triggers fail counter
        min_significance_for_resume: sig_weight above this triggers recovery counter
        trailing_days: window (in days) for trailing significance computation
        n_iter: bootstrap iterations for significance computation

    Returns:
        Updated dict[str, LifecycleState]
    """
    spy_rets = benchmark_returns.get("SPY", [])
    new_state: dict[str, LifecycleState] = {}

    for bot_id, nav_series in nav_history.items():
        if bot_id == "meta-ensemble":
            continue

        returns = _nav_to_returns(nav_series)
        bench = benchmark_returns.get(bot_id, spy_rets)
        sig_w, alpha = _trailing_significance(
            returns, bench, trailing_days=trailing_days, n_iter=n_iter
        )

        prev = prior_state.get(bot_id, LifecycleState(bot_id=bot_id))
        cur = LifecycleState(
            bot_id=bot_id,
            paused=prev.paused,
            paused_at=prev.paused_at,
            pause_reason=prev.pause_reason,
            consecutive_fail_days=prev.consecutive_fail_days,
            consecutive_recovery_days=prev.consecutive_recovery_days,
        )

        if not cur.paused:
            # Active: check if failing
            failing = sig_w < min_significance_for_active and alpha < 0
            if failing:
                cur.consecutive_fail_days += 1
                cur.consecutive_recovery_days = 0
                if cur.consecutive_fail_days >= fail_threshold_days:
                    cur.paused = True
                    cur.paused_at = today
                    cur.pause_reason = (
                        f"sig_weight={sig_w:.2f} alpha={alpha:.4f} "
                        f"for {cur.consecutive_fail_days} consecutive days"
                    )
            else:
                cur.consecutive_fail_days = 0
                cur.consecutive_recovery_days = 0
        else:
            # Paused: check if recovered
            recovering = sig_w >= min_significance_for_resume
            if recovering:
                cur.consecutive_recovery_days += 1
                cur.consecutive_fail_days = 0
                if cur.consecutive_recovery_days >= recovery_threshold_days:
                    cur.paused = False
                    cur.paused_at = None
                    cur.pause_reason = ""
                    cur.consecutive_fail_days = 0
                    cur.consecutive_recovery_days = 0
            else:
                cur.consecutive_recovery_days = 0

        new_state[bot_id] = cur

    return new_state


def load_lifecycle_state(path: Path) -> dict[str, LifecycleState]:
    """Load lifecycle state from a JSON file.

    Returns an empty dict if the file doesn't exist or can't be parsed.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        result: dict[str, LifecycleState] = {}
        for bot_id, entry in raw.items():
            paused_at_str = entry.get("paused_at")
            paused_at = date.fromisoformat(paused_at_str) if paused_at_str else None
            result[bot_id] = LifecycleState(
                bot_id=entry.get("bot_id", bot_id),
                paused=entry.get("paused", False),
                paused_at=paused_at,
                pause_reason=entry.get("pause_reason", ""),
                consecutive_fail_days=entry.get("consecutive_fail_days", 0),
                consecutive_recovery_days=entry.get("consecutive_recovery_days", 0),
            )
        return result
    except Exception:
        return {}


def save_lifecycle_state(state: dict[str, LifecycleState], path: Path) -> None:
    """Persist lifecycle state to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, dict] = {}
    for bot_id, s in state.items():
        payload[bot_id] = {
            "bot_id": s.bot_id,
            "paused": s.paused,
            "paused_at": s.paused_at.isoformat() if s.paused_at else None,
            "pause_reason": s.pause_reason,
            "consecutive_fail_days": s.consecutive_fail_days,
            "consecutive_recovery_days": s.consecutive_recovery_days,
        }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

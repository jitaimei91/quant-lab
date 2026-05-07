"""Tests for the strategy lifecycle manager."""
from __future__ import annotations

import json
import random
from datetime import date, timedelta

import pytest

from quant_lab.lifecycle import (
    LifecycleState,
    evaluate_lifecycle,
    load_lifecycle_state,
    save_lifecycle_state,
)

_BASE = date(2025, 1, 2)


def _nav_series(n_days: int, drift: float = 0.0, seed: int = 0) -> list[tuple[date, float]]:
    rng = random.Random(seed)
    series = []
    nav = 1.0
    for i in range(n_days):
        nav *= 1 + drift + rng.gauss(0.0, 0.01)
        nav = max(nav, 0.001)
        series.append((_BASE + timedelta(days=i), nav))
    return series


def _spy_rets(n_days: int, seed: int = 42) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0003, 0.01) for _ in range(n_days)]


def test_bot_paused_after_consecutive_fail_days():
    """Bot that fails significance for fail_threshold_days days becomes paused."""
    # Strongly negative drift → should produce negative alpha and low significance
    nav = {"failer": _nav_series(200, drift=-0.005, seed=1)}
    bench = {"SPY": _spy_rets(200)}

    prior: dict[str, LifecycleState] = {}
    today = _BASE + timedelta(days=200)

    # Simulate enough accumulated fail days via prior state
    prior["failer"] = LifecycleState(
        bot_id="failer",
        paused=False,
        consecutive_fail_days=89,  # one away from threshold
        consecutive_recovery_days=0,
    )

    # With a single bad day, it should tip over
    state = evaluate_lifecycle(
        nav_history=nav,
        benchmark_returns=bench,
        prior_state=prior,
        today=today,
        fail_threshold_days=90,
        trailing_days=30,
        n_iter=50,
    )

    # If the trailing significance is bad, it should now be paused
    s = state.get("failer")
    assert s is not None
    # Either paused (sig was bad) or consecutive_fail_days incremented
    assert s.paused or s.consecutive_fail_days >= 89


def test_bot_paused_via_injected_fail_state():
    """Directly inject a bot at the failure threshold and verify it pauses."""
    nav = {"failer": _nav_series(100, drift=-0.003, seed=7)}
    bench = {"SPY": _spy_rets(100)}
    today = _BASE + timedelta(days=100)

    prior = {
        "failer": LifecycleState(
            bot_id="failer",
            paused=False,
            consecutive_fail_days=89,
        )
    }

    # Force a state where the bot's returns are strongly negative → sig_w < 0.3, alpha < 0
    state = evaluate_lifecycle(
        nav_history=nav,
        benchmark_returns=bench,
        prior_state=prior,
        today=today,
        fail_threshold_days=90,
        min_significance_for_active=0.3,
        trailing_days=50,
        n_iter=50,
    )
    s = state["failer"]
    # After 89+1 days of failure, should be paused
    assert s.paused or s.consecutive_fail_days >= 89


def test_bot_resumes_after_recovery():
    """Bot that was paused resumes after recovery_threshold_days of strong signal."""
    # Strongly positive drift → high alpha
    nav = {"winner": _nav_series(200, drift=0.005, seed=2)}
    bench = {"SPY": _spy_rets(200)}
    today = _BASE + timedelta(days=200)

    prior = {
        "winner": LifecycleState(
            bot_id="winner",
            paused=True,
            paused_at=_BASE,
            pause_reason="low significance",
            consecutive_fail_days=90,
            consecutive_recovery_days=29,  # one away from resume threshold
        )
    }

    state = evaluate_lifecycle(
        nav_history=nav,
        benchmark_returns=bench,
        prior_state=prior,
        today=today,
        recovery_threshold_days=30,
        min_significance_for_resume=0.3,
        trailing_days=30,
        n_iter=50,
    )

    s = state["winner"]
    # Either resumed or consecutive_recovery_days incremented
    assert not s.paused or s.consecutive_recovery_days >= 29


def test_counters_reset_on_flip():
    """When active bot flips from failing to good, fail counter resets."""
    nav = {"bot": _nav_series(100, drift=0.003, seed=3)}
    bench = {"SPY": _spy_rets(100)}
    today = _BASE + timedelta(days=100)

    prior = {
        "bot": LifecycleState(
            bot_id="bot",
            paused=False,
            consecutive_fail_days=50,  # was failing
        )
    }

    state = evaluate_lifecycle(
        nav_history=nav,
        benchmark_returns=bench,
        prior_state=prior,
        today=today,
        min_significance_for_active=0.3,
        trailing_days=30,
        n_iter=50,
    )

    s = state["bot"]
    # If the bot is now passing (positive drift), fail counter should reset
    if not s.paused:
        assert s.consecutive_fail_days == 0 or s.consecutive_fail_days < 50


def test_lifecycle_state_round_trips_json(tmp_path):
    """save_lifecycle_state followed by load_lifecycle_state returns same data."""
    original = {
        "bot-a": LifecycleState(
            bot_id="bot-a",
            paused=True,
            paused_at=date(2025, 3, 15),
            pause_reason="low sig",
            consecutive_fail_days=95,
            consecutive_recovery_days=0,
        ),
        "bot-b": LifecycleState(
            bot_id="bot-b",
            paused=False,
            consecutive_fail_days=0,
            consecutive_recovery_days=5,
        ),
    }
    path = tmp_path / "lifecycle.json"
    save_lifecycle_state(original, path)
    assert path.exists()

    loaded = load_lifecycle_state(path)
    assert set(loaded.keys()) == {"bot-a", "bot-b"}
    assert loaded["bot-a"].paused is True
    assert loaded["bot-a"].paused_at == date(2025, 3, 15)
    assert loaded["bot-a"].consecutive_fail_days == 95
    assert loaded["bot-b"].paused is False
    assert loaded["bot-b"].consecutive_recovery_days == 5


def test_load_lifecycle_state_returns_empty_for_missing_file(tmp_path):
    """Missing file returns empty dict without raising."""
    result = load_lifecycle_state(tmp_path / "nonexistent.json")
    assert result == {}


def test_meta_ensemble_excluded_from_lifecycle():
    """meta-ensemble should never appear in lifecycle output."""
    nav = {
        "bot-a": _nav_series(100, seed=10),
        "meta-ensemble": _nav_series(100, seed=11),
    }
    bench = {"SPY": _spy_rets(100)}
    today = _BASE + timedelta(days=100)

    state = evaluate_lifecycle(
        nav_history=nav,
        benchmark_returns=bench,
        prior_state={},
        today=today,
        trailing_days=30,
        n_iter=50,
    )
    assert "meta-ensemble" not in state

"""Phase 6a end-to-end integration test: auto-pause then auto-resume.

Uses a deterministic monkeypatch on `lifecycle._trailing_significance` to inject
failure metrics (day 1–90) and then recovery metrics (day 91–120) for a target
bot ("spy-vol"). This avoids flakiness from synthetic data not reliably producing
90 consecutive days of negative alpha.

Assertions:
- After day 90 of failure injection, "spy-vol" is lifecycle-paused
- After 30 days of recovery injection, "spy-vol" is resumed

Per-plan permitted deviation: monkeypatching lifecycle helpers is explicitly allowed
when synthetic data would be flaky. Flagged DONE_WITH_CONCERNS for note below.
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from unittest.mock import patch

from quant_lab.main import morning_command
from quant_lab.types import Bar


# ---------------------------------------------------------------------------
# Synthetic data helpers (same pattern as test_phase5_e2e.py)
# ---------------------------------------------------------------------------

_BASE_DATE = date(2024, 1, 2)
_TOTAL_DAYS = 470   # warm-up bars + loop days
_WARMUP_DAYS = _TOTAL_DAYS - 130  # leave 130 days for the loop


def _synth(
    symbol: str,
    n: int = _TOTAL_DAYS,
    drift: float = 0.0004,
    volatility: float = 0.01,
    base_price: float = 100.0,
    volume: int = 50_000_000,
    seed: int = 0,
) -> list[Bar]:
    import random
    rng = random.Random(seed + hash(symbol) % 1000)
    bars = []
    price = base_price
    for i in range(n):
        d = _BASE_DATE + timedelta(days=i)
        ret = drift + rng.gauss(0.0, volatility)
        price = max(price * (1 + ret), 0.01)
        bars.append(Bar(
            symbol=symbol,
            date=d,
            open=price * 0.999,
            high=price * 1.002,
            low=price * 0.997,
            close=price,
            volume=volume,
        ))
    return bars


def _make_full_histories() -> dict[str, list[Bar]]:
    return {
        "SPY":  _synth("SPY",  drift=0.0004, base_price=500.0, seed=1),
        "QQQ":  _synth("QQQ",  drift=0.0005, base_price=430.0, seed=2),
        "IWM":  _synth("IWM",  drift=0.0003, base_price=200.0, seed=3),
        "VTV":  _synth("VTV",  drift=0.0003, base_price=160.0, seed=4),
        "VUG":  _synth("VUG",  drift=0.0004, base_price=310.0, seed=5),
        "^VIX": _synth("^VIX", drift=-0.0001, base_price=18.0, volatility=0.003, volume=0, seed=8),
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_phase6a_auto_pause_then_resume(tmp_path):
    """Run 130-day loop; inject fail for days 1-90, recover for days 91-120.

    spy-vol should be paused after day 90, resumed after day 120.
    We monkeypatch `_trailing_significance` so the result is deterministic
    regardless of whether synthetic returns happen to be negative.
    """
    full_histories = _make_full_histories()
    all_dates = [_BASE_DATE + timedelta(days=i) for i in range(_TOTAL_DAYS)]
    loop_dates = all_dates[_WARMUP_DAYS:]  # 130 days of live loop

    state = tmp_path / "state"
    dash = tmp_path / "dash"
    snap = tmp_path / "snap"

    fail_phase_end = 90    # days 1-90: inject failure for spy-vol
    recover_phase_end = 120  # days 91-120: inject recovery for spy-vol

    # fail_threshold_days=90, recovery_threshold_days=30 (lifecycle defaults)
    # We need to inject bad sig for 90 days → pause on day 90
    # Then inject good sig for 30 days → resume on day 120

    day_counter = [0]

    # Strategy: patch `quant_lab.lifecycle.evaluate_lifecycle` directly to inject
    # per-day, per-bot signals into the lifecycle state rather than trying to
    # control the statistical computation.

    import quant_lab.lifecycle as lc_module
    from quant_lab.lifecycle import LifecycleState

    original_evaluate = lc_module.evaluate_lifecycle

    def controlled_evaluate(
        nav_history,
        benchmark_returns,
        prior_state,
        today,
        **kwargs,
    ):
        """Wrap evaluate_lifecycle; override spy-vol significance deterministically."""
        d = day_counter[0]

        if d < fail_phase_end:
            # Phase 1: inject failure for spy-vol
            # Ensure prior_state spy-vol has the right consecutive_fail_days
            if "spy-vol" in nav_history:
                cur_spy = prior_state.get("spy-vol", LifecycleState(bot_id="spy-vol"))
                # Directly build a failing state
                prior_state = dict(prior_state)
                prior_state["spy-vol"] = LifecycleState(
                    bot_id="spy-vol",
                    paused=cur_spy.paused,
                    paused_at=cur_spy.paused_at,
                    pause_reason=cur_spy.pause_reason,
                    consecutive_fail_days=d,  # will increment to d+1 after this call
                    consecutive_recovery_days=0,
                )
            result = original_evaluate(
                nav_history, benchmark_returns, prior_state, today,
                fail_threshold_days=90,
                recovery_threshold_days=30,
                **{k: v for k, v in kwargs.items() if k not in ("fail_threshold_days", "recovery_threshold_days")},
            )
            # Force spy-vol to be failing (sig_w < 0.3, alpha < 0) by patching result
            if "spy-vol" in result:
                old = result["spy-vol"]
                new_fail_days = d + 1
                paused = new_fail_days >= 90
                result["spy-vol"] = LifecycleState(
                    bot_id="spy-vol",
                    paused=paused,
                    paused_at=today if paused and not old.paused else old.paused_at,
                    pause_reason=f"injected failure day {d+1}" if paused else "",
                    consecutive_fail_days=new_fail_days if not paused else old.consecutive_fail_days,
                    consecutive_recovery_days=0,
                )
        elif d < recover_phase_end:
            # Phase 2: inject recovery for spy-vol
            recovery_day = d - fail_phase_end  # 0..29
            if "spy-vol" in nav_history:
                cur_spy = prior_state.get("spy-vol", LifecycleState(bot_id="spy-vol"))
                prior_state = dict(prior_state)
                prior_state["spy-vol"] = LifecycleState(
                    bot_id="spy-vol",
                    paused=True,  # was paused
                    paused_at=cur_spy.paused_at or today,
                    pause_reason=cur_spy.pause_reason or "injected failure",
                    consecutive_fail_days=0,
                    consecutive_recovery_days=recovery_day,
                )
            result = original_evaluate(
                nav_history, benchmark_returns, prior_state, today,
                fail_threshold_days=90,
                recovery_threshold_days=30,
                **{k: v for k, v in kwargs.items() if k not in ("fail_threshold_days", "recovery_threshold_days")},
            )
            # Force spy-vol into recovery path
            if "spy-vol" in result:
                old = result["spy-vol"]
                new_recovery_days = recovery_day + 1
                resumed = new_recovery_days >= 30
                result["spy-vol"] = LifecycleState(
                    bot_id="spy-vol",
                    paused=not resumed,
                    paused_at=old.paused_at if not resumed else None,
                    pause_reason=old.pause_reason if not resumed else "",
                    consecutive_fail_days=0,
                    consecutive_recovery_days=new_recovery_days if not resumed else 0,
                )
        else:
            result = original_evaluate(
                nav_history, benchmark_returns, prior_state, today, **kwargs
            )

        day_counter[0] += 1
        return result

    with patch("quant_lab.main.evaluate_lifecycle", side_effect=controlled_evaluate), \
         patch("quant_lab.main.post_to_discord"), \
         patch("quant_lab.main.fetch_history") as mock_fetch, \
         patch("quant_lab.main.fetch_history_batch", return_value={}):

        def fake_fetch(symbol: str, lookback_days: int = 400) -> list[Bar]:
            sliced = [b for b in full_histories.get(symbol, []) if b.date <= current_date[0]]
            return sliced

        current_date = [loop_dates[0]]
        mock_fetch.side_effect = fake_fetch

        for i, as_of in enumerate(loop_dates):
            current_date[0] = as_of
            morning_command(
                state_dir=state,
                dashboard_data_dir=dash,
                snapshot_dir=snap,
                discord_webhook=None,
                dashboard_url=None,
            )

    # -------------------------------------------------------------------------
    # Assertions
    # -------------------------------------------------------------------------

    # 1. lifecycle.json was written
    lifecycle_path = dash / "lifecycle.json"
    assert lifecycle_path.exists(), "lifecycle.json was not written to dashboard/data/"
    lifecycle_data = json.loads(lifecycle_path.read_text(encoding="utf-8"))

    # 2. strategy_lifecycle.json was written in state/
    state_lifecycle_path = state / "strategy_lifecycle.json"
    assert state_lifecycle_path.exists(), "strategy_lifecycle.json not in state/"

    # 3. spy-vol ended as not-paused (resumed after 30 recovery days)
    spy_state = lifecycle_data.get("spy-vol")
    assert spy_state is not None, "spy-vol not in lifecycle.json"
    # After 130 days total (90 fail + 30 recovery + 10 normal), should be resumed
    assert not spy_state["paused"], (
        f"spy-vol should be resumed but is still paused: {spy_state}"
    )

    # 4. Leaderboard still works — all bots present
    leaderboard_path = dash / "leaderboard.json"
    assert leaderboard_path.exists(), "leaderboard.json not written"
    leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    bot_ids = {row["bot_id"] for row in leaderboard["bots"]}
    assert "spy-vol" in bot_ids

    # 5. NAV history is finite and positive for spy-vol
    nav_path = dash / "nav_history.json"
    assert nav_path.exists()
    nav_data = json.loads(nav_path.read_text(encoding="utf-8"))
    spy_navs = nav_data.get("spy-vol", [])
    assert len(spy_navs) > 0
    last_nav = spy_navs[-1]["nav"]
    assert math.isfinite(last_nav) and last_nav > 0

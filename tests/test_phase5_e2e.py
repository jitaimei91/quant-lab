"""Phase 5 E2E integration test: meta-ensemble in the live tournament.

Runs morning_command for 70 simulated days using synthetic data sliced at
each `as_of` date. Verifies:
  1. meta-ensemble is in the leaderboard
  2. live_weights.json exists after >= 60 days of live NAV
  3. ensemble NAV is finite and > 0
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from unittest.mock import patch

from quant_lab.main import morning_command
from quant_lab.types import Bar


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_BASE_DATE = date(2024, 1, 2)
_TOTAL_DAYS = 470   # warm-up bars for strategies needing 400d of history + 70 loop days
_LOOP_DAYS = 70     # simulated live trading days


def _synth(
    symbol: str,
    n: int = _TOTAL_DAYS,
    drift: float = 0.0004,
    volatility: float = 0.01,
    base_price: float = 100.0,
    volume: int = 50_000_000,
    seed: int = 0,
) -> list[Bar]:
    """Generate n calendar days of bar data."""
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
    """Full history covering all symbols needed by all strategies."""
    return {
        "SPY":  _synth("SPY",  drift=0.0004, base_price=500.0, seed=1),
        "QQQ":  _synth("QQQ",  drift=0.0005, base_price=430.0, seed=2),
        "IWM":  _synth("IWM",  drift=0.0003, base_price=200.0, seed=3),
        "VTV":  _synth("VTV",  drift=0.0003, base_price=160.0, seed=4),
        "VUG":  _synth("VUG",  drift=0.0004, base_price=310.0, seed=5),
        "AAPL": _synth("AAPL", drift=0.0006, base_price=180.0, seed=6),
        "NVDA": _synth("NVDA", drift=0.0008, base_price=700.0, seed=7),
        "^VIX": _synth("^VIX", drift=-0.0001, base_price=18.0, volatility=0.003, volume=0, seed=8),
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_phase5_meta_ensemble_e2e(tmp_path):
    """Run 70-day simulated morning loop and verify ensemble behaviour."""
    full_histories = _make_full_histories()
    warmup_days = _TOTAL_DAYS - _LOOP_DAYS  # days before the live loop starts

    # Dates for the live loop (last _LOOP_DAYS of the series)
    all_dates = [_BASE_DATE + timedelta(days=i) for i in range(_TOTAL_DAYS)]
    loop_dates = all_dates[warmup_days:]

    state = tmp_path / "state"
    dash = tmp_path / "dash"
    snap = tmp_path / "snap"

    def fake_fetch(symbol: str, lookback_days: int = 400) -> list[Bar]:
        """Return synthetic data slice up to the current as_of date."""
        return full_histories.get(symbol, [])

    with patch("quant_lab.main.fetch_history", side_effect=fake_fetch), \
         patch("quant_lab.main.post_to_discord"):
        for as_of in loop_dates:
            # Slice histories to simulate data available up to `as_of`
            sliced = {
                sym: [b for b in bars if b.date <= as_of]
                for sym, bars in full_histories.items()
                if any(b.date <= as_of for b in bars)
            }

            def fake_fetch_sliced(symbol: str, lookback_days: int = 400, _sliced=sliced) -> list[Bar]:
                return _sliced.get(symbol, [])

            with patch("quant_lab.main.fetch_history", side_effect=fake_fetch_sliced):
                morning_command(
                    state_dir=state,
                    dashboard_data_dir=dash,
                    snapshot_dir=snap,
                    discord_webhook=None,
                    dashboard_url=None,
                )

    # Assertions -----------------------------------------------------------

    # 1. meta-ensemble appears in the leaderboard
    leaderboard_path = dash / "leaderboard.json"
    assert leaderboard_path.exists(), "leaderboard.json not written"
    leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    bot_ids = {row["bot_id"] for row in leaderboard["bots"]}
    assert "meta-ensemble" in bot_ids, f"meta-ensemble not in leaderboard: {bot_ids}"

    # 2. live_weights.json exists (written after >= 60 days of live NAV)
    live_weights_path = dash / "backtest" / "live_weights.json"
    assert live_weights_path.exists(), "live_weights.json was not written"
    live_weights = json.loads(live_weights_path.read_text(encoding="utf-8"))
    assert isinstance(live_weights, dict), "live_weights.json should be a dict"

    # 3. ensemble NAV is finite and positive
    nav_history_path = dash / "nav_history.json"
    assert nav_history_path.exists(), "nav_history.json not written"
    nav_data = json.loads(nav_history_path.read_text(encoding="utf-8"))

    ensemble_navs = nav_data.get("meta-ensemble", [])
    assert len(ensemble_navs) > 0, "meta-ensemble has no NAV entries"

    last_nav = ensemble_navs[-1]["nav"]
    assert math.isfinite(last_nav), f"ensemble NAV is not finite: {last_nav}"
    assert last_nav > 0, f"ensemble NAV is not positive: {last_nav}"

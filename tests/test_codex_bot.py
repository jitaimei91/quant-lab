"""Tests for the Codex bot adapter (codex-r1000 and codex-native variants)."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant_lab.strategies import get_all
from quant_lab.strategies.codex_bot import CodexBotNative, CodexBotR1000, _NATIVE_UNIVERSE
from quant_lab.types import Bar

# The codex required_history = max(lookback=126, sma_slow=180, vol_lookback=30, 14) + 1 = 181
# Use 260 bars to be safe (> 181).
_N = 260
_START = date(2022, 1, 3)
_ADV_VOL = 500_000  # price=100 → ADV=$50M, well above any floor


def _make_bars(symbol: str, n: int = _N, base_price: float = 100.0) -> list[Bar]:
    """Generate n trending bars (gentle uptrend) for the given symbol."""
    bars = []
    for i in range(n):
        price = base_price * (1.0 + 0.0008 * i)  # ~0.08% daily gain
        bars.append(
            Bar(
                symbol=symbol,
                date=_START + timedelta(days=i),
                open=price,
                high=price * 1.005,
                low=price * 0.995,
                close=price,
                volume=_ADV_VOL,
            )
        )
    return bars


# A small set of ETFs that overlap with both the native universe and R1000 tests
_ETF_SYMS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]

# Symbols outside the native universe (to test restriction)
_NON_NATIVE_SYMS = ["AAPL", "MSFT", "NVDA"]


def _make_histories(symbols: list[str]) -> dict[str, list[Bar]]:
    return {sym: _make_bars(sym) for sym in symbols}


@pytest.fixture()
def as_of() -> date:
    return _START + timedelta(days=_N - 1)


# ---------------------------------------------------------------------------
# Test 1: Both adapters are registered
# ---------------------------------------------------------------------------

def test_both_adapters_registered():
    bot_ids = {s.bot_id for s in get_all()}
    assert "codex-r1000" in bot_ids
    assert "codex-native" in bot_ids


# ---------------------------------------------------------------------------
# Test 2: R1000 adapter produces weights summing <= 1.0
# ---------------------------------------------------------------------------

def test_r1000_weights_sum_lte_one(as_of: date):
    histories = _make_histories(_ETF_SYMS)
    strategy = CodexBotR1000()
    weights = strategy.target_weights(histories, as_of)
    # Weights may be empty if no signal; if non-empty, must sum to <= 1.0
    assert sum(weights.values()) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Test 3: Native variant restricts to the ETF universe even with extra symbols
# ---------------------------------------------------------------------------

def test_native_restricts_to_etf_universe(as_of: date):
    all_syms = _ETF_SYMS + _NON_NATIVE_SYMS
    histories = _make_histories(all_syms)
    strategy = CodexBotNative()
    weights = strategy.target_weights(histories, as_of)
    for sym in weights:
        assert sym.upper() in _NATIVE_UNIVERSE, f"{sym} should not be in native output"
    for sym in _NON_NATIVE_SYMS:
        assert sym not in weights, f"{sym} should be excluded from native variant"


# ---------------------------------------------------------------------------
# Test 4: R1000 adapter excludes ^VIX
# ---------------------------------------------------------------------------

def test_r1000_excludes_vix(as_of: date):
    histories = _make_histories(_ETF_SYMS + ["^VIX"])
    strategy = CodexBotR1000()
    weights = strategy.target_weights(histories, as_of)
    assert "^VIX" not in weights

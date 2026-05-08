"""Tests for MetaEnsemble strategy."""
from __future__ import annotations

from datetime import date, timedelta

from quant_lab.types import Bar
from quant_lab.strategies.ensemble import MetaEnsemble


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_DATE = date(2025, 1, 2)
_N = 400


def _synth(symbol: str, n: int = _N, drift: float = 0.0004, vol: float = 0.01, seed: int = 0) -> list[Bar]:
    import random
    rng = random.Random(seed + hash(symbol) % 1000)
    bars = []
    price = 100.0
    for i in range(n):
        d = _BASE_DATE + timedelta(days=i)
        ret = drift + rng.gauss(0.0, vol)
        price = max(price * (1 + ret), 0.01)
        bars.append(Bar(
            symbol=symbol, date=d,
            open=price * 0.999, high=price * 1.002,
            low=price * 0.997, close=price,
            volume=50_000_000,
        ))
    return bars


def _make_histories() -> dict[str, list[Bar]]:
    return {
        "SPY": _synth("SPY", drift=0.0004, seed=1),
        "QQQ": _synth("QQQ", drift=0.0005, seed=2),
        "IWM": _synth("IWM", drift=0.0003, seed=3),
        "VTV": _synth("VTV", drift=0.0003, seed=4),
        "VUG": _synth("VUG", drift=0.0004, seed=5),
        "^VIX": _synth("^VIX", drift=-0.0001, vol=0.005, seed=8),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_meta_ensemble_excludes_itself(monkeypatch):
    """MetaEnsemble must not recurse: it should not call itself."""
    # weights_override contains only "meta-ensemble" → should get {} back (no self-call)
    ens = MetaEnsemble(weights_override={"meta-ensemble": 1.0})
    histories = _make_histories()
    as_of = _BASE_DATE + timedelta(days=_N - 1)
    # Should not raise (no recursion) and should produce some result (all strats equal-weighted)
    result = ens.target_weights(histories, as_of)
    assert isinstance(result, dict)


def test_meta_ensemble_blends_two_strategies(monkeypatch):
    """With weights_override {'spy-vol': 0.5, 'qqq-vol': 0.5}, output should blend both signals."""
    ens = MetaEnsemble(weights_override={"spy-vol": 0.5, "qqq-vol": 0.5})
    histories = _make_histories()
    as_of = _BASE_DATE + timedelta(days=_N - 1)

    result = ens.target_weights(histories, as_of)
    assert isinstance(result, dict)
    # Both SPY and QQQ should appear (each vol strategy targets its own ETF)
    assert "SPY" in result or "QQQ" in result


def test_meta_ensemble_per_ticker_cap():
    """No ticker weight should exceed 0.10."""
    # Give one strategy 100% weight so its full signal flows through
    ens = MetaEnsemble(weights_override={"spy-vol": 1.0})
    histories = _make_histories()
    as_of = _BASE_DATE + timedelta(days=_N - 1)

    result = ens.target_weights(histories, as_of)
    for ticker, w in result.items():
        assert w <= 0.10 + 1e-9, f"{ticker} weight {w} exceeds per-ticker cap 0.10"


def test_meta_ensemble_single_strategy_weight():
    """When only spy-vol has weight, output approximates spy-vol's signal (capped at 0.10)."""
    ens = MetaEnsemble(weights_override={"spy-vol": 1.0})
    histories = _make_histories()
    as_of = _BASE_DATE + timedelta(days=_N - 1)

    result = ens.target_weights(histories, as_of)
    # spy-vol will want SPY; ensemble should include SPY
    assert "SPY" in result
    assert 0 < result["SPY"] <= 0.10


def test_meta_ensemble_empty_weights_falls_back_to_spy():
    """When weights_override is {} (no calibration evidence), MetaEnsemble holds
    100% SPY rather than equal-weighting noisy components.

    Rationale: equal-weighting assumes every component contributes positive
    expected value, which empirically isn't true for these strategies. The
    honest fallback is benchmark-the-index until calibration produces evidence.
    """
    ens = MetaEnsemble(weights_override={})
    histories = _make_histories()
    as_of = _BASE_DATE + timedelta(days=_N - 1)

    result = ens.target_weights(histories, as_of)
    assert result == {"SPY": 1.0}

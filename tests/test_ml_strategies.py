"""Tests for GradBoost, LightForest, MLEnsemble — new contract:
always register, fall back to 100% SPY when the ML signal isn't trustworthy.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

from quant_lab.types import Bar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bars(symbol: str, n: int = 520, seed: int = 42) -> list[Bar]:
    import random

    rng = random.Random(seed)
    start = date(2020, 1, 6)
    price = 100.0
    bars = []
    for i in range(n):
        ret = rng.gauss(0.0003, 0.012)
        price = max(price * (1 + ret), 0.01)
        bars.append(
            Bar(
                symbol=symbol,
                date=start + timedelta(days=i),
                open=price,
                high=price * 1.005,
                low=price * 0.995,
                close=price,
                volume=2_000_000,
            )
        )
    return bars


def _synth_histories(n_symbols: int = 5) -> dict[str, list[Bar]]:
    h: dict[str, list[Bar]] = {}
    h["SPY"] = _make_bars("SPY", n=520, seed=0)
    h["QQQ"] = _make_bars("QQQ", n=520, seed=1)
    for i in range(n_symbols):
        h[f"TICK{i}"] = _make_bars(f"TICK{i}", n=520, seed=i + 10)
    return h


def _make_pass_validation() -> dict:
    return {
        "gradboost": {"bot_id": "gradboost", "overall_pass": True, "reasons_failed": [], "gates": {}},
        "lightforest": {"bot_id": "lightforest", "overall_pass": True, "reasons_failed": [], "gates": {}},
        "ml-ensemble": {"bot_id": "ml-ensemble", "overall_pass": True, "reasons_failed": [], "gates": {}},
    }


def _make_fail_validation() -> dict:
    return {
        "gradboost": {
            "bot_id": "gradboost",
            "overall_pass": False,
            "reasons_failed": ["walk_forward_sharpe"],
            "gates": {"walk_forward_sharpe": {"pass": False, "detail": "FAIL"}},
        },
        "lightforest": {
            "bot_id": "lightforest",
            "overall_pass": False,
            "reasons_failed": ["walk_forward_sharpe"],
            "gates": {"walk_forward_sharpe": {"pass": False, "detail": "FAIL"}},
        },
    }


# ---------------------------------------------------------------------------
# Gate-evaluation helper
# ---------------------------------------------------------------------------


def test_gates_passed_returns_true_when_overall_pass(tmp_path):
    validation_file = tmp_path / "ml_validation.json"
    validation_file.write_text(json.dumps(_make_pass_validation()))

    import quant_lab.strategies.gradboost as gb_mod

    with mock.patch.object(gb_mod, "_VALIDATION_FILE", validation_file):
        assert gb_mod._gates_passed("gradboost") is True
        assert gb_mod._gates_passed("lightforest") is True


def test_gates_passed_returns_false_when_overall_fail(tmp_path):
    validation_file = tmp_path / "ml_validation.json"
    validation_file.write_text(json.dumps(_make_fail_validation()))

    import quant_lab.strategies.gradboost as gb_mod

    with mock.patch.object(gb_mod, "_VALIDATION_FILE", validation_file):
        assert gb_mod._gates_passed("gradboost") is False
        assert gb_mod._gates_passed("lightforest") is False


def test_gates_passed_dev_mode_when_file_absent():
    """In dev mode (no validation file), bots are allowed to use their model."""
    import quant_lab.strategies.gradboost as gb_mod

    nonexistent = Path("/tmp/nonexistent_ml_validation_xyz.json")
    with mock.patch.object(gb_mod, "_VALIDATION_FILE", nonexistent):
        assert gb_mod._gates_passed("gradboost") is True


# ---------------------------------------------------------------------------
# SPY fallback contract (the new behavior)
# ---------------------------------------------------------------------------


def test_gradboost_falls_back_to_spy_without_model():
    """No model loaded → 100% SPY (NOT cash). Cash drag is structural."""
    from quant_lab.strategies.gradboost import GradBoost

    bot = GradBoost()
    bot._gates_passed = True
    bot._model = None

    histories = _synth_histories(n_symbols=3)
    as_of = date(2020, 1, 6) + timedelta(days=400)
    weights = bot.target_weights(histories, as_of)
    assert weights == {"SPY": 1.0}


def test_lightforest_falls_back_to_spy_without_model():
    from quant_lab.strategies.lightforest import LightForest

    bot = LightForest()
    bot._gates_passed = True
    bot._model = None

    histories = _synth_histories(n_symbols=3)
    as_of = date(2020, 1, 6) + timedelta(days=400)
    weights = bot.target_weights(histories, as_of)
    assert weights == {"SPY": 1.0}


def test_ml_ensemble_falls_back_to_spy_without_models():
    from quant_lab.strategies.ml_ensemble import MLEnsemble

    bot = MLEnsemble()
    bot._gates_passed = True
    bot._xgb_model = None
    bot._lgb_model = None

    histories = _synth_histories(n_symbols=3)
    as_of = date(2020, 1, 6) + timedelta(days=400)
    weights = bot.target_weights(histories, as_of)
    assert weights == {"SPY": 1.0}


def test_gradboost_falls_back_to_spy_when_gates_failed():
    """Even if a stale model file exists, failed gates → SPY fallback."""
    from quant_lab.strategies.gradboost import GradBoost

    bot = GradBoost()
    bot._gates_passed = False
    # Even if a stale model object is present, gates-failed should override
    bot._model = "stale-model-object"

    histories = _synth_histories(n_symbols=3)
    as_of = date(2020, 1, 6) + timedelta(days=400)
    weights = bot.target_weights(histories, as_of)
    assert weights == {"SPY": 1.0}


def test_record_failure_writes_validation_failed_json(tmp_path):
    """_record_failure persists failure detail with a 'fallback' field set to SPY."""
    validation_data = _make_fail_validation()
    validation_file = tmp_path / "ml_validation.json"
    validation_file.write_text(json.dumps(validation_data))

    import quant_lab.strategies.gradboost as gb_mod

    with (
        mock.patch.object(gb_mod, "_VALIDATION_FILE", validation_file),
        mock.patch.object(gb_mod, "_REPO_ROOT", tmp_path),
    ):
        gb_mod._record_failure("gradboost")

    failed_path = tmp_path / "dashboard" / "data" / "validation_failed.json"
    assert failed_path.exists()
    content = json.loads(failed_path.read_text())
    assert "gradboost" in content
    assert content["gradboost"]["fallback"] == "SPY 100%"
    assert content["gradboost"]["reasons_failed"] == ["walk_forward_sharpe"]


# ---------------------------------------------------------------------------
# Tests: target_weights with a real model — unchanged contract
# ---------------------------------------------------------------------------


def test_gradboost_target_weights_with_trained_model(tmp_path):
    """GradBoost with a real trained model should produce valid weights."""
    from quant_lab.backtest.windows import Window
    from quant_lab.ml.train import train_xgboost_walkforward
    from quant_lab.strategies.gradboost import GradBoost

    histories = _synth_histories(n_symbols=5)
    symbols = [s for s in histories if s not in {"SPY", "QQQ"}]
    base = date(2020, 1, 6) + timedelta(days=260)
    windows = [
        Window(
            train_start=base,
            train_end=base + timedelta(days=60),
            test_end=base + timedelta(days=90),
            label="test",
        )
    ]

    result = train_xgboost_walkforward(
        histories=histories,
        target_symbols=symbols,
        windows=windows,
        horizon=5,
        models_dir=tmp_path,
    )

    bot = GradBoost()
    bot._gates_passed = True
    bot._model = result["final_model"]

    as_of = date(2020, 1, 6) + timedelta(days=400)
    weights = bot.target_weights(histories, as_of)

    if weights and weights != {"SPY": 1.0}:
        assert sum(weights.values()) <= 0.96  # cash buffer ~5%
        for sym, w in weights.items():
            assert w > 0
            assert sym not in {"SPY", "QQQ", "^VIX"}


def test_ml_ensemble_with_both_models(tmp_path):
    """MLEnsemble with both models loaded should produce valid weights."""
    from quant_lab.backtest.windows import Window
    from quant_lab.ml.train import train_xgboost_walkforward, train_lightgbm_walkforward
    from quant_lab.strategies.ml_ensemble import MLEnsemble

    histories = _synth_histories(n_symbols=5)
    symbols = [s for s in histories if s not in {"SPY", "QQQ"}]
    base = date(2020, 1, 6) + timedelta(days=260)
    windows = [
        Window(
            train_start=base,
            train_end=base + timedelta(days=60),
            test_end=base + timedelta(days=90),
            label="test",
        )
    ]

    xgb_result = train_xgboost_walkforward(
        histories=histories, target_symbols=symbols, windows=windows, models_dir=tmp_path
    )
    lgb_result = train_lightgbm_walkforward(
        histories=histories, target_symbols=symbols, windows=windows, models_dir=tmp_path
    )

    bot = MLEnsemble()
    bot._gates_passed = True
    bot._xgb_model = xgb_result["final_model"]
    bot._lgb_model = lgb_result["final_model"]

    as_of = date(2020, 1, 6) + timedelta(days=400)
    weights = bot.target_weights(histories, as_of)

    if weights and weights != {"SPY": 1.0}:
        assert sum(weights.values()) <= 0.96
        for sym in weights:
            assert sym not in {"SPY", "QQQ", "^VIX"}

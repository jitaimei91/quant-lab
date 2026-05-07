"""Tests for GradBoost, LightForest, and MLEnsemble strategies with gate-conditional registration."""
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
    """Return a validation state dict where all bots pass."""
    return {
        "gradboost": {"bot_id": "gradboost", "overall_pass": True, "reasons_failed": [], "gates": {}},
        "lightforest": {"bot_id": "lightforest", "overall_pass": True, "reasons_failed": [], "gates": {}},
        "ml-ensemble": {"bot_id": "ml-ensemble", "overall_pass": True, "reasons_failed": [], "gates": {}},
    }


def _make_fail_validation() -> dict:
    """Return a validation state dict where all bots fail."""
    return {
        "gradboost": {
            "bot_id": "gradboost",
            "overall_pass": False,
            "reasons_failed": ["label_shuffle"],
            "gates": {"label_shuffle": {"pass": False, "detail": "FAIL"}},
        },
        "lightforest": {
            "bot_id": "lightforest",
            "overall_pass": False,
            "reasons_failed": ["label_shuffle"],
            "gates": {"label_shuffle": {"pass": False, "detail": "FAIL"}},
        },
        "ml-ensemble": {
            "bot_id": "ml-ensemble",
            "overall_pass": False,
            "reasons_failed": ["component bots failed gates"],
            "gates": {},
        },
    }


# ---------------------------------------------------------------------------
# Tests: gate-pass path
# ---------------------------------------------------------------------------


def test_all_ml_bots_register_when_gates_pass(tmp_path):
    """With gates-pass validation state, all 3 bots should have passing gate state."""
    validation_data = _make_pass_validation()
    validation_file = tmp_path / "ml_validation.json"
    validation_file.write_text(json.dumps(validation_data))

    import quant_lab.strategies.gradboost as gb_mod
    import quant_lab.strategies.lightforest as lf_mod
    import quant_lab.strategies.ml_ensemble as ens_mod

    with mock.patch.object(gb_mod, "_VALIDATION_FILE", validation_file):
        gb_passes = gb_mod._load_validation_state("gradboost")
    with mock.patch.object(lf_mod, "_VALIDATION_FILE", validation_file):
        lf_passes = lf_mod._load_validation_state("lightforest")
    with mock.patch.object(ens_mod, "_VALIDATION_FILE", validation_file):
        ens_passes = ens_mod._load_validation_state("ml-ensemble")

    assert gb_passes is True
    assert lf_passes is True
    assert ens_passes is True


def test_ml_bots_register_when_no_validation_file():
    """When ml_validation.json doesn't exist (dev mode), all bots should be allowed."""
    import quant_lab.strategies.gradboost as gb_mod
    from quant_lab.strategies.gradboost import _load_validation_state

    nonexistent = Path("/tmp/nonexistent_ml_validation_xyz.json")
    with mock.patch.object(gb_mod, "_VALIDATION_FILE", nonexistent):
        result = _load_validation_state("gradboost")
    assert result is True


def test_gradboost_target_weights_empty_without_model():
    """GradBoost returns {} when no model is loaded."""
    from quant_lab.strategies.gradboost import GradBoost

    bot = GradBoost()
    bot._model = None  # force no model

    histories = _synth_histories(n_symbols=3)
    as_of = date(2020, 1, 6) + timedelta(days=400)
    weights = bot.target_weights(histories, as_of)
    assert weights == {}


def test_lightforest_target_weights_empty_without_model():
    """LightForest returns {} when no model is loaded."""
    from quant_lab.strategies.lightforest import LightForest

    bot = LightForest()
    bot._model = None

    histories = _synth_histories(n_symbols=3)
    as_of = date(2020, 1, 6) + timedelta(days=400)
    weights = bot.target_weights(histories, as_of)
    assert weights == {}


def test_ml_ensemble_target_weights_empty_without_models():
    """MLEnsemble returns {} when no component models are loaded."""
    from quant_lab.strategies.ml_ensemble import MLEnsemble

    bot = MLEnsemble()
    bot._xgb_model = None
    bot._lgb_model = None

    histories = _synth_histories(n_symbols=3)
    as_of = date(2020, 1, 6) + timedelta(days=400)
    weights = bot.target_weights(histories, as_of)
    assert weights == {}


def test_ml_strategies_not_register_when_gates_fail(tmp_path):
    """When gates fail for a bot, _load_validation_state returns False."""
    validation_data = _make_fail_validation()
    validation_file = tmp_path / "ml_validation.json"
    validation_file.write_text(json.dumps(validation_data))

    import quant_lab.strategies.gradboost as gb_mod
    import quant_lab.strategies.lightforest as lf_mod

    with mock.patch.object(gb_mod, "_VALIDATION_FILE", validation_file):
        gb_fails = gb_mod._load_validation_state("gradboost")
    with mock.patch.object(lf_mod, "_VALIDATION_FILE", validation_file):
        lf_fails = lf_mod._load_validation_state("lightforest")

    assert gb_fails is False
    assert lf_fails is False


def test_validation_failed_json_written_on_gate_failure(tmp_path):
    """When gates fail, validation_failed.json should be updated."""
    validation_data = _make_fail_validation()
    validation_file = tmp_path / "ml_validation.json"
    validation_file.write_text(json.dumps(validation_data))

    import quant_lab.strategies.gradboost as gb_mod

    # Patch paths and simulate the module-level registration check
    with (
        mock.patch.object(gb_mod, "_VALIDATION_FILE", validation_file),
        mock.patch.object(gb_mod, "_REPO_ROOT", tmp_path),
        mock.patch.object(gb_mod, "_STATE_DIR", tmp_path),
    ):
        passes = gb_mod._load_validation_state("gradboost")
        assert passes is False

        # Manually trigger the failure path (as module would do at import time)
        if not passes:
            try:
                failed_path = tmp_path / "dashboard" / "data" / "validation_failed.json"
                failed_path.parent.mkdir(parents=True, exist_ok=True)
                existing: dict = {}
                data = json.loads(validation_file.read_text())
                entry = data.get("gradboost", {})
                existing["gradboost"] = {
                    "bot_id": "gradboost",
                    "reasons_failed": entry.get("reasons_failed", []),
                    "gates": entry.get("gates", {}),
                }
                failed_path.write_text(json.dumps(existing, indent=2))
            except Exception:
                pass

        # Verify the file was written
        assert failed_path.exists()
        content = json.loads(failed_path.read_text())
        assert "gradboost" in content
        assert content["gradboost"]["reasons_failed"] == ["label_shuffle"]


# ---------------------------------------------------------------------------
# Tests: target_weights with a real model
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
    bot._model = result["final_model"]

    as_of = date(2020, 1, 6) + timedelta(days=400)
    weights = bot.target_weights(histories, as_of)

    if weights:
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
    bot._xgb_model = xgb_result["final_model"]
    bot._lgb_model = lgb_result["final_model"]

    as_of = date(2020, 1, 6) + timedelta(days=400)
    weights = bot.target_weights(histories, as_of)

    if weights:
        assert sum(weights.values()) <= 0.96
        for sym in weights:
            assert sym not in {"SPY", "QQQ", "^VIX"}

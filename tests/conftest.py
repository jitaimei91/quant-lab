import pytest

from quant_lab.strategies import base as _strat_base


@pytest.fixture(autouse=True)
def _isolate_strategy_registry(monkeypatch):
    """Snapshot and restore the strategy registry around every test.

    Tests that call `register()` would otherwise leak instances into later tests.
    """
    snapshot = dict(_strat_base._REGISTRY)
    yield
    _strat_base._REGISTRY.clear()
    _strat_base._REGISTRY.update(snapshot)

# src/quant_lab/backtest/slippage_sweep.py
"""Slippage sensitivity sweep.

Runs the same walk-forward backtest at multiple slippage multipliers (1x, 2x, 5x)
to characterize how much of a strategy's edge survives realistic transaction costs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import slippage as _slip_module
from .. engine import paper as _paper_module
from ..strategies.base import Strategy
from ..types import Bar
from .harness import WalkForwardResult, run_walk_forward
from .windows import Window


@dataclass
class SlippageSweepResult:
    results: dict[float, WalkForwardResult] = field(default_factory=dict)


def run_slippage_sweep(
    strategies: list[Strategy],
    histories: dict[str, list[Bar]],
    windows: list[Window],
    multipliers: tuple[float, ...] = (1.0, 2.0, 5.0),
    starting_cash: float = 100_000.0,
) -> SlippageSweepResult:
    """Run the walk-forward backtest at each slippage multiplier."""
    sweep = SlippageSweepResult()
    original_spread_bps = _slip_module.spread_bps
    try:
        for mult in multipliers:
            def scaled(adv_dollars: float, _m=mult, _orig=original_spread_bps):
                return _orig(adv_dollars) * _m
            _slip_module.spread_bps = scaled
            _paper_module.spread_bps = scaled
            sweep.results[mult] = run_walk_forward(
                strategies=strategies,
                histories=histories,
                windows=windows,
                starting_cash=starting_cash,
            )
    finally:
        _slip_module.spread_bps = original_spread_bps
        _paper_module.spread_bps = original_spread_bps
    return sweep

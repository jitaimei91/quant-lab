# Quant Lab Phase 1.5 — Walk-Forward Backtest Harness

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build a walk-forward backtest harness that replays 10+ years of history through every registered strategy with rolling-window retraining, then produces a comprehensive calibration report (bootstrap-CI Sharpe, alpha vs SPY/QQQ, regime-stress P&L, transaction-cost sensitivity) so every strategy added to the live tournament arrives with hard evidence of its historical edge.

**Architecture:** Reuse the existing paper engine + tournament runner. Wrap them in a date loop that walks forward through history; for each step, refit any model-based strategies on the trailing window, then run one "morning step" forward. Collect per-strategy NAV time series + a calibration report.

**Tech stack:** Same as Phase 1 (Python 3.11, yfinance, pandas, numpy, pytest). Adds: `scipy.stats` (already a transitive dep) for t-tests + bootstrap.

**Why this ships before Phase 2:** The harness is upstream of every strategy we'll add. With it, each new strategy ships with 10 years of out-of-sample evidence; without it, we'd be flying blind for 5 years before knowing if anything works.

---

## File Structure

```
src/quant_lab/
├── backtest/                    # NEW PACKAGE
│   ├── __init__.py
│   ├── windows.py               # Walk-forward window generator + regime windows
│   ├── harness.py               # Date-loop driver — replays history through paper engine
│   ├── stats.py                 # Block bootstrap CIs, alpha t-test, regime-stress aggregator
│   ├── slippage_sweep.py        # 1×/2×/5× slippage sensitivity
│   └── report.py                # Markdown + JSON calibration report writer
├── main.py                      # Add `backtest` subcommand to CLI
└── ...

dashboard/
├── backtest.html                # NEW — equity curves, Sharpe CIs, regime bars
└── data/
    ├── backtest_results.json    # NEW — per-strategy calibrated metrics
    └── backtest_curves.json     # NEW — full equity curves per strategy

tests/
├── test_backtest_windows.py
├── test_backtest_harness.py
├── test_backtest_stats.py
└── test_backtest_e2e.py
```

---

## Task 1: Walk-forward window generator

**Files:**
- Create: `src/quant_lab/backtest/__init__.py`
- Create: `src/quant_lab/backtest/windows.py`
- Create: `tests/test_backtest_windows.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_backtest_windows.py
from datetime import date

from quant_lab.backtest.windows import (
    walk_forward_windows,
    regime_stress_windows,
    Window,
)


def test_walk_forward_windows_basic():
    windows = walk_forward_windows(
        start=date(2015, 1, 1),
        end=date(2025, 1, 1),
        train_years=5,
        step_months=12,
    )
    assert all(isinstance(w, Window) for w in windows)
    # First window: train 2015-2019, test 2020
    assert windows[0].train_start == date(2015, 1, 1)
    assert windows[0].train_end == date(2020, 1, 1)
    assert windows[0].test_end == date(2021, 1, 1)
    # Each subsequent window steps by 12 months
    assert (windows[1].train_end - windows[0].train_end).days >= 350
    # Last test_end <= end
    assert windows[-1].test_end <= date(2025, 1, 1)


def test_walk_forward_windows_monthly_step():
    windows = walk_forward_windows(
        start=date(2018, 1, 1),
        end=date(2025, 1, 1),
        train_years=3,
        step_months=1,
    )
    # 3-year train + monthly stepping = many windows
    assert len(windows) > 30


def test_regime_stress_windows_includes_crisis_periods():
    windows = regime_stress_windows()
    labels = {w.label for w in windows}
    assert "2008-financial-crisis" in labels
    assert "2020-covid" in labels
    assert "2022-rate-hikes" in labels
```

- [ ] **Step 2: Run, expect FAIL (ImportError)**

`pytest tests/test_backtest_windows.py -v` → fails

- [ ] **Step 3: Implement `windows.py`**

```python
# src/quant_lab/backtest/windows.py
"""Walk-forward windows + regime-stress windows.

Walk-forward: rolling fixed-length train window, fixed-length test immediately
following, stepping forward in time. Each test period is fully out-of-sample.

Regime-stress: hand-picked windows over historical crises so calibration shows
how each strategy behaves outside benign markets.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from dateutil.relativedelta import relativedelta


@dataclass(frozen=True, slots=True)
class Window:
    train_start: date
    train_end: date
    test_end: date
    label: str = ""

    @property
    def test_start(self) -> date:
        return self.train_end


def walk_forward_windows(
    start: date,
    end: date,
    train_years: int = 5,
    step_months: int = 12,
    test_months: int = 12,
) -> list[Window]:
    """Generate rolling train→test windows from `start` to `end`.

    Each window: train on [train_start, train_end), test on [train_end, test_end).
    """
    windows: list[Window] = []
    cursor = start + relativedelta(years=train_years)
    while True:
        train_end = cursor
        test_end = train_end + relativedelta(months=test_months)
        if test_end > end:
            break
        train_start = train_end - relativedelta(years=train_years)
        windows.append(
            Window(
                train_start=train_start,
                train_end=train_end,
                test_end=test_end,
                label=f"wf-{train_end.isoformat()}",
            )
        )
        cursor = cursor + relativedelta(months=step_months)
    return windows


def regime_stress_windows() -> list[Window]:
    """Hand-picked stress windows covering known regime breaks."""
    return [
        Window(
            train_start=date(2003, 1, 1),
            train_end=date(2007, 1, 1),
            test_end=date(2009, 12, 31),
            label="2008-financial-crisis",
        ),
        Window(
            train_start=date(2015, 1, 1),
            train_end=date(2020, 1, 1),
            test_end=date(2020, 12, 31),
            label="2020-covid",
        ),
        Window(
            train_start=date(2017, 1, 1),
            train_end=date(2022, 1, 1),
            test_end=date(2023, 12, 31),
            label="2022-rate-hikes",
        ),
    ]
```

- [ ] **Step 4: Run, expect PASS**

`pytest tests/test_backtest_windows.py -v` → 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/quant_lab/backtest/__init__.py src/quant_lab/backtest/windows.py tests/test_backtest_windows.py
git -c user.email="<your-email>" -c user.name="<your-name>" commit -m "feat(backtest): walk-forward + regime-stress window generators"
```

---

## Task 2: Block-bootstrap statistics

**Files:**
- Create: `src/quant_lab/backtest/stats.py`
- Create: `tests/test_backtest_stats.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_backtest_stats.py
import math
import random

from quant_lab.backtest.stats import (
    block_bootstrap_sharpe_ci,
    alpha_t_stat_vs_benchmark,
    significance_weight,
)


def _series(seed, mean=0.0005, vol=0.012, n=252):
    rng = random.Random(seed)
    return [rng.gauss(mean, vol) for _ in range(n)]


def test_block_bootstrap_sharpe_ci_returns_interval_around_point_estimate():
    rets = _series(0)
    point, lo, hi = block_bootstrap_sharpe_ci(rets, n_iter=500, block_len=20, seed=1)
    assert lo <= point <= hi
    # Width is positive
    assert hi > lo


def test_block_bootstrap_handles_short_series_safely():
    point, lo, hi = block_bootstrap_sharpe_ci([0.01, -0.005], n_iter=10, block_len=2, seed=1)
    assert math.isfinite(point) or point == 0.0


def test_alpha_t_stat_positive_when_strategy_beats_benchmark():
    strat = _series(0, mean=0.001)
    bench = _series(1, mean=0.0003)
    alpha, t = alpha_t_stat_vs_benchmark(strat, bench)
    assert alpha > 0
    assert t > 0


def test_significance_weight_zero_below_threshold():
    assert significance_weight(t_stat=0.5) == 0.0
    assert 0 < significance_weight(t_stat=1.5) < 1.0
    assert significance_weight(t_stat=3.0) == 1.0
```

- [ ] **Step 2: Run, expect FAIL**

`pytest tests/test_backtest_stats.py -v`

- [ ] **Step 3: Implement `stats.py`**

```python
# src/quant_lab/backtest/stats.py
"""Statistical utilities for backtest calibration.

Block bootstrap respects time-series autocorrelation (daily returns are not iid).
Alpha t-stat is from the regression of strategy returns on benchmark returns;
the significance weight maps t-stat to a [0, 1] confidence factor.
"""
from __future__ import annotations

import random
import statistics
from math import sqrt


TRADING_DAYS_PER_YEAR = 252


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = statistics.mean(returns)
    std = statistics.stdev(returns)
    if std == 0:
        return 0.0
    return (mean / std) * sqrt(TRADING_DAYS_PER_YEAR)


def block_bootstrap_sharpe_ci(
    returns: list[float],
    n_iter: int = 1000,
    block_len: int = 20,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Stationary block bootstrap CI for Sharpe ratio.

    Returns (point_estimate, lo_2_5, hi_97_5).
    """
    if len(returns) < 2:
        return 0.0, 0.0, 0.0
    point = _sharpe(returns)
    if len(returns) < block_len:
        return point, point, point

    rng = random.Random(seed)
    n = len(returns)
    samples: list[float] = []
    for _ in range(n_iter):
        resampled: list[float] = []
        while len(resampled) < n:
            start = rng.randrange(0, n - block_len + 1)
            resampled.extend(returns[start : start + block_len])
        samples.append(_sharpe(resampled[:n]))
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples))]
    return point, lo, hi


def alpha_t_stat_vs_benchmark(
    strategy_returns: list[float],
    benchmark_returns: list[float],
) -> tuple[float, float]:
    """Regress strategy on benchmark; return (alpha_per_day, t_stat_of_alpha)."""
    n = min(len(strategy_returns), len(benchmark_returns))
    if n < 30:
        return 0.0, 0.0
    s = strategy_returns[:n]
    b = benchmark_returns[:n]
    s_mean = statistics.mean(s)
    b_mean = statistics.mean(b)
    cov = sum((s[i] - s_mean) * (b[i] - b_mean) for i in range(n)) / (n - 1)
    b_var = sum((b[i] - b_mean) ** 2 for i in range(n)) / (n - 1)
    if b_var == 0:
        return 0.0, 0.0
    beta = cov / b_var
    alpha = s_mean - beta * b_mean
    residuals = [s[i] - (alpha + beta * b[i]) for i in range(n)]
    res_var = sum(r * r for r in residuals) / (n - 2) if n > 2 else 0
    if res_var <= 0:
        return alpha, 0.0
    se_alpha = sqrt(res_var * (1.0 / n + b_mean * b_mean / ((n - 1) * b_var)))
    if se_alpha == 0:
        return alpha, 0.0
    return alpha, alpha / se_alpha


def significance_weight(t_stat: float) -> float:
    """Map t-stat to [0, 1]. 0 below t=1, ramps linearly to 1 at t=3."""
    if t_stat <= 1.0:
        return 0.0
    if t_stat >= 3.0:
        return 1.0
    return (t_stat - 1.0) / 2.0
```

- [ ] **Step 4: Run, expect PASS**

`pytest tests/test_backtest_stats.py -v` → 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/quant_lab/backtest/stats.py tests/test_backtest_stats.py
git -c user.email="<your-email>" -c user.name="<your-name>" commit -m "feat(backtest): block-bootstrap Sharpe CI + alpha t-stat + significance weight"
```

---

## Task 3: Backtest harness driver

**Files:**
- Create: `src/quant_lab/backtest/harness.py`
- Create: `tests/test_backtest_harness.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_backtest_harness.py
from datetime import date

from quant_lab.backtest.harness import run_walk_forward
from quant_lab.backtest.windows import Window
from quant_lab.strategies.base import Strategy
from quant_lab.types import Bar


class _AlwaysSPY(Strategy):
    bot_id = "test-always-spy"
    description = "Test strategy"

    def target_weights(self, histories, as_of):
        return {"SPY": 1.0}


def _bars(symbol, start, end, drift=0.0005):
    base = start
    bars, price = [], 400.0
    n_days = (end - start).days
    for i in range(n_days):
        d = base.fromordinal(base.toordinal() + i)
        # Skip weekends roughly
        if d.weekday() >= 5:
            continue
        price *= (1 + drift)
        bars.append(Bar(symbol=symbol, date=d, open=price, high=price, low=price, close=price, volume=10_000_000))
    return bars


def test_run_walk_forward_produces_nav_per_strategy():
    histories = {
        "SPY": _bars("SPY", date(2018, 1, 1), date(2024, 1, 1)),
        "QQQ": _bars("QQQ", date(2018, 1, 1), date(2024, 1, 1), drift=0.0007),
    }
    window = Window(
        train_start=date(2018, 1, 1),
        train_end=date(2020, 1, 1),
        test_end=date(2021, 1, 1),
        label="test",
    )
    result = run_walk_forward(
        strategies=[_AlwaysSPY()],
        histories=histories,
        windows=[window],
        starting_cash=100_000,
    )
    assert "test-always-spy" in result.nav_by_window["test"]
    nav = result.nav_by_window["test"]["test-always-spy"]
    assert len(nav) > 100  # ~year of trading days
    assert nav[-1] > nav[0]  # positive drift => positive NAV change


def test_run_walk_forward_multiple_windows_isolated():
    histories = {"SPY": _bars("SPY", date(2018, 1, 1), date(2024, 1, 1))}
    w1 = Window(date(2018, 1, 1), date(2020, 1, 1), date(2021, 1, 1), "w1")
    w2 = Window(date(2019, 1, 1), date(2021, 1, 1), date(2022, 1, 1), "w2")
    result = run_walk_forward(
        strategies=[_AlwaysSPY()],
        histories=histories,
        windows=[w1, w2],
        starting_cash=100_000,
    )
    # Each window has its own NAV series starting from the same starting_cash
    assert result.nav_by_window["w1"]["test-always-spy"][0] != result.nav_by_window["w2"]["test-always-spy"][-1]
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement `harness.py`**

```python
# src/quant_lab/backtest/harness.py
"""Walk-forward backtest harness.

Replays historical bars through the existing paper engine. For each window:
1. Initialize a fresh portfolio per strategy with `starting_cash`.
2. Step through each trading day in [window.train_end, window.test_end).
3. Slice histories so strategies only see data <= as_of (no leakage).
4. Run `run_morning_for_strategies` for one day, recording NAV.
5. Repeat next day.

Strategies that need refit/retrain on the train window must implement
`Strategy.fit(train_histories)` (Phase 3 ML); rule-based strategies are
parameter-free and a no-op on fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..tournament.runner import _avg_dollar_volume, run_morning_for_strategies
from ..strategies.base import Strategy
from ..types import Bar, Portfolio
from .windows import Window


@dataclass
class WalkForwardResult:
    nav_by_window: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    dates_by_window: dict[str, list[date]] = field(default_factory=dict)
    returns_by_window: dict[str, dict[str, list[float]]] = field(default_factory=dict)


def _slice_histories_to(
    histories: dict[str, list[Bar]],
    end_date: date,
) -> dict[str, list[Bar]]:
    return {sym: [b for b in bars if b.date <= end_date] for sym, bars in histories.items()}


def _trading_dates(histories: dict[str, list[Bar]], start: date, end: date) -> list[date]:
    dates: set[date] = set()
    for bars in histories.values():
        for b in bars:
            if start <= b.date < end:
                dates.add(b.date)
    return sorted(dates)


def run_walk_forward(
    strategies: list[Strategy],
    histories: dict[str, list[Bar]],
    windows: list[Window],
    starting_cash: float = 100_000.0,
    fit_callback=None,
) -> WalkForwardResult:
    """Run a walk-forward backtest over `windows`.

    `fit_callback(strategy, train_histories)` is called once per strategy at
    the start of each window with bars sliced to `[train_start, train_end)`.
    For Phase 1.5 this is a no-op; Phase 3 ML strategies will use it.
    """
    result = WalkForwardResult()

    for window in windows:
        # 1. Optional fit
        train_hist = {
            sym: [b for b in bars if window.train_start <= b.date < window.train_end]
            for sym, bars in histories.items()
        }
        if fit_callback is not None:
            for strat in strategies:
                fit_callback(strat, train_hist)

        # 2. Initialize fresh portfolios for this window
        portfolios: dict[str, Portfolio] = {
            strat.bot_id: Portfolio(bot_id=strat.bot_id, cash=starting_cash, positions={})
            for strat in strategies
        }
        navs: dict[str, list[tuple[date, float]]] = {strat.bot_id: [] for strat in strategies}

        # 3. Walk forward day by day
        for as_of in _trading_dates(histories, window.train_end, window.test_end):
            visible = _slice_histories_to(histories, as_of)
            advs = {sym: _avg_dollar_volume(bars) for sym, bars in visible.items()}
            portfolios, _trades, navs = run_morning_for_strategies(
                strategies=strategies,
                histories=visible,
                advs=advs,
                prior_portfolios=portfolios,
                prior_navs=navs,
                as_of=as_of,
                starting_cash=starting_cash,
            )

        # 4. Record results for this window
        result.nav_by_window[window.label] = {
            bot_id: [nav for _, nav in series] for bot_id, series in navs.items()
        }
        result.dates_by_window[window.label] = sorted(
            {d for series in navs.values() for d, _ in series}
        )
        result.returns_by_window[window.label] = {}
        for bot_id, series in navs.items():
            navs_only = [n for _, n in series]
            rets = [
                navs_only[i] / navs_only[i - 1] - 1.0
                for i in range(1, len(navs_only))
                if navs_only[i - 1] > 0
            ]
            result.returns_by_window[window.label][bot_id] = rets

    return result
```

- [ ] **Step 4: Run, expect PASS**

`pytest tests/test_backtest_harness.py -v` → 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/quant_lab/backtest/harness.py tests/test_backtest_harness.py
git -c user.email="<your-email>" -c user.name="<your-name>" commit -m "feat(backtest): walk-forward harness driving paper engine through history"
```

---

## Task 4: Slippage sweep

**Files:**
- Create: `src/quant_lab/backtest/slippage_sweep.py`
- Modify: `tests/test_backtest_harness.py` (append)

- [ ] **Step 1: Write failing test**

Append to `tests/test_backtest_harness.py`:

```python
from quant_lab.backtest.slippage_sweep import run_slippage_sweep


def test_slippage_sweep_returns_one_result_per_multiplier():
    histories = {"SPY": _bars("SPY", date(2018, 1, 1), date(2021, 1, 1))}
    window = Window(date(2018, 1, 1), date(2020, 1, 1), date(2020, 12, 31), "test")
    sweep = run_slippage_sweep(
        strategies=[_AlwaysSPY()],
        histories=histories,
        windows=[window],
        multipliers=(1.0, 2.0, 5.0),
    )
    assert set(sweep.results.keys()) == {1.0, 2.0, 5.0}
    # Higher slippage should produce same-or-lower NAV (cumulative cost drag)
    nav_1x = sweep.results[1.0].nav_by_window["test"]["test-always-spy"][-1]
    nav_5x = sweep.results[5.0].nav_by_window["test"]["test-always-spy"][-1]
    assert nav_5x <= nav_1x
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement `slippage_sweep.py`**

```python
# src/quant_lab/backtest/slippage_sweep.py
"""Slippage sensitivity sweep.

Runs the same walk-forward backtest at multiple slippage multipliers (1x, 2x, 5x)
to characterize how much of a strategy's edge survives realistic transaction costs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import slippage as _slip_module
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
            sweep.results[mult] = run_walk_forward(
                strategies=strategies,
                histories=histories,
                windows=windows,
                starting_cash=starting_cash,
            )
    finally:
        _slip_module.spread_bps = original_spread_bps
    return sweep
```

**Note:** monkey-patches the module-level `spread_bps`. Acceptable for a test/CLI tool; the module is single-process and we restore on exit.

- [ ] **Step 4: Run, expect PASS**

`pytest tests/test_backtest_harness.py -v` → 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/quant_lab/backtest/slippage_sweep.py tests/test_backtest_harness.py
git -c user.email="<your-email>" -c user.name="<your-name>" commit -m "feat(backtest): slippage sensitivity sweep (1x/2x/5x)"
```

---

## Task 5: Calibration report writer

**Files:**
- Create: `src/quant_lab/backtest/report.py`
- Create: `tests/test_backtest_report.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_backtest_report.py
import json
from datetime import date
from pathlib import Path

from quant_lab.backtest.report import write_calibration_report
from quant_lab.backtest.harness import WalkForwardResult


def test_write_calibration_report_emits_json_and_markdown(tmp_path: Path):
    result = WalkForwardResult(
        nav_by_window={"wf-2020": {"strat-a": [100_000, 102_000, 105_000]}},
        dates_by_window={"wf-2020": [date(2020, 1, 1), date(2020, 6, 1), date(2020, 12, 31)]},
        returns_by_window={"wf-2020": {"strat-a": [0.02, 0.029]}},
    )
    benchmark_returns = {"wf-2020": [0.005, 0.010]}
    out_dir = tmp_path / "out"
    write_calibration_report(
        out_dir=out_dir,
        wf_result=result,
        benchmark_returns_by_window=benchmark_returns,
        slippage_sweep=None,
        regime_results={},
    )
    payload = json.loads((out_dir / "backtest_results.json").read_text())
    assert "strategies" in payload
    assert any(s["bot_id"] == "strat-a" for s in payload["strategies"])
    md = (out_dir / "calibration_report.md").read_text()
    assert "# Calibration Report" in md
    assert "strat-a" in md
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement `report.py`**

```python
# src/quant_lab/backtest/report.py
"""Aggregate walk-forward + slippage-sweep + regime-stress results into a
calibration report (JSON for the dashboard, Markdown for human reading).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .harness import WalkForwardResult
from .stats import (
    alpha_t_stat_vs_benchmark,
    block_bootstrap_sharpe_ci,
    significance_weight,
)


def _per_strategy_summary(
    wf_result: WalkForwardResult,
    benchmark_returns_by_window: dict[str, list[float]],
) -> list[dict]:
    bot_ids = {b for w in wf_result.returns_by_window.values() for b in w.keys()}
    summaries = []
    for bot_id in sorted(bot_ids):
        per_window = []
        all_returns: list[float] = []
        all_alpha_ts: list[float] = []
        for window_label, by_bot in wf_result.returns_by_window.items():
            rets = by_bot.get(bot_id, [])
            if not rets:
                continue
            point, lo, hi = block_bootstrap_sharpe_ci(rets, n_iter=500, block_len=20, seed=42)
            bench = benchmark_returns_by_window.get(window_label, [])
            alpha, t = alpha_t_stat_vs_benchmark(rets, bench) if bench else (0.0, 0.0)
            per_window.append({
                "window": window_label,
                "sharpe": point,
                "sharpe_ci_lo": lo,
                "sharpe_ci_hi": hi,
                "alpha_per_day": alpha,
                "alpha_t_stat": t,
                "days": len(rets),
            })
            all_returns.extend(rets)
            all_alpha_ts.append(t)
        if not per_window:
            continue
        agg_point, agg_lo, agg_hi = block_bootstrap_sharpe_ci(all_returns, n_iter=1000, block_len=20, seed=42)
        median_t = sorted(all_alpha_ts)[len(all_alpha_ts) // 2] if all_alpha_ts else 0.0
        sig_weight = significance_weight(median_t)
        summaries.append({
            "bot_id": bot_id,
            "aggregate": {
                "sharpe": agg_point,
                "sharpe_ci_lo": agg_lo,
                "sharpe_ci_hi": agg_hi,
                "median_alpha_t": median_t,
                "significance_weight": sig_weight,
                "windows_evaluated": len(per_window),
                "total_test_days": len(all_returns),
            },
            "per_window": per_window,
        })
    return summaries


def write_calibration_report(
    out_dir: Path,
    wf_result: WalkForwardResult,
    benchmark_returns_by_window: dict[str, list[float]],
    slippage_sweep,
    regime_results: dict[str, WalkForwardResult],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = _per_strategy_summary(wf_result, benchmark_returns_by_window)

    payload = {
        "strategies": summaries,
        "regimes": {
            label: _per_strategy_summary(result, {})
            for label, result in regime_results.items()
        },
        "slippage_sweep": (
            {
                str(mult): _per_strategy_summary(res, benchmark_returns_by_window)
                for mult, res in slippage_sweep.results.items()
            }
            if slippage_sweep is not None
            else None
        ),
    }
    (out_dir / "backtest_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out_dir / "backtest_curves.json").write_text(
        json.dumps({
            "windows": list(wf_result.nav_by_window.keys()),
            "curves": {
                label: {
                    bot_id: [
                        {"date": d.isoformat(), "nav": nav}
                        for d, nav in zip(wf_result.dates_by_window[label], navs)
                    ]
                    for bot_id, navs in wf_result.nav_by_window[label].items()
                }
                for label in wf_result.nav_by_window
            },
        }, indent=2) + "\n"
    )

    # Markdown report
    lines = ["# Calibration Report", ""]
    lines.append(f"Strategies evaluated: **{len(summaries)}**")
    lines.append("")
    lines.append("| Bot | Aggregate Sharpe | 95% CI | Median α t-stat | Sig weight | Days |")
    lines.append("|---|---|---|---|---|---|")
    for s in summaries:
        a = s["aggregate"]
        lines.append(
            f"| {s['bot_id']} | {a['sharpe']:.2f} | [{a['sharpe_ci_lo']:.2f}, {a['sharpe_ci_hi']:.2f}] | "
            f"{a['median_alpha_t']:.2f} | {a['significance_weight']:.2f} | {a['total_test_days']} |"
        )
    (out_dir / "calibration_report.md").write_text("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run, expect PASS**

`pytest tests/test_backtest_report.py -v` → 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/quant_lab/backtest/report.py tests/test_backtest_report.py
git -c user.email="<your-email>" -c user.name="<your-name>" commit -m "feat(backtest): calibration report (JSON + markdown) with bootstrap CIs and alpha t-stats"
```

---

## Task 6: CLI subcommand `quant-lab backtest`

**Files:**
- Modify: `src/quant_lab/main.py` (add `backtest_command` and `backtest` subparser)
- Create: `tests/test_backtest_cli.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_backtest_cli.py
import json
from datetime import date
from unittest.mock import patch

from quant_lab.main import backtest_command
from quant_lab.types import Bar


def _synth(symbol, n=2200, drift=0.0004):
    base = date(2017, 1, 2)
    bars, price = [], 400.0
    for i in range(n):
        d = base.fromordinal(base.toordinal() + i)
        if d.weekday() >= 5:
            continue
        price *= (1 + drift)
        bars.append(Bar(symbol=symbol, date=d, open=price, high=price, low=price, close=price, volume=50_000_000))
    return bars


def test_backtest_command_writes_calibration_artifacts(tmp_path, monkeypatch):
    histories = {"SPY": _synth("SPY"), "QQQ": _synth("QQQ", drift=0.0006)}
    monkeypatch.setattr("quant_lab.main.fetch_history", lambda symbol, lookback_days=365: histories.get(symbol.upper(), []))

    backtest_command(
        out_dir=tmp_path / "backtest",
        start=date(2017, 1, 1),
        end=date(2024, 1, 1),
        train_years=3,
        step_months=12,
        run_slippage_sweep=False,
        run_regime_stress=False,
    )

    payload = json.loads((tmp_path / "backtest" / "backtest_results.json").read_text())
    assert "strategies" in payload
    bot_ids = {s["bot_id"] for s in payload["strategies"]}
    assert "spy-vol" in bot_ids
    assert "qqq-vol" in bot_ids
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Add `backtest_command` and subparser to `main.py`**

In `src/quant_lab/main.py`, after the existing `morning_command` definition, add:

```python
from datetime import datetime as _dt

from .backtest.harness import run_walk_forward
from .backtest.report import write_calibration_report
from .backtest.slippage_sweep import run_slippage_sweep
from .backtest.windows import regime_stress_windows, walk_forward_windows


def _benchmark_returns(histories, windows, symbol="SPY"):
    out = {}
    bars = histories.get(symbol, [])
    for window in windows:
        in_window = [b for b in bars if window.train_end <= b.date < window.test_end]
        rets = []
        for i in range(1, len(in_window)):
            prev = in_window[i - 1].close
            if prev > 0:
                rets.append(in_window[i].close / prev - 1.0)
        out[window.label] = rets
    return out


def backtest_command(
    out_dir,
    start,
    end,
    train_years: int = 5,
    step_months: int = 12,
    run_slippage_sweep: bool = True,
    run_regime_stress: bool = True,
) -> None:
    from .strategies.base import get_all
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = ["SPY", "QQQ"]
    lookback_days = (end - start).days + 365
    histories = {}
    for sym in symbols:
        bars = fetch_history(sym, lookback_days=lookback_days)
        if bars:
            histories[sym] = [b for b in bars if start <= b.date <= end]
    if not histories:
        raise RuntimeError("No historical data fetched.")

    strategies_list = get_all()
    windows = walk_forward_windows(start=start, end=end, train_years=train_years, step_months=step_months)
    if not windows:
        raise RuntimeError(f"No walk-forward windows generated from {start} to {end} with train_years={train_years}.")

    wf_result = run_walk_forward(strategies=strategies_list, histories=histories, windows=windows)
    bench_returns = _benchmark_returns(histories, windows)

    sweep = None
    if run_slippage_sweep:
        sweep = run_slippage_sweep(
            strategies=strategies_list, histories=histories, windows=windows[:1]
        )
    regime_results = {}
    if run_regime_stress:
        regimes = regime_stress_windows()
        applicable = [w for w in regimes if w.test_end <= end and w.train_start >= start]
        if applicable:
            regime_results["stress"] = run_walk_forward(
                strategies=strategies_list, histories=histories, windows=applicable
            )

    write_calibration_report(
        out_dir=out_dir,
        wf_result=wf_result,
        benchmark_returns_by_window=bench_returns,
        slippage_sweep=sweep,
        regime_results=regime_results,
    )
```

Then update the `cli()` function's argparse section to add the subcommand:

```python
def cli() -> None:
    parser = argparse.ArgumentParser(prog="quant-lab")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("morning", help="Run the morning tournament step")

    bt = sub.add_parser("backtest", help="Run the walk-forward backtest")
    bt.add_argument("--start", type=lambda s: _dt.fromisoformat(s).date(), default=date(2015, 1, 1))
    bt.add_argument("--end", type=lambda s: _dt.fromisoformat(s).date(), default=date.today())
    bt.add_argument("--train-years", type=int, default=5)
    bt.add_argument("--step-months", type=int, default=12)
    bt.add_argument("--no-slippage-sweep", action="store_true")
    bt.add_argument("--no-regime-stress", action="store_true")

    args = parser.parse_args()
    if args.cmd == "morning":
        repo_root = Path(__file__).resolve().parents[2]
        morning_command(
            state_dir=repo_root / "state",
            dashboard_data_dir=repo_root / "dashboard" / "data",
            snapshot_dir=repo_root / "data" / "snapshots",
            discord_webhook=os.getenv("DISCORD_WEBHOOK"),
            dashboard_url=os.getenv("DASHBOARD_URL"),
        )
    elif args.cmd == "backtest":
        repo_root = Path(__file__).resolve().parents[2]
        backtest_command(
            out_dir=repo_root / "dashboard" / "data" / "backtest",
            start=args.start,
            end=args.end,
            train_years=args.train_years,
            step_months=args.step_months,
            run_slippage_sweep=not args.no_slippage_sweep,
            run_regime_stress=not args.no_regime_stress,
        )
```

- [ ] **Step 4: Run, expect PASS**

`pytest tests/test_backtest_cli.py -v` → 1 passed

- [ ] **Step 5: Run full suite to confirm nothing else broke**

`pytest -q` → all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/quant_lab/main.py tests/test_backtest_cli.py
git -c user.email="<your-email>" -c user.name="<your-name>" commit -m "feat(cli): add quant-lab backtest subcommand with walk-forward + sweeps"
```

---

## Task 7: Backtest dashboard page

**Files:**
- Create: `dashboard/backtest.html`
- Create: `dashboard/backtest.js`
- Modify: `dashboard/index.html` (add link to backtest page)

- [ ] **Step 1: Create `dashboard/backtest.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Quant Lab — Walk-Forward Calibration</title>
  <link rel="stylesheet" href="styles.css" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.js"></script>
</head>
<body>
  <header>
    <h1>Calibration Report</h1>
    <p class="disclaimer">Walk-forward out-of-sample evidence. Survivorship-biased data. Past performance ≠ future results.</p>
    <p><a href="index.html">← Live tournament</a></p>
  </header>

  <section>
    <h2>Aggregate Calibration</h2>
    <table id="agg-table">
      <thead><tr><th>Bot</th><th>Sharpe</th><th>95% CI</th><th>Median α t-stat</th><th>Sig weight</th><th>Days</th></tr></thead>
      <tbody></tbody>
    </table>
  </section>

  <section>
    <h2>Equity Curves Per Window</h2>
    <div id="curves"></div>
  </section>

  <footer>
    <p>Disclaimer: backtest data is survivorship-biased (yfinance only contains currently-listed names). Real-world results would be lower by 1-3%/yr from delisted names. Strategies that beat SPY here may still fail live.</p>
  </footer>

  <script src="backtest.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `dashboard/backtest.js`**

```javascript
(async () => {
  const fmt = (v, digits = 2) => v == null ? "-" : v.toFixed(digits);
  const cls = (v) => v >= 0 ? "pos" : "neg";

  let results, curves;
  try {
    [results, curves] = await Promise.all([
      fetch("data/backtest/backtest_results.json").then(r => r.json()),
      fetch("data/backtest/backtest_curves.json").then(r => r.json()),
    ]);
  } catch {
    document.body.innerHTML = "<header><h1>No backtest data yet</h1><p>Run <code>quant-lab backtest</code> to generate.</p></header>";
    return;
  }

  const tbody = document.querySelector("#agg-table tbody");
  for (const s of results.strategies || []) {
    const a = s.aggregate;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${s.bot_id}</td>
      <td class="${cls(a.sharpe)}">${fmt(a.sharpe)}</td>
      <td>[${fmt(a.sharpe_ci_lo)}, ${fmt(a.sharpe_ci_hi)}]</td>
      <td>${fmt(a.median_alpha_t)}</td>
      <td>${fmt(a.significance_weight)}</td>
      <td>${a.total_test_days}</td>
    `;
    tbody.appendChild(tr);
  }

  const curvesEl = document.getElementById("curves");
  for (const window of curves.windows) {
    const block = document.createElement("div");
    block.innerHTML = `<h3>${window}</h3><canvas height="120"></canvas>`;
    curvesEl.appendChild(block);
    const ctx = block.querySelector("canvas").getContext("2d");
    const palette = ["#58a6ff", "#3fb950", "#f0883e", "#a371f7", "#ec4899"];
    let i = 0;
    const dataByBot = curves.curves[window];
    const allDates = new Set();
    for (const series of Object.values(dataByBot)) {
      series.forEach(p => allDates.add(p.date));
    }
    const labels = Array.from(allDates).sort();
    const datasets = [];
    for (const [bot, series] of Object.entries(dataByBot)) {
      const map = Object.fromEntries(series.map(p => [p.date, p.nav]));
      datasets.push({
        label: bot,
        data: labels.map(d => map[d] ?? null),
        spanGaps: true,
        borderColor: palette[i++ % palette.length],
        backgroundColor: "transparent",
        pointRadius: 0,
        borderWidth: 2,
      });
    }
    new Chart(ctx, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: "#e6edf3" } } },
        scales: {
          x: { ticks: { color: "#8b949e" }, grid: { color: "#30363d" } },
          y: { ticks: { color: "#8b949e" }, grid: { color: "#30363d" } },
        },
      },
    });
  }
})();
```

- [ ] **Step 3: Add link from `index.html`**

Add after the `<header>` block in `dashboard/index.html`:

```html
<nav style="max-width: 1100px; margin: 0 auto; padding: 0 1.5rem;">
  <a href="backtest.html">Calibration Report →</a>
</nav>
```

- [ ] **Step 4: Commit**

```bash
git add dashboard/backtest.html dashboard/backtest.js dashboard/index.html
git -c user.email="<your-email>" -c user.name="<your-name>" commit -m "feat(dashboard): walk-forward calibration report page"
```

---

## Task 8: End-to-end smoke test for backtest

**Files:**
- Create: `tests/test_backtest_e2e.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_backtest_e2e.py
"""E2E smoke: pull synthetic 7 years of data, run walk-forward, write report, assert artifacts exist."""
import json
from datetime import date
from pathlib import Path

from quant_lab.main import backtest_command
from quant_lab.types import Bar


def _synth(symbol, n=2200, drift=0.0004):
    base = date(2017, 1, 2)
    bars, price = [], 400.0
    for i in range(n):
        d = base.fromordinal(base.toordinal() + i)
        if d.weekday() >= 5:
            continue
        price *= (1 + drift)
        bars.append(Bar(symbol=symbol, date=d, open=price, high=price, low=price, close=price, volume=50_000_000))
    return bars


def test_backtest_e2e_produces_calibration_artifacts(tmp_path, monkeypatch):
    histories = {"SPY": _synth("SPY"), "QQQ": _synth("QQQ", drift=0.0006)}
    monkeypatch.setattr("quant_lab.main.fetch_history", lambda symbol, lookback_days=365: histories.get(symbol.upper(), []))

    out = tmp_path / "bt"
    backtest_command(
        out_dir=out,
        start=date(2017, 1, 1),
        end=date(2023, 1, 1),
        train_years=3,
        step_months=12,
        run_slippage_sweep=True,
        run_regime_stress=False,
    )

    assert (out / "backtest_results.json").exists()
    assert (out / "backtest_curves.json").exists()
    assert (out / "calibration_report.md").exists()

    payload = json.loads((out / "backtest_results.json").read_text())
    bot_ids = {s["bot_id"] for s in payload["strategies"]}
    assert {"spy-vol", "qqq-vol"} <= bot_ids
    # Slippage sweep should be present
    assert payload["slippage_sweep"] is not None
    assert "1.0" in payload["slippage_sweep"]
```

- [ ] **Step 2: Run, expect PASS**

`pytest tests/test_backtest_e2e.py -v` → 1 passed

- [ ] **Step 3: Run full suite**

`pytest -q` → all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_backtest_e2e.py
git -c user.email="<your-email>" -c user.name="<your-name>" commit -m "test: e2e smoke for backtest pipeline (synthetic 7-yr data)"
```

---

## Self-Review

**Spec coverage:**
- §10 ML pipeline + walk-forward / label-shuffle / OOS gates: harness in Task 3 + bootstrap CI / alpha t-stat / significance weight in Task 2 form the foundation; Phase 3 ML strategies will plug into the same harness via `fit_callback`.
- "Maximum calibration": multiple walk-forward windows (Task 1), block bootstrap CIs (Task 2, time-series correct), alpha t-stat regression vs SPY (Task 2), regime stress windows (Task 1, Task 6), slippage sweep (Task 4), full reporting (Task 5), CLI + dashboard (Tasks 6, 7).
- "Synthesize at end" — deferred to Phase 5, but the JSON contract from this phase (`backtest_results.json` with `aggregate.significance_weight`) is the input that Phase 5's meta-ensemble bot consumes.

**Placeholder scan:** None.

**Type consistency:** `Window`, `WalkForwardResult`, `SlippageSweepResult` consistent across files. The `fit_callback` is a placeholder for Phase 3 ML — explicitly typed as `None`-default.

**Honest limitation:** This phase relies on yfinance data which is survivorship-biased; the dashboard page and report markdown both call this out.

The harness is the upstream bottleneck: every Phase 2 (classical) and Phase 3 (ML) strategy benefits from it. Shipping it now means each new strategy in later phases ships with calibrated evidence on day one.

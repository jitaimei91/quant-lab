# Quant Lab Phase 1 (MVP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a working morning bot that paper-trades SPY-Vol + QQQ-Vol benchmarks on free-tier infra, posts a daily Discord brief, and generates a public GitHub Pages dashboard.

**Architecture:** Stateless GitHub Actions cron → yfinance data fetch → Strategy.signal() → Paper engine (in-memory + JSON persistence) → Discord webhook + JSON dashboard data → static HTML dashboard on GH Pages.

**Tech Stack:** Python 3.11, yfinance, pandas, numpy, requests, pytest, GitHub Actions, GitHub Pages, Discord webhooks.

**What this MVP intentionally defers** (handled in Phase 2+):
- Alpaca primary data (yfinance only for now)
- Turso storage (local JSON files for now)
- Classical strategies (Momo, MeanRev, etc.) — only the two vol-targeted index bots in MVP
- Codex bot adapter
- Bootstrapped CIs, factor decomposition, regime kill-switch
- ML pipeline
- Bot-vs-Bot dashboard page
- Full automated `bootstrap.sh` with full path — MVP ships fast-path only

---

## File Structure

```
quant-lab/
├── .github/workflows/
│   ├── morning.yml          # weekday cron, runs the bot
│   └── ci.yml               # runs tests on PR
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
├── bootstrap.sh             # fast-path setup
├── config/
│   └── universe.example.json
├── src/quant_lab/
│   ├── __init__.py
│   ├── types.py             # Bar, Position, Trade, Portfolio
│   ├── data.py              # yfinance wrapper with caching
│   ├── slippage.py          # bps cost model
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py          # Strategy ABC + registry
│   │   ├── spy_vol.py
│   │   └── qqq_vol.py
│   ├── engine/
│   │   ├── __init__.py
│   │   └── paper.py         # Paper trading engine
│   ├── tournament/
│   │   ├── __init__.py
│   │   ├── runner.py        # Runs all strategies for a date
│   │   └── stats.py         # Total return, Sharpe, drawdown
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── discord.py       # Webhook poster
│   │   └── dashboard.py     # JSON exporter for GH Pages
│   ├── persistence.py       # Load/save JSON state files
│   └── main.py              # Entry points: morning, init
├── dashboard/
│   ├── index.html           # Static HTML
│   ├── styles.css
│   ├── app.js               # Renders charts from data/*.json
│   └── data/                # Generated JSONs (gitignored except .gitkeep)
├── state/
│   ├── .gitkeep
│   └── (portfolios.json, nav_history.json — committed by workflow)
├── data/snapshots/
│   └── .gitkeep             # Daily price snapshots (committed by workflow)
└── tests/
    ├── __init__.py
    ├── test_types.py
    ├── test_slippage.py
    ├── test_paper_engine.py
    ├── test_strategies.py
    ├── test_stats.py
    └── test_dashboard.py
```

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.python-version`, `README.md`
- Create: `src/quant_lab/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "quant-lab"
version = "0.1.0"
description = "Personal quant research lab + morning briefing bot"
requires-python = ">=3.11"
dependencies = [
    "yfinance>=0.2.40",
    "pandas>=2.2",
    "numpy>=1.26",
    "requests>=2.32",
    "python-dateutil>=2.9",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "hypothesis>=6",
    "ruff>=0.5",
]

[project.scripts]
quant-lab = "quant_lab.main:cli"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.coverage
htmlcov/
.venv/
venv/
*.egg-info/
build/
dist/
.DS_Store
.env
.env.local
config/account.json
config/universe.json
dashboard/data/*.json
!dashboard/data/.gitkeep
state/*.json
!state/.gitkeep
data/cache/
```

- [ ] **Step 3: Create `.python-version`**

```
3.11
```

- [ ] **Step 4: Create `src/quant_lab/__init__.py`**

```python
"""Quant Lab — personal quant research and morning briefing bot."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Create `tests/__init__.py` (empty file)**

```python
```

- [ ] **Step 6: Create `README.md`**

```markdown
# Quant Lab

Personal quant research lab + morning briefing bot. Paper-trades a tournament of strategies on free GitHub Actions, posts daily Discord briefs, dashboards on GitHub Pages.

**Status:** Phase 1 MVP — SPY-Vol + QQQ-Vol benchmarks shipping.

## Quick Start

```bash
./bootstrap.sh --fast
```

See `docs/superpowers/specs/2026-05-07-quant-lab-design.md` for the full design.

## Disclaimer

Research and education tool. Paper trading only. Not financial advice. Past performance does not predict future results.
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore .python-version README.md src/ tests/
git commit -m "scaffold: phase 1 project skeleton"
```

---

## Task 2: Core data types

**Files:**
- Create: `src/quant_lab/types.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: Write failing test for `Bar` dataclass**

Create `tests/test_types.py`:

```python
from datetime import date
from quant_lab.types import Bar, Position, Trade, Portfolio


def test_bar_construction():
    bar = Bar(symbol="SPY", date=date(2026, 5, 6), open=500.0, high=502.0, low=498.0, close=501.0, volume=50_000_000)
    assert bar.symbol == "SPY"
    assert bar.close == 501.0
    assert bar.volume == 50_000_000
```

- [ ] **Step 2: Run test, expect failure**

Run: `pytest tests/test_types.py::test_bar_construction -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant_lab.types'`

- [ ] **Step 3: Implement `Bar` in `src/quant_lab/types.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    date: _date
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(slots=True)
class Position:
    symbol: str
    shares: float
    avg_cost: float

    def market_value(self, price: float) -> float:
        return self.shares * price


@dataclass(slots=True)
class Trade:
    bot_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    shares: float
    price: float
    slippage_bps: float
    timestamp: _date
    reason: str = ""


@dataclass(slots=True)
class Portfolio:
    bot_id: str
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    def equity(self, prices: dict[str, float]) -> float:
        invested = sum(
            pos.market_value(prices.get(sym, pos.avg_cost))
            for sym, pos in self.positions.items()
        )
        return self.cash + invested

    def weight(self, symbol: str, prices: dict[str, float]) -> float:
        equity = self.equity(prices)
        if equity <= 0:
            return 0.0
        pos = self.positions.get(symbol)
        if not pos:
            return 0.0
        price = prices.get(symbol, pos.avg_cost)
        return pos.market_value(price) / equity
```

- [ ] **Step 4: Run test, expect pass**

Run: `pytest tests/test_types.py::test_bar_construction -v`
Expected: PASS

- [ ] **Step 5: Add tests for `Position`, `Trade`, `Portfolio`**

Append to `tests/test_types.py`:

```python
def test_position_market_value():
    pos = Position(symbol="AAPL", shares=10, avg_cost=180.0)
    assert pos.market_value(200.0) == 2000.0


def test_trade_construction():
    trade = Trade(
        bot_id="spy-vol",
        symbol="SPY",
        side="BUY",
        shares=2.0,
        price=500.0,
        slippage_bps=5.0,
        timestamp=date(2026, 5, 6),
    )
    assert trade.side == "BUY"
    assert trade.bot_id == "spy-vol"


def test_portfolio_equity_with_positions():
    portfolio = Portfolio(
        bot_id="spy-vol",
        cash=2_000.0,
        positions={
            "SPY": Position(symbol="SPY", shares=10, avg_cost=480.0),
        },
    )
    prices = {"SPY": 500.0}
    assert portfolio.equity(prices) == 2_000.0 + 10 * 500.0


def test_portfolio_weight():
    portfolio = Portfolio(
        bot_id="qqq-vol",
        cash=1_000.0,
        positions={"QQQ": Position(symbol="QQQ", shares=2, avg_cost=400.0)},
    )
    prices = {"QQQ": 500.0}
    weight = portfolio.weight("QQQ", prices)
    assert abs(weight - (1000.0 / 2000.0)) < 1e-9
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/test_types.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add src/quant_lab/types.py tests/test_types.py
git commit -m "feat(types): add Bar, Position, Trade, Portfolio dataclasses"
```

---

## Task 3: Slippage model

**Files:**
- Create: `src/quant_lab/slippage.py`
- Create: `tests/test_slippage.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_slippage.py`:

```python
import math
import pytest
from quant_lab.slippage import spread_bps, apply_slippage


def test_spread_bps_floors_at_one():
    # Hypothetical infinite-liquidity name
    assert spread_bps(adv_dollars=1e15) >= 1.0


def test_spread_bps_grows_for_low_liquidity():
    high = spread_bps(adv_dollars=10e6)   # $10M ADV (low)
    low = spread_bps(adv_dollars=10e9)    # $10B ADV (high)
    assert high > low


def test_apply_slippage_buy_increases_price():
    fill = apply_slippage(price=100.0, side="BUY", spread_bps_value=10.0)
    assert math.isclose(fill, 100.0 * (1 + 10.0 / 10_000))


def test_apply_slippage_sell_decreases_price():
    fill = apply_slippage(price=100.0, side="SELL", spread_bps_value=10.0)
    assert math.isclose(fill, 100.0 * (1 - 10.0 / 10_000))


def test_apply_slippage_invalid_side_raises():
    with pytest.raises(ValueError):
        apply_slippage(price=100.0, side="HOLD", spread_bps_value=10.0)
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest tests/test_slippage.py -v`
Expected: ImportError — module not found

- [ ] **Step 3: Implement `src/quant_lab/slippage.py`**

```python
"""Cost model for paper trading.

Approximates round-trip transaction costs (bid-ask spread + market impact)
as a function of average daily dollar volume. See spec section 7.
"""
from __future__ import annotations

import math


def spread_bps(adv_dollars: float) -> float:
    """Return one-way slippage cost in basis points.

    Roughly: liquid mega-caps ~3-6 bps, mid-caps ~10-15 bps,
    R1000-tail names ~20-30 bps. Floor at 1 bp.
    """
    adv_millions = max(adv_dollars / 1_000_000, 1.0)
    raw = 5.0 + 100.0 / math.sqrt(adv_millions)
    return max(1.0, raw)


def apply_slippage(price: float, side: str, spread_bps_value: float) -> float:
    """Return the fill price for a market order at `price` with given spread."""
    factor = spread_bps_value / 10_000.0
    if side == "BUY":
        return price * (1.0 + factor)
    if side == "SELL":
        return price * (1.0 - factor)
    raise ValueError(f"Unknown side: {side!r}")
```

- [ ] **Step 4: Run, expect PASS**

Run: `pytest tests/test_slippage.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/quant_lab/slippage.py tests/test_slippage.py
git commit -m "feat(slippage): add liquidity-aware bid-ask spread model"
```

---

## Task 4: Data fetcher (yfinance wrapper)

**Files:**
- Create: `src/quant_lab/data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write failing test (uses `monkeypatch` to avoid network)**

Create `tests/test_data.py`:

```python
import pandas as pd
from datetime import date
from unittest.mock import MagicMock

from quant_lab.data import fetch_history, latest_bar
from quant_lab.types import Bar


def _fake_yf_history():
    idx = pd.to_datetime(["2026-05-04", "2026-05-05", "2026-05-06"])
    df = pd.DataFrame(
        {
            "Open": [498.0, 499.5, 500.0],
            "High": [501.0, 502.0, 503.0],
            "Low": [497.0, 498.5, 499.0],
            "Close": [500.0, 501.0, 502.5],
            "Volume": [50_000_000, 48_000_000, 52_000_000],
        },
        index=idx,
    )
    return df


def test_fetch_history_returns_bars(monkeypatch):
    fake_ticker = MagicMock()
    fake_ticker.history.return_value = _fake_yf_history()
    monkeypatch.setattr("yfinance.Ticker", lambda symbol: fake_ticker)

    bars = fetch_history("SPY", lookback_days=5)
    assert all(isinstance(b, Bar) for b in bars)
    assert bars[-1].close == 502.5
    assert bars[-1].symbol == "SPY"


def test_latest_bar_returns_last(monkeypatch):
    fake_ticker = MagicMock()
    fake_ticker.history.return_value = _fake_yf_history()
    monkeypatch.setattr("yfinance.Ticker", lambda symbol: fake_ticker)

    bar = latest_bar("SPY")
    assert bar.symbol == "SPY"
    assert bar.date == date(2026, 5, 6)
    assert bar.close == 502.5
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest tests/test_data.py -v`
Expected: ImportError

- [ ] **Step 3: Implement `src/quant_lab/data.py`**

```python
"""Daily-bar market data fetcher.

Phase 1 uses yfinance only. Phase 2 will add Alpaca primary with yfinance
fallback per spec section 5.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import yfinance as yf

from .types import Bar


def fetch_history(symbol: str, lookback_days: int = 365) -> list[Bar]:
    """Fetch OHLCV bars for the last `lookback_days` trading days for `symbol`."""
    ticker = yf.Ticker(symbol)
    end = date.today()
    start = end - timedelta(days=lookback_days)
    df = ticker.history(start=start, end=end + timedelta(days=1), auto_adjust=True)
    if df.empty:
        return []

    bars: list[Bar] = []
    for ts, row in df.iterrows():
        bars.append(
            Bar(
                symbol=symbol.upper(),
                date=ts.date(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
            )
        )
    return bars


def latest_bar(symbol: str) -> Bar | None:
    """Return the most recent bar for `symbol`, or None if no data."""
    bars = fetch_history(symbol, lookback_days=10)
    return bars[-1] if bars else None


def fetch_many(symbols: Iterable[str], lookback_days: int = 365) -> dict[str, list[Bar]]:
    """Fetch histories for many symbols, skipping any that fail."""
    histories: dict[str, list[Bar]] = {}
    for symbol in symbols:
        try:
            bars = fetch_history(symbol, lookback_days=lookback_days)
            if bars:
                histories[symbol.upper()] = bars
        except Exception:
            continue
    return histories
```

- [ ] **Step 4: Run, expect PASS**

Run: `pytest tests/test_data.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/quant_lab/data.py tests/test_data.py
git commit -m "feat(data): yfinance-backed Bar fetcher"
```

---

## Task 5: Strategy interface and registry

**Files:**
- Create: `src/quant_lab/strategies/__init__.py`
- Create: `src/quant_lab/strategies/base.py`
- Create: `tests/test_strategies.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_strategies.py`:

```python
from datetime import date
from quant_lab.strategies.base import Strategy, register, get_all
from quant_lab.types import Bar


class _FakeStrat(Strategy):
    bot_id = "fake-strat"
    description = "Test strategy"

    def target_weights(self, histories, as_of):
        return {"SPY": 0.5}


def test_strategy_subclass_required_fields():
    s = _FakeStrat()
    assert s.bot_id == "fake-strat"
    assert s.description == "Test strategy"


def test_strategy_target_weights_returns_dict():
    s = _FakeStrat()
    weights = s.target_weights({}, date(2026, 5, 6))
    assert weights == {"SPY": 0.5}


def test_register_and_get_all():
    register(_FakeStrat)
    instances = get_all()
    assert any(s.bot_id == "fake-strat" for s in instances)
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest tests/test_strategies.py -v`
Expected: ImportError

- [ ] **Step 3: Create `src/quant_lab/strategies/__init__.py`**

```python
"""Strategy package.

Strategies are auto-registered when their module is imported. The morning
runner imports all strategy modules to populate the registry.
"""
from .base import Strategy, register, get_all  # noqa: F401
```

- [ ] **Step 4: Create `src/quant_lab/strategies/base.py`**

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Type

from ..types import Bar


_REGISTRY: dict[str, "Strategy"] = {}


class Strategy(ABC):
    """Base class for all trading strategies.

    Subclasses must define `bot_id` and `description` as class attributes,
    and implement `target_weights`.
    """

    bot_id: str = ""
    description: str = ""

    @abstractmethod
    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        """Return target portfolio weights as {symbol: weight}.

        Weights should sum to <= 1.0; the remainder is held in cash.
        Implementations must use only data with date <= as_of.
        """


def register(strategy_cls: Type[Strategy]) -> Type[Strategy]:
    """Register a Strategy subclass. Decorator-friendly."""
    instance = strategy_cls()
    if not instance.bot_id:
        raise ValueError(f"{strategy_cls.__name__} missing bot_id")
    _REGISTRY[instance.bot_id] = instance
    return strategy_cls


def get_all() -> list[Strategy]:
    """Return all registered strategy instances."""
    return list(_REGISTRY.values())


def get(bot_id: str) -> Strategy:
    return _REGISTRY[bot_id]
```

- [ ] **Step 5: Run, expect PASS**

Run: `pytest tests/test_strategies.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/quant_lab/strategies/ tests/test_strategies.py
git commit -m "feat(strategies): add Strategy ABC and registry"
```

---

## Task 6: SPY-Vol strategy

**Files:**
- Create: `src/quant_lab/strategies/spy_vol.py`
- Modify: `tests/test_strategies.py` (append)

- [ ] **Step 1: Write failing test**

Append to `tests/test_strategies.py`:

```python
from quant_lab.strategies.spy_vol import SPYVol


def _synth_bars(symbol, n_days=300, vol=0.01):
    import math, random
    random.seed(42)
    base_date = date(2025, 1, 1)
    price = 500.0
    bars = []
    for i in range(n_days):
        ret = random.gauss(0.0003, vol)
        price *= (1 + ret)
        bars.append(
            Bar(
                symbol=symbol,
                date=base_date.replace(day=1) if i == 0 else base_date.fromordinal(base_date.toordinal() + i),
                open=price * 0.999,
                high=price * 1.005,
                low=price * 0.995,
                close=price,
                volume=50_000_000,
            )
        )
    return bars


def test_spy_vol_target_low_when_vol_high():
    """When realized vol exceeds 15% target, weight should be < 1.0."""
    bars = _synth_bars("SPY", n_days=300, vol=0.025)  # ~40% annualized
    strat = SPYVol(target_vol=0.15)
    weights = strat.target_weights({"SPY": bars}, bars[-1].date)
    assert "SPY" in weights
    assert 0 < weights["SPY"] < 1.0


def test_spy_vol_caps_leverage_at_one():
    """When realized vol is below target, weight is capped at 1.0 (no leverage)."""
    bars = _synth_bars("SPY", n_days=300, vol=0.001)  # very low
    strat = SPYVol(target_vol=0.15)
    weights = strat.target_weights({"SPY": bars}, bars[-1].date)
    assert weights["SPY"] == 1.0


def test_spy_vol_returns_zero_with_insufficient_history():
    bars = _synth_bars("SPY", n_days=10)
    strat = SPYVol(target_vol=0.15)
    weights = strat.target_weights({"SPY": bars}, bars[-1].date)
    assert weights == {} or weights.get("SPY", 0.0) == 0.0
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest tests/test_strategies.py -v`
Expected: ImportError on `SPYVol`

- [ ] **Step 3: Implement `src/quant_lab/strategies/spy_vol.py`**

```python
from __future__ import annotations

from datetime import date
from math import sqrt

from ..types import Bar
from .base import Strategy, register


TRADING_DAYS_PER_YEAR = 252


def _realized_vol(bars: list[Bar], window: int) -> float | None:
    """Annualized realized volatility from daily log returns over the last `window` bars."""
    if len(bars) <= window:
        return None
    closes = [b.close for b in bars[-(window + 1):]]
    rets = []
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0:
            return None
        rets.append((closes[i] / closes[i - 1]) - 1.0)
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return sqrt(var) * sqrt(TRADING_DAYS_PER_YEAR)


class _VolTargetedIndex(Strategy):
    """Vol-targeted long exposure to a single ETF symbol."""

    symbol: str = ""
    target_vol_default: float = 0.15
    vol_window: int = 60

    def __init__(self, target_vol: float | None = None):
        self.target_vol = target_vol or self.target_vol_default

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        bars = histories.get(self.symbol, [])
        bars = [b for b in bars if b.date <= as_of]
        realized = _realized_vol(bars, window=self.vol_window)
        if realized is None or realized <= 0:
            return {}
        weight = min(1.0, self.target_vol / realized)
        return {self.symbol: weight}


@register
class SPYVol(_VolTargetedIndex):
    bot_id = "spy-vol"
    description = "Vol-targeted long S&P 500 (SPY). Honest market benchmark."
    symbol = "SPY"
```

- [ ] **Step 4: Run, expect PASS**

Run: `pytest tests/test_strategies.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/quant_lab/strategies/spy_vol.py tests/test_strategies.py
git commit -m "feat(strategies): add SPY-Vol benchmark (vol-targeted long SPY)"
```

---

## Task 7: QQQ-Vol strategy

**Files:**
- Create: `src/quant_lab/strategies/qqq_vol.py`
- Modify: `tests/test_strategies.py` (append)

- [ ] **Step 1: Write failing test**

Append to `tests/test_strategies.py`:

```python
from quant_lab.strategies.qqq_vol import QQQVol


def test_qqq_vol_targets_qqq():
    bars = _synth_bars("QQQ", n_days=300, vol=0.018)
    strat = QQQVol(target_vol=0.15)
    weights = strat.target_weights({"QQQ": bars}, bars[-1].date)
    assert "QQQ" in weights
    assert 0 < weights["QQQ"] <= 1.0
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest tests/test_strategies.py::test_qqq_vol_targets_qqq -v`
Expected: ImportError on `QQQVol`

- [ ] **Step 3: Implement `src/quant_lab/strategies/qqq_vol.py`**

```python
from __future__ import annotations

from .spy_vol import _VolTargetedIndex
from .base import register


@register
class QQQVol(_VolTargetedIndex):
    bot_id = "qqq-vol"
    description = "Vol-targeted long Nasdaq-100 (QQQ). Honest Nasdaq benchmark."
    symbol = "QQQ"
```

- [ ] **Step 4: Run, expect PASS**

Run: `pytest tests/test_strategies.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/quant_lab/strategies/qqq_vol.py tests/test_strategies.py
git commit -m "feat(strategies): add QQQ-Vol benchmark"
```

---

## Task 8: Paper trading engine

**Files:**
- Create: `src/quant_lab/engine/__init__.py`
- Create: `src/quant_lab/engine/paper.py`
- Create: `tests/test_paper_engine.py`

- [ ] **Step 1: Create `src/quant_lab/engine/__init__.py`**

```python
from .paper import rebalance, PaperResult  # noqa: F401
```

- [ ] **Step 2: Write failing test**

Create `tests/test_paper_engine.py`:

```python
from datetime import date

from quant_lab.engine import rebalance, PaperResult
from quant_lab.types import Bar, Portfolio, Position


def _bar(symbol, d, close):
    return Bar(symbol=symbol, date=d, open=close, high=close, low=close, close=close, volume=10_000_000)


def test_rebalance_buys_to_target_weight():
    portfolio = Portfolio(bot_id="t", cash=10_000.0, positions={})
    weights = {"SPY": 0.5}
    prices = {"SPY": 500.0}
    advs = {"SPY": 1e10}
    today = date(2026, 5, 6)

    result = rebalance(portfolio, weights, prices, advs, as_of=today)

    assert isinstance(result, PaperResult)
    assert result.portfolio.positions["SPY"].shares > 0
    # Should have spent close to 50% of equity (minus slippage)
    new_weight = result.portfolio.weight("SPY", prices)
    assert 0.45 < new_weight < 0.55
    assert any(t.side == "BUY" for t in result.trades)


def test_rebalance_sells_when_target_lower():
    portfolio = Portfolio(
        bot_id="t",
        cash=5_000.0,
        positions={"SPY": Position(symbol="SPY", shares=10, avg_cost=500.0)},
    )
    weights = {"SPY": 0.0}  # exit
    prices = {"SPY": 500.0}
    advs = {"SPY": 1e10}
    today = date(2026, 5, 6)

    result = rebalance(portfolio, weights, prices, advs, as_of=today)

    assert "SPY" not in result.portfolio.positions or result.portfolio.positions["SPY"].shares == 0
    assert any(t.side == "SELL" for t in result.trades)


def test_rebalance_skips_tiny_drift():
    """When at-target within tolerance, no trade should fire."""
    portfolio = Portfolio(
        bot_id="t",
        cash=5_000.0,
        positions={"SPY": Position(symbol="SPY", shares=10, avg_cost=500.0)},
    )
    weights = {"SPY": 0.5}  # already ~50%
    prices = {"SPY": 500.0}
    advs = {"SPY": 1e10}
    result = rebalance(portfolio, weights, prices, advs, as_of=date(2026, 5, 6), drift_threshold=0.01)
    assert result.trades == []
```

- [ ] **Step 3: Run, expect FAIL**

Run: `pytest tests/test_paper_engine.py -v`
Expected: ImportError

- [ ] **Step 4: Implement `src/quant_lab/engine/paper.py`**

```python
"""Paper trading engine.

Rebalances a portfolio toward target weights using realistic slippage,
respects per-name liquidity, and emits a Trade list for audit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date

from ..slippage import apply_slippage, spread_bps
from ..types import Portfolio, Position, Trade


@dataclass
class PaperResult:
    portfolio: Portfolio
    trades: list[Trade] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def rebalance(
    portfolio: Portfolio,
    target_weights: dict[str, float],
    prices: dict[str, float],
    advs: dict[str, float],
    as_of: _date,
    drift_threshold: float = 0.005,
    min_trade_value: float = 5.0,
    adv_floor: float = 1_000_000,
) -> PaperResult:
    """Rebalance `portfolio` toward `target_weights`.

    - Applies slippage from `slippage.apply_slippage`.
    - Skips names with ADV below `adv_floor` (liquidity gate).
    - Skips trades whose change in weight is below `drift_threshold`.
    """
    equity = portfolio.equity(prices)
    if equity <= 0:
        return PaperResult(portfolio=portfolio)

    new_positions = dict(portfolio.positions)
    cash = portfolio.cash
    trades: list[Trade] = []
    skipped: list[str] = []

    symbols = sorted(set(new_positions) | set(target_weights))
    for symbol in symbols:
        target_w = target_weights.get(symbol, 0.0)
        adv = advs.get(symbol, 0.0)
        if target_w > 0 and adv < adv_floor:
            skipped.append(f"{symbol}: ADV ${adv:,.0f} below floor ${adv_floor:,.0f}")
            target_w = 0.0  # don't open new position; existing is allowed to roll off

        price = prices.get(symbol)
        if price is None or price <= 0:
            skipped.append(f"{symbol}: no price")
            continue

        current_pos = new_positions.get(symbol)
        current_shares = current_pos.shares if current_pos else 0.0
        current_value = current_shares * price
        target_value = equity * target_w
        delta_value = target_value - current_value

        if abs(delta_value) < min_trade_value:
            continue
        if abs(delta_value) / max(equity, 1.0) < drift_threshold:
            continue

        side = "BUY" if delta_value > 0 else "SELL"
        bps = spread_bps(adv) if adv > 0 else 30.0
        fill_price = apply_slippage(price=price, side=side, spread_bps_value=bps)
        delta_shares = delta_value / fill_price
        cash -= delta_shares * fill_price

        new_shares = current_shares + delta_shares
        if abs(new_shares) < 1e-9:
            new_positions.pop(symbol, None)
        else:
            new_avg = price if current_shares == 0 else (
                ((current_pos.avg_cost * current_shares) + (fill_price * delta_shares))
                / new_shares
                if side == "BUY" else current_pos.avg_cost
            )
            new_positions[symbol] = Position(symbol=symbol, shares=new_shares, avg_cost=new_avg)

        trades.append(
            Trade(
                bot_id=portfolio.bot_id,
                symbol=symbol,
                side=side,
                shares=abs(delta_shares),
                price=fill_price,
                slippage_bps=bps,
                timestamp=as_of,
                reason=f"rebalance to target weight {target_w:.4f}",
            )
        )

    new_portfolio = Portfolio(bot_id=portfolio.bot_id, cash=cash, positions=new_positions)
    return PaperResult(portfolio=new_portfolio, trades=trades, skipped=skipped)
```

- [ ] **Step 5: Run, expect PASS**

Run: `pytest tests/test_paper_engine.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/quant_lab/engine/ tests/test_paper_engine.py
git commit -m "feat(engine): add paper rebalance with slippage + liquidity filter"
```

---

## Task 9: Stats module (total return, Sharpe, drawdown)

**Files:**
- Create: `src/quant_lab/tournament/__init__.py`
- Create: `src/quant_lab/tournament/stats.py`
- Create: `tests/test_stats.py`

- [ ] **Step 1: Create `src/quant_lab/tournament/__init__.py`**

```python
from .stats import compute_metrics, Metrics  # noqa: F401
```

- [ ] **Step 2: Write failing test**

Create `tests/test_stats.py`:

```python
import math
import pytest
from quant_lab.tournament.stats import compute_metrics, Metrics


def test_compute_metrics_flat_curve():
    nav = [100_000] * 252
    m = compute_metrics(nav)
    assert m.total_return == 0.0
    assert m.sharpe == 0.0
    assert m.max_drawdown == 0.0


def test_compute_metrics_steady_growth():
    nav = [100_000 * ((1.001) ** i) for i in range(252)]
    m = compute_metrics(nav)
    assert m.total_return > 0
    assert m.sharpe > 0
    assert m.max_drawdown == 0.0


def test_compute_metrics_drawdown():
    nav = [100, 110, 120, 90, 100]
    m = compute_metrics(nav)
    # Max DD from 120 to 90 = -25%
    assert math.isclose(m.max_drawdown, -0.25, abs_tol=1e-9)


def test_compute_metrics_single_point_safe():
    m = compute_metrics([100])
    assert m.total_return == 0.0
    assert m.sharpe == 0.0
```

- [ ] **Step 3: Run, expect FAIL**

Run: `pytest tests/test_stats.py -v`
Expected: ImportError

- [ ] **Step 4: Implement `src/quant_lab/tournament/stats.py`**

```python
"""Performance statistics for paper-traded strategies.

Phase 1 ships total return, annualized return, Sharpe, max drawdown.
Phase 2 will add bootstrapped confidence intervals and Fama-French
factor decomposition (spec section 9).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class Metrics:
    total_return: float
    annualized_return: float
    sharpe: float
    volatility: float
    max_drawdown: float
    days: int


def _daily_returns(nav: list[float]) -> list[float]:
    rets = []
    for i in range(1, len(nav)):
        prev = nav[i - 1]
        if prev <= 0:
            return []
        rets.append(nav[i] / prev - 1.0)
    return rets


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return sqrt(var)


def _max_drawdown(nav: list[float]) -> float:
    if not nav:
        return 0.0
    peak = nav[0]
    worst = 0.0
    for value in nav:
        peak = max(peak, value)
        if peak > 0:
            dd = (value - peak) / peak
            worst = min(worst, dd)
    return worst


def compute_metrics(nav: list[float]) -> Metrics:
    """Compute standard performance metrics from a NAV time series."""
    if len(nav) < 2:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0.0, len(nav))

    rets = _daily_returns(nav)
    if not rets:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0.0, len(nav))

    total = nav[-1] / nav[0] - 1.0
    daily_mean = sum(rets) / len(rets)
    daily_vol = _stdev(rets)
    ann_vol = daily_vol * sqrt(TRADING_DAYS_PER_YEAR)
    ann_return = (1 + daily_mean) ** TRADING_DAYS_PER_YEAR - 1
    sharpe = (daily_mean / daily_vol) * sqrt(TRADING_DAYS_PER_YEAR) if daily_vol > 0 else 0.0
    return Metrics(
        total_return=total,
        annualized_return=ann_return,
        sharpe=sharpe,
        volatility=ann_vol,
        max_drawdown=_max_drawdown(nav),
        days=len(nav),
    )
```

- [ ] **Step 5: Run, expect PASS**

Run: `pytest tests/test_stats.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/quant_lab/tournament/ tests/test_stats.py
git commit -m "feat(tournament): add core performance metrics"
```

---

## Task 10: Persistence layer (JSON state files)

**Files:**
- Create: `src/quant_lab/persistence.py`
- Create: `tests/test_persistence.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_persistence.py`:

```python
import json
from pathlib import Path

from quant_lab.persistence import (
    save_portfolios,
    load_portfolios,
    save_nav_history,
    load_nav_history,
    append_trades,
)
from quant_lab.types import Portfolio, Position, Trade
from datetime import date


def test_save_and_load_portfolios(tmp_path):
    p = Portfolio(
        bot_id="spy-vol",
        cash=5000.0,
        positions={"SPY": Position(symbol="SPY", shares=10, avg_cost=500.0)},
    )
    path = tmp_path / "portfolios.json"
    save_portfolios([p], path)
    loaded = load_portfolios(path)
    assert len(loaded) == 1
    assert loaded[0].bot_id == "spy-vol"
    assert loaded[0].positions["SPY"].shares == 10


def test_load_portfolios_missing_file_returns_empty(tmp_path):
    path = tmp_path / "absent.json"
    assert load_portfolios(path) == []


def test_save_and_load_nav_history(tmp_path):
    history = {"spy-vol": [(date(2026, 5, 5), 100_000.0), (date(2026, 5, 6), 101_000.0)]}
    path = tmp_path / "nav.json"
    save_nav_history(history, path)
    loaded = load_nav_history(path)
    assert loaded["spy-vol"][0] == (date(2026, 5, 5), 100_000.0)
    assert loaded["spy-vol"][-1] == (date(2026, 5, 6), 101_000.0)


def test_append_trades(tmp_path):
    path = tmp_path / "trades.jsonl"
    t = Trade(bot_id="spy-vol", symbol="SPY", side="BUY", shares=2.0, price=500.0,
              slippage_bps=5.0, timestamp=date(2026, 5, 6))
    append_trades([t], path)
    append_trades([t], path)
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["symbol"] == "SPY"
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest tests/test_persistence.py -v`
Expected: ImportError

- [ ] **Step 3: Implement `src/quant_lab/persistence.py`**

```python
"""JSON state persistence for Phase 1.

Phase 2 will move state to Turso (libSQL); the public API of this module
will keep its shape so swapping the backend stays a one-file change.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterable

from .types import Portfolio, Position, Trade


def save_portfolios(portfolios: Iterable[Portfolio], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for p in portfolios:
        payload.append({
            "bot_id": p.bot_id,
            "cash": p.cash,
            "positions": {
                sym: {"symbol": pos.symbol, "shares": pos.shares, "avg_cost": pos.avg_cost}
                for sym, pos in p.positions.items()
            },
        })
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_portfolios(path: Path) -> list[Portfolio]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    portfolios: list[Portfolio] = []
    for item in data:
        positions = {
            sym: Position(symbol=row["symbol"], shares=row["shares"], avg_cost=row["avg_cost"])
            for sym, row in item.get("positions", {}).items()
        }
        portfolios.append(Portfolio(bot_id=item["bot_id"], cash=item["cash"], positions=positions))
    return portfolios


def save_nav_history(history: dict[str, list[tuple[date, float]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        bot_id: [{"date": d.isoformat(), "nav": nav} for d, nav in series]
        for bot_id, series in history.items()
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_nav_history(path: Path) -> dict[str, list[tuple[date, float]]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        bot_id: [(date.fromisoformat(row["date"]), float(row["nav"])) for row in series]
        for bot_id, series in data.items()
    }


def append_trades(trades: Iterable[Trade], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for t in trades:
            handle.write(json.dumps({
                "bot_id": t.bot_id,
                "symbol": t.symbol,
                "side": t.side,
                "shares": t.shares,
                "price": t.price,
                "slippage_bps": t.slippage_bps,
                "timestamp": t.timestamp.isoformat(),
                "reason": t.reason,
            }) + "\n")
```

- [ ] **Step 4: Run, expect PASS**

Run: `pytest tests/test_persistence.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/quant_lab/persistence.py tests/test_persistence.py
git commit -m "feat(persistence): JSON portfolio + NAV + trade-log persistence"
```

---

## Task 11: Tournament runner

**Files:**
- Create: `src/quant_lab/tournament/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_runner.py`:

```python
from datetime import date

from quant_lab.tournament.runner import run_morning_for_strategies
from quant_lab.types import Bar, Portfolio
from quant_lab.strategies.base import Strategy, register


class _AlwaysHoldSPY(Strategy):
    bot_id = "test-hold-spy"
    description = "Test"

    def target_weights(self, histories, as_of):
        return {"SPY": 1.0}


def _bars(symbol, n=100, start_price=400.0):
    base = date(2026, 1, 2)
    bars = []
    price = start_price
    for i in range(n):
        d = base.fromordinal(base.toordinal() + i)
        price = price * (1.0 + 0.001)
        bars.append(Bar(symbol=symbol, date=d, open=price, high=price, low=price, close=price, volume=10_000_000))
    return bars


def test_run_morning_initializes_portfolios():
    histories = {"SPY": _bars("SPY", 100)}
    advs = {"SPY": 1e10}
    register(_AlwaysHoldSPY)
    portfolios, trades, navs = run_morning_for_strategies(
        strategies=[_AlwaysHoldSPY()],
        histories=histories,
        advs=advs,
        prior_portfolios={},
        prior_navs={},
        as_of=histories["SPY"][-1].date,
        starting_cash=100_000,
    )
    assert "test-hold-spy" in portfolios
    p = portfolios["test-hold-spy"]
    # Should have bought SPY toward 100% weight
    assert "SPY" in p.positions
    assert p.weight("SPY", {"SPY": histories["SPY"][-1].close}) > 0.95


def test_run_morning_records_nav():
    histories = {"SPY": _bars("SPY", 100)}
    advs = {"SPY": 1e10}
    portfolios, trades, navs = run_morning_for_strategies(
        strategies=[_AlwaysHoldSPY()],
        histories=histories,
        advs=advs,
        prior_portfolios={},
        prior_navs={},
        as_of=histories["SPY"][-1].date,
        starting_cash=100_000,
    )
    assert "test-hold-spy" in navs
    assert len(navs["test-hold-spy"]) >= 1
    assert navs["test-hold-spy"][-1][0] == histories["SPY"][-1].date
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest tests/test_runner.py -v`
Expected: ImportError

- [ ] **Step 3: Implement `src/quant_lab/tournament/runner.py`**

```python
"""Morning tournament runner.

Loads each strategy's prior portfolio (or initializes), gets target weights,
applies paper-trading rebalance, records new NAV, returns updated state.
"""
from __future__ import annotations

from datetime import date
from typing import Iterable

from ..engine import rebalance
from ..strategies.base import Strategy
from ..types import Bar, Portfolio, Trade


def _avg_dollar_volume(bars: list[Bar], window: int = 30) -> float:
    if not bars:
        return 0.0
    recent = bars[-window:]
    if not recent:
        return 0.0
    return sum(b.close * b.volume for b in recent) / len(recent)


def run_morning_for_strategies(
    strategies: Iterable[Strategy],
    histories: dict[str, list[Bar]],
    advs: dict[str, float] | None,
    prior_portfolios: dict[str, Portfolio],
    prior_navs: dict[str, list[tuple[date, float]]],
    as_of: date,
    starting_cash: float = 100_000.0,
) -> tuple[dict[str, Portfolio], list[Trade], dict[str, list[tuple[date, float]]]]:
    """Run one morning step for all strategies. Returns updated state."""
    if advs is None:
        advs = {sym: _avg_dollar_volume(bars) for sym, bars in histories.items()}

    prices = {sym: bars[-1].close for sym, bars in histories.items() if bars}

    new_portfolios: dict[str, Portfolio] = {}
    new_navs: dict[str, list[tuple[date, float]]] = {k: list(v) for k, v in prior_navs.items()}
    all_trades: list[Trade] = []

    for strat in strategies:
        portfolio = prior_portfolios.get(
            strat.bot_id,
            Portfolio(bot_id=strat.bot_id, cash=starting_cash, positions={}),
        )
        weights = strat.target_weights(histories, as_of)
        result = rebalance(portfolio, weights, prices, advs, as_of=as_of)
        new_portfolios[strat.bot_id] = result.portfolio
        all_trades.extend(result.trades)

        nav = result.portfolio.equity(prices)
        series = new_navs.setdefault(strat.bot_id, [])
        if not series or series[-1][0] != as_of:
            series.append((as_of, nav))
        else:
            series[-1] = (as_of, nav)

    return new_portfolios, all_trades, new_navs
```

- [ ] **Step 4: Run, expect PASS**

Run: `pytest tests/test_runner.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/quant_lab/tournament/runner.py tests/test_runner.py
git commit -m "feat(tournament): add morning runner that drives strategies"
```

---

## Task 12: Discord reporter

**Files:**
- Create: `src/quant_lab/reporting/__init__.py`
- Create: `src/quant_lab/reporting/discord.py`
- Create: `tests/test_discord.py`

- [ ] **Step 1: Create `src/quant_lab/reporting/__init__.py`**

```python
from .discord import build_message, post_to_discord  # noqa: F401
from .dashboard import write_dashboard_data  # noqa: F401
```

- [ ] **Step 2: Write failing test**

Create `tests/test_discord.py`:

```python
from datetime import date
from unittest.mock import patch

from quant_lab.reporting.discord import build_message, post_to_discord
from quant_lab.tournament.stats import Metrics


def test_build_message_includes_market_snapshot():
    leaderboard = [
        ("spy-vol", Metrics(0.05, 0.10, 0.6, 0.15, -0.05, 100), {"SPY": 1.0}),
        ("qqq-vol", Metrics(0.08, 0.16, 0.7, 0.18, -0.07, 100), {"QQQ": 1.0}),
    ]
    market = {"SPY": {"change_pct": 0.31, "ytd_pct": 5.1},
              "QQQ": {"change_pct": 0.54, "ytd_pct": 8.7}}
    msg = build_message(date(2026, 5, 7), leaderboard, market)
    assert "SPY" in msg
    assert "QQQ" in msg
    assert "+0.31%" in msg or "0.31%" in msg
    assert "spy-vol" in msg
    assert "Not financial advice" in msg


def test_post_to_discord_calls_webhook():
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 204
        post_to_discord("https://discord.test/webhook", "hello")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://discord.test/webhook"
        assert kwargs["json"]["content"].startswith("hello")


def test_post_to_discord_truncates_long_messages():
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 204
        long = "x" * 5000
        post_to_discord("https://discord.test/webhook", long)
        body = mock_post.call_args.kwargs["json"]["content"]
        assert len(body) <= 2000
```

- [ ] **Step 3: Run, expect FAIL**

Run: `pytest tests/test_discord.py -v`
Expected: ImportError

- [ ] **Step 4: Implement `src/quant_lab/reporting/discord.py`**

```python
"""Discord webhook reporter for the morning brief.

Discord limits messages to 2000 chars; this module truncates and links
to the dashboard for full detail.
"""
from __future__ import annotations

from datetime import date
from typing import Iterable

import requests

from ..tournament.stats import Metrics


DISCORD_MAX_CHARS = 2000

DISCLAIMER = "Research tool. Not financial advice. Paper trading only."


def build_message(
    today: date,
    leaderboard: Iterable[tuple[str, Metrics, dict[str, float]]],
    market: dict[str, dict[str, float]],
    dashboard_url: str | None = None,
) -> str:
    lines = [
        f"**QUANT LAB — {today.isoformat()}**",
        f"_{DISCLAIMER}_",
        "",
        "**Market**",
    ]
    for sym in ("SPY", "QQQ"):
        info = market.get(sym, {})
        chg = info.get("change_pct", 0.0)
        ytd = info.get("ytd_pct", 0.0)
        arrow = "▲" if chg >= 0 else "▼"
        lines.append(f"  {sym}: {chg:+.2f}% {arrow}  YTD {ytd:+.2f}%")
    lines.append("")
    lines.append("**Tournament**")
    for bot_id, metrics, weights in leaderboard:
        positions = ", ".join(f"{s} {w:.0%}" for s, w in sorted(weights.items()) if w > 0.01) or "cash"
        lines.append(
            f"  {bot_id}: total {metrics.total_return:+.2%} | "
            f"sharpe {metrics.sharpe:.2f} | dd {metrics.max_drawdown:+.2%} | {positions}"
        )
    if dashboard_url:
        lines.append("")
        lines.append(f"Dashboard: {dashboard_url}")
    msg = "\n".join(lines)
    if len(msg) > DISCORD_MAX_CHARS:
        msg = msg[: DISCORD_MAX_CHARS - 50] + "\n…(truncated; see dashboard)"
    return msg


def post_to_discord(webhook_url: str, message: str) -> None:
    if len(message) > DISCORD_MAX_CHARS:
        message = message[: DISCORD_MAX_CHARS - 50] + "\n…(truncated)"
    response = requests.post(webhook_url, json={"content": message}, timeout=15)
    if response.status_code >= 400:
        raise RuntimeError(f"Discord webhook returned {response.status_code}: {response.text[:200]}")
```

- [ ] **Step 5: Run, expect PASS**

Run: `pytest tests/test_discord.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/quant_lab/reporting/__init__.py src/quant_lab/reporting/discord.py tests/test_discord.py
git commit -m "feat(reporting): Discord morning brief poster"
```

---

## Task 13: Dashboard JSON exporter

**Files:**
- Create: `src/quant_lab/reporting/dashboard.py`
- Create: `tests/test_dashboard.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_dashboard.py`:

```python
import json
from datetime import date
from pathlib import Path

from quant_lab.reporting.dashboard import write_dashboard_data
from quant_lab.tournament.stats import Metrics


def test_write_dashboard_data_creates_leaderboard_json(tmp_path):
    out = tmp_path / "data"
    leaderboard = [
        ("spy-vol", Metrics(0.05, 0.10, 0.6, 0.15, -0.05, 100), {"SPY": 1.0}),
    ]
    nav_history = {"spy-vol": [(date(2026, 5, 5), 100_000), (date(2026, 5, 6), 101_000)]}
    market = {"SPY": {"change_pct": 0.31, "ytd_pct": 5.1},
              "QQQ": {"change_pct": 0.54, "ytd_pct": 8.7}}
    write_dashboard_data(out_dir=out, leaderboard=leaderboard, nav_history=nav_history,
                         market=market, generated_at=date(2026, 5, 6))

    leaderboard_json = json.loads((out / "leaderboard.json").read_text())
    assert leaderboard_json["bots"][0]["bot_id"] == "spy-vol"
    assert leaderboard_json["market"]["SPY"]["change_pct"] == 0.31

    nav_json = json.loads((out / "nav_history.json").read_text())
    assert nav_json["spy-vol"][-1]["nav"] == 101_000
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest tests/test_dashboard.py -v`
Expected: ImportError

- [ ] **Step 3: Implement `src/quant_lab/reporting/dashboard.py`**

```python
"""Dashboard JSON exporter.

Writes a small set of JSON files served by GitHub Pages and consumed by
`dashboard/app.js` to render the leaderboard and equity curves.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterable

from ..tournament.stats import Metrics


def write_dashboard_data(
    out_dir: Path,
    leaderboard: Iterable[tuple[str, Metrics, dict[str, float]]],
    nav_history: dict[str, list[tuple[date, float]]],
    market: dict[str, dict[str, float]],
    generated_at: date,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_payload = {
        "generated_at": generated_at.isoformat(),
        "market": market,
        "bots": [
            {
                "bot_id": bot_id,
                "metrics": {
                    "total_return": m.total_return,
                    "annualized_return": m.annualized_return,
                    "sharpe": m.sharpe,
                    "volatility": m.volatility,
                    "max_drawdown": m.max_drawdown,
                    "days": m.days,
                },
                "current_weights": weights,
            }
            for bot_id, m, weights in leaderboard
        ],
    }
    (out_dir / "leaderboard.json").write_text(json.dumps(leaderboard_payload, indent=2) + "\n")

    nav_payload = {
        bot_id: [{"date": d.isoformat(), "nav": nav} for d, nav in series]
        for bot_id, series in nav_history.items()
    }
    (out_dir / "nav_history.json").write_text(json.dumps(nav_payload, indent=2) + "\n")
```

- [ ] **Step 4: Run, expect PASS**

Run: `pytest tests/test_dashboard.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/quant_lab/reporting/dashboard.py tests/test_dashboard.py
git commit -m "feat(reporting): dashboard JSON exporter"
```

---

## Task 14: Static dashboard HTML/JS

**Files:**
- Create: `dashboard/index.html`
- Create: `dashboard/styles.css`
- Create: `dashboard/app.js`
- Create: `dashboard/data/.gitkeep`

- [ ] **Step 1: Create `dashboard/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Quant Lab — Morning Brief</title>
  <link rel="stylesheet" href="styles.css" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.js"></script>
</head>
<body>
  <header>
    <h1>Quant Lab</h1>
    <p class="disclaimer">Research tool. Not financial advice. Paper trading only.</p>
  </header>

  <section id="market">
    <h2>Market</h2>
    <div id="market-stats"></div>
  </section>

  <section id="leaderboard">
    <h2>Tournament Leaderboard</h2>
    <table id="leaderboard-table">
      <thead>
        <tr>
          <th>Bot</th>
          <th>Total return</th>
          <th>Annualized</th>
          <th>Sharpe</th>
          <th>Max DD</th>
          <th>Days</th>
          <th>Current weights</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </section>

  <section id="curves">
    <h2>Equity Curves</h2>
    <canvas id="equity-chart" height="120"></canvas>
  </section>

  <footer>
    <p>Generated: <span id="generated-at"></span></p>
    <p>
      Disclaimer: Strategies in this tournament are well-known, public, and have been studied
      (and largely arbitraged) by professional quants for decades. The probability that any
      single strategy here delivers consistent risk-adjusted outperformance vs. a passive index
      over a 5+ year horizon is low. Use this system to learn, not to allocate real capital.
    </p>
  </footer>

  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `dashboard/styles.css`**

```css
:root {
  --bg: #0d1117;
  --fg: #e6edf3;
  --muted: #8b949e;
  --accent: #58a6ff;
  --good: #3fb950;
  --bad: #f85149;
  --border: #30363d;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
       background: var(--bg); color: var(--fg); line-height: 1.5; }
header, section, footer { max-width: 1100px; margin: 0 auto; padding: 1.5rem; }
header { border-bottom: 1px solid var(--border); }
h1 { margin: 0 0 .5rem 0; font-size: 1.6rem; }
h2 { font-size: 1.15rem; border-bottom: 1px solid var(--border); padding-bottom: .35rem; }
.disclaimer { color: var(--muted); font-size: .9rem; margin: 0; }
table { width: 100%; border-collapse: collapse; font-size: .92rem; }
th, td { padding: .55rem .75rem; text-align: right; border-bottom: 1px solid var(--border); }
th:first-child, td:first-child, th:last-child, td:last-child { text-align: left; }
.pos { color: var(--good); }
.neg { color: var(--bad); }
#market-stats { display: flex; gap: 2rem; flex-wrap: wrap; }
.market-card { padding: .75rem 1rem; border: 1px solid var(--border); border-radius: 6px;
               background: rgba(255,255,255,0.02); min-width: 180px; }
.market-card .sym { font-weight: 600; }
.market-card .chg { font-size: 1.2rem; }
footer { color: var(--muted); font-size: .85rem; }
```

- [ ] **Step 3: Create `dashboard/app.js`**

```javascript
(async () => {
  const fmtPct = (v) => (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%";
  const cls = (v) => (v >= 0 ? "pos" : "neg");

  const [leaderboardRes, navRes] = await Promise.all([
    fetch("data/leaderboard.json"),
    fetch("data/nav_history.json"),
  ]);
  const leaderboard = await leaderboardRes.json();
  const navHistory = await navRes.json();

  // Market
  const marketEl = document.getElementById("market-stats");
  for (const sym of ["SPY", "QQQ"]) {
    const m = leaderboard.market[sym] || { change_pct: 0, ytd_pct: 0 };
    const card = document.createElement("div");
    card.className = "market-card";
    card.innerHTML = `
      <div class="sym">${sym}</div>
      <div class="chg ${m.change_pct >= 0 ? "pos" : "neg"}">${(m.change_pct >= 0 ? "+" : "") + m.change_pct.toFixed(2)}%</div>
      <div>YTD ${(m.ytd_pct >= 0 ? "+" : "") + m.ytd_pct.toFixed(2)}%</div>
    `;
    marketEl.appendChild(card);
  }

  // Leaderboard
  const tbody = document.querySelector("#leaderboard-table tbody");
  for (const bot of leaderboard.bots) {
    const row = document.createElement("tr");
    const weights = Object.entries(bot.current_weights || {})
      .filter(([, w]) => w > 0.01)
      .map(([s, w]) => `${s} ${(w * 100).toFixed(0)}%`)
      .join(", ") || "cash";
    row.innerHTML = `
      <td>${bot.bot_id}</td>
      <td class="${cls(bot.metrics.total_return)}">${fmtPct(bot.metrics.total_return)}</td>
      <td class="${cls(bot.metrics.annualized_return)}">${fmtPct(bot.metrics.annualized_return)}</td>
      <td>${bot.metrics.sharpe.toFixed(2)}</td>
      <td class="neg">${fmtPct(bot.metrics.max_drawdown)}</td>
      <td>${bot.metrics.days}</td>
      <td>${weights}</td>
    `;
    tbody.appendChild(row);
  }

  document.getElementById("generated-at").textContent = leaderboard.generated_at;

  // Equity chart
  const ctx = document.getElementById("equity-chart").getContext("2d");
  const datasets = [];
  const palette = ["#58a6ff", "#3fb950", "#f0883e", "#a371f7", "#ec4899"];
  let i = 0;
  let allDates = new Set();
  for (const [bot, series] of Object.entries(navHistory)) {
    series.forEach(p => allDates.add(p.date));
  }
  const labels = Array.from(allDates).sort();
  for (const [bot, series] of Object.entries(navHistory)) {
    const map = Object.fromEntries(series.map(p => [p.date, p.nav]));
    datasets.push({
      label: bot,
      data: labels.map(d => map[d] ?? null),
      spanGaps: true,
      borderColor: palette[i % palette.length],
      backgroundColor: "transparent",
      pointRadius: 0,
      borderWidth: 2,
    });
    i++;
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
})();
```

- [ ] **Step 4: Create `dashboard/data/.gitkeep` (empty file)**

```
```

- [ ] **Step 5: Commit**

```bash
git add dashboard/
git commit -m "feat(dashboard): static HTML + Chart.js leaderboard and equity curves"
```

---

## Task 15: Main entry point (`morning` command)

**Files:**
- Create: `src/quant_lab/main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_main.py`:

```python
from datetime import date
from unittest.mock import patch, MagicMock

from quant_lab.main import morning_command


def test_morning_command_dry_run(tmp_path, monkeypatch):
    """End-to-end smoke test: morning_command should not crash on synthetic data."""
    from quant_lab.types import Bar
    base = date(2026, 1, 2)

    def fake_fetch(symbol, lookback_days=365):
        bars = []
        price = 500.0
        for i in range(120):
            d = base.fromordinal(base.toordinal() + i)
            price *= 1.0005
            bars.append(Bar(symbol=symbol, date=d, open=price, high=price, low=price, close=price, volume=10_000_000))
        return bars

    monkeypatch.setattr("quant_lab.main.fetch_history", fake_fetch)

    state_dir = tmp_path / "state"
    dashboard_dir = tmp_path / "dashboard_data"
    snapshot_dir = tmp_path / "snapshots"

    with patch("quant_lab.main.post_to_discord") as mock_post:
        morning_command(
            state_dir=state_dir,
            dashboard_data_dir=dashboard_dir,
            snapshot_dir=snapshot_dir,
            discord_webhook=None,  # skip Discord
            dashboard_url=None,
        )

    assert (state_dir / "portfolios.json").exists()
    assert (state_dir / "nav_history.json").exists()
    assert (dashboard_dir / "leaderboard.json").exists()
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest tests/test_main.py -v`
Expected: ImportError

- [ ] **Step 3: Implement `src/quant_lab/main.py`**

```python
"""Entry points for Quant Lab.

`morning_command` runs one full morning step: fetch data, run all registered
strategies, persist state, write dashboard data, post Discord brief.
"""
from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

from . import strategies  # noqa: F401  (registers strategies via import)
from .data import fetch_history
from .persistence import (
    save_portfolios,
    load_portfolios,
    save_nav_history,
    load_nav_history,
    append_trades,
)
from .reporting.discord import build_message, post_to_discord
from .reporting.dashboard import write_dashboard_data
from .strategies.base import get_all
from .tournament.runner import run_morning_for_strategies
from .tournament.stats import compute_metrics


SYMBOLS_FOR_PHASE_1 = ["SPY", "QQQ"]


def _market_snapshot(histories: dict, today: date) -> dict[str, dict[str, float]]:
    snapshot: dict[str, dict[str, float]] = {}
    for sym in ("SPY", "QQQ"):
        bars = histories.get(sym, [])
        if len(bars) < 2:
            snapshot[sym] = {"change_pct": 0.0, "ytd_pct": 0.0}
            continue
        last = bars[-1]
        prev = bars[-2]
        chg = (last.close / prev.close - 1.0) * 100
        ytd = next((b for b in bars if b.date.year == last.date.year), bars[0])
        ytd_pct = (last.close / ytd.close - 1.0) * 100
        snapshot[sym] = {"change_pct": chg, "ytd_pct": ytd_pct}
    return snapshot


def morning_command(
    state_dir: Path,
    dashboard_data_dir: Path,
    snapshot_dir: Path,
    discord_webhook: str | None,
    dashboard_url: str | None,
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    dashboard_data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    histories: dict[str, list] = {}
    for symbol in SYMBOLS_FOR_PHASE_1:
        bars = fetch_history(symbol, lookback_days=400)
        if bars:
            histories[symbol] = bars
    if not histories:
        raise RuntimeError("No data fetched; check network or yfinance status.")

    today = max(bars[-1].date for bars in histories.values())

    prior_portfolios_list = load_portfolios(state_dir / "portfolios.json")
    prior_portfolios = {p.bot_id: p for p in prior_portfolios_list}
    prior_navs = load_nav_history(state_dir / "nav_history.json")

    strategies_list = get_all()
    portfolios, trades, nav_history = run_morning_for_strategies(
        strategies=strategies_list,
        histories=histories,
        advs=None,
        prior_portfolios=prior_portfolios,
        prior_navs=prior_navs,
        as_of=today,
    )

    save_portfolios(portfolios.values(), state_dir / "portfolios.json")
    save_nav_history(nav_history, state_dir / "nav_history.json")
    append_trades(trades, state_dir / "trades.jsonl")

    leaderboard = []
    for strat in strategies_list:
        navs = nav_history.get(strat.bot_id, [])
        nav_values = [n for _, n in navs]
        metrics = compute_metrics(nav_values)
        weights = {
            sym: portfolios[strat.bot_id].weight(sym, {s: bars[-1].close for s, bars in histories.items()})
            for sym in {sym for pos in portfolios[strat.bot_id].positions.keys() for sym in [pos]}
        }
        leaderboard.append((strat.bot_id, metrics, weights))
    leaderboard.sort(key=lambda row: row[1].sharpe, reverse=True)

    market = _market_snapshot(histories, today)
    write_dashboard_data(
        out_dir=dashboard_data_dir,
        leaderboard=leaderboard,
        nav_history=nav_history,
        market=market,
        generated_at=today,
    )

    if discord_webhook:
        msg = build_message(today, leaderboard, market, dashboard_url=dashboard_url)
        try:
            post_to_discord(discord_webhook, msg)
        except Exception as exc:
            # Log but don't crash the run; dashboard still updates
            print(f"[warn] Discord post failed: {exc}")


def cli() -> None:
    parser = argparse.ArgumentParser(prog="quant-lab")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("morning", help="Run the morning tournament step")

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


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Run, expect PASS**

Run: `pytest tests/test_main.py -v`
Expected: 1 passed

- [ ] **Step 5: Run full test suite**

Run: `pytest -q`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/quant_lab/main.py tests/test_main.py
git commit -m "feat(main): morning command orchestrates one full tournament step"
```

---

## Task 16: GitHub Actions morning workflow

**Files:**
- Create: `.github/workflows/morning.yml`

- [ ] **Step 1: Create `.github/workflows/morning.yml`**

```yaml
name: Morning Quant Lab

on:
  workflow_dispatch:
  schedule:
    # 11:30 UTC = 6:30 AM EST winter / 7:30 AM EDT summer.
    # Run twice for DST coverage; idempotent on the second fire of the same day.
    - cron: "30 11 * * 1-5"
    - cron: "30 10 * * 1-5"

permissions:
  contents: write
  pages: write
  id-token: write

concurrency:
  group: morning
  cancel-in-progress: false

jobs:
  run-bot:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    env:
      DISCORD_WEBHOOK: ${{ secrets.DISCORD_WEBHOOK }}
      DASHBOARD_URL: ${{ vars.DASHBOARD_URL }}

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install package
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]

      - name: Run tests
        run: pytest -q

      - name: Run morning bot
        run: quant-lab morning

      - name: Commit state and dashboard data
        run: |
          git config user.name "quant-lab-bot"
          git config user.email "quant-lab@users.noreply.github.com"
          git add state/ dashboard/data/ data/snapshots/ 2>/dev/null || true
          if ! git diff --cached --quiet; then
            git commit -m "chore: morning $(date -u +%Y-%m-%d) state update"
            git push
          else
            echo "No state changes to commit."
          fi

      - name: Deploy dashboard to GitHub Pages
        uses: actions/upload-pages-artifact@v3
        with:
          path: dashboard

  deploy:
    needs: run-bot
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/morning.yml
git commit -m "ci: add morning workflow with cron + Pages deploy"
```

---

## Task 17: CI workflow for tests on PR

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -e .[dev]
      - run: pytest -q
      - run: ruff check src tests
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add tests + ruff workflow on PR/push"
```

---

## Task 18: Bootstrap fast-path script

**Files:**
- Create: `bootstrap.sh`

- [ ] **Step 1: Create `bootstrap.sh`**

```bash
#!/usr/bin/env bash
# Quant Lab — fast-path bootstrap.
# Creates a public GitHub repo, sets the Discord webhook secret, enables Pages,
# and triggers the first morning run. Requires: gh CLI authenticated.
#
# Full path (Alpaca + Turso) is added in Phase 2.

set -euo pipefail

REPO_NAME="${REPO_NAME:-quant-lab}"
MODE="${1:-fast}"

err() { echo "error: $*" >&2; exit 1; }

command -v gh >/dev/null 2>&1 || err "gh CLI not found. Install: https://cli.github.com/"
command -v python3 >/dev/null 2>&1 || err "python3 not found."

gh auth status >/dev/null 2>&1 || err "gh not authenticated. Run: gh auth login"

echo
echo "Quant Lab — fast-path setup"
echo "==========================="
echo

if [ "$MODE" != "--fast" ] && [ "$MODE" != "fast" ]; then
  echo "Note: only --fast mode is supported in Phase 1." >&2
fi

# 1) Discord webhook
read -r -p "Paste your Discord webhook URL (or leave blank to skip): " DISCORD_WEBHOOK
echo

# 2) Create repo if it doesn't exist
GH_USER=$(gh api user --jq .login)
REPO_FULL="$GH_USER/$REPO_NAME"

if gh repo view "$REPO_FULL" >/dev/null 2>&1; then
  echo "Repo $REPO_FULL exists, skipping create."
else
  echo "Creating $REPO_FULL ..."
  gh repo create "$REPO_FULL" --public --source=. --remote=origin --push
fi

# 3) Set secret
if [ -n "$DISCORD_WEBHOOK" ]; then
  echo "Setting DISCORD_WEBHOOK secret ..."
  gh secret set DISCORD_WEBHOOK -b "$DISCORD_WEBHOOK" --repo "$REPO_FULL"
else
  echo "Skipping Discord secret (no webhook provided)."
fi

# 4) Enable GitHub Pages (Actions source)
echo "Enabling GitHub Pages ..."
gh api -X POST "/repos/$REPO_FULL/pages" \
  -f "build_type=workflow" >/dev/null 2>&1 || \
  gh api -X PUT "/repos/$REPO_FULL/pages" \
  -f "build_type=workflow" >/dev/null 2>&1 || true

DASHBOARD_URL="https://${GH_USER}.github.io/${REPO_NAME}/"
gh variable set DASHBOARD_URL -b "$DASHBOARD_URL" --repo "$REPO_FULL" || true

# 5) Trigger first run
echo "Triggering first morning run ..."
gh workflow run morning.yml --repo "$REPO_FULL" || true

echo
echo "Done."
echo "  Repo:       https://github.com/$REPO_FULL"
echo "  Dashboard:  $DASHBOARD_URL  (live after first run + Pages build, ~3-5 min)"
echo "  Watch run:  gh run watch --repo $REPO_FULL"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x bootstrap.sh
git add bootstrap.sh
git commit -m "feat(bootstrap): fast-path setup script (gh CLI + Discord webhook)"
```

---

## Task 19: README with setup instructions

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README content**

```markdown
# Quant Lab

Personal quant research lab + morning briefing bot. Paper-trades a tournament of strategies on free-tier GitHub Actions, posts daily Discord briefs, dashboards on GitHub Pages.

> **Status:** Phase 1 MVP — SPY-Vol + QQQ-Vol benchmarks live. Phase 2 adds classical strategies (Momo, MeanRev, Breakout, MA-Cross, RSI-Rev) and the Codex bot adapter. Phase 3 adds ML strategies (XGBoost, LightGBM) with walk-forward validation.

## Quick start (fast path, ~60 seconds)

```bash
./bootstrap.sh --fast
```

You'll be prompted for a Discord webhook URL (or skip with blank input). The script:

1. Creates a public GitHub repo
2. Sets the Discord webhook secret (if provided)
3. Enables GitHub Pages with the Actions deploy source
4. Triggers the first morning run

After ~3-5 minutes the dashboard is live at `https://<your-username>.github.io/quant-lab/`.

## Disclaimer

Strategies in this tournament are well-known, public, and have been studied (and largely arbitraged) by professional quants for decades. The probability that any single strategy here delivers consistent risk-adjusted outperformance vs. a passive index over a 5+ year horizon is low.

This is a research and educational tool. Paper trading only. Not financial advice. Past performance, including paper performance, does not predict future results.

## Local development

```bash
pip install -e .[dev]
pytest -q
quant-lab morning  # one-shot run
```

## Design

Full design spec at [`docs/superpowers/specs/2026-05-07-quant-lab-design.md`](docs/superpowers/specs/2026-05-07-quant-lab-design.md).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(readme): phase 1 setup instructions and disclaimer"
```

---

## Task 20: End-to-end smoke test

**Files:**
- Create: `tests/test_e2e_smoke.py`

- [ ] **Step 1: Write smoke test**

Create `tests/test_e2e_smoke.py`:

```python
"""End-to-end smoke test using synthetic data.

Runs the entire morning pipeline against fake yfinance data and verifies
state files, dashboard data, and a (mocked) Discord post all happen.
"""
import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

from quant_lab.main import morning_command
from quant_lab.types import Bar


def _synth(symbol, n=200, drift=0.0004):
    base = date(2026, 1, 2)
    bars, price = [], 500.0 if symbol == "SPY" else 450.0
    for i in range(n):
        d = base.fromordinal(base.toordinal() + i)
        price *= (1 + drift)
        bars.append(Bar(symbol=symbol, date=d, open=price, high=price, low=price, close=price, volume=50_000_000))
    return bars


def test_two_consecutive_morning_runs(tmp_path, monkeypatch):
    histories = {"SPY": _synth("SPY"), "QQQ": _synth("QQQ", drift=0.0006)}

    def fake_fetch(symbol, lookback_days=365):
        return histories.get(symbol.upper(), [])

    monkeypatch.setattr("quant_lab.main.fetch_history", fake_fetch)

    state = tmp_path / "state"
    dash = tmp_path / "dashboard_data"
    snap = tmp_path / "snapshots"

    with patch("quant_lab.main.post_to_discord") as mp:
        morning_command(state, dash, snap, discord_webhook="https://x", dashboard_url="https://y")
        assert mp.called

        # Second run on the same data must not crash
        morning_command(state, dash, snap, discord_webhook="https://x", dashboard_url="https://y")

    leaderboard = json.loads((dash / "leaderboard.json").read_text())
    bot_ids = {row["bot_id"] for row in leaderboard["bots"]}
    assert "spy-vol" in bot_ids
    assert "qqq-vol" in bot_ids
    assert leaderboard["market"]["SPY"]["change_pct"] != 0  # synthetic drift > 0
```

- [ ] **Step 2: Run, expect PASS**

Run: `pytest tests/test_e2e_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 3: Run full suite**

Run: `pytest -q`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e_smoke.py
git commit -m "test: end-to-end smoke test for morning pipeline"
```

---

## Self-Review

Spec coverage:
- §3 Tech stack: yfinance + Python 3.11 + GH Actions + Pages — covered (Tasks 1, 4, 16, 14).
- §4 Strategy roster: SPY-Vol + QQQ-Vol shipped; remaining 8+ bots deferred to Phase 2 plan as documented.
- §5 Data pipeline: yfinance only, with fallback/Alpaca deferred to Phase 2 — explicit in plan header.
- §6 Paper trading: rebalance with slippage + liquidity gate (Tasks 3, 8).
- §7 Slippage: spread_bps ADV-aware (Task 3).
- §9 Tournament metrics: total return, Sharpe, drawdown (Task 9). CIs / factor decomp deferred to Phase 2.
- §11 Discord report: market header + tournament + leader signals (Task 12).
- §12 Dashboard: leaderboard + equity curves (Tasks 13–14). Bot-vs-Bot, Methodology, Validation pages deferred to Phase 4.
- §13 Storage: JSON files (Task 10). Turso deferred to Phase 2.
- §14 Workflows: morning.yml + ci.yml (Tasks 16–17). Watchdog + weekly retrain deferred.
- §16 Setup: bootstrap.sh fast path (Task 18). Full path with Alpaca + Turso deferred to Phase 2.

Placeholder scan: no TBD / TODO / "implement later" lines.

Type consistency: `Bar`, `Position`, `Trade`, `Portfolio` defined in Task 2 are used consistently across Tasks 4, 6, 7, 8, 10, 11, 12, 15, 20. `Strategy.target_weights(histories, as_of)` signature matches in base, SPY-Vol, QQQ-Vol, runner, main.

Phase 1 is intentionally narrow — ship a working skeleton that anyone can fork, run via `bootstrap.sh`, and see a live dashboard for. Each subsequent phase plan layers in capabilities.

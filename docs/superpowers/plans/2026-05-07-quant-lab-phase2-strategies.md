# Quant Lab Phase 2 — Classical Strategies + Codex Adapter + Live Tournament Upgrades

> **For agentic workers:** Use superpowers:subagent-driven-development. Steps use `- [ ]`. Plan format is intentionally compact — function signatures + behaviors + acceptance tests. Implementers fill in bodies following Phase 1 patterns.

**Goal:** Ship 5 classical strategies (Momo, MeanRev, Breakout, MA-Cross, RSI-Rev), the Codex-bot adapter (frozen snapshot at `competitors/codex-bot-snapshot-2026-05-07/`), an R1000 universe + watchlist loader, and live-tournament rigor upgrades (bootstrap CIs, 3-factor decomposition, significance flags, VIX regime kill-switch, watchdog). Every new strategy ships with a walk-forward calibration report from the harness.

**Architecture:** All new strategies plug into the existing `Strategy` ABC + registry. Universe loader replaces the hard-coded `["SPY", "QQQ"]` list in `main.py`. Tournament stats extend the existing `Metrics` dataclass with bootstrap CI fields and factor decomposition. Regime kill-switch reads VIX from yfinance (`^VIX`) and pauses bots based on VIX threshold + per-bot drawdown. Watchdog is a separate GH Actions workflow.

**Tech stack:** Same as Phase 1.5 + nothing new. Universe data from Wikipedia archived R1000 lists (free, point-in-time approximation). VIX from yfinance.

---

## File Structure

```
src/quant_lab/
├── data/
│   ├── universe.py              # NEW — load R1000 + watchlist
│   └── (existing data.py stays as-is)
├── strategies/
│   ├── momo.py                  # NEW — cross-sectional 6mo momentum
│   ├── meanrev.py               # NEW — short-term mean reversion
│   ├── breakout.py              # NEW — 52w-high breakout
│   ├── ma_cross.py              # NEW — 50/200 SMA golden cross
│   ├── rsi_rev.py               # NEW — RSI<30 reversal (negative control)
│   ├── codex_bot.py             # NEW — adapter for frozen Codex snapshot
│   └── (existing __init__, base, spy_vol, qqq_vol stay)
├── tournament/
│   ├── stats.py                 # EXTEND — bootstrap CIs + factor decomp + sig flags
│   ├── factors.py               # NEW — 3-factor decomposition (mkt, size, value)
│   └── (existing __init__, runner stay)
├── engine/
│   ├── regime.py                # NEW — VIX kill-switch
│   └── (existing __init__, paper stay)
├── main.py                      # MODIFY — wire R1000 universe + regime check + extended metrics
└── (everything else stays)

config/
├── watchlist.example.txt        # NEW — sample tickers per line
└── universe_r1000.txt           # NEW — committed snapshot of R1000 constituents

competitors/codex-bot-snapshot-2026-05-07/  # already exists; we read from it

.github/workflows/
└── watchdog.yml                 # NEW — checks if morning ran; alerts if not

tests/
├── test_universe.py
├── test_momo.py
├── test_meanrev.py
├── test_breakout.py
├── test_ma_cross.py
├── test_rsi_rev.py
├── test_codex_bot.py
├── test_factors.py
├── test_regime.py
└── test_phase2_e2e.py
```

---

## Task 1: R1000 universe + watchlist loader

**Files:**
- Create: `src/quant_lab/data/__init__.py` (re-export `fetch_history`, `fetch_many`, `latest_bar` from existing `data.py`)
- Create: `src/quant_lab/data/universe.py`
- Create: `config/universe_r1000.txt` (start with ~150 mega-cap stocks committed; loader pads from yfinance later)
- Create: `config/watchlist.example.txt`
- Create: `tests/test_universe.py`

**Important refactor:** Phase 1 has `src/quant_lab/data.py`; we want a `src/quant_lab/data/` package. Option A: rename `data.py` → `data/_yfinance.py` and re-export from `data/__init__.py`. Option B: keep `data.py` and create `data/universe.py` as a sibling — Python allows both `data.py` and `data/` only if `data.py` is renamed first.

**Go with Option A.** Move `src/quant_lab/data.py` → `src/quant_lab/data/_fetcher.py`. New `src/quant_lab/data/__init__.py` re-exports everything from `_fetcher`. Add `src/quant_lab/data/universe.py`. All existing imports (`from .data import fetch_history` etc.) continue to work.

**`universe.py` API:**

```python
def load_universe(
    universe_path: Path,
    watchlist_path: Path | None = None,
) -> list[str]:
    """Load tickers. Strip whitespace, skip lines starting with '#' or empty.
    Always include SPY, QQQ, ^VIX (case-insensitive dedupe). Returns sorted unique."""

def parse_universe_text(content: str) -> list[str]:
    """Test-friendly: parse content into list of tickers (no I/O)."""
```

**`config/universe_r1000.txt` initial seed** — start with this header + ~150 lines of mega-cap tickers (no need for actual full R1000 in v1; we add an Importing-note comment that future runs may pull from Wikipedia):

```
# Quant Lab universe — currently a curated list of large/mid caps.
# Phase 2 ships with this seed; Phase 2.5 may extend by scraping Wikipedia's
# Russell 1000 list. Add tickers one per line, lines starting with # are ignored.
SPY
QQQ
^VIX
AAPL
MSFT
NVDA
GOOGL
AMZN
META
TSLA
... (extend to ~150 large-cap tickers — implementer can pull from a static list of S&P 500 constituents at https://en.wikipedia.org/wiki/List_of_S%26P_500_companies; commit the snapshot to repo)
```

**Acceptance tests** (`tests/test_universe.py`):
- `parse_universe_text` skips comments and blanks
- `load_universe` always includes SPY, QQQ, ^VIX even if missing from file
- Empty watchlist path returns just universe
- Duplicates deduped case-insensitively (`AAPL` and `aapl` collapse)
- Sorted output

**Steps:**
- [ ] Move `data.py` → `data/_fetcher.py`; create `data/__init__.py` re-exporting all symbols
- [ ] Verify `pytest -q` still passes (38+ tests should still pass after refactor)
- [ ] Write `tests/test_universe.py` with 4-5 tests covering parse + load
- [ ] Implement `data/universe.py`
- [ ] Add `config/universe_r1000.txt` (~150 tickers — large-cap focus, since liquidity matters most for paper trading realism)
- [ ] Add `config/watchlist.example.txt` (5-10 sample tickers)
- [ ] Verify `pytest -q` shows 4-5 new tests passing
- [ ] Commit: `feat(universe): R1000 + watchlist loader, package data module`

---

## Task 2: Classical strategy — Momo (cross-sectional momentum)

**File:** `src/quant_lab/strategies/momo.py`
**Test:** `tests/test_momo.py`

**Strategy logic:**
- Universe: all tickers in `histories` excluding the index proxies (SPY, QQQ, ^VIX)
- Compute trailing 6-month (126 trading day) return for each ticker
- Filter: ADV (computed from histories) > $5M; price history > 250 days
- Rank by trailing return descending; pick top decile (or top 10, whichever is smaller)
- Equal-weight across selected names; remainder cash
- Rebalance happens daily (paper engine has drift threshold so it won't churn)
- Returns `dict[str, float]` of weights, sums to <= 0.95 (5% cash buffer)

**API:**
```python
@register
class Momo(Strategy):
    bot_id = "momo"
    description = "Cross-sectional 6-month momentum, top decile, equal-weight"
    lookback_days: int = 126
    cash_buffer: float = 0.05
    adv_floor: float = 5_000_000
    min_history_days: int = 250

    def target_weights(self, histories, as_of) -> dict[str, float]: ...
```

**Acceptance tests:**
- With synthetic data where one ticker has clear positive momentum and others flat → that ticker is in the top selection
- With insufficient history → returns `{}`
- Sum of weights ≤ 0.95
- Index proxies (SPY, QQQ, ^VIX) excluded from selection
- ADV-floor filter respected

**Steps:**
- [ ] Write 4-5 failing tests
- [ ] Implement strategy
- [ ] Add `from . import momo` to `strategies/__init__.py`
- [ ] Verify tests pass
- [ ] Commit: `feat(strategies): add Momo (cross-sectional momentum)`

---

## Task 3: Classical strategy — MeanRev (short-term mean reversion)

**File:** `src/quant_lab/strategies/meanrev.py`
**Test:** `tests/test_meanrev.py`

**Strategy logic:**
- For each ticker in universe (excluding indices):
  - Compute 5-day cumulative return
  - If 5-day return < -5% AND price > 200-day SMA (still in uptrend) AND ADV > $10M → entry signal
- Hold up to 5 names equal-weight, max 30 days per position, exit if return > +3%
- Return target weights; if no signals, return `{}`
- Note: doesn't include news filter in v1 — flagged in description as a known limitation
- Position-management is stateless: this implementation derives current intended weights from signals rather than tracking entry dates. The paper engine handles holding/rebalancing.

**API:**
```python
@register
class MeanRev(Strategy):
    bot_id = "meanrev"
    description = "5-day mean reversion in uptrending names (no news filter in v1)"
    lookback_days: int = 5
    drop_threshold: float = -0.05
    target_uplift: float = 0.03
    max_hold_days: int = 30
    adv_floor: float = 10_000_000
    max_positions: int = 5

    def target_weights(self, histories, as_of) -> dict[str, float]: ...
```

**Acceptance tests:**
- Synthetic data with a 7% drop in last 5 days while staying above SMA → ticker selected
- 5-day drop but below SMA → not selected
- No qualifying signals → returns `{}`

**Steps:** TDD same pattern. Commit: `feat(strategies): add MeanRev (5d drop in uptrend)`

---

## Task 4: Classical strategy — Breakout (52-week high)

**File:** `src/quant_lab/strategies/breakout.py`
**Test:** `tests/test_breakout.py`

**Strategy logic:**
- For each ticker:
  - Today's close == max of last 252 trading days (52-week high)
  - Today's volume >= 1.5 × 20-day avg volume
  - ADV > $5M
- Equal-weight selected names, max 5 positions, 5% cash buffer

**API:**
```python
@register
class Breakout(Strategy):
    bot_id = "breakout"
    description = "52-week high on volume >= 1.5x 20d avg"
    high_lookback: int = 252
    volume_lookback: int = 20
    volume_multiplier: float = 1.5
    adv_floor: float = 5_000_000
    max_positions: int = 5
    cash_buffer: float = 0.05

    def target_weights(self, histories, as_of) -> dict[str, float]: ...
```

**Acceptance tests:**
- Synthetic ticker with monotonic-rising prices closing at all-time high with volume spike → selected
- Same prices but flat volume → not selected
- Insufficient history → `{}`

**Steps:** Same pattern. Commit: `feat(strategies): add Breakout (52w-high on volume)`

---

## Task 5: Classical strategy — MA-Cross (50/200 golden cross)

**File:** `src/quant_lab/strategies/ma_cross.py`
**Test:** `tests/test_ma_cross.py`

**Strategy logic:**
- For each ticker: compute 50-day and 200-day SMA
- "Long" condition: 50-day > 200-day (golden cross persists, not just the crossing day) AND price > 50-day
- Hold up to 10 names equal-weight, 5% cash buffer
- ADV > $5M filter

**Acceptance tests:**
- Synthetic ticker with 50>200 and price>50 → selected
- 50<200 → not selected
- Insufficient history (<200 days) → `{}`

**Steps:** Same. Commit: `feat(strategies): add MA-Cross (50/200 golden cross)`

---

## Task 6: Classical strategy — RSI-Rev (negative control)

**File:** `src/quant_lab/strategies/rsi_rev.py`
**Test:** `tests/test_rsi_rev.py`

**Strategy logic:**
- For each ticker: compute 14-period RSI
- Entry: RSI < 30 with confirmation candle (today's close > yesterday's close)
- Exit: RSI > 70 (paper engine handles this through next-day rebalance — strategy returns weight 0 when RSI > 70)
- Equal-weight up to 3 positions, 10% cash buffer
- ADV > $10M filter
- **Description includes "negative control — included to demonstrate weak-evidence strategies losing"**

**Acceptance tests:**
- Standard RSI calculation: 14-period rolling, Wilder's smoothing
- Synthetic data designed to trigger RSI < 30 → selected
- RSI > 70 → not selected (returns 0 weight)

**Steps:** Same. Commit: `feat(strategies): add RSI-Rev (negative control)`

---

## Task 7: Codex bot adapter

**Files:**
- Create: `src/quant_lab/strategies/codex_bot.py`
- Create: `tests/test_codex_bot.py`
- Modify: `pyproject.toml` to add the local snapshot package as a dependency (use `uv pip install -e ./competitors/codex-bot-snapshot-2026-05-07` in the bootstrap script; for tests we'll modify `sys.path`)

**Strategy logic:**
- Adapter wraps `morning_quant_bot.strategy.target_weights(...)` and `morning_quant_bot.evolver.StrategyEvolver` from the frozen snapshot
- Two registered variants:
  - `CodexBotR1000` (`bot_id="codex-r1000"`) — runs on full R1000 universe
  - `CodexBotNative` (`bot_id="codex-native"`) — restricted to the 19 ETFs in the snapshot's `config/universe.txt`
- On each call to `target_weights`:
  - Load (or evolve, sparingly) the strategy params via `StrategyEvolver`
  - Adapt our `Bar` types to morning_quant_bot's `Bar` types
  - Call their `target_weights(...)` and return the result mapped back

**Important:** The Codex snapshot uses its own `morning_quant_bot.models.Bar` dataclass. Our adapter must convert. Build a tiny helper:

```python
def _to_codex_bars(our_bars: list[Bar]) -> list[CodexBar]:
    """Convert our Bar instances to morning_quant_bot.Bar instances."""
```

**Test fixtures:** Tests mock the codex evolver so we don't run a 30s genetic algorithm during pytest.

**Acceptance tests:**
- Adapter is registered as both `codex-r1000` and `codex-native`
- Running `target_weights` on synthetic ETF data returns weights summing to ≤ 1.0
- The native variant restricts symbols to the 19 ETFs even if more are in `histories`

**Steps:**
- [ ] Add the snapshot path to test setup so imports work (use `pyproject.toml`'s `[tool.pytest.ini_options].pythonpath` to add `competitors/codex-bot-snapshot-2026-05-07/src`)
- [ ] Write 3-4 tests with a mocked evolver
- [ ] Implement adapter
- [ ] Verify imports resolve from snapshot path
- [ ] Run full suite
- [ ] Commit: `feat(strategies): Codex bot adapter (R1000 + native variants)`

---

## Task 8: Live tournament upgrades — bootstrap CIs + factor decomposition

**Files:**
- Modify: `src/quant_lab/tournament/stats.py` — extend `Metrics` to include `sharpe_ci_lo`, `sharpe_ci_hi`, `alpha_t_stat_vs_spy`, `alpha_t_stat_vs_qqq`, `significance_weight`. Reuse `backtest/stats.py` functions.
- Create: `src/quant_lab/tournament/factors.py` — 3-factor decomposition (market beta, size beta, value beta) using yfinance proxies (SPY, IWM-SPY, VTV-VUG)
- Test: `tests/test_factors.py`
- Modify: `tests/test_stats.py` to cover extended Metrics fields

**Extended `Metrics`:**

```python
@dataclass(frozen=True, slots=True)
class Metrics:
    total_return: float
    annualized_return: float
    sharpe: float
    sharpe_ci_lo: float          # NEW
    sharpe_ci_hi: float          # NEW
    volatility: float
    max_drawdown: float
    days: int
    alpha_t_stat_vs_spy: float   # NEW
    alpha_t_stat_vs_qqq: float   # NEW
    significance_weight: float   # NEW — derived from alpha_t_stat_vs_spy
    factor_loadings: dict[str, float] | None = None  # NEW — beta_mkt, beta_size, beta_value, alpha_per_day
```

`compute_metrics(nav, benchmark_returns_by_symbol=None)` accepts optional benchmark dict; if provided, computes alpha t-stats and significance weight.

**Factor decomposition (`factors.py`):**

```python
def compute_factor_loadings(
    strategy_returns: list[float],
    factor_returns: dict[str, list[float]],  # keys: "MKT", "SIZE", "VALUE"
) -> dict[str, float]:
    """Multivariate OLS regression of strategy on factors. Returns
    {alpha_per_day, beta_mkt, beta_size, beta_value, r_squared}.
    """

def factor_proxies_from_histories(histories: dict[str, list[Bar]]) -> dict[str, list[float]]:
    """Build factor return series from yfinance proxies:
        MKT = SPY daily returns
        SIZE = IWM - SPY (small-cap minus large-cap)
        VALUE = VTV - VUG (value minus growth)
    Returns aligned daily-return series ready for regression.
    """
```

**Acceptance tests:**
- `compute_factor_loadings` on synthetic data where strategy = 1.5 * MKT (beta=1.5) → returns beta_mkt ≈ 1.5, alpha ≈ 0
- `factor_proxies_from_histories` returns dict with keys MKT, SIZE, VALUE
- Extended `Metrics.compute_metrics` produces sane CIs and alpha t-stats

**Steps:**
- [ ] Write tests for factor regression
- [ ] Implement `factors.py`
- [ ] Extend `Metrics` and `compute_metrics`
- [ ] Update existing tests for new field names (just add defaults)
- [ ] Verify all tests pass
- [ ] Commit: `feat(tournament): bootstrap CIs + 3-factor decomposition + significance weights`

---

## Task 9: Regime kill-switch

**File:** `src/quant_lab/engine/regime.py`
**Test:** `tests/test_regime.py`

**Logic:**
```python
def regime_state(histories: dict[str, list[Bar]]) -> dict[str, Any]:
    """Returns {'vix': float, 'regime': 'NORMAL'|'CAUTION'|'PANIC',
               'halt_new_entries': bool, 'liquidate_all': bool}"""

def per_bot_drawdown(nav_series: list[tuple[date, float]], window_days: int = 30) -> float:
    """Trailing-window drawdown."""

def should_pause_bot(
    bot_id: str,
    nav_series: list[tuple[date, float]],
    sharpe_window_days: int = 60,
) -> tuple[bool, str]:
    """Returns (should_pause, reason). Pauses on:
      - 30-day DD > 25%
      - 60-day Sharpe < -1.0
    """
```

**Wired into `main.py`:**
- `morning_command` checks regime state at the start
- If `liquidate_all=True` (VIX > 50) → set all bot weights to {} for the rebalance pass (forces SELL of all positions)
- If `halt_new_entries=True` (VIX > 35) → strategies still produce signals but `paper.rebalance()` is called with a flag that prevents *opening new positions* (existing positions managed normally)
- If a bot is paused → it's excluded from leaderboard ranking (still shown as paused)

**Engine change for halt_new_entries:**

Modify `engine/paper.py:rebalance()` to accept a new parameter `block_new_entries: bool = False`. When True: any tick where `current_shares == 0` and `target_value > 0` is skipped (logged in `skipped`).

**Acceptance tests:**
- VIX = 25 → NORMAL regime
- VIX = 40 → CAUTION, halt_new_entries=True
- VIX = 55 → PANIC, liquidate_all=True
- Per-bot DD > 25% → should_pause=True
- Test that `paper.rebalance(block_new_entries=True)` doesn't open new positions but still rebalances existing

**Steps:**
- [ ] Tests for regime state and pause logic
- [ ] Implement `regime.py`
- [ ] Modify `engine/paper.py` for `block_new_entries` parameter
- [ ] Wire into `main.morning_command`
- [ ] Run full suite
- [ ] Commit: `feat(engine): regime kill-switch (VIX + per-bot drawdown halts)`

---

## Task 10: Watchdog workflow

**File:** `.github/workflows/watchdog.yml`

**Logic:**
- Cron: `0 12,14,16,20 * * 1-5` (4 times during weekdays)
- Reads `state/last_morning.json` (we'll add this file write to morning_command, recording timestamp + status)
- If last morning run > 24h ago AND it's a weekday → POST to Discord webhook with alert
- Doesn't modify any state, just notifies

**`main.py` change:**
After `morning_command` completes successfully, write `state/last_morning.json`:
```json
{"timestamp": "2026-05-07T11:30:00Z", "status": "success", "strategies": ["momo", ...]}
```

If morning fails partway through, write `status: "partial"` or `"failed"` instead.

**Steps:**
- [ ] Add `last_morning.json` write to `main.morning_command` finally block
- [ ] Create watchdog.yml
- [ ] Test the YAML parses
- [ ] Commit: `ci: watchdog workflow alerts when morning run is missed`

---

## Task 11: End-to-end Phase 2 integration test + calibration run

**Files:**
- Create: `tests/test_phase2_e2e.py`

**Test:**
- Set up synthetic data for SPY, QQQ, AAPL, NVDA, IWM, VTV, VUG (factor proxies + a couple stocks for momentum)
- Run `morning_command` on synthetic data — assert all 8 strategies (SPYVol, QQQVol, Momo, MeanRev, Breakout, MACross, RSIRev, CodexBotR1000) have NAV recorded
- Verify `leaderboard.json` includes new fields (`sharpe_ci_lo`, `sharpe_ci_hi`, `alpha_t_stat_vs_spy`, etc.)

**After integration test passes, run the calibration on real data:**

```bash
quant-lab backtest --start 2018-01-01 --end 2025-05-01 --train-years 3 --step-months 12
```

This produces a calibration report for all strategies. Commit the artifacts (`dashboard/data/backtest/`).

**Steps:**
- [ ] Implement integration test
- [ ] Run full pytest — all tests pass
- [ ] Run real-data backtest, commit artifacts
- [ ] Commit (1): `test: Phase 2 e2e integration test`
- [ ] Commit (2): `data: Phase 2 calibration report (real yfinance data, 2018-2025)`

---

## Self-review

**Spec coverage:**
- §4 strategy roster (Momo, MeanRev, Breakout, MA-Cross, RSI-Rev): Tasks 2-6 ✓
- §4 Codex bot adapter (R1000 + native): Task 7 ✓
- §9 tournament stats (bootstrap CIs, factor decomp, sig flags): Task 8 ✓
- §8 regime kill-switch: Task 9 ✓
- §14 watchdog workflow: Task 10 ✓
- Universe extension: Task 1 ✓

**Type consistency:** `Bar`, `Position`, `Trade`, `Portfolio`, `Strategy` all unchanged. `Metrics` extended with new fields (existing fields preserved with defaults).

**No placeholders.**

After Phase 2: ~10+ strategies in the live tournament, every strategy with a calibration report, factor decomposition exposing whether outperformance is real alpha or market beta. Ready for Phase 3 (ML) and Phase 5 (meta-ensemble).

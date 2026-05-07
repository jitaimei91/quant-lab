# Quant Research Lab + Morning Briefing — Design Spec

**Date:** 2026-05-07
**Status:** Approved design, ready for implementation planning
**Owner:** User (US-equity-only, free-tier-only personal project)

---

## 1. Goal & Non-Goals

### Goal

A personal **quant research lab + morning briefing tool** that:

1. Runs a tournament of paper-trading strategies (classical + ML) on the Russell 1000 + a personal watchlist
2. Provides each weekday morning: portfolio summary, market snapshot, news on holdings, tournament leaderboard, and the leader's advisory signals
3. Hosts a public, interactive web dashboard showing live results, bot-vs-bot comparisons, and methodology
4. Generates evidence about which strategies actually work on a multi-year out-of-sample basis

### Explicit Non-Goals

- This is **not** a "tells me what to buy" service. Strategies are evidence to study, not orders to follow.
- This is **not** a high-frequency or intraday system. Daily EOD bars only.
- This is **not** a real-money execution system. Paper-trading only in v1-v2.
- This is **not** a deep-learning project. CPU-friendly gradient boosting only.
- This is **not** a wealth-management tool. Index funds remain the rational default for capital.

### Disclosures (must be visible everywhere)

- Strategies are well-known, public, and academically documented. Most have been arbitraged by professional quants.
- Survivorship bias exists in the universe data despite mitigation.
- Past performance, including paper performance, does not predict future results.
- Statistical significance gates apply: bots with t-stat ≤ 2.0 on alpha or fewer than 120 trading days of history are flagged as not statistically distinguishable from luck.

---

## 2. Architecture Overview

```
                ┌─────────────────────────────────────────────────────┐
                │  GH Actions: morning.yml (M–F 6:30 AM ET)           │
                │  GH Actions: watchdog.yml (4×/day market hours)     │
                │  GH Actions: weekly-retrain.yml (Sun 6 PM ET)       │
                └─────────────────────┬───────────────────────────────┘
                                      │
       ┌────────────────────┬─────────┼─────────────────┬─────────────────┐
       ▼                    ▼         ▼                 ▼                 ▼
 ┌──────────┐        ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────┐
 │  Data    │ Alpaca │ Universe │  │ Strategy │  │ Paper engine │  │ Regime   │
 │ fetcher  │ ←→yfin │ + factor │→ │   bus    │→ │ + slippage   │  │ kill-    │
 │          │        │  data    │  │ (10 bots)│  │  + liquidity │  │ switch   │
 └──────────┘        └──────────┘  └──────────┘  └──────┬───────┘  └────┬─────┘
                                                        │               │
                                                        ▼               │
                                                ┌──────────────┐        │
                                                │  Turso DB    │        │
                                                │ (trades,     │ ◀──────┘
                                                │  positions)  │
                                                └──────┬───────┘
                                                       │
                          ┌────────────────────────────┼─────────────────────┐
                          ▼                            ▼                     ▼
                  ┌───────────────┐           ┌───────────────┐      ┌──────────────┐
                  │ Tournament:   │           │ Discord       │      │ Dashboard    │
                  │ Sharpe + CIs  │ ────────▶ │ morning brief │      │ generator    │
                  │ Factor decomp │           └───────────────┘      │ → GH Pages   │
                  │ Stat tests    │                                  │ + JSON API   │
                  └───────────────┘                                  └──────────────┘
```

Everything runs in stateless GitHub Actions workflows. Persistent state lives in **Turso** (free SQLite-on-edge) and committed parquet/JSON in the repo.

---

## 3. Tech Stack (all free)

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11 | Standard for quant; mature ecosystem |
| Data primary | Alpaca Markets free tier | Official, real-time IEX, paper-trading API |
| Data fallback | yfinance | Fundamentals + redundancy when Alpaca rate-limits |
| Universe data | Wikipedia historical R1000 monthly snapshots | Best free approximation of point-in-time constituents |
| Index data | Alpaca SPY, QQQ daily bars | Both benchmarks |
| News | Alpaca News API (free) | Per-ticker headlines |
| Storage (state) | Turso (libSQL, 9 GB free) | No repo bloat; queryable from any runner |
| Storage (snapshots) | Parquet files in repo | Versioned price snapshots |
| Storage (models) | git-LFS (1 GB free) | Pickled XGBoost/LightGBM artifacts |
| ML | XGBoost + LightGBM | CPU-friendly, beats deep learning on tabular finance |
| Stats / factors | statsmodels, pandas, numpy, scipy | Standard scientific Python |
| Compute | GitHub Actions (public repo, unlimited mins) | $0 forever |
| Notifications | Discord webhook | User preference, free, instant |
| Dashboard | HTML + Tailwind + Chart.js, served by GH Pages | Static, fast, no build step |
| Tests | pytest + hypothesis | Property-based for paper-engine math |

---

## 4. Strategy Roster

### v1 (Weeks 1-2) — Classical, hand-coded, transparent

| # | Bot | Strategy | Liquidity filter | Notes |
|---|---|---|---|---|
| 1 | Momo | Cross-sectional 6-month momentum, top decile, monthly rebal | ADV > $5M | Crowded but real factor |
| 2 | MeanRev | Buy on >5% drop with no news, exit at +3% or 5 days | ADV > $10M | HFT-crowded, regime-sensitive |
| 3 | Breakout | 52-week high closing on volume ≥ 1.5× 20-day avg | ADV > $5M | False-breakout risk |
| 4 | MA-Cross | 50-day SMA crosses above 200-day SMA (golden cross) | ADV > $5M | Trend-only edge |
| 5 | RSI-Rev | RSI < 30 with confirmation candle, sell at RSI > 70 | ADV > $10M | **Negative control** — included to demonstrate weak-evidence strategies losing |
| 6 | SPY-Vol | Vol-targeted long SPY (15% annualized vol target) | n/a | **S&P 500 honest benchmark** |
| 7 | QQQ-Vol | Vol-targeted long QQQ (15% annualized vol target) | n/a | **Nasdaq honest benchmark** |

### v2 (Weeks 3-4) — ML, validation-gated

Each ML bot **must pass all three** gates before joining the live tournament:

1. **Walk-forward Sharpe** > SPY-Vol Sharpe over 5+ years of out-of-sample backtest
2. **Label-shuffle test**: same model trained on shuffled labels must score near zero (no spurious signal from lookahead/leakage)
3. **Out-of-sample stability**: live Sharpe within ±30% of backtest Sharpe

Failing bots are **excluded from the leaderboard** with a public report on the dashboard's Validation page.

| # | Bot | Model | Features | Target |
|---|---|---|---|---|
| 8 | GradBoost | XGBoost | ~40 technicals, point-in-time | 5-day forward return; trade top decile |
| 9 | LightForest | LightGBM | Same features, different hyperparams | Same target |
| 10 | Ensemble | Average(8, 9) | — | Blended signal |

### Bot #11 — Codex Bot (the rival)

The user has an existing Codex-built bot at `../morning-quant-bot/` (`morning-quant-bot` package, ~2,300 LOC). It uses a genetic algorithm to tune a single momentum/trend/vol/RSI strategy on a 19-ETF universe.

**We include it as bot #11 in our tournament.** Adapter:

- `src/strategies/codex_bot.py` imports `morning_quant_bot.strategy.target_weights` and `morning_quant_bot.evolver.StrategyEvolver`
- Wraps it to conform to our `Strategy` interface
- Runs on the SAME universe (R1000 + watchlist), data feed, slippage model, and paper-trading engine as our bots — fair comparison on identical infrastructure
- A second variant runs it on its NATIVE universe (19 ETFs only) to check whether it does better on its home turf — both variants compete

This lets the tournament objectively answer: does the Codex bot's GA-tuned strategy beat hand-coded classical strategies, vol-targeted indexes, or ML strategies? The dashboard's "Bot vs Bot" page will feature a default view of "Codex Bot vs Tournament Leader."

### Custom user bots (extensibility)

- **v1**: Drop-in support — create `src/strategies/<name>.py` implementing the `Strategy` base class. Auto-discovered, joins tournament next morning. Works for any rule-based bot in ~30 lines.
- **v2 stretch (out of v1-v2 scope)**: Bot Builder UI — webpage form for non-code bot configuration. Deferred until core system has run for 2+ weeks and informed which knobs the UI should expose.

---

## 5. Data Pipeline

### Daily fetch (in `morning.yml`)

1. **Alpaca primary**: pull EOD bars for R1000 + watchlist + SPY + QQQ (~3 min)
2. **yfinance fallback**: fill any tickers Alpaca returns empty for
3. **Health check**: if either fetcher fails > 5% of universe, log to `data_health` table and post Discord alert
4. **Data freshness gate**: if data is stale (last bar > 1 trading day old), mark all today's signals as **"advisory only — stale data"** and continue. Do not silently fail.
5. **Cache**: write to Turso `prices` table + a daily parquet snapshot in `data/snapshots/YYYY-MM-DD.parquet` (committed to repo, ~1 MB compressed)

### Universe rebalance (monthly, in `weekly-retrain.yml`)

1. Scrape Wikipedia's archived R1000 / R3000 list for the prior month-end (page revision API)
2. Apply that constituent set retroactively in backtests for that month → point-in-time approximation
3. Survivorship bias remains for delisted-then-relisted edge cases — **disclosed prominently on dashboard's Methodology page**

### News fetch (daily)

- Alpaca News API: pull last 24h headlines for user holdings + watchlist + S&P 500 top 50 movers
- Cache in Turso `news` table with 7-day TTL
- Used in: Discord report (holdings news section), MeanRev bot's "no news" filter

---

## 6. Paper Trading Engine

### Per-bot portfolios

- Each bot starts with $100,000 in an isolated portfolio in Turso
- Schema: `portfolios(bot_id, cash, last_updated)`, `positions(bot_id, ticker, shares, avg_cost)`, `trades(bot_id, ticker, side, shares, price, timestamp, slippage_bps)`, `nav_history(bot_id, date, nav)`
- All values denominated in USD

### Execution model

- Signals generated at end of day N
- Trades execute at the **opening price of day N+1** — eliminates lookahead, models a realistic retail timeline
- **No same-day execution**: a strategy that uses today's close to make today's trade is not allowed (gate enforced in code)
- Order types: market orders only, sized by position-sizing rules

### Position sizing

- Per-trade size = (portfolio_NAV × signal_strength × vol_target) / stock_30d_vol
- **Hard caps**: 5% of portfolio per ticker, 30% per GICS sector
- Cash buffer: minimum 2% cash retained at all times

### Corporate actions

- Use Alpaca's split- and dividend-adjusted prices for all calculations
- Dividend payments credited as cash to the receiving bot's portfolio

---

## 7. Slippage & Liquidity Model

### Slippage cost (applied to every paper trade)

```
spread_bps = max(1, 5 + 100 / sqrt(ADV_in_millions))
trade_cost = trade_value × spread_bps / 10000
```

Roughly: liquid mega-caps cost ~6 bps round-trip, mid-caps ~10-15 bps, small-caps in R1000 tail ~20-30 bps.

### Liquidity filter

- Each strategy has an `adv_floor` (minimum 30-day avg daily dollar volume)
- Tickers below the floor are **rejected at signal generation** with a logged reason
- Default floors: $5M for slow-turnover strategies (Momo, Breakout, MA-Cross), $10M for fast-turnover (MeanRev, RSI-Rev), $1M for ML strategies (will be tightened if ML signals end up in micro-caps)

### Commissions

- $0 (matches Robinhood, Schwab, Fidelity retail)
- Per-share fee proxy for SEC fees: $0.000166/share (negligible but tracked for realism)

---

## 8. Regime Kill-Switch

Cross-bot safety rails, evaluated at start of each `morning.yml` run:

| Condition | Action |
|---|---|
| VIX close > 35 | No new entries across all bots; existing positions managed normally |
| VIX close > 50 | Liquidate all bot positions to cash, log "regime panic" event |
| Per-bot 30-day rolling drawdown > 25% | Pause that bot, post Discord alert, exclude from leader signals |
| Per-bot 60-day rolling Sharpe < -1.0 | Pause that bot, flag for review |
| `morning.yml` failure rate > 30% in last 7 days | Halt automated trades; mode = "advisory only" |

Manual override: `workflow_dispatch` with input `force_unhalt=true`.

---

## 9. Tournament + Statistical Framework

### Per-bot daily metrics (computed in `tournament/stats.py`)

- Total return, annualized return
- **Excess return vs SPY** and **vs QQQ** (both, separately)
- Sharpe ratio with **bootstrapped 95% CI** (5,000 resamples of daily returns)
- Sortino, Calmar, max drawdown, current drawdown
- Win rate (with binomial CI)
- Average gain / average loss / profit factor
- **Fama-French 5-factor decomposition** (market, size, value, profitability, investment) — alpha + factor loadings + t-stats
- **Significance flag**: green only if `alpha_t_stat > 2.0 AND trading_days >= 120`
- Turnover (annualized)

### Leaderboard ranking

Default sort: **risk-adjusted excess return with significance**, computed as:

```
score = (sharpe_excess_vs_SPY * sig_weight) where
sig_weight = clamp(0, 1, (alpha_t_stat - 1.0) / 1.0) if t_stat > 1 else 0
```

This means: a bot with high Sharpe but low statistical significance ranks below a bot with lower Sharpe but high significance. **Discourages chasing noise.**

### Bootstrapped CIs

- Resample daily returns with replacement, 5,000 iterations
- Compute Sharpe, max DD, win rate on each resample
- Report 2.5th and 97.5th percentiles as 95% CI
- Display CIs on every leaderboard / detail row — never just point estimates

---

## 10. ML Pipeline

### Feature engineering (`ml/features.py`)

~40 features per (ticker, date), strict point-in-time computation:

- Price-based: returns over 1/5/20/60/120/252 days
- Momentum / mean-reversion: trailing-window z-scores
- Volume: relative volume vs 20-day avg, on-balance volume
- Volatility: rolling 20/60-day realized vol, vol-of-vol
- Technical: RSI(14), MACD, Bollinger band z-score, ATR
- Cross-sectional: sector-relative, market-relative versions of the above
- Liquidity: ADV, spread proxy

**No fundamental features in v2** — fundamentals introduce point-in-time complexity (when was the data filed vs reported?). Add in v3.

### Walk-forward training (`ml/train.py`)

- Training window: rolling 5 years
- Validation window: 1 year (out-of-sample, immediately after train)
- Step: 1 month
- Each fold trains a fresh model on its train window, evaluates on validation, records OOS Sharpe
- Final live model = trained on all data up to last completed walk-forward fold

### Validation gates (`ml/validate.py`)

Run before any ML bot is added to live tournament:

1. **Walk-forward Sharpe gate**: median walk-forward OOS Sharpe must exceed SPY-Vol median Sharpe over the same period
2. **Label-shuffle test**: train identical model on shuffled forward-return labels. Resulting Sharpe should be near zero (mean Sharpe across 10 shuffles within ±0.1). Failing this means feature pipeline has lookahead leakage.
3. **OOS stability test**: live Sharpe (last 60 days) within ±30% of backtest Sharpe (same period the year before). Drift outside this range = paused, flagged for retraining/review.

Failures are **public** on the Validation dashboard page — that's the honest frame.

### Retraining cadence

- Weekly (Sunday 6 PM ET): retrain on data through Friday's close
- Monthly: full walk-forward re-evaluation, regenerate validation report
- Manual: `workflow_dispatch` for ad-hoc retrains

---

## 11. Discord Morning Report

Posted M-F at 6:45 AM ET (after `morning.yml` completes).

Format:

```
═══════════════════════════════════════════
QUANT LAB — Morning Brief — 2026-05-07
═══════════════════════════════════════════
⚠ Research tool. Not financial advice. Paper trading only.

 YOUR PORTFOLIO
  NAV: $XX,XXX  (overnight: +0.4% | YTD: +6.2%)
  vs SPY YTD:  +1.1pp   |  vs QQQ YTD:  -2.5pp
  Movers:
    NVDA  +2.1%  (earnings beat)
    TSLA  -1.3%

 MARKET SNAPSHOT
  S&P 500 (SPY):  +0.31% ▲   YTD +5.1%
  Nasdaq (QQQ):   +0.54% ▲   YTD +8.7%
  VIX:            14.2       Regime: NORMAL

 NEWS ON YOUR HOLDINGS
  • AAPL — [headline summary]
  • NVDA — [headline summary]

 TOURNAMENT — Day 47
  ┌────────────┬───────┬─────────┬──────────────┬─────┐
  │ Bot        │ Total │ vs SPY  │ Sharpe (CI)  │ Sig │
  ├────────────┼───────┼─────────┼──────────────┼─────┤
  │ SPY-Vol    │ +5.4% │   +0.3% │ 0.61 [.34,.91]│  ✓  │
  │ Momo       │ +4.8% │  -0.3%  │ 0.42 [.10,.78]│  ⚠  │
  │ QQQ-Vol    │ +8.5% │  +3.4%  │ 0.71 [.41,1.0]│  ✓  │
  │ Breakout   │ +3.2% │  -1.9%  │ 0.31 [-.05,.71]│  ⚠  │
  │ ...        │       │         │               │     │
  │ RSI-Rev    │ -2.3% │  -7.4%  │ -0.18 [...]   │  —  │
  └────────────┴───────┴─────────┴──────────────┴─────┘
  ✓ statistically significant alpha (t > 2)
  ⚠ low confidence (CI spans zero or t ≤ 2)
  — negative-control bot, expected underperformance

 LEADER'S SIGNALS (advisory only)
  Leader: QQQ-Vol  (significance: ✓)
  BUY:    None today (regime filter NORMAL, target weight unchanged)
  HOLD:   QQQ at current weight
  SELL:   None

 📊 Dashboard: https://<user>.github.io/quant-lab/
═══════════════════════════════════════════
```

---

## 12. Dashboard (GitHub Pages)

### Pages

1. **Home / Leaderboard** (`index.html`)
   - Top banner: SPY +/-, QQQ +/- (both today and YTD)
   - Sortable leaderboard table (default sort: significance-weighted score)
   - Sig flags, factor decomposition column expandable per row
   - Permanent disclaimer block in footer

2. **Bot Detail** (`bot/<name>.html`)
   - Equity curve overlaid against SPY-Vol and QQQ-Vol baselines
   - Daily returns histogram, drawdown chart
   - Trade log (paginated, filterable by ticker/date)
   - Factor exposures table with t-stats
   - Strategy description and code link

3. **Bot vs Bot** (`compare.html`)
   - Two dropdowns to pick any two bots (or SPY, QQQ, user portfolio)
   - Overlaid equity curves
   - Daily +/- table: who won today, this week, this month
   - Side-by-side stats with CIs
   - Statistical significance test of difference (paired t-test on daily excess returns)

4. **Methodology** (`methodology.html`)
   - Every formula, assumption, slippage model, liquidity filter
   - All disclosures — survivorship bias, regime risk, statistical limits
   - Data lineage diagram

5. **Validation** (`validation.html`)
   - Walk-forward backtest results for each ML bot
   - Label-shuffle test outcomes (with raw numbers)
   - OOS stability charts
   - **Excluded bots prominently listed** — failed validation, why

### Data feeds (JSON API)

Each page reads from JSON files in `dashboard/data/`:
- `leaderboard.json`, `bot/<name>.json`, `compare/<a>-vs-<b>.json`, `validation.json`, `market.json`
- Public, free to fetch from anywhere — anyone can build alternative views

### Embedding

- Iframe support: `<iframe src="https://<user>.github.io/quant-lab/embed/leaderboard">` etc.
- Responsive layouts for embeds

---

## 13. Storage Layout

### Turso (libSQL, 9 GB free)

Tables:
- `portfolios(bot_id, cash, last_updated)`
- `positions(bot_id, ticker, shares, avg_cost, opened_at)`
- `trades(bot_id, ticker, side, shares, price, slippage_bps, timestamp, signal_meta_json)`
- `nav_history(bot_id, date, nav)`
- `signals(bot_id, ticker, date, signal_strength, decision, reason)`
- `prices(ticker, date, open, high, low, close, adj_close, volume)`
- `news(ticker, headline, summary, url, published_at)`
- `data_health(run_id, fetcher, success_rate, started_at, completed_at)`
- `regime_state(date, vix, regime, halted_bots_json)`

### Repo (parquet + git-LFS)

- `data/snapshots/YYYY-MM-DD.parquet` — daily price snapshots, ~1 MB each
- `data/universe/YYYY-MM.json` — historical R1000 constituents per month
- `models/<bot>/<YYYY-MM-DD>.pkl` (git-LFS) — pickled trained models, ~10-50 MB each
- `dashboard/data/*.json` — generated dashboard data
- `logs/run-<timestamp>.log` — per-run logs (rotated, last 30 days kept)

---

## 14. GitHub Actions Workflows

### `morning.yml`

- Trigger: `cron: '30 11 * * 1-5'` (11:30 UTC = 6:30 AM EST / 7:30 AM EDT — adjust DST manually or run twice)
- Steps:
  1. Checkout repo, set up Python 3.11, install deps
  2. Pull data (Alpaca → yfinance fallback)
  3. Run all strategies (signal generation)
  4. Execute paper trades through paper engine
  5. Compute tournament stats + factor decomp + CIs
  6. Generate dashboard JSONs
  7. Commit JSONs back to repo (via `peter-evans/create-pull-request` or direct commit on main)
  8. Send Discord webhook
  9. Log success to Turso

### `watchdog.yml`

- Trigger: `cron: '0 12,14,16,20 * * 1-5'` (4 times during weekdays)
- Steps: query Turso for last successful `morning.yml` run; if > 24h ago and it's a weekday, post Discord alert.

### `weekly-retrain.yml`

- Trigger: `cron: '0 22 * * 0'` (Sunday 10 PM UTC = 6 PM ET)
- Steps:
  1. Pull historical data (5+ years)
  2. Run walk-forward training for ML bots
  3. Run validation gates (walk-forward Sharpe, label-shuffle, OOS stability)
  4. If passing: pickle model to git-LFS, commit
  5. If failing: keep prior model, post Discord alert with diagnostic

### `monthly-rebalance.yml`

- Trigger: `cron: '0 22 1 * *'` (1st of each month, 10 PM UTC)
- Pull updated R1000 constituents from Wikipedia, commit to `data/universe/`

---

## 15. Repo Skeleton

```
quant-lab/
├── bootstrap.sh                # automated setup (full + --fast paths)
├── .github/workflows/
│   ├── morning.yml
│   ├── watchdog.yml
│   ├── weekly-retrain.yml
│   └── monthly-rebalance.yml
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-05-07-quant-lab-design.md   (this file)
├── src/
│   ├── data/
│   │   ├── fetcher.py          # Alpaca primary, yfinance fallback
│   │   ├── universe.py         # R1000 historical constituents
│   │   ├── news.py             # Alpaca news fetcher
│   │   └── cache.py            # Turso + parquet caching
│   ├── strategies/
│   │   ├── base.py             # Strategy interface
│   │   ├── momo.py
│   │   ├── meanrev.py
│   │   ├── breakout.py
│   │   ├── ma_cross.py
│   │   ├── rsi_rev.py
│   │   ├── spy_vol.py
│   │   ├── qqq_vol.py
│   │   ├── gradboost.py        # ML, v2
│   │   ├── lightforest.py      # ML, v2
│   │   ├── ensemble.py         # ML, v2
│   │   └── codex_bot.py        # Adapter — wraps morning_quant_bot package
│   ├── engine/
│   │   ├── paper.py            # Per-bot portfolio + executor
│   │   ├── slippage.py         # Spread + liquidity model
│   │   └── regime.py           # VIX kill-switch
│   ├── tournament/
│   │   ├── leaderboard.py
│   │   ├── stats.py            # Sharpe, drawdown, CIs
│   │   ├── factor_decomp.py    # Fama-French 5-factor
│   │   └── compare.py          # Bot-vs-bot
│   ├── ml/
│   │   ├── features.py         # Walk-forward-safe features
│   │   ├── train.py            # Walk-forward training
│   │   └── validate.py         # Label-shuffle, OOS gates
│   ├── reporting/
│   │   ├── discord.py          # Webhook poster
│   │   └── dashboard.py        # JSON + HTML generator
│   └── main.py                 # Entry points (morning, retrain, watchdog)
├── tests/
│   ├── test_strategies.py
│   ├── test_paper_engine.py
│   ├── test_slippage.py
│   ├── test_label_shuffle.py
│   ├── test_walk_forward.py
│   └── test_factor_decomp.py
├── dashboard/
│   ├── index.html
│   ├── bot.html
│   ├── compare.html
│   ├── methodology.html
│   ├── validation.html
│   ├── embed/                  # iframe-friendly variants
│   ├── styles.css
│   ├── app.js
│   └── data/                   # generated JSONs
├── data/
│   ├── snapshots/              # parquet daily snapshots
│   └── universe/               # R1000 constituents per month
├── models/                     # git-LFS pickled models
├── pyproject.toml
├── README.md
├── .gitignore
└── .gitattributes              # git-LFS rules
```

---

## 16. User Setup — Automated (`./bootstrap.sh`, ~3 min)

A single shell script handles all configuration with two modes:

### Fast path — `./bootstrap.sh --fast` (60 seconds, zero accounts)

- Uses yfinance only as data source (no Alpaca account needed)
- Uses local SQLite file (no Turso account needed)
- Only required input: Discord webhook URL (or `--no-discord` flag for fully zero-input mode)
- Creates GitHub repo via `gh` CLI, enables Pages, triggers first run
- Trade-off: data is less reliable (Yahoo can break), state is local-only (no cross-device dashboard)

### Full path — `./bootstrap.sh` (3 minutes, ~6 clicks, paste-back of 3 credentials)

The script:

1. Installs missing dependencies via `brew` or `pipx` (`gh`, `turso`, Python 3.11+, `uv`)
2. Authenticates with GitHub (`gh auth login`)
3. Opens browser tabs in sequence:
   a. **Alpaca paper account signup** — user fills form → paste back `KEY` + `SECRET`
   b. **Turso signup** (one-click GitHub OAuth) → paste back `URL` + `TOKEN`
   c. **Discord webhook generator** (custom HTML helper page hosted in repo) → paste back webhook URL
4. Auto-creates the GitHub repo (`gh repo create`)
5. Auto-sets all secrets (`gh secret set ALPACA_KEY`, etc.)
6. Auto-enables GitHub Pages with Actions source (`gh api`)
7. Auto-prompts for initial holdings (`{ ticker: shares }`) and watchlist tickers — written to `config/account.json`
8. Triggers first morning run (`gh workflow run morning.yml`)
9. Watches the run, posts the resulting Discord message inline so user sees their first brief immediately
10. Validates everything: checks Discord received, dashboard URL reachable, Turso DB seeded

User clicks: ~6. Manual typing: 0 (all paste-back). Total time including KYC on Alpaca (the slowest piece): ~3 min.

### Unautomatable bits (inherent to third-party services)

- Alpaca KYC verification (~90 sec form)
- Turso GitHub-OAuth click (~10 sec)
- Discord webhook authorization (~10 sec)
- GitHub OAuth (~10 sec, one-time)

### Re-running setup

`./bootstrap.sh --reconfigure` rotates secrets, repushes config without re-creating the repo. Idempotent.

---

## 17. Roadmap

| Week | Milestone |
|---|---|
| **1** | Project scaffold; `bootstrap.sh` setup script; Alpaca + yfinance data fetcher; paper engine + slippage; SPY-Vol + QQQ-Vol benchmarks; Discord webhook; basic leaderboard JSON |
| **2** | 5 classical strategies (Momo, MeanRev, Breakout, MA-Cross, RSI-Rev); Codex bot adapter (`codex_bot.py`); tournament stats + bootstrapped CIs + factor decomp; regime kill-switch; watchdog; basic dashboard pages |
| **3** | ML feature pipeline (walk-forward safe); GradBoost + LightForest training; validation gates (walk-forward Sharpe, label-shuffle, OOS stability) |
| **4** | Ensemble bot; full dashboard polish (Bot vs Bot — default view "Codex vs Leader"; Methodology; Validation pages); error handling hardening; complete test coverage |
| **Ongoing** | Monthly universe rebalance; weekly retrains; on-going strategy research; v2-stretch Bot Builder UI |

---

## 18. Out of Scope (v3+)

- Bot Builder UI (rules-based bot creation in browser)
- Fundamental factor strategies (point-in-time fundamentals)
- Options strategies (covered calls, cash-secured puts)
- Multi-asset (crypto, FX, futures)
- Real-money execution via Alpaca Trade API
- Sentiment-based bots (news / Reddit / Twitter NLP)
- Reinforcement learning agents
- Live retrain (currently weekly batch)
- Custom domain for dashboard

---

## 19. Risk & Disclosure Block (verbatim — must appear on every public surface)

> This is a research and educational tool. All trading is performed in a simulated paper account using historical and delayed market data. No real money is executed. Past performance, including paper performance, does not guarantee future results.
>
> Strategies in this tournament are well-known, public, and have been studied (and largely arbitraged) by professional quantitative investors for decades. The probability that any single strategy here delivers consistent risk-adjusted outperformance vs. a passive index over a 5+ year horizon is low.
>
> Universe data is subject to survivorship bias despite mitigation. Backtest results are gross of taxes, advisory fees, and any transaction costs not modeled in the slippage assumption. Statistical significance flags are based on point-in-time t-statistics and may be unstable.
>
> Bots flagged "low confidence" or "no significance" should be interpreted as **statistically indistinguishable from luck**. The negative-control bot (RSI-Rev) is included specifically to demonstrate the noise floor of strategies with weak academic support.
>
> Use this system to learn quantitative methodology, not to allocate real capital. For real-money investing, low-cost broad-market index funds remain the rational default for the overwhelming majority of investors.

---

**End of design spec.**

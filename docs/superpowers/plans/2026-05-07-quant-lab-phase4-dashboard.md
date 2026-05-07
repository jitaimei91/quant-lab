# Quant Lab Phase 4 — Dashboard Polish (ship-ready)

> **For agentic workers:** Use superpowers:subagent-driven-development. Compact plan format.

**Goal:** Add the three missing dashboard pages — Bot vs Bot (head-to-head competition view), Methodology (full disclosures + formulas), Validation (per-strategy calibration evidence + significance flags). Polish existing pages. Ensure the dashboard is production-quality and uploadable to GitHub Pages.

**Architecture:** All pages are static HTML + JS reading JSON from `dashboard/data/`. No build step. The morning workflow already commits dashboard JSONs. New pages need new JSON fields exported from `reporting/dashboard.py`.

---

## File Structure

```
dashboard/
├── index.html                # EXTEND — add nav menu, top market banner with QQQ alongside SPY
├── compare.html              # NEW — bot-vs-bot head-to-head comparison
├── methodology.html          # NEW — formulas, disclosures, slippage model, factor decomposition
├── validation.html           # NEW — per-strategy calibration: bootstrap CIs, alpha t-stats, factor loadings
├── bot.html                  # NEW — single-bot deep dive (equity curve, trades, factor exposures)
├── styles.css                # EXTEND — nav styles, comparison view, validation badges
├── compare.js                # NEW
├── methodology.js            # NEW (minimal — mostly static)
├── validation.js             # NEW
├── bot.js                    # NEW
└── (existing app.js stays for index.html)

src/quant_lab/reporting/
└── dashboard.py              # EXTEND — write per-bot JSON, comparison-ready data, methodology-static page

tests/
└── test_dashboard_polish.py  # NEW — test new JSON fields export
```

---

## Task 1: Extend dashboard JSON exports

**Files:**
- Modify: `src/quant_lab/reporting/dashboard.py`
- Modify: `tests/test_dashboard.py` (extend coverage) or create `tests/test_dashboard_polish.py`

**New JSON files to write:**
- `dashboard/data/bots/<bot_id>.json` — full per-bot detail: NAV history, trade log (last 100), factor loadings, current weights, paused state
- `dashboard/data/methodology.json` — static-ish content: formulas, parameter values, slippage model, factor model
- `dashboard/data/validation.json` — per-strategy calibration evidence: pulls from existing `dashboard/data/backtest/backtest_results.json` aggregated with live metrics

**API additions to `reporting/dashboard.py`:**

```python
def write_per_bot_files(
    out_dir: Path,
    portfolios: dict[str, Portfolio],
    nav_history: dict[str, list[tuple[date, float]]],
    metrics_by_bot: dict[str, Metrics],
    trades_log_path: Path | None,
) -> None:
    """For each bot, write a per-bot JSON to out_dir/bots/<bot_id>.json"""

def write_validation_data(
    out_dir: Path,
    backtest_results_path: Path,
    live_metrics: dict[str, Metrics],
) -> None:
    """Aggregate backtest + live metrics into out_dir/validation.json."""
```

**Wire into `main.morning_command`** to call both after the existing `write_dashboard_data`.

**Acceptance tests:**
- `write_per_bot_files` creates `bots/<bot_id>.json` for every bot in `portfolios`
- Each per-bot JSON includes: bot_id, equity curve, current weights, paused state, factor loadings if present, last N trades
- `write_validation_data` produces `validation.json` with per-strategy calibration + live evidence

**Steps:** TDD pattern. Commit: `feat(dashboard): per-bot + validation JSON exporters`

---

## Task 2: Single-bot detail page (`bot.html`)

**Files:**
- Create: `dashboard/bot.html`
- Create: `dashboard/bot.js`
- Modify: `dashboard/styles.css`

**Behavior:**
- URL params: `?id=<bot_id>` selects which bot to render
- Loads `data/bots/<bot_id>.json`
- Renders:
  - Header: bot_id + description + paused state badge
  - Top stats row: total return, annualized, Sharpe + 95% CI, max DD, days, significance badge (✓ green / ⚠ yellow / ✗ gray)
  - Equity curve chart (Chart.js)
  - Factor loadings table (alpha_per_day, beta_mkt, beta_size, beta_value, R²)
  - Current weights table (ticker → weight %)
  - Recent trades table (last 30)
- Disclaimer footer

**Acceptance:** can navigate to `bot.html?id=spy-vol` and see SPY-Vol's full detail.

**Steps:** Build the static HTML + JS. Single commit: `feat(dashboard): per-bot detail page`

---

## Task 3: Bot vs Bot comparison page (`compare.html`)

**Files:**
- Create: `dashboard/compare.html`
- Create: `dashboard/compare.js`

**Behavior:**
- Two dropdowns to select any pair of bots (or SPY, QQQ as benchmarks)
- Loads both bots' per-bot JSONs
- Renders:
  - Both equity curves overlaid on a single chart
  - Side-by-side stats table (return, Sharpe + CIs, max DD, alpha t-stat)
  - Daily +/- comparison: who won today, this week, this month
  - Pair t-test on daily return differences (using existing `alpha_t_stat_vs_benchmark` math against the OTHER bot as benchmark)
  - **Default view: `codex-r1000` vs `meta-ensemble`** — the headline competition

**Acceptance:** can compare any two bots, see overlay + stats.

**Steps:** Single commit: `feat(dashboard): Bot vs Bot head-to-head comparison page`

---

## Task 4: Methodology page (`methodology.html`)

**Files:**
- Create: `dashboard/methodology.html`
- Create: `dashboard/methodology.js` (minimal — could be inline)

**Content (static markdown-style HTML):**
- What this is + what it isn't (research tool, not advice)
- Strategies listed with one-line descriptions + academic references
- Slippage model formula
- Factor decomposition formulas
- Bootstrap CI methodology
- Significance gates
- Survivorship bias disclosure
- Regime kill-switch rules
- Meta-ensemble weighting formula
- Data sources (yfinance + Wikipedia universe)
- Limitations section

**Acceptance:** all sections present, links to academic papers (Jegadeesh-Titman 1993, etc.) where applicable.

**Steps:** Single commit: `feat(dashboard): methodology + disclosure page`

---

## Task 5: Validation page (`validation.html`)

**Files:**
- Create: `dashboard/validation.html`
- Create: `dashboard/validation.js`

**Behavior:**
- Loads `data/validation.json`
- For each strategy, shows:
  - Backtest aggregate Sharpe with CI
  - Median alpha t-stat vs SPY
  - Significance weight (significance badge: ≥0.7 green, 0.3-0.7 yellow, <0.3 gray)
  - Per-window Sharpes mini-chart
  - Live Sharpe (when available)
  - Pass/fail flags for the implicit gates (Sharpe > SPY-Vol, sig_weight > 0.3, etc.)
- Highlights ML strategies that "failed validation" (when Phase 3 ships) with red badges

**Acceptance:** all strategies in calibration report appear.

**Steps:** Single commit: `feat(dashboard): per-strategy validation evidence page`

---

## Task 6: Top-of-dashboard nav + market banner

**Files:**
- Modify: `dashboard/index.html` — add a `<nav>` linking to compare/validation/methodology pages
- Modify: `dashboard/styles.css` — nav bar styling
- Modify: `dashboard/app.js` — render top market banner with QQQ alongside SPY (existing code shows SPY only)

**Acceptance:** nav links work, market banner shows both indexes.

**Steps:** Single commit: `feat(dashboard): nav menu + dual-index market banner`

---

## Task 7: Polish + ship-readiness check

**Files:**
- Modify: any minor polish
- Run a real morning step against synthetic-then-real data
- Verify all dashboard pages load (open files in a browser via `file://` for sanity)
- Verify no broken links between pages
- Verify the dashboard JSON contracts haven't drifted (run e2e tests)

**Final commit:** `chore: dashboard polish + cross-page link audit`

---

## Self-review

After Phase 4:
- 4 new dashboard pages (bot detail, compare, methodology, validation)
- Existing index.html has nav + dual-index banner
- All JSON exports produced by morning_command
- Default Bot-vs-Bot view shows the headline competition: `codex-r1000` vs `meta-ensemble`
- Ready to upload via `bootstrap.sh` — GH Pages will serve the full multi-page site

This makes the project upload-ready.

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

### Dashboard pages

| Page | URL path | Description |
|------|----------|-------------|
| Home | `/` | Live tournament leaderboard + equity curves |
| Bot detail | `/bot.html?id=<bot_id>` | Per-bot equity curve, factor loadings, weights, trades |
| Compare | `/compare.html` | Head-to-head bot comparison (default: `codex-r1000` vs `meta-ensemble`) |
| Validation | `/validation.html` | Walk-forward backtest evidence with significance badges |
| Methodology | `/methodology.html` | Strategy descriptions, formulas, disclosures |
| Calibration | `/backtest.html` | Walk-forward equity curves per window |

To preview locally:

```bash
cd dashboard && python3 -m http.server 8080
# open http://localhost:8080
```

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

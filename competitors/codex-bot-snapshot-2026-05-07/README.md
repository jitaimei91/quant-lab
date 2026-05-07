# Morning Quant Bot

Local, no-key paper trading assistant for daily market action reports.

This bot is a decision-support and simulation tool. It does not connect to a
broker, does not place trades, and does not provide guaranteed investment
advice. It ranks actions from transparent rules that are backtested and
walk-forward evaluated on free public daily market data.

## What It Does

- Pulls free daily OHLCV data from no-key public chart endpoints and caches it locally.
- Simulates your account from a JSON snapshot.
- Evolves a population of long-only strategy rules with walk-forward scoring.
- Backtests candidate strategies with slippage and drawdown penalties.
- Produces a morning Markdown report with suggested paper actions.
- Optionally applies the report's target allocation to the paper account.

No paid market-data API key is required. In practice, scheduled cloud runs are
more reliable with a free data key because no-key public endpoints can rate
limit aggressively.

Optional free keys can improve reliability:

- `ALPHA_VANTAGE_API_KEY`
- `STOOQ_API_KEY`

Set `MARKET_DATA_PROVIDER=auto`, `yahoo`, `alphavantage`, or `stooq`.
See `config/.env.example` for the environment variable names.

For Alpha Vantage, the default function is `TIME_SERIES_DAILY`, which is the
free-compatible daily OHLCV endpoint. Set `ALPHA_VANTAGE_FUNCTION` only if you
know your key supports a different endpoint.

## Quick Start

From this directory:

```bash
python3 run_bot.py init-account --cash 10000 --positions "SPY:2:500,QQQ:1:430"
python3 run_bot.py morning
```

Your report will be written under `reports/`.

If you want the bot to update its own simulated account after generating the
report:

```bash
python3 run_bot.py morning --paper-fill
```

The bot also writes dashboard artifacts to `public/`:

- `latest.json`
- `history.json`
- `index.html`

## Account Format

Edit `config/account.json` after running `init-account`, or copy
`config/account.example.json`.

```json
{
  "cash": 10000,
  "positions": {
    "SPY": { "shares": 2, "avg_cost": 500 },
    "QQQ": { "shares": 1, "avg_cost": 430 }
  }
}
```

Use this for simulation only. Keep your real brokerage credentials out of this
project.

## Universe

Edit `config/universe.txt` to change the symbols the bot can choose from. The
default is a broad ETF universe so the first version has diversified candidates
without needing paid fundamentals data.

## Commands

```bash
python3 run_bot.py init-account --cash 25000 --positions "AAPL:5:180,MSFT:3:410"
python3 run_bot.py backtest --symbols SPY,QQQ,IWM,TLT,GLD
python3 run_bot.py evolve --generations 6 --population 50
python3 run_bot.py morning --paper-fill
python3 run_bot.py schedule
python3 run_bot.py export-site
```

## Morning Automation

The `schedule` command prints a local cron entry. It runs the bot every weekday
morning and saves reports locally. Cron does not require a paid service.

For GitHub Actions and Vercel deployment, see `docs/DEPLOYMENT.md`.

## How "Self Evolve" Works

The bot does not blindly rewrite itself. Each run:

1. Loads the current strategy leaderboard from `state/leaderboard.json`.
2. Generates mutated strategy candidates.
3. Backtests each candidate.
4. Scores train, validation, and test windows separately.
5. Penalizes drawdown, turnover, and weak out-of-sample results.
6. Promotes only the highest-scoring strategy parameters.

This is intentionally conservative. A strategy that looks good only in-sample
is treated as fragile.

## Comparing Bots

Run future variants with different IDs:

```bash
python3 run_bot.py morning --bot-id momentum-v1
python3 run_bot.py morning --bot-id defensive-v1
```

Each run appends to `state/bot_runs.jsonl`. The dashboard compares the latest
run per `bot_id`.

See `docs/BOT_SCHEMA.md` for the comparison schema.

## Limitations

- Public free data can be delayed or unavailable.
- Daily bars are not suitable for intraday execution.
- Backtests are approximations and cannot guarantee future results.
- Slippage, taxes, borrow costs, liquidity, and spreads can change results.
- This first version is long-only and does not trade options.

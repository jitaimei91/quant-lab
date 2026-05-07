# Private Scheduled Deployment

This bot can run privately in GitHub Actions and publish a static dashboard to
Vercel.

## GitHub Secrets

Required for a real simulation:

- `ACCOUNT_JSON`: your simulated account JSON.

Optional data keys:

- `ALPHA_VANTAGE_API_KEY`: free Alpha Vantage key.
- `STOOQ_API_KEY`: free Stooq key if you use its CSV endpoint.

Optional Vercel deployment:

- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

Optional GitHub variables:

- `BOT_ID`: display/comparison id, such as `momentum-v1`.
- `MARKET_DATA_PROVIDER`: `auto`, `yahoo`, `alphavantage`, or `stooq`.
- `PUBLIC_REDACT`: set to `1` to hide cash, equity, and share counts in the
  deployed JSON.

## Workflow

The workflow at `.github/workflows/morning-quant-bot.yml`:

1. Restores cached market data and bot state.
2. Writes `config/account.json` from the `ACCOUNT_JSON` secret.
3. Runs the tests.
4. Generates a morning report.
5. Uploads reports/state/dashboard files as GitHub artifacts.
6. Deploys the `public/` folder to Vercel if `VERCEL_TOKEN` is set.

State is persisted with the GitHub Actions cache and uploaded as an artifact.
That is good enough for a private paper-trading bot, but it is not a permanent
database. For a production-grade version, move `state/` into a private database
or encrypted object store.

## Vercel Privacy

A Vercel deployment can be public unless you enable deployment protection,
password protection, team access controls, or another private access layer.
Use `--redact-public` if you want the dashboard JSON to hide cash, equity, and
share counts:

```bash
python3 run_bot.py morning --paper-fill --redact-public
```

## Cross-Comparing Future Bots

Every run appends one JSON object to `state/bot_runs.jsonl` and rebuilds:

- `public/latest.json`
- `public/history.json`
- `public/index.html`

Future bots should write the same schema with a distinct `bot_id`. The dashboard
groups by `bot_id` and compares the most recent walk-forward metrics.

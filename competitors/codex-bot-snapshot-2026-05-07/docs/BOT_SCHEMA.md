# Bot Run Schema

Future bots can be compared by writing JSON lines to `state/bot_runs.jsonl`
with this shape:

```json
{
  "schema_version": 1,
  "bot_id": "momentum-v1",
  "generated_at": "2026-05-07T08:00:00",
  "report_date": "2026-05-07",
  "data_window": {
    "start": "2018-05-08",
    "end": "2026-05-06"
  },
  "account": {
    "cash": 1000.0,
    "equity": 12500.0,
    "positions": {
      "SPY": {
        "shares": 10.0,
        "avg_cost": 500.0
      }
    }
  },
  "strategy": {
    "lookback": 126,
    "sma_fast": 40,
    "sma_slow": 180,
    "vol_lookback": 30,
    "max_positions": 5,
    "min_momentum": 0.02,
    "max_symbol_vol": 0.45,
    "cash_buffer": 0.08,
    "max_weight": 0.3,
    "rebalance_days": 10
  },
  "walk_forward": {
    "total_return": 0.2,
    "cagr": 0.08,
    "sharpe": 0.9,
    "volatility": 0.15,
    "max_drawdown": -0.12,
    "trades": 50,
    "turnover": 3.4,
    "final_equity": 12000.0
  },
  "recommendations": []
}
```

The dashboard groups by `bot_id` and compares the most recent run for each bot.


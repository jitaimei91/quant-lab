from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .models import Account, Position, Recommendation


def apply_paper_fills(
    account: Account,
    recommendations: list[Recommendation],
    trade_log_path: Path,
    slippage_bps: float = 5.0,
) -> Account:
    slippage = slippage_bps / 10_000.0
    fills: list[dict] = []

    for rec in recommendations:
        if abs(rec.delta_shares) < 1e-9:
            continue
        fill_price = rec.latest_price * (1 + slippage if rec.delta_shares > 0 else 1 - slippage)
        cost = rec.delta_shares * fill_price
        account.cash -= cost
        position = account.positions.get(rec.symbol, Position(rec.symbol, 0.0, fill_price))
        new_shares = position.shares + rec.delta_shares
        if new_shares <= 1e-9:
            account.positions.pop(rec.symbol, None)
        else:
            if rec.delta_shares > 0:
                previous_cost = position.shares * position.avg_cost
                added_cost = rec.delta_shares * fill_price
                avg_cost = (previous_cost + added_cost) / new_shares
            else:
                avg_cost = position.avg_cost
            account.positions[rec.symbol] = Position(rec.symbol, new_shares, avg_cost)

        fills.append(
            {
                "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "symbol": rec.symbol,
                "action": rec.action,
                "shares": rec.delta_shares,
                "fill_price": fill_price,
                "notional": cost,
            }
        )

    if fills:
        trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        with trade_log_path.open("a", encoding="utf-8") as handle:
            for fill in fills:
                handle.write(json.dumps(fill, sort_keys=True) + "\n")

    return account


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

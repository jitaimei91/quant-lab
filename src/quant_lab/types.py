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

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class Bar:
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class Position:
    symbol: str
    shares: float
    avg_cost: float = 0.0

    @classmethod
    def from_dict(cls, symbol: str, raw: dict[str, Any]) -> "Position":
        return cls(
            symbol=symbol.upper(),
            shares=float(raw.get("shares", 0.0)),
            avg_cost=float(raw.get("avg_cost", 0.0)),
        )

    def to_dict(self) -> dict[str, float]:
        return {"shares": self.shares, "avg_cost": self.avg_cost}


@dataclass
class Account:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Account":
        positions = {
            symbol.upper(): Position.from_dict(symbol, data)
            for symbol, data in raw.get("positions", {}).items()
        }
        return cls(
            cash=float(raw.get("cash", 0.0)),
            positions=positions,
            notes=str(raw.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cash": round(self.cash, 2),
            "positions": {
                symbol: position.to_dict()
                for symbol, position in sorted(self.positions.items())
                if abs(position.shares) > 1e-9
            },
            "notes": self.notes,
        }

    def equity(self, prices: dict[str, float]) -> float:
        total = self.cash
        for symbol, position in self.positions.items():
            total += position.shares * prices.get(symbol, position.avg_cost)
        return total


@dataclass(frozen=True)
class StrategyParams:
    lookback: int
    sma_fast: int
    sma_slow: int
    vol_lookback: int
    max_positions: int
    min_momentum: float
    max_symbol_vol: float
    cash_buffer: float
    max_weight: float
    rebalance_days: int

    def key(self) -> str:
        parts = [
            self.lookback,
            self.sma_fast,
            self.sma_slow,
            self.vol_lookback,
            self.max_positions,
            round(self.min_momentum, 4),
            round(self.max_symbol_vol, 4),
            round(self.cash_buffer, 4),
            round(self.max_weight, 4),
            self.rebalance_days,
        ]
        return "|".join(str(part) for part in parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "sma_fast": self.sma_fast,
            "sma_slow": self.sma_slow,
            "vol_lookback": self.vol_lookback,
            "max_positions": self.max_positions,
            "min_momentum": self.min_momentum,
            "max_symbol_vol": self.max_symbol_vol,
            "cash_buffer": self.cash_buffer,
            "max_weight": self.max_weight,
            "rebalance_days": self.rebalance_days,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StrategyParams":
        return cls(
            lookback=int(raw["lookback"]),
            sma_fast=int(raw["sma_fast"]),
            sma_slow=int(raw["sma_slow"]),
            vol_lookback=int(raw["vol_lookback"]),
            max_positions=int(raw["max_positions"]),
            min_momentum=float(raw["min_momentum"]),
            max_symbol_vol=float(raw["max_symbol_vol"]),
            cash_buffer=float(raw["cash_buffer"]),
            max_weight=float(raw["max_weight"]),
            rebalance_days=int(raw["rebalance_days"]),
        )


@dataclass
class BacktestMetrics:
    total_return: float
    cagr: float
    sharpe: float
    volatility: float
    max_drawdown: float
    trades: int
    turnover: float
    final_equity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "sharpe": self.sharpe,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "trades": self.trades,
            "turnover": self.turnover,
            "final_equity": self.final_equity,
        }


@dataclass
class Recommendation:
    symbol: str
    action: str
    current_shares: float
    target_shares: float
    delta_shares: float
    latest_price: float
    target_weight: float
    reason: str

    def to_markdown_row(self) -> str:
        return (
            f"| {self.symbol} | {self.action} | "
            f"{self.current_shares:.4f} | {self.target_shares:.4f} | "
            f"{self.delta_shares:+.4f} | ${self.latest_price:,.2f} | "
            f"{self.target_weight:.1%} | {self.reason} |"
        )


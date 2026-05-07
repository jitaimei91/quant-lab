"""Paper trading engine.

Rebalances a portfolio toward target weights using realistic slippage,
respects per-name liquidity, and emits a Trade list for audit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date

from ..slippage import apply_slippage, spread_bps
from ..types import Portfolio, Position, Trade


@dataclass
class PaperResult:
    portfolio: Portfolio
    trades: list[Trade] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def rebalance(
    portfolio: Portfolio,
    target_weights: dict[str, float],
    prices: dict[str, float],
    advs: dict[str, float],
    as_of: _date,
    drift_threshold: float = 0.005,
    min_trade_value: float = 5.0,
    adv_floor: float = 1_000_000,
    block_new_entries: bool = False,
) -> PaperResult:
    """Rebalance `portfolio` toward `target_weights`.

    - Applies slippage from `slippage.apply_slippage`.
    - Skips names with ADV below `adv_floor` (liquidity gate).
    - Skips trades whose change in weight is below `drift_threshold`.
    - When `block_new_entries=True`, skips opening brand-new positions
      (current_shares == 0 and target_value > 0). Existing positions
      are still managed (trimmed or closed) normally.
    """
    equity = portfolio.equity(prices)
    if equity <= 0:
        return PaperResult(portfolio=portfolio)

    new_positions = dict(portfolio.positions)
    cash = portfolio.cash
    trades: list[Trade] = []
    skipped: list[str] = []

    symbols = sorted(set(new_positions) | set(target_weights))
    for symbol in symbols:
        target_w = target_weights.get(symbol, 0.0)
        adv = advs.get(symbol, 0.0)
        if target_w > 0 and adv < adv_floor:
            skipped.append(f"{symbol}: ADV ${adv:,.0f} below floor ${adv_floor:,.0f}")
            target_w = 0.0  # don't open new position; existing is allowed to roll off

        price = prices.get(symbol)
        if price is None or price <= 0:
            skipped.append(f"{symbol}: no price")
            continue

        current_pos = new_positions.get(symbol)
        current_shares = current_pos.shares if current_pos else 0.0
        current_value = current_shares * price
        target_value = equity * target_w

        # Regime kill-switch: skip opening brand-new positions
        if block_new_entries and current_shares == 0 and target_value > 0:
            skipped.append(f"{symbol}: new entry blocked (regime halt_new_entries)")
            continue
        delta_value = target_value - current_value

        if abs(delta_value) < min_trade_value:
            continue
        if abs(delta_value) / max(equity, 1.0) < drift_threshold:
            continue

        side = "BUY" if delta_value > 0 else "SELL"
        bps = spread_bps(adv) if adv > 0 else 30.0
        fill_price = apply_slippage(price=price, side=side, spread_bps_value=bps)
        delta_shares = delta_value / fill_price
        # Cap sells at existing shares to avoid going short
        if side == "SELL" and current_shares > 0:
            delta_shares = max(delta_shares, -current_shares)
        cash -= delta_shares * fill_price

        new_shares = current_shares + delta_shares
        if abs(new_shares) < 1e-9:
            new_positions.pop(symbol, None)
        else:
            new_avg = price if current_shares == 0 else (
                ((current_pos.avg_cost * current_shares) + (fill_price * delta_shares))
                / new_shares
                if side == "BUY" else current_pos.avg_cost
            )
            new_positions[symbol] = Position(symbol=symbol, shares=new_shares, avg_cost=new_avg)

        trades.append(
            Trade(
                bot_id=portfolio.bot_id,
                symbol=symbol,
                side=side,
                shares=abs(delta_shares),
                price=fill_price,
                slippage_bps=bps,
                timestamp=as_of,
                reason=f"rebalance to target weight {target_w:.4f}",
            )
        )

    new_portfolio = Portfolio(bot_id=portfolio.bot_id, cash=cash, positions=new_positions)
    return PaperResult(portfolio=new_portfolio, trades=trades, skipped=skipped)

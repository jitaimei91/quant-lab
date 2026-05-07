"""Cost model for paper trading.

Approximates round-trip transaction costs (bid-ask spread + market impact)
as a function of average daily dollar volume. See spec section 7.
"""
from __future__ import annotations

import math


def spread_bps(adv_dollars: float) -> float:
    """Return one-way slippage cost in basis points.

    Roughly: liquid mega-caps ~3-6 bps, mid-caps ~10-15 bps,
    R1000-tail names ~20-30 bps. Floor at 1 bp.
    """
    adv_millions = max(adv_dollars / 1_000_000, 1.0)
    raw = 5.0 + 100.0 / math.sqrt(adv_millions)
    return max(1.0, raw)


def apply_slippage(price: float, side: str, spread_bps_value: float) -> float:
    """Return the fill price for a market order at `price` with given spread."""
    factor = spread_bps_value / 10_000.0
    if side == "BUY":
        return price * (1.0 + factor)
    if side == "SELL":
        return price * (1.0 - factor)
    raise ValueError(f"Unknown side: {side!r}")

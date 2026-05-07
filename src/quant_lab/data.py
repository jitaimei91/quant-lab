"""Daily-bar market data fetcher.

Phase 1 uses yfinance only. Phase 2 will add Alpaca primary with yfinance
fallback per spec section 5.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import yfinance as yf

from .types import Bar


def fetch_history(symbol: str, lookback_days: int = 365) -> list[Bar]:
    """Fetch OHLCV bars for the last `lookback_days` trading days for `symbol`."""
    ticker = yf.Ticker(symbol)
    end = date.today()
    start = end - timedelta(days=lookback_days)
    df = ticker.history(start=start, end=end + timedelta(days=1), auto_adjust=True)
    if df.empty:
        return []

    bars: list[Bar] = []
    for ts, row in df.iterrows():
        bars.append(
            Bar(
                symbol=symbol.upper(),
                date=ts.date(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
            )
        )
    return bars


def latest_bar(symbol: str) -> Bar | None:
    """Return the most recent bar for `symbol`, or None if no data."""
    bars = fetch_history(symbol, lookback_days=10)
    return bars[-1] if bars else None


def fetch_many(symbols: Iterable[str], lookback_days: int = 365) -> dict[str, list[Bar]]:
    """Fetch histories for many symbols, skipping any that fail."""
    histories: dict[str, list[Bar]] = {}
    for symbol in symbols:
        try:
            bars = fetch_history(symbol, lookback_days=lookback_days)
            if bars:
                histories[symbol.upper()] = bars
        except Exception:
            continue
    return histories

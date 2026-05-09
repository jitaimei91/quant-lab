"""Daily-bar market data fetcher.

Phase 1 uses yfinance only. Phase 2 will add Alpaca primary with yfinance
fallback per spec section 5.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import yfinance as yf

from ..types import Bar


def fetch_history(symbol: str, lookback_days: int = 365) -> list[Bar]:
    """Fetch OHLCV bars for the last `lookback_days` trading days for `symbol`.

    Returns an empty list on any fetch failure (network, rate limit, etc.).
    The caller is responsible for handling missing data.
    """
    end = date.today()
    start = end - timedelta(days=lookback_days)
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end + timedelta(days=1), auto_adjust=True)
    except Exception:
        return []
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


def fetch_history_range(symbol: str, start: date, end: date) -> list[Bar]:
    """Fetch OHLCV bars for `symbol` between `start` and `end` (inclusive).

    Returns an empty list on any fetch failure.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end + timedelta(days=1), auto_adjust=True)
    except Exception:
        return []
    if df.empty:
        return []

    bars: list[Bar] = []
    for ts, row in df.iterrows():
        d = ts.date()
        if start <= d <= end:
            bars.append(
                Bar(
                    symbol=symbol.upper(),
                    date=d,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                )
            )
    return bars


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


def fetch_history_batch(
    symbols: Iterable[str],
    lookback_days: int = 365,
) -> dict[str, list[Bar]]:
    """Batch-fetch many symbols in one yfinance call. Use for 100+ symbol lists.

    yfinance's `download()` can pull hundreds of tickers concurrently in one
    HTTP request, which is dramatically faster (10-100x) than sequential
    per-symbol calls. Returns the same shape as `fetch_many`: a dict of
    {SYMBOL: [Bar, ...]} with failed symbols silently skipped.

    Notes:
    - yfinance returns a MultiIndex columns DataFrame keyed (ticker, field).
    - When a single symbol is requested, yfinance returns a flat columns
      DataFrame instead — handle both shapes.
    - Volume can be NaN on holidays/halts; coerce to 0.
    """
    syms = sorted({s.upper() for s in symbols if s})
    if not syms:
        return {}

    end = date.today()
    start = end - timedelta(days=lookback_days)
    try:
        df = yf.download(
            tickers=" ".join(syms),
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception:
        return {}

    if df is None or df.empty:
        return {}

    out: dict[str, list[Bar]] = {}
    # Flat-columns case: only one ticker requested
    if len(syms) == 1:
        sym = syms[0]
        bars: list[Bar] = []
        for ts, row in df.iterrows():
            try:
                bars.append(
                    Bar(
                        symbol=sym,
                        date=ts.date(),
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=int(row["Volume"]) if not _is_nan(row["Volume"]) else 0,
                    )
                )
            except (ValueError, KeyError, TypeError):
                continue
        if bars:
            out[sym] = bars
        return out

    # MultiIndex case: (ticker, field) on columns
    for sym in syms:
        if sym not in df.columns.get_level_values(0):
            continue
        sub = df[sym]
        if sub is None or sub.empty:
            continue
        bars = []
        for ts, row in sub.iterrows():
            close = row.get("Close")
            if close is None or _is_nan(close):
                continue
            try:
                bars.append(
                    Bar(
                        symbol=sym,
                        date=ts.date(),
                        open=float(row.get("Open", close)) if not _is_nan(row.get("Open", close)) else float(close),
                        high=float(row.get("High", close)) if not _is_nan(row.get("High", close)) else float(close),
                        low=float(row.get("Low", close)) if not _is_nan(row.get("Low", close)) else float(close),
                        close=float(close),
                        volume=int(row.get("Volume", 0)) if not _is_nan(row.get("Volume", 0)) else 0,
                    )
                )
            except (ValueError, TypeError):
                continue
        if bars:
            out[sym] = bars
    return out


def _is_nan(x) -> bool:
    """NaN check that doesn't raise on non-numerics."""
    try:
        return x != x  # NaN is the only value that fails self-equality
    except Exception:
        return True

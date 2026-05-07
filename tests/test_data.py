import pandas as pd
from datetime import date
from unittest.mock import MagicMock

from quant_lab.data import fetch_history, latest_bar
from quant_lab.types import Bar


def _fake_yf_history():
    idx = pd.to_datetime(["2026-05-04", "2026-05-05", "2026-05-06"])
    df = pd.DataFrame(
        {
            "Open": [498.0, 499.5, 500.0],
            "High": [501.0, 502.0, 503.0],
            "Low": [497.0, 498.5, 499.0],
            "Close": [500.0, 501.0, 502.5],
            "Volume": [50_000_000, 48_000_000, 52_000_000],
        },
        index=idx,
    )
    return df


def test_fetch_history_returns_bars(monkeypatch):
    fake_ticker = MagicMock()
    fake_ticker.history.return_value = _fake_yf_history()
    monkeypatch.setattr("yfinance.Ticker", lambda symbol: fake_ticker)

    bars = fetch_history("SPY", lookback_days=5)
    assert all(isinstance(b, Bar) for b in bars)
    assert bars[-1].close == 502.5
    assert bars[-1].symbol == "SPY"


def test_latest_bar_returns_last(monkeypatch):
    fake_ticker = MagicMock()
    fake_ticker.history.return_value = _fake_yf_history()
    monkeypatch.setattr("yfinance.Ticker", lambda symbol: fake_ticker)

    bar = latest_bar("SPY")
    assert bar.symbol == "SPY"
    assert bar.date == date(2026, 5, 6)
    assert bar.close == 502.5

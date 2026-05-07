from __future__ import annotations

import csv
import json
import os
from datetime import date, datetime, time, timedelta, timezone
from json import JSONDecodeError
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import Bar


class MarketDataError(RuntimeError):
    pass


class MarketDataClient:
    """Daily market data client with local CSV caching.

    Provider order defaults to no-key Yahoo chart data first, then optional
    keyed free tiers when the relevant environment variables are present.
    """

    def __init__(self, cache_dir: Path, timeout: int = 20, provider: str | None = None) -> None:
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.provider = (provider or os.getenv("MARKET_DATA_PROVIDER", "auto")).lower()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_history(self, symbol: str, start: date, end: date) -> list[Bar]:
        symbol = symbol.upper()
        cached = self._read_cache(symbol)
        if self._cache_covers(cached, start, end):
            return self._slice(cached, start, end)

        failures: list[str] = []
        for provider_name, fetcher in self._provider_order():
            try:
                fresh = fetcher(symbol, start, end)
            except (URLError, TimeoutError, HTTPError, JSONDecodeError, MarketDataError) as exc:
                failures.append(f"{provider_name}: {exc}")
                continue
            if fresh:
                merged = self._merge(cached, fresh)
                self._write_cache(symbol, merged)
                return self._slice(merged, start, end)
            failures.append(f"{provider_name}: no rows")

        if cached:
            return self._slice(cached, start, end)
        raise MarketDataError(f"Could not fetch {symbol}. {'; '.join(failures)}")

    def _provider_order(self):
        providers = {
            "yahoo": self._fetch_yahoo,
            "alphavantage": self._fetch_alpha_vantage,
            "stooq": self._fetch_stooq,
        }
        if self.provider != "auto":
            if self.provider not in providers:
                raise MarketDataError(f"Unknown data provider: {self.provider}")
            return [(self.provider, providers[self.provider])]

        order = [("yahoo", self._fetch_yahoo)]
        if os.getenv("ALPHA_VANTAGE_API_KEY"):
            order.append(("alphavantage", self._fetch_alpha_vantage))
        if os.getenv("STOOQ_API_KEY"):
            order.append(("stooq", self._fetch_stooq))
        return order

    def _cache_path(self, symbol: str) -> Path:
        safe = "".join(ch if ch.isalnum() else "_" for ch in symbol.upper())
        return self.cache_dir / f"{safe}.csv"

    def _read_cache(self, symbol: str) -> list[Bar]:
        path = self._cache_path(symbol)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [self._row_to_bar(symbol, row) for row in reader]

    def _write_cache(self, symbol: str, bars: list[Bar]) -> None:
        path = self._cache_path(symbol)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["date", "open", "high", "low", "close", "volume"],
            )
            writer.writeheader()
            for bar in bars:
                writer.writerow(
                    {
                        "date": bar.date.isoformat(),
                        "open": f"{bar.open:.8f}",
                        "high": f"{bar.high:.8f}",
                        "low": f"{bar.low:.8f}",
                        "close": f"{bar.close:.8f}",
                        "volume": str(bar.volume),
                    }
                )

    def _fetch_stooq(self, symbol: str, start: date, end: date) -> list[Bar]:
        params = {
            "s": self._stooq_symbol(symbol),
            "d1": start.strftime("%Y%m%d"),
            "d2": end.strftime("%Y%m%d"),
            "i": "d",
        }
        api_key = os.getenv("STOOQ_API_KEY")
        if api_key:
            params["apikey"] = api_key
        query = urlencode(params)
        url = f"https://stooq.com/q/d/l/?{query}"
        request = Request(url, headers={"User-Agent": "morning-quant-bot/0.1"})
        with urlopen(request, timeout=self.timeout) as response:
            text = response.read().decode("utf-8")
        rows = list(csv.DictReader(text.splitlines()))
        bars = [self._row_to_bar(symbol, row) for row in rows if row.get("Close")]
        if not bars:
            raise MarketDataError(f"No rows returned for {symbol}")
        return bars

    def _fetch_yahoo(self, symbol: str, start: date, end: date) -> list[Bar]:
        errors: list[str] = []
        for host in ("query2.finance.yahoo.com", "query1.finance.yahoo.com"):
            try:
                return self._fetch_yahoo_period(host, symbol, start, end)
            except (URLError, TimeoutError, HTTPError, JSONDecodeError, MarketDataError) as exc:
                errors.append(f"{host}/period: {exc}")
        for host in ("query2.finance.yahoo.com", "query1.finance.yahoo.com"):
            try:
                return self._fetch_yahoo_range(host, symbol, start, end)
            except (URLError, TimeoutError, HTTPError, JSONDecodeError, MarketDataError) as exc:
                errors.append(f"{host}/range: {exc}")
        raise MarketDataError("; ".join(errors))

    def _fetch_yahoo_period(
        self,
        host: str,
        symbol: str,
        start: date,
        end: date,
    ) -> list[Bar]:
        period1 = int(datetime.combine(start, time.min, tzinfo=timezone.utc).timestamp())
        period2 = int(datetime.combine(end + timedelta(days=1), time.min, tzinfo=timezone.utc).timestamp())
        query = urlencode(
            {
                "period1": period1,
                "period2": period2,
                "interval": "1d",
                "events": "history",
                "includeAdjustedClose": "true",
            }
        )
        url = f"https://{host}/v8/finance/chart/{symbol}?{query}"
        return self._read_yahoo_chart(url, symbol, start, end)

    def _fetch_yahoo_range(
        self,
        host: str,
        symbol: str,
        start: date,
        end: date,
    ) -> list[Bar]:
        query = urlencode(
            {
                "range": self._yahoo_range(start, end),
                "interval": "1d",
                "events": "history",
                "includeAdjustedClose": "true",
            }
        )
        url = f"https://{host}/v8/finance/chart/{symbol}?{query}"
        return self._read_yahoo_chart(url, symbol, start, end)

    def _read_yahoo_chart(self, url: str, symbol: str, start: date, end: date) -> list[Bar]:
        request = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            },
        )
        with urlopen(request, timeout=self.timeout) as response:
            text = response.read().decode("utf-8")
        if text.strip().lower().startswith("too many requests"):
            raise MarketDataError("Yahoo rate limit")
        payload = json.loads(text)

        chart = payload.get("chart", {})
        if chart.get("error"):
            raise MarketDataError(str(chart["error"]))
        result = (chart.get("result") or [None])[0]
        if not result:
            raise MarketDataError(f"No Yahoo chart result for {symbol}")

        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [None])[0]
        if not timestamps or not quote:
            raise MarketDataError(f"No Yahoo quote rows for {symbol}")

        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []
        bars: list[Bar] = []
        for index, stamp in enumerate(timestamps):
            values = (
                opens[index] if index < len(opens) else None,
                highs[index] if index < len(highs) else None,
                lows[index] if index < len(lows) else None,
                closes[index] if index < len(closes) else None,
            )
            if any(value is None for value in values):
                continue
            row_date = datetime.fromtimestamp(stamp, tz=timezone.utc).date()
            if row_date < start or row_date > end:
                continue
            volume = volumes[index] if index < len(volumes) and volumes[index] is not None else 0
            bars.append(
                Bar(
                    symbol=symbol.upper(),
                    date=row_date,
                    open=float(values[0]),
                    high=float(values[1]),
                    low=float(values[2]),
                    close=float(values[3]),
                    volume=int(volume),
                )
            )
        if not bars:
            raise MarketDataError(f"No Yahoo rows returned for {symbol}")
        return bars

    def _fetch_alpha_vantage(self, symbol: str, start: date, end: date) -> list[Bar]:
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            raise MarketDataError("ALPHA_VANTAGE_API_KEY is not set")
        query = urlencode(
            {
                "function": os.getenv("ALPHA_VANTAGE_FUNCTION", "TIME_SERIES_DAILY"),
                "symbol": symbol,
                "outputsize": "full",
                "apikey": api_key,
            }
        )
        url = f"https://www.alphavantage.co/query?{query}"
        request = Request(url, headers={"User-Agent": "morning-quant-bot/0.1"})
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if "Error Message" in payload:
            raise MarketDataError(payload["Error Message"])
        if "Note" in payload:
            raise MarketDataError(payload["Note"])
        if "Information" in payload:
            raise MarketDataError(payload["Information"])
        series = payload.get("Time Series (Daily)")
        if not series:
            raise MarketDataError("No Alpha Vantage daily series returned")
        bars: list[Bar] = []
        for raw_date, row in series.items():
            row_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
            if row_date < start or row_date > end:
                continue
            bars.append(
                Bar(
                    symbol=symbol.upper(),
                    date=row_date,
                    open=float(row["1. open"]),
                    high=float(row["2. high"]),
                    low=float(row["3. low"]),
                    close=float(row["4. close"]),
                    volume=int(float(row.get("5. volume") or row.get("6. volume") or 0)),
                )
            )
        bars.sort(key=lambda item: item.date)
        if not bars:
            raise MarketDataError(f"No Alpha Vantage rows returned for {symbol}")
        return bars

    @staticmethod
    def _stooq_symbol(symbol: str) -> str:
        lowered = symbol.lower().replace("-", ".")
        if "." not in lowered:
            lowered = f"{lowered}.us"
        return lowered

    @staticmethod
    def _yahoo_range(start: date, end: date) -> str:
        days = max(1, (end - start).days)
        if days <= 31:
            return "1mo"
        if days <= 93:
            return "3mo"
        if days <= 186:
            return "6mo"
        if days <= 366:
            return "1y"
        if days <= 366 * 2:
            return "2y"
        if days <= 366 * 5:
            return "5y"
        if days <= 366 * 10:
            return "10y"
        return "max"

    @staticmethod
    def _row_to_bar(symbol: str, row: dict[str, str]) -> Bar:
        return Bar(
            symbol=symbol.upper(),
            date=datetime.strptime(row["Date"] if "Date" in row else row["date"], "%Y-%m-%d").date(),
            open=float(row["Open"] if "Open" in row else row["open"]),
            high=float(row["High"] if "High" in row else row["high"]),
            low=float(row["Low"] if "Low" in row else row["low"]),
            close=float(row["Close"] if "Close" in row else row["close"]),
            volume=int(float(row.get("Volume") or row.get("volume") or 0)),
        )

    @staticmethod
    def _cache_covers(bars: list[Bar], start: date, end: date) -> bool:
        if not bars:
            return False
        return bars[0].date <= start and bars[-1].date >= end

    @staticmethod
    def _slice(bars: list[Bar], start: date, end: date) -> list[Bar]:
        return [bar for bar in bars if start <= bar.date <= end]

    @staticmethod
    def _merge(old: list[Bar], new: list[Bar]) -> list[Bar]:
        by_date = {bar.date: bar for bar in old}
        by_date.update({bar.date: bar for bar in new})
        return [by_date[item] for item in sorted(by_date)]

def fetch_histories(
    client: MarketDataClient,
    symbols: list[str],
    start: date,
    end: date,
    min_rows: int = 260,
) -> dict[str, list[Bar]]:
    histories: dict[str, list[Bar]] = {}
    failures: list[str] = []
    for symbol in symbols:
        try:
            bars = client.get_history(symbol, start, end)
        except MarketDataError as exc:
            failures.append(f"{symbol}: {exc}")
            continue
        if len(bars) >= min_rows:
            histories[symbol] = bars
        else:
            failures.append(f"{symbol}: only {len(bars)} rows")
    if len(histories) < 2:
        joined = "; ".join(failures[:5])
        raise MarketDataError(f"Not enough usable histories. {joined}")
    return histories

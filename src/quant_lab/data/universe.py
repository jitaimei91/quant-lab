"""Universe loader — parse flat ticker files into sorted, deduplicated ticker lists.

Lines starting with '#' or empty lines are ignored.
SPY, QQQ, ^VIX are always included (for strategy benchmarking and regime data).
"""
from __future__ import annotations

from pathlib import Path

_ALWAYS_INCLUDE = {"SPY", "QQQ", "^VIX"}


def parse_universe_text(content: str) -> list[str]:
    """Parse a newline-separated ticker list from a string.

    Skips blank lines and lines starting with '#'. Returns sorted unique tickers
    (uppercased). Does not inject the always-include set — that is load_universe's job.
    """
    tickers: set[str] = set()
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tickers.add(line.upper())
    return sorted(tickers)


def load_universe(
    universe_path: Path,
    watchlist_path: Path | None = None,
) -> list[str]:
    """Load tickers from one or two flat text files.

    Always includes SPY, QQQ, and ^VIX.  Deduplicates case-insensitively and
    returns a sorted list of unique uppercase tickers.
    """
    tickers: set[str] = set(_ALWAYS_INCLUDE)

    tickers.update(parse_universe_text(universe_path.read_text()))

    if watchlist_path is not None:
        tickers.update(parse_universe_text(watchlist_path.read_text()))

    return sorted(tickers)

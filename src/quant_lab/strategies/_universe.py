"""Shared universe-loading helpers for strategies.

`STOCK_UNIVERSE` is the union of all single-stock tickers managed by the lab.
Used both by the stock-momo bot (which trades these names) and by every
ETF/asset-class bot (which excludes them, so e.g. apex's dual-momentum
doesn't try to pick AAPL when it's looking for the next sleeve).

Loaded once at module import — the file is small (~500 lines) so re-reading
it on every strategy call would be wasteful but loading once is fine.
"""
from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SP500_FILE = _REPO_ROOT / "config" / "universe_sp500.txt"


def load_universe(path: Path) -> frozenset[str]:
    """Load tickers from a universe file. Skips blank lines and # comments."""
    if not path.exists():
        return frozenset()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line.upper())
    return frozenset(out)


# Loaded once at import. Empty frozenset if the file is missing (tests, dev).
STOCK_UNIVERSE: frozenset[str] = load_universe(_SP500_FILE)

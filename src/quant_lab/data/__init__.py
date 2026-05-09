"""Market data package.

Re-exports all public symbols from _fetcher so existing imports remain unchanged.
"""
from ._fetcher import (  # noqa: F401
    fetch_history,
    fetch_history_range,
    latest_bar,
    fetch_many,
    fetch_history_batch,
)

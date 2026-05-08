"""Market data package.

Re-exports all public symbols from _fetcher so existing imports remain unchanged.
"""
from ._fetcher import fetch_history, fetch_history_range, latest_bar, fetch_many  # noqa: F401

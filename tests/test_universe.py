"""Tests for quant_lab.data.universe — parse_universe_text + load_universe."""
from __future__ import annotations

from quant_lab.data.universe import load_universe, parse_universe_text


# ---------------------------------------------------------------------------
# parse_universe_text
# ---------------------------------------------------------------------------


def test_parse_skips_comments_and_blanks():
    text = """# header comment
AAPL

MSFT
# another comment
NVDA
"""
    result = parse_universe_text(text)
    assert result == ["AAPL", "MSFT", "NVDA"]


def test_parse_uppercases_and_deduplicates():
    text = "aapl\nAAPL\nMSFT\nmsft"
    result = parse_universe_text(text)
    assert result == ["AAPL", "MSFT"]


def test_parse_empty_content_returns_empty():
    assert parse_universe_text("") == []
    assert parse_universe_text("# only comments\n\n") == []


# ---------------------------------------------------------------------------
# load_universe
# ---------------------------------------------------------------------------


def test_load_always_includes_required_tickers(tmp_path):
    universe_file = tmp_path / "universe.txt"
    universe_file.write_text("AAPL\nMSFT\n")
    result = load_universe(universe_file)
    assert "SPY" in result
    assert "QQQ" in result
    assert "^VIX" in result


def test_load_merges_watchlist(tmp_path):
    universe_file = tmp_path / "universe.txt"
    universe_file.write_text("AAPL\n")
    watchlist_file = tmp_path / "watchlist.txt"
    watchlist_file.write_text("NVDA\nTSLA\n")
    result = load_universe(universe_file, watchlist_path=watchlist_file)
    assert "AAPL" in result
    assert "NVDA" in result
    assert "TSLA" in result


def test_load_returns_sorted_unique(tmp_path):
    universe_file = tmp_path / "universe.txt"
    universe_file.write_text("MSFT\nAAPL\nMSFT\n")
    result = load_universe(universe_file)
    assert result == sorted(set(result))
    assert result.count("MSFT") == 1


def test_load_none_watchlist_path_returns_just_universe(tmp_path):
    universe_file = tmp_path / "universe.txt"
    universe_file.write_text("AAPL\n")
    result = load_universe(universe_file, watchlist_path=None)
    assert "AAPL" in result
    assert "SPY" in result

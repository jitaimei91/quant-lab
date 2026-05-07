"""Codex bot adapter — wraps the frozen morning_quant_bot snapshot.

Two registered variants:
  - CodexBotR1000  (bot_id="codex-r1000") — runs on full histories, excluding
    ^VIX which is not a tradeable instrument.
  - CodexBotNative (bot_id="codex-native") — restricts to the ETFs listed in
    competitors/codex-bot-snapshot-2026-05-07/config/universe.txt.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from ..types import Bar
from .base import Strategy, register

# ---------------------------------------------------------------------------
# Ensure the frozen snapshot is importable regardless of pytest pythonpath
# ---------------------------------------------------------------------------
_SNAPSHOT_SRC = (
    Path(__file__).parent.parent.parent.parent
    / "competitors"
    / "codex-bot-snapshot-2026-05-07"
    / "src"
)
if _SNAPSHOT_SRC.exists() and str(_SNAPSHOT_SRC) not in sys.path:
    sys.path.insert(0, str(_SNAPSHOT_SRC))

from morning_quant_bot.models import Bar as CodexBar  # noqa: E402
from morning_quant_bot.strategy import DEFAULT_STRATEGY, target_weights as codex_target_weights  # noqa: E402

# ---------------------------------------------------------------------------
# Load the native ETF universe once at class-load time
# ---------------------------------------------------------------------------
_NATIVE_UNIVERSE_FILE = (
    Path(__file__).parent.parent.parent.parent
    / "competitors"
    / "codex-bot-snapshot-2026-05-07"
    / "config"
    / "universe.txt"
)


def _load_native_universe() -> frozenset[str]:
    """Load tickers from the codex snapshot's universe.txt, skipping comments."""
    if not _NATIVE_UNIVERSE_FILE.exists():
        return frozenset()
    tickers: set[str] = set()
    for line in _NATIVE_UNIVERSE_FILE.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            tickers.add(stripped.upper())
    return frozenset(tickers)


_NATIVE_UNIVERSE: frozenset[str] = _load_native_universe()

# ---------------------------------------------------------------------------
# Bar conversion helper
# ---------------------------------------------------------------------------

def _to_codex_bars(our_bars: list[Bar]) -> list[CodexBar]:
    """Convert our Bar instances to morning_quant_bot.Bar instances."""
    return [
        CodexBar(
            symbol=b.symbol,
            date=b.date,
            open=b.open,
            high=b.high,
            low=b.low,
            close=b.close,
            volume=b.volume,
        )
        for b in our_bars
    ]


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class _CodexAdapter(Strategy):
    """Internal base — subclasses define which symbols to include."""

    bot_id = ""
    description = ""

    def _filter_histories(
        self, histories: dict[str, list[Bar]], as_of: date
    ) -> dict[str, list[Bar]]:
        """Subclasses override to restrict the symbol set."""
        return {
            sym: [b for b in bars if b.date <= as_of]
            for sym, bars in histories.items()
        }

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        filtered = self._filter_histories(histories, as_of)
        if not filtered:
            return {}

        # Convert to codex Bar format, keyed by symbol
        codex_bars_by_sym: dict[str, list[CodexBar]] = {
            sym: _to_codex_bars(bars) for sym, bars in filtered.items() if bars
        }
        if not codex_bars_by_sym:
            return {}

        # Build date_indexes: {symbol: {date: index}}
        date_indexes = {
            sym: {bar.date: i for i, bar in enumerate(bars)}
            for sym, bars in codex_bars_by_sym.items()
        }

        weights, _reasons = codex_target_weights(
            codex_bars_by_sym,
            date_indexes,
            as_of,
            DEFAULT_STRATEGY,
        )
        return weights


# ---------------------------------------------------------------------------
# Registered variants
# ---------------------------------------------------------------------------

@register
class CodexBotR1000(_CodexAdapter):
    """Codex bot running on the full R1000 histories (excluding ^VIX)."""

    bot_id = "codex-r1000"
    description = "Codex bot adapter — R1000 universe variant"

    def _filter_histories(
        self, histories: dict[str, list[Bar]], as_of: date
    ) -> dict[str, list[Bar]]:
        return {
            sym: [b for b in bars if b.date <= as_of]
            for sym, bars in histories.items()
            if sym != "^VIX"
        }


@register
class CodexBotNative(_CodexAdapter):
    """Codex bot running on the 20 ETFs from the snapshot's universe.txt."""

    bot_id = "codex-native"
    description = "Codex bot adapter — native ETF universe variant"

    def _filter_histories(
        self, histories: dict[str, list[Bar]], as_of: date
    ) -> dict[str, list[Bar]]:
        return {
            sym: [b for b in bars if b.date <= as_of]
            for sym, bars in histories.items()
            if sym.upper() in _NATIVE_UNIVERSE
        }

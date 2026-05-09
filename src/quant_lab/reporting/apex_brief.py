"""Daily apex brief — actionable portfolio + buy/sell diffs for Discord.

This is the "what to actually do today" message, distilled from the lab's
meta-ensemble. It collapses 13 bots' opinions into one portfolio and
diffs against yesterday so the user sees just the trades.

Confidence levels are explicit and honest:
  - **PAPER ONLY** (< 30 days of data): pure noise, do not act.
  - **low confidence** (30-89 days): rolling Sharpes computable but huge
    confidence intervals. Signal is barely above noise.
  - **medium confidence** (90-179 days): first statistically meaningful
    signal. Lifecycle has begun pruning bad bots.
  - **live-calibrated** (180+ days): trustworthy enough to base a small
    real-money allocation on. Still not a guarantee — the past 6 months
    might not represent the next 6.

Format prioritises buy/sell deltas because that's what the user acts on.
The full target portfolio + leaderboard remain available on the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence


# Trades smaller than this fraction are suppressed — both because they're
# below realistic execution lot-size and because they're typically just
# weight-noise from the inverse-vol scaling rather than real conviction.
_MIN_TRADE_DELTA = 0.01  # 1% of NAV


@dataclass(frozen=True)
class ConfidenceTier:
    label: str
    emoji: str
    note: str


def _confidence_for(days_of_data: int) -> ConfidenceTier:
    if days_of_data < 30:
        return ConfidenceTier(
            label="PAPER ONLY",
            emoji="🚫",
            note=(
                f"Only {days_of_data} days of live data — these picks are NOISE. "
                "Do not act on them yet."
            ),
        )
    if days_of_data < 90:
        return ConfidenceTier(
            label="LOW CONFIDENCE",
            emoji="⚠️",
            note=(
                f"{days_of_data} days of live data. Rolling Sharpes are still "
                "noisy (CI ±1.5). Treat as directional guidance, not signal."
            ),
        )
    if days_of_data < 180:
        return ConfidenceTier(
            label="MEDIUM CONFIDENCE",
            emoji="🟡",
            note=(
                f"{days_of_data} days of live data. First statistically "
                "meaningful signal — lifecycle has begun pruning."
            ),
        )
    return ConfidenceTier(
        label="LIVE-CALIBRATED",
        emoji="✅",
        note=(
            f"{days_of_data} days of live data, full regime cycle covered. "
            "Suggestions are evidence-weighted from real performance."
        ),
    )


def _format_pct(weight: float) -> str:
    """Render a portfolio weight (0..1) as a percentage string."""
    return f"{weight * 100:.1f}%"


def diff_portfolios(
    target: dict[str, float],
    current: dict[str, float],
    *,
    min_delta: float = _MIN_TRADE_DELTA,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Return (holds, buys, sells) maps keyed by ticker.

    - holds: tickers in both with delta below `min_delta`
    - buys:  tickers where target - current >= +min_delta (value is the new target weight)
    - sells: tickers where current - target >= +min_delta (value is the OLD weight,
             since "what to sell" is "the position you currently have")
    """
    tickers = set(target) | set(current)
    holds: dict[str, float] = {}
    buys: dict[str, float] = {}
    sells: dict[str, float] = {}
    for sym in tickers:
        t = target.get(sym, 0.0)
        c = current.get(sym, 0.0)
        delta = t - c
        if delta >= min_delta:
            buys[sym] = t
        elif delta <= -min_delta:
            sells[sym] = c
        elif t > 0.005:
            # Genuinely held, no meaningful change — show as HOLD with target weight
            holds[sym] = t
    return holds, buys, sells


def build_brief(
    *,
    today: date,
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    days_of_data: int,
    market_snapshot: dict[str, dict[str, float]] | None = None,
    spy_benchmark_return: float | None = None,
    portfolio_return: float | None = None,
    dashboard_url: str | None = None,
) -> str:
    """Build the Discord-ready apex brief message.

    `current_weights` is what the meta-ensemble HELD as of yesterday's close
    (i.e. what the user already owns if they've been mirroring). `target_weights`
    is today's recommendation. The diff between them is the trade list.

    `days_of_data` is the number of NAV observations the meta-ensemble has
    accumulated — drives the confidence tier.
    """
    tier = _confidence_for(days_of_data)
    holds, buys, sells = diff_portfolios(target_weights, current_weights)

    lines: list[str] = []
    lines.append(f"**📊 QUANT LAB BRIEF — {today.isoformat()}**")
    lines.append(f"{tier.emoji} **{tier.label}** — {tier.note}")
    lines.append("")

    # Quick market context
    if market_snapshot:
        market_bits = []
        for sym in ("SPY", "QQQ"):
            info = market_snapshot.get(sym) or {}
            chg = info.get("change_pct", 0.0)
            arrow = "▲" if chg >= 0 else "▼"
            market_bits.append(f"{sym} {chg:+.2f}% {arrow}")
        if market_bits:
            lines.append(f"_Market: {' · '.join(market_bits)}_")
            lines.append("")

    # Portfolio P&L vs benchmark when available
    if portfolio_return is not None and spy_benchmark_return is not None:
        diff = portfolio_return - spy_benchmark_return
        verdict = "beat SPY" if diff > 0 else "lagged SPY"
        lines.append(
            f"_Yesterday: portfolio {portfolio_return:+.2%} vs SPY "
            f"{spy_benchmark_return:+.2%} → {verdict} by {abs(diff):.2%}_"
        )
        lines.append("")

    # The actionable diff — this is the whole point
    if not buys and not sells:
        lines.append("**No trades today** — hold current positions.")
    else:
        if buys:
            lines.append("**🟢 BUY (target weight after trade):**")
            for sym, w in sorted(buys.items(), key=lambda x: -x[1]):
                lines.append(f"  • {sym} → {_format_pct(w)}")
        if sells:
            lines.append("**🔴 SELL (current position to close/trim):**")
            for sym, w in sorted(sells.items(), key=lambda x: -x[1]):
                lines.append(f"  • {sym} (was {_format_pct(w)})")

    if holds:
        lines.append("")
        held_str = ", ".join(
            f"{sym} {_format_pct(w)}"
            for sym, w in sorted(holds.items(), key=lambda x: -x[1])[:8]
        )
        lines.append(f"**Holding:** {held_str}")

    if dashboard_url:
        lines.append("")
        lines.append(f"_Full breakdown: {dashboard_url}_")

    lines.append("")
    lines.append("_Research tool. Not financial advice._")

    return "\n".join(lines)

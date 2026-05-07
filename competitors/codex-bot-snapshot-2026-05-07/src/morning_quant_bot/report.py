from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from .models import Account, BacktestMetrics, Recommendation, StrategyParams


def build_report(
    report_date: date,
    account: Account,
    equity: float,
    strategy: StrategyParams,
    metrics: BacktestMetrics,
    recommendations: list[Recommendation],
    report_path: Path,
    data_start: date,
    data_end: date,
) -> str:
    lines: list[str] = []
    lines.append(f"# Morning Quant Report - {report_date.isoformat()}")
    lines.append("")
    lines.append("Simulation only. This report is generated from public daily data,")
    lines.append("walk-forward backtests, and explicit long-only rules. It is not a")
    lines.append("guarantee of future returns or a broker execution instruction.")
    lines.append("")
    lines.append("## Account")
    lines.append("")
    lines.append(f"- Cash: ${account.cash:,.2f}")
    lines.append(f"- Estimated equity: ${equity:,.2f}")
    lines.append(f"- Positions: {len(account.positions)}")
    lines.append("")
    lines.append("## Strategy Selected")
    lines.append("")
    lines.append(f"- Lookback: {strategy.lookback} trading days")
    lines.append(f"- Moving averages: {strategy.sma_fast}/{strategy.sma_slow}")
    lines.append(f"- Volatility lookback: {strategy.vol_lookback}")
    lines.append(f"- Max positions: {strategy.max_positions}")
    lines.append(f"- Cash buffer: {strategy.cash_buffer:.1%}")
    lines.append(f"- Max single-name weight: {strategy.max_weight:.1%}")
    lines.append(f"- Rebalance cadence: {strategy.rebalance_days} trading days")
    lines.append("")
    lines.append("## Walk-Forward Test")
    lines.append("")
    lines.append(f"- Data window: {data_start.isoformat()} to {data_end.isoformat()}")
    lines.append(f"- Test CAGR: {metrics.cagr:.1%}")
    lines.append(f"- Test Sharpe: {metrics.sharpe:.2f}")
    lines.append(f"- Test max drawdown: {metrics.max_drawdown:.1%}")
    lines.append(f"- Test total return: {metrics.total_return:.1%}")
    lines.append(f"- Simulated trades: {metrics.trades}")
    lines.append("")
    lines.append("## Morning Actions")
    lines.append("")
    lines.append("| Symbol | Action | Current Shares | Target Shares | Delta | Price | Target Weight | Reason |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for recommendation in recommendations:
        lines.append(recommendation.to_markdown_row())
    if not recommendations:
        lines.append("| CASH | HOLD | 0 | 0 | 0 | $0.00 | 100.0% | No candidates passed risk filters |")
    lines.append("")
    lines.append("## Operating Rule")
    lines.append("")
    lines.append("Use the paper account first. If real-money trading is ever considered,")
    lines.append("compare fills, taxes, slippage, liquidity, and your own risk limits before")
    lines.append("acting. Do not override risk controls because a single report looks strong.")
    lines.append("")
    lines.append(f"Generated at {datetime.now().replace(microsecond=0).isoformat()}.")
    text = "\n".join(lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    return text


from __future__ import annotations

import random

from .features import annualized_volatility, rsi, sma
from .models import Bar, StrategyParams


DEFAULT_STRATEGY = StrategyParams(
    lookback=126,
    sma_fast=40,
    sma_slow=180,
    vol_lookback=30,
    max_positions=5,
    min_momentum=0.02,
    max_symbol_vol=0.45,
    cash_buffer=0.08,
    max_weight=0.30,
    rebalance_days=10,
)


def required_history(params: StrategyParams) -> int:
    return max(params.lookback, params.sma_slow, params.vol_lookback, 14) + 1


def score_symbol(bars: list[Bar], index: int, params: StrategyParams) -> tuple[float, str] | None:
    closes = [bar.close for bar in bars]
    needed = required_history(params)
    if index < needed:
        return None

    close = closes[index]
    past = closes[index - params.lookback]
    momentum = close / past - 1.0 if past > 0 else 0.0
    fast = sma(closes, params.sma_fast, index)
    slow = sma(closes, params.sma_slow, index)
    if fast is None or slow is None or slow <= 0:
        return None

    trend = fast / slow - 1.0
    vol = annualized_volatility(closes, params.vol_lookback, index)
    current_rsi = rsi(closes, 14, index)
    if current_rsi is None:
        return None

    if momentum < params.min_momentum:
        return None
    if close < slow:
        return None
    if vol > params.max_symbol_vol:
        return None
    if current_rsi < 42 or current_rsi > 92:
        return None

    score = 0.60 * momentum + 0.30 * trend - 0.10 * vol
    reason = (
        f"mom {momentum:.1%}, trend {trend:.1%}, "
        f"vol {vol:.1%}, RSI {current_rsi:.0f}"
    )
    return score, reason


def target_weights(
    bars_by_symbol: dict[str, list[Bar]],
    date_indexes: dict[str, dict],
    current_date,
    params: StrategyParams,
) -> tuple[dict[str, float], dict[str, str]]:
    ranked: list[tuple[str, float, float, str]] = []
    for symbol, bars in bars_by_symbol.items():
        index = date_indexes[symbol].get(current_date)
        if index is None:
            continue
        scored = score_symbol(bars, index, params)
        if scored is None:
            continue
        score, reason = scored
        closes = [bar.close for bar in bars]
        vol = annualized_volatility(closes, params.vol_lookback, index)
        risk_unit = 1.0 / max(vol, 0.08)
        ranked.append((symbol, score, risk_unit, reason))

    ranked.sort(key=lambda item: item[1], reverse=True)
    selected = ranked[: params.max_positions]
    if not selected:
        return {}, {}

    total_risk_unit = sum(item[2] for item in selected)
    investable = max(0.0, min(1.0, 1.0 - params.cash_buffer))
    weights: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for symbol, _score, risk_unit, reason in selected:
        raw_weight = investable * risk_unit / total_risk_unit
        weights[symbol] = min(raw_weight, params.max_weight)
        reasons[symbol] = reason

    total_weight = sum(weights.values())
    if total_weight > investable and total_weight > 0:
        scale = investable / total_weight
        weights = {symbol: weight * scale for symbol, weight in weights.items()}
    return weights, reasons


def random_params(rng: random.Random) -> StrategyParams:
    sma_slow = rng.choice([120, 150, 180, 200, 220])
    sma_fast = rng.choice([20, 30, 40, 50, 60])
    if sma_fast >= sma_slow:
        sma_fast = max(15, sma_slow // 4)
    return StrategyParams(
        lookback=rng.choice([63, 84, 105, 126, 168, 189, 252]),
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        vol_lookback=rng.choice([20, 30, 45, 60]),
        max_positions=rng.choice([3, 4, 5, 6, 8]),
        min_momentum=rng.uniform(-0.01, 0.08),
        max_symbol_vol=rng.uniform(0.25, 0.65),
        cash_buffer=rng.uniform(0.02, 0.20),
        max_weight=rng.uniform(0.18, 0.40),
        rebalance_days=rng.choice([5, 10, 15, 20, 21]),
    )


def mutate_params(params: StrategyParams, rng: random.Random) -> StrategyParams:
    data = params.to_dict()
    field = rng.choice(list(data))
    if field in {"lookback", "sma_slow"}:
        data[field] = max(50, int(data[field] + rng.choice([-42, -21, 21, 42])))
    elif field == "sma_fast":
        data[field] = max(10, int(data[field] + rng.choice([-10, -5, 5, 10])))
    elif field == "vol_lookback":
        data[field] = max(10, int(data[field] + rng.choice([-15, -5, 5, 15])))
    elif field == "max_positions":
        data[field] = min(10, max(2, int(data[field] + rng.choice([-1, 1]))))
    elif field in {"min_momentum", "cash_buffer"}:
        data[field] = max(-0.05, min(0.25, float(data[field]) + rng.uniform(-0.02, 0.02)))
    elif field == "max_symbol_vol":
        data[field] = max(0.15, min(0.80, float(data[field]) + rng.uniform(-0.08, 0.08)))
    elif field == "max_weight":
        data[field] = max(0.10, min(0.50, float(data[field]) + rng.uniform(-0.04, 0.04)))
    elif field == "rebalance_days":
        data[field] = rng.choice([5, 10, 15, 20, 21])

    if data["sma_fast"] >= data["sma_slow"]:
        data["sma_fast"] = max(10, data["sma_slow"] // 4)
    return StrategyParams.from_dict(data)

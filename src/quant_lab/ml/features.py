"""Point-in-time-safe feature engineering for ML strategies.

All features are computed strictly using data with bar.date <= as_of.
No future data leaks into feature computation.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from ..types import Bar

_INDEX_PROXIES = {"SPY", "QQQ", "^VIX"}


# ---------------------------------------------------------------------------
# Low-level helper functions
# ---------------------------------------------------------------------------


def _closes(bars: list[Bar]) -> np.ndarray:
    return np.array([b.close for b in bars], dtype=float)


def _volumes(bars: list[Bar]) -> np.ndarray:
    return np.array([b.volume for b in bars], dtype=float)


def _dollar_volumes(bars: list[Bar]) -> np.ndarray:
    return np.array([b.close * b.volume for b in bars], dtype=float)


def _rsi_wilder(closes: np.ndarray, period: int = 14) -> float:
    """Compute Wilder's RSI using exponential smoothing."""
    if len(closes) < period + 1:
        return float("nan")
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    # Seed with simple average
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    # Apply Wilder smoothing on the rest
    for g, loss in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average with pandas-style span."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line) as scalars (last values)."""
    if len(closes) < slow + signal:
        return float("nan"), float("nan")
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    return float(macd_line[-1]), float(signal_line[-1])


def _atr(bars: list[Bar], period: int = 14) -> float:
    """Average True Range over the last `period` bars."""
    if len(bars) < period + 1:
        return float("nan")
    recent = bars[-(period + 1):]
    trs = []
    for i in range(1, len(recent)):
        hi = recent[i].high
        lo = recent[i].low
        prev_close = recent[i - 1].close
        tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        trs.append(tr)
    return float(np.mean(trs[-period:]))


def _obv_change(bars: list[Bar], window: int = 20) -> float:
    """OBV change over the last `window` bars, normalized by initial OBV."""
    if len(bars) < window + 1:
        return float("nan")
    recent = bars[-(window + 1):]
    obv = 0.0
    obv_series = [0.0]
    for i in range(1, len(recent)):
        if recent[i].close > recent[i - 1].close:
            obv += recent[i].volume
        elif recent[i].close < recent[i - 1].close:
            obv -= recent[i].volume
        obv_series.append(obv)
    start_obv = obv_series[0]
    end_obv = obv_series[-1]
    if abs(start_obv) < 1:
        return end_obv / max(1.0, recent[0].volume)
    return (end_obv - start_obv) / abs(start_obv)


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------


def _compute_single_symbol_features(
    bars: list[Bar],
    spy_bars: list[Bar],
    as_of: date,
) -> Optional[dict[str, float]]:
    """Compute ~30 features for a single symbol, strictly using data <= as_of."""
    eligible = [b for b in bars if b.date <= as_of]
    if len(eligible) < 252:
        return None

    closes = _closes(eligible)
    vols = _volumes(eligible)
    dvols = _dollar_volumes(eligible)
    price = closes[-1]

    features: dict[str, float] = {}

    # -- Return features --
    for n, label in [(1, "ret_1d"), (5, "ret_5d"), (20, "ret_20d"), (60, "ret_60d"), (120, "ret_120d"), (252, "ret_252d")]:
        if len(closes) > n:
            features[label] = float(closes[-1] / closes[-(n + 1)] - 1.0)
        else:
            features[label] = float("nan")

    # -- Volatility features --
    for n, label in [(20, "vol_20d"), (60, "vol_60d")]:
        if len(closes) > n + 1:
            rets = np.diff(np.log(closes[-(n + 1):]))
            features[label] = float(np.std(rets) * math.sqrt(252))
        else:
            features[label] = float("nan")

    # vol of vol (rolling 20-day vol computed each day)
    if len(closes) >= 80:
        vols_series = []
        for i in range(60, len(closes)):
            window_rets = np.diff(np.log(closes[i - 20:i + 1]))
            vols_series.append(float(np.std(window_rets) * math.sqrt(252)))
        features["vol_of_vol"] = float(np.std(vols_series[-20:])) if len(vols_series) >= 20 else float("nan")
    else:
        features["vol_of_vol"] = float("nan")

    # -- Z-scores: (price - SMA_n) / std_n --
    for n, label in [(20, "zscore_20"), (60, "zscore_60"), (200, "zscore_200")]:
        if len(closes) >= n:
            sma = float(np.mean(closes[-n:]))
            std = float(np.std(closes[-n:]))
            features[label] = float((price - sma) / std) if std > 0 else 0.0
        else:
            features[label] = float("nan")

    # -- Distance to SMA in vol units --
    for n, label in [(20, "dist_sma_20_vol"), (60, "dist_sma_60_vol")]:
        if len(closes) > n + 1:
            sma = float(np.mean(closes[-n:]))
            rets = np.diff(np.log(closes[-(n + 1):]))
            vol = float(np.std(rets) * math.sqrt(252))
            features[label] = float((price - sma) / sma / max(vol, 1e-8))
        else:
            features[label] = float("nan")

    # -- Volume features --
    if len(vols) >= 21 and len(vols) > 1:
        avg_vol_20 = float(np.mean(vols[-21:-1]))
        features["rel_volume"] = float(vols[-1] / avg_vol_20) if avg_vol_20 > 0 else 1.0
    else:
        features["rel_volume"] = float("nan")

    features["obv_change"] = _obv_change(eligible, window=20)

    # log ADV (dollar volume)
    if len(dvols) >= 20:
        adv = float(np.mean(dvols[-20:]))
        features["log_adv"] = math.log(adv) if adv > 0 else float("nan")
    else:
        features["log_adv"] = float("nan")

    # -- RSI(14) --
    features["rsi_14"] = _rsi_wilder(closes, period=14)

    # -- MACD --
    macd_val, macd_sig = _macd(closes)
    features["macd_line"] = macd_val
    features["macd_signal"] = macd_sig

    # -- Bollinger band z-score (20-day, 2 std) --
    if len(closes) >= 20:
        bb_mean = float(np.mean(closes[-20:]))
        bb_std = float(np.std(closes[-20:]))
        features["bb_zscore"] = float((price - bb_mean) / bb_std) if bb_std > 0 else 0.0
    else:
        features["bb_zscore"] = float("nan")

    # -- ATR --
    features["atr"] = _atr(eligible, period=14)
    # ATR as % of price
    if not math.isnan(features["atr"]) and price > 0:
        features["atr_pct"] = features["atr"] / price
    else:
        features["atr_pct"] = float("nan")

    # -- Sector-relative (subtract SPY's returns) --
    spy_eligible = [b for b in spy_bars if b.date <= as_of] if spy_bars else []
    spy_closes = _closes(spy_eligible) if len(spy_eligible) >= 252 else None

    for n, label in [(5, "rel_ret_5d"), (20, "rel_ret_20d")]:
        sym_ret = features.get(f"ret_{n}d", float("nan"))
        if spy_closes is not None and len(spy_closes) > n and not math.isnan(sym_ret):
            spy_ret = float(spy_closes[-1] / spy_closes[-(n + 1)] - 1.0)
            features[label] = sym_ret - spy_ret
        else:
            features[label] = float("nan")

    return features


def compute_features(
    histories: dict[str, list[Bar]],
    target_symbols: list[str],
    as_of: date,
    horizon: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """For each target symbol, compute ~30 features as of `as_of`.

    The label is the `horizon`-day forward return. NaN if insufficient future data.
    Features use ONLY data with bar.date <= as_of (point-in-time safe).

    Returns (X_df, y_series) indexed by symbol.
    """
    spy_bars = histories.get("SPY", [])
    rows: dict[str, dict[str, float]] = {}
    labels: dict[str, float] = {}

    for symbol in target_symbols:
        if symbol in _INDEX_PROXIES:
            continue
        bars = histories.get(symbol, [])
        if not bars:
            continue

        feats = _compute_single_symbol_features(bars, spy_bars, as_of)
        if feats is None:
            continue

        # Compute forward label: horizon-day return using data AFTER as_of
        eligible_all = [b for b in bars if b.date <= as_of]
        future = [b for b in bars if b.date > as_of]
        if len(future) >= horizon:
            fwd_return = float(future[horizon - 1].close / eligible_all[-1].close - 1.0)
        else:
            fwd_return = float("nan")

        rows[symbol] = feats
        labels[symbol] = fwd_return

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=float)

    X_df = pd.DataFrame.from_dict(rows, orient="index")
    y_series = pd.Series(labels, name="fwd_return")
    return X_df, y_series


def build_training_set(
    histories: dict[str, list[Bar]],
    target_symbols: list[str],
    train_start: date,
    train_end: date,
    horizon: int = 5,
    sample_every_days: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """For each (symbol, sampled_date) in [train_start, train_end), compute features + label.

    Concatenates into a single (X, y) for training.
    Strict point-in-time: features use only data <= sample_date.
    """
    # Collect all relevant dates from any symbol's bars
    all_dates: list[date] = sorted(
        {b.date for bars in histories.values() for b in bars if train_start <= b.date < train_end}
    )
    # Sample every N days
    sampled = [d for i, d in enumerate(all_dates) if i % sample_every_days == 0]

    all_X: list[pd.DataFrame] = []
    all_y: list[pd.Series] = []

    for sample_date in sampled:
        X_df, y_series = compute_features(
            histories=histories,
            target_symbols=target_symbols,
            as_of=sample_date,
            horizon=horizon,
        )
        if X_df.empty:
            continue
        # Add date to index for disambiguation
        X_df = X_df.copy()
        X_df.index = pd.MultiIndex.from_tuples(
            [(sym, sample_date) for sym in X_df.index],
            names=["symbol", "date"],
        )
        y_series.index = pd.MultiIndex.from_tuples(
            [(sym, sample_date) for sym in y_series.index],
            names=["symbol", "date"],
        )
        all_X.append(X_df)
        all_y.append(y_series)

    if not all_X:
        return pd.DataFrame(), pd.Series(dtype=float)

    X_all = pd.concat(all_X)
    y_all = pd.concat(all_y)

    # Drop rows where label is NaN (no future data available)
    valid_mask = y_all.notna()
    return X_all[valid_mask], y_all[valid_mask]

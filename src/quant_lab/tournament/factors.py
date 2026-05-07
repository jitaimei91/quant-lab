"""3-factor decomposition for live tournament strategies.

Factor proxies (all from yfinance via histories):
    MKT   = SPY daily returns
    SIZE  = IWM - SPY  (small-cap minus large-cap)
    VALUE = VTV - VUG  (value minus growth)

OLS regression is pure-Python (no scipy/statsmodels dependency).
"""
from __future__ import annotations

import statistics
from math import sqrt

from ..types import Bar


def compute_factor_loadings(
    strategy_returns: list[float],
    factor_returns: dict[str, list[float]],
) -> dict[str, float]:
    """Multivariate OLS regression of strategy on 3 factors.

    Parameters
    ----------
    strategy_returns:
        Aligned daily return series for the strategy.
    factor_returns:
        Dict with keys "MKT", "SIZE", "VALUE", each a list of daily returns
        of the same length as strategy_returns.

    Returns
    -------
    Dict with keys alpha_per_day, beta_mkt, beta_size, beta_value, r_squared.
    All zeros if insufficient data.
    """
    n = len(strategy_returns)
    for key in ("MKT", "SIZE", "VALUE"):
        n = min(n, len(factor_returns.get(key, [])))

    if n < 30:
        return {
            "alpha_per_day": 0.0,
            "beta_mkt": 0.0,
            "beta_size": 0.0,
            "beta_value": 0.0,
            "r_squared": 0.0,
        }

    y = strategy_returns[:n]
    mkt = factor_returns["MKT"][:n]
    size = factor_returns["SIZE"][:n]
    value = factor_returns["VALUE"][:n]

    # Build design matrix columns: constant + 3 factors
    # X shape: n x 4  (intercept, MKT, SIZE, VALUE)
    X: list[list[float]] = [[1.0, mkt[i], size[i], value[i]] for i in range(n)]

    # Normal equations: beta = (X'X)^{-1} X'y
    # X is n×4; compute X'X (4×4) and X'y (4×1)
    p = 4
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for i in range(n):
        row = X[i]
        for a in range(p):
            Xty[a] += row[a] * y[i]
            for b in range(p):
                XtX[a][b] += row[a] * row[b]

    coeffs = _solve_4x4(XtX, Xty)
    if coeffs is None:
        return {
            "alpha_per_day": 0.0,
            "beta_mkt": 0.0,
            "beta_size": 0.0,
            "beta_value": 0.0,
            "r_squared": 0.0,
        }

    alpha, beta_mkt, beta_size, beta_value = coeffs

    # R-squared
    y_mean = statistics.mean(y)
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    y_hat = [alpha + beta_mkt * mkt[i] + beta_size * size[i] + beta_value * value[i] for i in range(n)]
    ss_res = sum((y[i] - y_hat[i]) ** 2 for i in range(n))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "alpha_per_day": alpha,
        "beta_mkt": beta_mkt,
        "beta_size": beta_size,
        "beta_value": beta_value,
        "r_squared": max(0.0, r_squared),
    }


def factor_proxies_from_histories(histories: dict[str, list[Bar]]) -> dict[str, list[float]]:
    """Build aligned factor return series from yfinance proxy symbols.

    MKT   = SPY daily returns
    SIZE  = IWM - SPY (each day)
    VALUE = VTV - VUG (each day)

    Returns an empty dict if any required proxy is missing.
    """
    required = ("SPY", "IWM", "VTV", "VUG")
    for sym in required:
        if sym not in histories or len(histories[sym]) < 2:
            return {}

    def daily_rets(sym: str) -> list[tuple]:
        bars = histories[sym]
        out = []
        for i in range(1, len(bars)):
            prev = bars[i - 1].close
            if prev > 0:
                out.append((bars[i].date, bars[i].close / prev - 1.0))
        return out

    spy_rets = {d: r for d, r in daily_rets("SPY")}
    iwm_rets = {d: r for d, r in daily_rets("IWM")}
    vtv_rets = {d: r for d, r in daily_rets("VTV")}
    vug_rets = {d: r for d, r in daily_rets("VUG")}

    # Intersect dates
    common_dates = sorted(
        set(spy_rets) & set(iwm_rets) & set(vtv_rets) & set(vug_rets)
    )
    if not common_dates:
        return {}

    mkt_series = [spy_rets[d] for d in common_dates]
    size_series = [iwm_rets[d] - spy_rets[d] for d in common_dates]
    value_series = [vtv_rets[d] - vug_rets[d] for d in common_dates]

    return {"MKT": mkt_series, "SIZE": size_series, "VALUE": value_series}


# ---------------------------------------------------------------------------
# Internal: pure-Python 4x4 Gaussian elimination with partial pivoting
# ---------------------------------------------------------------------------

def _solve_4x4(A: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve Ax = b for 4x4 system using Gaussian elimination.

    Returns solution list or None if singular.
    """
    n = len(b)
    # Augmented matrix [A | b]
    aug = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        # Partial pivot
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[max_row][col]) < 1e-14:
            return None
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]

    # Back-substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(aug[i][i]) < 1e-14:
            return None
        x[i] = aug[i][n]
        for j in range(i + 1, n):
            x[i] -= aug[i][j] * x[j]
        x[i] /= aug[i][i]

    return x

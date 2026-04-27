"""
Delta-hedge backtest engine — Week 4.

Strategy: buy a call option, delta-hedge daily using the spot, track P&L.
P&L breakdown: theta decay vs. gamma scalping (realized vs. implied vol spread).

Usage (once implemented):
    from backtest.delta_hedge import run_backtest
    results = run_backtest("SPY", "2024-01-05", strike=480, expiry="2024-02-16")
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm


# ── Black-Scholes Greeks (self-contained, no circular import) ─────────────────

def _bs_delta(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if (opt == "call" and S > K) else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if opt == "call" else norm.cdf(d1) - 1.0


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if opt == "call" else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ── Backtest engine ────────────────────────────────────────────────────────────

def run_backtest(
    price_series: pd.Series,
    K: float,
    T0: float,
    sigma_implied: float,
    r: float = 0.05,
    opt: str = "call",
    hedge_freq: int = 1,
) -> pd.DataFrame:
    """
    Run a delta-hedge backtest on a pre-loaded price series.

    Parameters
    ----------
    price_series    : pd.Series of daily closing prices, indexed by date
    K               : option strike
    T0              : initial time to expiry in years
    sigma_implied   : implied vol used to compute delta (constant for now)
    r               : risk-free rate
    opt             : 'call' or 'put'
    hedge_freq      : rebalance every N days (default daily)

    Returns
    -------
    pd.DataFrame with columns: date, S, delta, option_pnl, hedge_pnl, total_pnl
    """
    prices = price_series.values
    dates = price_series.index
    n = len(prices)
    dt = 1 / 252.0

    records = []
    delta_prev = 0.0
    option_value_prev = _bs_price(prices[0], K, T0, r, sigma_implied, opt)
    hedge_cash = 0.0

    for i in range(n):
        T = max(T0 - i * dt, 1e-6)
        S = prices[i]
        option_value = _bs_price(S, K, T, r, sigma_implied, opt)

        if i % hedge_freq == 0:
            delta = _bs_delta(S, K, T, r, sigma_implied, opt)
            hedge_cash += (delta - delta_prev) * S  # cash spent on hedge
            delta_prev = delta

        option_pnl = option_value - option_value_prev if i > 0 else 0.0
        hedge_pnl = delta_prev * (S - prices[i - 1]) if i > 0 else 0.0

        records.append(
            {
                "date": dates[i],
                "spot": S,
                "T": T,
                "delta": delta_prev,
                "option_value": option_value,
                "option_pnl": option_pnl,
                "hedge_pnl": hedge_pnl,
                "total_pnl": option_pnl + hedge_pnl,
            }
        )
        option_value_prev = option_value

    result = pd.DataFrame(records).set_index("date")
    result["cumulative_pnl"] = result["total_pnl"].cumsum()
    return result


def summary_stats(results: pd.DataFrame) -> dict:
    pnl = results["total_pnl"]
    return {
        "total_pnl": pnl.sum(),
        "sharpe": pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0.0,
        "max_drawdown": (results["cumulative_pnl"].cummax() - results["cumulative_pnl"]).max(),
        "win_rate": (pnl > 0).mean(),
        "n_days": len(results),
    }

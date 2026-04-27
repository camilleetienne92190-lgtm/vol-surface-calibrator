# Vol Surface Calibrator — Delta Hedge Engine v2
"""
Delta-hedge backtest engine — Week 4.

Strategy: SHORT one ATM call, delta-hedge daily with the underlying.
P&L logic:
  option_pnl  = V(t-1) - V(t)        (short call profits when vol declines / time passes)
  hedge_pnl   = delta(t-1) * (S(t) - S(t-1))  (long delta hedge)
  daily_pnl   = option_pnl + hedge_pnl
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm


# ── Black-Scholes helpers (self-contained, no circular import) ────────────────

def _bs_price(S: float, K: float, T: float, r: float, sigma: float, opt: str = "call") -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if opt == "call" else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_delta(S: float, K: float, T: float, r: float, sigma: float, opt: str = "call") -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if (opt == "call" and S > K) else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if opt == "call" else norm.cdf(d1) - 1.0


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest(
    ticker: str = "SPY",
    start_date: str | None = None,
    expiry_days: int = 30,
    sigma: float | None = None,
    r: float = 0.05,
) -> dict:
    """
    Simulate shorting an ATM call and delta-hedging daily.

    Parameters
    ----------
    ticker       : underlying ticker (fetched via yfinance)
    start_date   : YYYY-MM-DD string; defaults to 1 year ago
    expiry_days  : option lifetime in calendar days
    sigma        : fixed implied vol; if None, use realised vol of first 5 days
    r            : risk-free rate (continuous)

    Returns
    -------
    dict with keys:
        pnl_series       pd.Series  cumulative P&L indexed by date
        daily_pnl        pd.Series  daily P&L
        delta_series     pd.Series  delta (hedge ratio) each day
        option_values    pd.Series  BS option value each day
        price_series     pd.Series  underlying closing prices
        strike           float      option strike used
        stats            dict       summary statistics
    """
    import yfinance as yf

    # ── Date setup ────────────────────────────────────────────────────────────
    if start_date is None:
        t0 = datetime.today() - timedelta(days=365)
    else:
        t0 = datetime.strptime(start_date, "%Y-%m-%d")

    # Fetch enough history: expiry_days of data + a small buffer
    end_dt = t0 + timedelta(days=expiry_days + 10)
    if end_dt > datetime.today():
        end_dt = datetime.today()

    raw = yf.Ticker(ticker).history(
        start=t0.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
    )
    if raw.empty:
        raise RuntimeError(f"No historical data for {ticker} between {t0.date()} and {end_dt.date()}")

    prices = raw["Close"].dropna()
    if len(prices) < 5:
        raise ValueError(f"Only {len(prices)} trading days in range — need at least 5.")

    # ── Strike: ATM rounded to nearest 5 ─────────────────────────────────────
    spot_0 = float(prices.iloc[0])
    K = round(spot_0 / 5) * 5

    # ── Sigma: realised vol from first 5 days if not provided ────────────────
    if sigma is None:
        log_rets = np.log(prices.iloc[1:6] / prices.iloc[0:5].values)
        sigma = max(float(log_rets.std() * np.sqrt(252)), 0.001)
    implied_vol_entry = sigma

    # ── Initial option value (collect premium — SHORT the call) ──────────────
    T0 = expiry_days / 365.0
    initial_premium = _bs_price(spot_0, K, T0, r, sigma, "call")

    # ── Daily simulation ──────────────────────────────────────────────────────
    dt = 1.0 / 252.0
    n = len(prices)

    dates         = prices.index
    spot_arr      = prices.values
    opt_vals      = np.zeros(n)
    delta_arr     = np.zeros(n)
    daily_pnl_arr = np.zeros(n)

    for i in range(n):
        T_rem = max(T0 - i * dt, 1e-6)
        S = spot_arr[i]
        opt_vals[i] = _bs_price(S, K, T_rem, r, sigma, "call")
        delta_arr[i] = _bs_delta(S, K, T_rem, r, sigma, "call")

        if i > 0:
            # Short call P&L: we sold the call, so we profit when value falls
            option_pnl = opt_vals[i - 1] - opt_vals[i]
            # Delta hedge P&L: we hold delta shares of the underlying (long)
            hedge_pnl = delta_arr[i - 1] * (S - spot_arr[i - 1])
            daily_pnl_arr[i] = option_pnl + hedge_pnl

    daily_pnl   = pd.Series(daily_pnl_arr, index=dates)
    pnl_series  = daily_pnl.cumsum()
    delta_series = pd.Series(delta_arr, index=dates)
    option_values = pd.Series(opt_vals, index=dates)

    # ── Stats ─────────────────────────────────────────────────────────────────
    pnl_nonzero = daily_pnl.iloc[1:]  # day 0 is always 0
    realised_vol = float(
        np.log(prices / prices.shift(1)).dropna().std() * np.sqrt(252)
    )
    total_pnl    = float(pnl_series.iloc[-1])
    sharpe       = (
        float(pnl_nonzero.mean() / pnl_nonzero.std() * np.sqrt(252))
        if pnl_nonzero.std() > 0 else 0.0
    )
    running_max  = pnl_series.cummax()
    max_drawdown = float((running_max - pnl_series).max())
    win_rate     = float((pnl_nonzero > 0).mean())

    stats = {
        "total_pnl":             round(total_pnl, 4),
        "sharpe_ratio":          round(sharpe, 4),
        "max_drawdown":          round(max_drawdown, 4),
        "win_rate":              round(win_rate, 4),
        "nb_rebalances":         n - 1,
        "initial_option_premium": round(initial_premium, 4),
        "realised_vol":          round(realised_vol, 4),
        "implied_vol_entry":     round(implied_vol_entry, 4),
    }

    return {
        "pnl_series":    pnl_series,
        "daily_pnl":     daily_pnl,
        "delta_series":  delta_series,
        "option_values": option_values,
        "price_series":  prices,
        "strike":        K,
        "stats":         stats,
    }

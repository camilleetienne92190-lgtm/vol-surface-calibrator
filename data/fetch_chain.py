from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
import yfinance as yf

RF_RATE = 0.05


# ── Black-Scholes helpers ──────────────────────────────────────────────────────

def _bs_price(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if opt == "call" else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


# ── IV solver: Newton-Raphson with Brentq fallback ────────────────────────────

def implied_vol(
    S: float, K: float, T: float, r: float, market_price: float, opt: str
) -> float | None:
    intrinsic = max(S - K, 0.0) if opt == "call" else max(K - S, 0.0)
    # Reject if price is below intrinsic or time is negligible
    if market_price <= intrinsic + 1e-6 or T <= 1e-6:
        return None

    # Newton-Raphson
    sigma = 0.3
    for _ in range(50):
        price = _bs_price(S, K, T, r, sigma, opt)
        vega = _bs_vega(S, K, T, r, sigma)
        if abs(vega) < 1e-10:
            break
        sigma -= (price - market_price) / vega
        sigma = max(1e-4, min(sigma, 5.0))
        if abs(_bs_price(S, K, T, r, sigma, opt) - market_price) < 1e-8:
            return sigma

    # Brentq fallback
    try:
        f = lambda s: _bs_price(S, K, T, r, s, opt) - market_price
        lo, hi = 1e-4, 5.0
        if f(lo) * f(hi) > 0:
            return None
        sigma = brentq(f, lo, hi, xtol=1e-8, maxiter=200)
        return sigma if 1e-4 < sigma < 5.0 else None
    except Exception:
        return None


# ── Main fetch ────────────────────────────────────────────────────────────────

def fetch_option_chain(ticker: str = "SPY", r: float = RF_RATE) -> pd.DataFrame:
    sep = "-" * 55
    print(f"\n{sep}")
    print(f"  fetch_chain  |  ticker={ticker}")
    print(sep)

    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1d")
        if hist.empty:
            raise ValueError(f"No price history returned for {ticker}")
        S = float(hist["Close"].iloc[-1])
        print(f"  Spot price          : {S:.2f}")
    except Exception as exc:
        print(f"  [ERROR] Cannot fetch spot price: {exc}")
        sys.exit(1)

    try:
        expirations = tk.options
    except Exception as exc:
        print(f"  [ERROR] Cannot fetch expirations: {exc}")
        sys.exit(1)

    print(f"  Expirations found   : {len(expirations)}")

    today = pd.Timestamp.today().normalize()
    rows: list[dict] = []
    skipped_iv = 0

    for expiry in expirations:
        expiry_date = pd.Timestamp(expiry)
        dte = (expiry_date - today).days
        T = dte / 365.0
        if dte <= 0:
            continue

        try:
            chain = tk.option_chain(expiry)
        except Exception as exc:
            print(f"  [WARN] {expiry}: {exc}")
            continue

        for opt_type, raw_df in (("call", chain.calls), ("put", chain.puts)):
            filtered = raw_df[
                (raw_df["volume"] > 0)
                & (raw_df["openInterest"] > 0)
                & (raw_df["bid"] > 0)
            ].copy()

            for _, row in filtered.iterrows():
                K = float(row["strike"])
                mid = (float(row["bid"]) + float(row["ask"])) / 2.0

                iv = implied_vol(S, K, T, r, mid, opt_type)
                if iv is None:
                    skipped_iv += 1
                    continue

                rows.append(
                    {
                        "strike": K,
                        "expiry": expiry,
                        "days_to_expiry": dte,
                        "option_type": opt_type,
                        "market_price": round(mid, 4),
                        "implied_vol": round(iv, 6),
                        "moneyness": round(np.log(K / S), 6),
                    }
                )

    df = pd.DataFrame(rows)
    print(f"  Options kept        : {len(df)}")
    print(f"  IV failures skipped : {skipped_iv}")
    print(f"{sep}\n")
    return df


# ── CSV helper ────────────────────────────────────────────────────────────────

def save_to_csv(df: pd.DataFrame, filename: str) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved -> {path.resolve()}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    df = fetch_option_chain(ticker)

    if df.empty:
        print("[WARNING] DataFrame is empty — nothing to save.")
        sys.exit(0)

    print(df.head(10).to_string(index=False))

    out = Path(__file__).parent / f"{ticker}_chain.csv"
    save_to_csv(df, str(out))

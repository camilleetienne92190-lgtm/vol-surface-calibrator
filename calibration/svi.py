"""
SVI (Stochastic Volatility Inspired) calibration — Gatheral (2004).

Raw SVI parametrisation:
    w(k) = a + b * (rho*(k - m) + sqrt((k - m)^2 + sigma^2))

where k = ln(K/S)  is log-moneyness,
      w = IV^2 * T is total implied variance.

No-arbitrage necessary conditions:
    b >= 0
    |rho| < 1
    sigma > 0
    a + b*sigma*sqrt(1 - rho^2) >= 0   (minimum total variance non-negative)
"""

from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize

# ── Dark theme ─────────────────────────────────────────────────────────────────

_DARK = {
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.edgecolor": "#444444",
    "grid.color": "#2a2a2a",
    "grid.alpha": 1.0,
    "axes.titlecolor": "white",
    "legend.facecolor": "#1a1d25",
    "legend.edgecolor": "#444444",
}

def _apply_dark() -> None:
    for k, v in _DARK.items():
        plt.rcParams[k] = v


# ── SVI core maths ─────────────────────────────────────────────────────────────

@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_var(self, k: np.ndarray) -> np.ndarray:
        km = k - self.m
        D = np.sqrt(km**2 + self.sigma**2)
        return self.a + self.b * (self.rho * km + D)

    def implied_vol(self, k: np.ndarray, T: float) -> np.ndarray:
        w = self.total_var(k)
        return np.sqrt(np.maximum(w, 1e-10) / T)

    def derivatives(self, k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (w, w', w'') using analytic SVI derivatives."""
        km = k - self.m
        D = np.sqrt(km**2 + self.sigma**2)
        w  = self.a + self.b * (self.rho * km + D)
        wp = self.b * (self.rho + km / D)                           # dw/dk
        wpp = self.b * self.sigma**2 / D**3                         # d²w/dk²
        return w, wp, wpp


# ── Calibration ────────────────────────────────────────────────────────────────

# Bounds: a, b, rho, m, sigma
_BOUNDS = [
    (-0.5, 2.0),         # a
    (1e-4, 4.0),         # b  ≥ 0
    (-0.9999, 0.9999),   # rho
    (-3.0, 3.0),         # m
    (1e-4, 3.0),         # sigma > 0
]


def _objective(raw: np.ndarray, k: np.ndarray, iv_mkt: np.ndarray, T: float) -> float:
    a, b, rho, m, sigma = raw
    # Penalty: min-variance constraint a + b*sigma*sqrt(1-rho^2) >= 0
    min_var = a + b * sigma * np.sqrt(max(1 - rho**2, 0.0))
    penalty = 1e5 * min(min_var, 0.0)**2

    km = k - m
    D = np.sqrt(km**2 + sigma**2)
    w = a + b * (rho * km + D)

    # Penalty: negative total variance
    neg_mask = w <= 0
    if neg_mask.any():
        penalty += 1e5 * np.sum(np.minimum(w[neg_mask], 0.0)**2)
        w = np.where(w <= 0, 1e-10, w)

    iv_fit = np.sqrt(w / T)
    mse = float(np.mean((iv_fit - iv_mkt) ** 2))
    return mse + penalty


def calibrate_slice(
    k: np.ndarray, iv_mkt: np.ndarray, T: float
) -> tuple[SVIParams, float]:
    """
    Fit SVI params to a single maturity slice.
    Strategy: differential_evolution (global) → L-BFGS-B polish.
    Fallback to Nelder-Mead if either stage fails.

    Returns (params, rmse_vol_pct)
    """
    args = (k, iv_mkt, T)

    # Stage 1 — global search
    try:
        de = differential_evolution(
            _objective, _BOUNDS, args=args,
            maxiter=2000, tol=1e-12, seed=42,
            mutation=(0.5, 1.5), recombination=0.9,
            popsize=20, polish=False,
            workers=1,
        )
        x0 = de.x
    except Exception:
        x0 = [0.04, 0.2, -0.5, 0.0, 0.3]

    # Stage 2 — local polish with L-BFGS-B
    try:
        res = minimize(
            _objective, x0, args=args,
            method="L-BFGS-B",
            bounds=_BOUNDS,
            options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10},
        )
        x_best = res.x if res.success or res.fun < _objective(x0, *args) else x0
    except Exception:
        x_best = x0

    # Nelder-Mead fallback if still poor
    if _objective(x_best, *args) > 0.01:
        try:
            nm = minimize(
                _objective, x_best, args=args,
                method="Nelder-Mead",
                options={"maxiter": 10000, "xatol": 1e-10, "fatol": 1e-12},
            )
            if nm.fun < _objective(x_best, *args):
                x_best = nm.x
        except Exception:
            pass

    a, b, rho, m, sigma = x_best
    params = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)

    iv_fit = params.implied_vol(k, T)
    rmse = float(np.sqrt(np.mean((iv_fit - iv_mkt) ** 2))) * 100.0  # vol points %
    return params, rmse


def fit_all_slices(
    df: pd.DataFrame, min_points: int = 5
) -> dict[int, tuple[SVIParams, float]]:
    """
    Fit one SVI slice per maturity. Returns {dte: (SVIParams, rmse_pct)}.
    Slices with < min_points are skipped with a warning.
    """
    results: dict[int, tuple[SVIParams, float]] = {}
    for dte in sorted(df["days_to_expiry"].unique()):
        sl = df[df["days_to_expiry"] == dte]
        if len(sl) < min_points:
            print(f"  [SKIP] DTE={dte:4d} — only {len(sl)} point(s), need >= {min_points}")
            continue
        T = dte / 365.0
        k = sl["moneyness"].values
        iv = sl["implied_vol"].values
        params, rmse = calibrate_slice(k, iv, T)
        results[dte] = (params, rmse)
    return results


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_svi_fits(
    df: pd.DataFrame,
    fits: dict[int, tuple[SVIParams, float]],
    output_path: Path,
) -> None:
    _apply_dark()
    fig, ax = plt.subplots(figsize=(14, 7))

    cmap = plt.cm.plasma
    dtes = sorted(fits.keys())
    n = max(len(dtes) - 1, 1)

    for i, dte in enumerate(dtes):
        params, rmse = fits[dte]
        T = dte / 365.0
        color = cmap(i / n)

        # Raw scatter
        sl = df[df["days_to_expiry"] == dte]
        ax.scatter(
            sl["moneyness"], sl["implied_vol"] * 100,
            color=color, s=30, zorder=5, alpha=0.9,
        )

        # Dense fitted curve over the actual data range + 10% buffer
        k_lo = sl["moneyness"].min()
        k_hi = sl["moneyness"].max()
        buf = (k_hi - k_lo) * 0.10
        k_grid = np.linspace(k_lo - buf, k_hi + buf, 400)

        iv_fit = params.implied_vol(k_grid, T) * 100
        ax.plot(
            k_grid, iv_fit,
            color=color, linewidth=1.6, alpha=0.85,
            label=f"{dte}d  (RMSE {rmse:.2f} vp)",
        )

    ax.axvline(0, color="#FFD700", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Moneyness  ln(K/S)", color="white")
    ax.set_ylabel("Implied Vol (%)", color="white")
    ax.set_title("SVI Calibration — Raw IV vs Fitted Curve per Maturity", color="white", fontsize=13)
    ax.grid(True, color="#2a2a2a", linewidth=0.7)
    ax.legend(fontsize=8, labelcolor="white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    plt.close()
    print(f"  Plot saved -> {output_path.resolve()}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/SPY_chain.csv")
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"\n  Loaded {len(df)} options from {csv_path}")
    print(f"  Maturities: {sorted(df['days_to_expiry'].unique())}\n")

    fits = fit_all_slices(df)

    # ── Print calibration summary table ────────────────────────────────────────
    print()
    print(f"  {'Expiry (DTE)':>14} | {'Points':>6} | {'RMSE (vol pts)':>14}")
    print(f"  {'-'*14}-+-{'-'*6}-+-{'-'*14}")
    for dte, (params, rmse) in sorted(fits.items()):
        n = len(df[df["days_to_expiry"] == dte])
        print(f"  {dte:>14} | {n:>6} | {rmse:>13.3f}%")
    print()

    plot_svi_fits(df, fits, Path("plots/plot_svi_fits.png"))

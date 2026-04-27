"""
Static arbitrage diagnostics on a calibrated SVI surface.

Checks:
  1. Calendar spread  — total variance w(k, T1) <= w(k, T2) for T1 < T2
  2. Butterfly        — risk-neutral density g(k) >= 0 for each maturity
                        g(k) = (1 - k*w'/(2w))^2 - w'^2/4*(1/4 + 1/w) + w''/2
                        (Gatheral & Jacquier, 2014)
"""

from __future__ import annotations
import sys
from pathlib import Path

# Allow import from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from calibration.svi import fit_all_slices, SVIParams

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


# ── Calendar spread check ──────────────────────────────────────────────────────

def check_calendar(
    fits: dict[int, tuple[SVIParams, float]],
    n_grid: int = 500,
) -> list[dict]:
    """
    For each consecutive pair (T1 < T2), verify w(k, T1) <= w(k, T2) for all k.
    Uses a shared dense grid spanning the union of both slices' natural range.

    Returns a list of violation records:
        {dte1, dte2, n_violations, max_violation_var, max_violation_vp, k_worst}
    """
    dtes = sorted(fits.keys())
    violations = []

    for idx in range(len(dtes) - 1):
        dte1, dte2 = dtes[idx], dtes[idx + 1]
        p1, _ = fits[dte1]
        p2, _ = fits[dte2]

        # Grid over the union of both slices' typical domain
        k_grid = np.linspace(-3.0, 1.0, n_grid)
        w1 = p1.total_var(k_grid)
        w2 = p2.total_var(k_grid)

        diff = w1 - w2          # positive means violation: w(T1) > w(T2)
        viol_mask = diff > 1e-8

        if viol_mask.any():
            max_diff_var = float(diff[viol_mask].max())
            k_worst = float(k_grid[diff == diff.max()][0])
            T1, T2 = dte1 / 365.0, dte2 / 365.0
            # Convert variance violation to vol points at ATM approximation
            # Delta_IV ≈ Delta_w / (2 * IV * T); use w2 ATM as reference
            k_atm = np.argmin(np.abs(k_grid))
            iv_ref = float(np.sqrt(max(w2[k_atm], 1e-6) / T2))
            max_vp = max_diff_var / (2 * iv_ref * T2) * 100.0 if iv_ref > 0 else 0.0

            violations.append({
                "dte1": dte1, "dte2": dte2,
                "n_violation_points": int(viol_mask.sum()),
                "max_violation_var": round(max_diff_var, 6),
                "max_violation_vp": round(max_vp, 3),
                "k_worst": round(k_worst, 4),
            })

    return violations


# ── Butterfly check ────────────────────────────────────────────────────────────

def check_butterfly(
    params: SVIParams, T: float, n_grid: int = 500
) -> dict:
    """
    Compute risk-neutral density proxy g(k) for a single maturity.
    Uses analytic SVI derivatives.

    g(k) = (1 - k*w'/(2w))^2 - w'^2/4*(1/4 + 1/w) + w''/2  >= 0

    Returns summary with min g, location of minimum, and violation flag.
    """
    k_grid = np.linspace(-3.0, 1.0, n_grid)
    w, wp, wpp = params.derivatives(k_grid)
    w = np.maximum(w, 1e-10)

    g = (1.0 - k_grid * wp / (2.0 * w))**2 \
        - wp**2 / 4.0 * (1.0 / 4.0 + 1.0 / w) \
        + wpp / 2.0

    min_g = float(g.min())
    k_min_g = float(k_grid[np.argmin(g)])
    violated = bool(min_g < -1e-6)

    return {
        "T": T,
        "min_g": round(min_g, 6),
        "k_min_g": round(k_min_g, 4),
        "violated": violated,
        "g_curve": (k_grid, g),
    }


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_arb_check(
    fits: dict[int, tuple[SVIParams, float]],
    cal_violations: list[dict],
    bfly_results: dict[int, dict],
    output_path: Path,
) -> None:
    _apply_dark()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left panel: total variance curves + calendar violations ────────────────
    ax = axes[0]
    dtes = sorted(fits.keys())
    n = max(len(dtes) - 1, 1)
    cmap = plt.cm.plasma
    k_grid = np.linspace(-3.0, 1.0, 500)

    violated_pairs = {(v["dte1"], v["dte2"]) for v in cal_violations}

    for i, dte in enumerate(dtes):
        params, _ = fits[dte]
        color = cmap(i / n)
        w = params.total_var(k_grid)
        ax.plot(k_grid, w, color=color, linewidth=1.8, label=f"{dte}d", alpha=0.9)

    # Shade violation regions between consecutive pairs
    for v in cal_violations:
        p1, _ = fits[v["dte1"]]
        p2, _ = fits[v["dte2"]]
        w1 = p1.total_var(k_grid)
        w2 = p2.total_var(k_grid)
        viol = w1 > w2 + 1e-8
        if viol.any():
            ax.fill_between(
                k_grid, w1, w2,
                where=viol, color="#FF4444", alpha=0.35, label="_nolegend_"
            )
            ax.axvline(v["k_worst"], color="#FF4444", linewidth=0.8,
                       linestyle=":", alpha=0.7)

    # Custom legend entry for violation shading
    if cal_violations:
        ax.add_artist(ax.legend(
            handles=[Line2D([0], [0], color="#FF4444", linewidth=6, alpha=0.35,
                            label="Calendar violation")],
            loc="upper right", fontsize=8, labelcolor="white",
        ))
        ax.legend(loc="upper left", fontsize=7, labelcolor="white")
    else:
        ax.legend(loc="upper left", fontsize=7, labelcolor="white")

    ax.axvline(0, color="#FFD700", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Moneyness  ln(K/S)")
    ax.set_ylabel("Total Variance  w = IV² · T")
    ax.set_title("Calendar Spread Check\n(red shading = violation)", fontsize=11)
    ax.set_xlim(-3.0, 1.0)
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#2a2a2a")

    # ── Right panel: g(k) density per maturity ─────────────────────────────────
    ax2 = axes[1]
    for i, dte in enumerate(dtes):
        res = bfly_results[dte]
        k_g, g = res["g_curve"]
        color = cmap(i / n)
        lw = 1.6
        ls = "-"
        if res["violated"]:
            # Draw the violation region in red
            viol_mask = g < 0
            ax2.fill_between(k_g, g, 0, where=viol_mask,
                             color="#FF4444", alpha=0.4, label="_nolegend_")
            ls = "--"
        ax2.plot(k_g, g, color=color, linewidth=lw, linestyle=ls,
                 label=f"{dte}d {'[BFLY VIOL]' if res['violated'] else ''}", alpha=0.85)

    ax2.axhline(0, color="#FF4444", linewidth=1.0, linestyle="-", alpha=0.8, label="g=0 floor")
    ax2.axvline(0, color="#FFD700", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Moneyness  ln(K/S)")
    ax2.set_ylabel("Risk-neutral density proxy  g(k)")
    ax2.set_title("Butterfly Arbitrage Check\n(g(k) >= 0 required)", fontsize=11)
    ax2.set_xlim(-3.0, 1.0)
    ax2.grid(True, color="#2a2a2a")
    ax2.legend(fontsize=7, labelcolor="white")

    plt.suptitle("SVI Surface — Static Arbitrage Diagnostics", color="white", fontsize=13, y=1.01)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    plt.close()
    print(f"  Plot saved -> {output_path.resolve()}")


# ── Report printer ─────────────────────────────────────────────────────────────

def print_report(
    cal_violations: list[dict],
    bfly_results: dict[int, dict],
) -> None:
    bar = "-" * 60
    print(f"\n{'='*60}")
    print("  ARBITRAGE CHECK REPORT")
    print(f"{'='*60}")

    # Calendar
    print(f"\n  [1] CALENDAR SPREAD ARBITRAGE")
    print(f"  {bar}")
    if not cal_violations:
        print("  CLEAN — no calendar violations detected.")
    else:
        print(f"  {'Pair (DTE1 / DTE2)':>20} | {'Viol pts':>8} | {'Max viol (var)':>14} | {'Max viol (vp)':>13} | {'k_worst':>8}")
        print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*14}-+-{'-'*13}-+-{'-'*8}")
        for v in cal_violations:
            print(
                f"  {v['dte1']:>6}d / {v['dte2']:<6}d    "
                f"| {v['n_violation_points']:>8} "
                f"| {v['max_violation_var']:>14.6f} "
                f"| {v['max_violation_vp']:>12.3f}% "
                f"| {v['k_worst']:>8.4f}"
            )

    # Butterfly
    print(f"\n  [2] BUTTERFLY ARBITRAGE (g(k) >= 0)")
    print(f"  {bar}")
    any_bfly = False
    print(f"  {'DTE':>6} | {'min g(k)':>10} | {'k @ min g':>10} | {'Status':>12}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
    for dte, res in sorted(bfly_results.items()):
        status = "VIOLATION" if res["violated"] else "CLEAN"
        mark = " <--" if res["violated"] else ""
        print(f"  {dte:>6} | {res['min_g']:>10.6f} | {res['k_min_g']:>10.4f} | {status:>12}{mark}")
        if res["violated"]:
            any_bfly = True
    if not any_bfly:
        print(f"\n  CLEAN — no butterfly violations detected.")

    print(f"\n{'='*60}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/SPY_chain.csv")
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"\n  Loaded {len(df)} options — calibrating SVI...")
    fits = fit_all_slices(df)
    print(f"  Calibrated {len(fits)} maturity slices: {sorted(fits.keys())}")

    # Run checks
    cal_violations = check_calendar(fits)

    bfly_results: dict[int, dict] = {}
    for dte, (params, _) in fits.items():
        bfly_results[dte] = check_butterfly(params, dte / 365.0)

    print_report(cal_violations, bfly_results)
    plot_arb_check(fits, cal_violations, bfly_results, Path("plots/plot_arb_check.png"))

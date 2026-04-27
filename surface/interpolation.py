from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

# ── Dark theme ────────────────────────────────────────────────────────────────

_DARK = {
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.edgecolor": "#444444",
    "grid.color": "#333333",
    "grid.alpha": 0.4,
    "axes.titlecolor": "white",
}

def _apply_dark() -> None:
    for k, v in _DARK.items():
        plt.rcParams[k] = v


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_chain(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"strike", "days_to_expiry", "moneyness", "implied_vol", "option_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


# ── 3-D scatter surface ───────────────────────────────────────────────────────

def plot_surface_3d(df: pd.DataFrame, output_path: Path) -> None:
    _apply_dark()
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    x = df["moneyness"].values
    y = df["days_to_expiry"].values
    z = df["implied_vol"].values * 100.0  # → percent

    sc = ax.scatter(x, y, z, c=z, cmap="plasma", s=6, alpha=0.85, depthshade=True)

    cbar = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.55, aspect=15)
    cbar.set_label("Implied Vol (%)", color="white", labelpad=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("Moneyness  ln(K/S)", color="white", labelpad=10)
    ax.set_ylabel("Days to Expiry", color="white", labelpad=10)
    ax.set_zlabel("Implied Vol (%)", color="white", labelpad=10)
    ax.set_title("Raw Implied Volatility Surface", color="white", fontsize=14, pad=22)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#333333")

    ax.tick_params(colors="white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    plt.close()
    print(f"  [surface] 3D plot saved -> {output_path.resolve()}")


# ── IV smile slices per maturity ──────────────────────────────────────────────

def plot_smile_slices(df: pd.DataFrame, output_path: Path) -> None:
    _apply_dark()
    fig, ax = plt.subplots(figsize=(13, 7))

    expirations = sorted(df["days_to_expiry"].unique())
    cmap = plt.cm.plasma
    n = max(len(expirations) - 1, 1)

    drawn = 0
    for i, dte in enumerate(expirations):
        sl = df[df["days_to_expiry"] == dte].sort_values("moneyness")
        if len(sl) < 3:
            continue
        color = cmap(i / n)
        ax.plot(
            sl["moneyness"],
            sl["implied_vol"] * 100.0,
            marker="o",
            markersize=2.5,
            linewidth=1.4,
            color=color,
            label=f"{dte}d",
            alpha=0.88,
        )
        drawn += 1

    ax.axvline(0, color="#FFD700", linewidth=0.9, linestyle="--", alpha=0.7, label="ATM")
    ax.set_xlabel("Moneyness  ln(K/S)", color="white")
    ax.set_ylabel("Implied Vol (%)", color="white")
    ax.set_title("IV Smile per Maturity", color="white", fontsize=14)
    ax.grid(True, alpha=0.2, color="#555555")

    if drawn <= 20:
        ax.legend(
            loc="upper right", fontsize=7, ncol=3,
            facecolor="#1a1d25", edgecolor="#555555", labelcolor="white"
        )
    else:
        # Too many maturities — add a colorbar legend instead
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(expirations[0], expirations[-1]))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Days to Expiry", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    plt.close()
    print(f"  [surface] Smile slices saved -> {output_path.resolve()}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/SPY_chain.csv"
    df = load_chain(csv_path)
    print(f"  Loaded {len(df)} options from {csv_path}")

    plots = Path("plots")
    plot_surface_3d(df, plots / "plot_surface_raw.png")
    plot_smile_slices(df, plots / "plot_smile_slices.png")

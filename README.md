# Volatility Surface Calibrator

A quantitative finance toolkit for building, calibrating, and visualising implied volatility surfaces from live option chains.

**Live app:** https://vol-surface-calibrator-icmwyjysjwydkdtluuvddv.streamlit.app

---

## Overview

This project fetches live option chains via `yfinance`, computes implied volatilities using Black-Scholes inversion (Newton-Raphson + Brentq), and plots the resulting IV surface in both 3D and per-maturity smile slices. Weeks 2–4 add SVI/SABR calibration, static arbitrage checks, a delta-hedge backtest engine, and a Streamlit dashboard.

---

## Features — Week 1

- **Live option chain fetch** — `yfinance` with quality filters (volume, OI, bid > 0)
- **IV solver** — Newton-Raphson with Brentq fallback; handles intrinsic floor, T→0, deep ITM/OTM edge cases
- **3D IV surface** — scatter plot with dark theme (matplotlib)
- **IV smile slices** — one curve per maturity on a single chart
- **CSV export** — persists cleaned chain data for offline use

---

## Installation

```bash
# Clone / navigate to the project
cd vol-surface-calibrator

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Fetch option chain + compute IVs

```bash
python data/fetch_chain.py          # defaults to SPY
python data/fetch_chain.py QQQ      # any ticker
```

Output: `data/SPY_chain.csv` and console summary.

### Build IV surface plots

```bash
python surface/interpolation.py data/SPY_chain.csv
```

Output: `plots/plot_surface_raw.png` and `plots/plot_smile_slices.png`.

### Launch Streamlit dashboard

```bash
streamlit run app.py
```

---

## Project Structure

```
vol-surface-calibrator/
├── data/
│   └── fetch_chain.py        # Live option chain + IV solver
├── calibration/
│   └── svi.py                # SVI calibration (Week 2)
├── surface/
│   ├── interpolation.py      # 3D surface + smile slice plots
│   └── arbitrage_check.py    # Static arbitrage checks (Week 2)
├── backtest/
│   └── delta_hedge.py        # Delta-hedge P&L engine (Week 4)
├── plots/                    # Generated plot outputs
├── app.py                    # Streamlit dashboard (Week 3)
├── requirements.txt
└── README.md
```

---

## Roadmap

| Week | Feature |
|------|---------|
| **1** | Live fetch · IV solver · 3D surface plot · smile slices ✅ |
| **2** | SVI calibration per maturity slice · static arbitrage checks (calendar, butterfly) |
| **3** | SABR model calibration · Streamlit interactive dashboard with Plotly |
| **4** | Delta-hedge backtest · realized vs. implied vol P&L attribution · performance report |

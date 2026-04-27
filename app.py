"""
Volatility Surface Calibrator — Streamlit Dashboard
5 tabs: Live Surface · SVI Calibration · Arbitrage Check · BS Pricer · Delta Hedge
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))

from data.fetch_chain import fetch_option_chain
from calibration.svi import fit_all_slices, calibrate_slice, SVIParams
from surface.arbitrage_check import check_calendar, check_butterfly
from backtest.delta_hedge import run_backtest, summary_stats

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Vol Surface Calibrator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    [data-testid="stSidebar"] { background-color: #13161f; border-right: 1px solid #2a2a2a; }
    .metric-card {
        background: #1a1d25; border: 1px solid #2a2a2a; border-radius: 8px;
        padding: 12px 18px; text-align: center;
    }
    .badge-clean {
        display:inline-block; background:#1a6b3c; color:#6eff9a;
        border:1px solid #6eff9a; border-radius:20px;
        padding:5px 18px; font-weight:700; font-size:14px;
    }
    .badge-viol {
        display:inline-block; background:#6b1a1a; color:#ff6e6e;
        border:1px solid #ff6e6e; border-radius:20px;
        padding:5px 18px; font-weight:700; font-size:14px;
    }
    div[data-testid="stTabs"] button { font-size: 14px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Plotly dark layout helpers ─────────────────────────────────────────────────

_PL = dict(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(color="white", size=12),
)
_AXIS = dict(gridcolor="#2a2a2a", linecolor="#444444", zerolinecolor="#444444", color="white")


def dark_layout(**kw) -> dict:
    base = dict(**_PL, xaxis=_AXIS, yaxis=_AXIS, margin=dict(l=50, r=30, t=50, b=50))
    base.update(kw)
    return base


def dark_3d_layout(**kw) -> dict:
    ax3 = dict(color="white", gridcolor="#2a2a2a", backgroundcolor="#0e1117",
               showbackground=True, zerolinecolor="#444444")
    base = dict(
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        scene=dict(xaxis=ax3, yaxis=ax3, zaxis=ax3, bgcolor="#0e1117"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    base.update(kw)
    return base


# ── Black-Scholes pricer + Greeks ──────────────────────────────────────────────

def bs_compute(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> dict:
    if T <= 0 or sigma <= 0:
        price = max(S - K, 0.0) if opt == "call" else max(K - S, 0.0)
        return dict(price=price, delta=float(S > K) if opt == "call" else float(S < K) - 1,
                    gamma=0.0, vega=0.0, theta=0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1, nd2 = norm.cdf(d1), norm.cdf(d2)
    disc = np.exp(-r * T)
    if opt == "call":
        price = S * nd1 - K * disc * nd2
        delta = nd1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * disc * nd2) / 365
    else:
        price = K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = nd1 - 1.0
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * disc * norm.cdf(-d2)) / 365
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100.0  # per 1 vol point
    return dict(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta)


# ── SVI surface lookup ─────────────────────────────────────────────────────────

def svi_vol_at(
    fits: dict, K: float, S: float, T_days: float
) -> tuple[float, int] | tuple[None, None]:
    if not fits:
        return None, None
    dte_arr = np.array(sorted(fits.keys()))
    closest = int(dte_arr[np.argmin(np.abs(dte_arr - T_days))])
    params, _ = fits[closest]
    iv = float(params.implied_vol(np.array([np.log(K / S)]), closest / 365.0)[0])
    return iv, closest


# ── Cached data fetch ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_chain_cached(ticker: str, r: float = 0.05) -> tuple[pd.DataFrame | None, float | None, str]:
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1d")
        if hist.empty:
            return None, None, f"No price data for {ticker}"
        spot = float(hist["Close"].iloc[-1])
    except Exception as exc:
        return None, None, str(exc)
    try:
        df = fetch_option_chain(ticker, r=r)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        return df, spot, ts
    except SystemExit:
        return None, None, "Option chain fetch failed"
    except Exception as exc:
        return None, None, str(exc)


# ── Session state defaults ─────────────────────────────────────────────────────

for _k, _v in [
    ("fits", {}),
    ("cal_violations", []),
    ("bfly_results", {}),
    ("arb_ran", False),
    ("backtest_result", None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 Vol Surface Calibrator")
    st.markdown("---")
    ticker = st.text_input("Ticker", value="SPY", max_chars=10).upper().strip()
    rf_rate = st.slider("Risk-free rate", 0.0, 0.12, 0.05, step=0.005, format="%.3f")
    st.markdown("---")
    st.markdown("**Links**")
    st.markdown("[GitHub](https://github.com/your-repo/vol-surface-calibrator)")
    st.markdown("---")
    st.caption("Built by [Your Name]")
    st.caption("Powered by yfinance · scipy · Streamlit")


# ── Title ──────────────────────────────────────────────────────────────────────

st.title("Volatility Surface Calibrator")
st.caption(f"Ticker: **{ticker}** · Risk-free rate: **{rf_rate:.1%}**")

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐 Live Surface",
    "📐 SVI Calibration",
    "🔍 Arbitrage Check",
    "💲 BS vs SVI Pricer",
    "📊 Delta Hedge Backtest",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE SURFACE
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Live Implied Volatility Surface")

    with st.spinner(f"Fetching option chain for {ticker}..."):
        df, spot, fetch_ts = fetch_chain_cached(ticker, rf_rate)

    if df is None or df.empty:
        st.error(f"Could not load data: {fetch_ts}")
        st.stop()

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spot Price", f"${spot:,.2f}")
    c2.metric("Options (filtered)", len(df))
    c3.metric("Maturities", df["days_to_expiry"].nunique())
    c4.metric("Fetched at", fetch_ts.split()[1] if fetch_ts else "—")

    st.markdown("---")

    # 3D scatter
    fig3d = go.Figure(data=go.Scatter3d(
        x=df["moneyness"],
        y=df["days_to_expiry"],
        z=df["implied_vol"] * 100,
        mode="markers",
        marker=dict(
            size=4,
            color=df["implied_vol"] * 100,
            colorscale="Plasma",
            colorbar=dict(title="IV (%)", tickfont=dict(color="white"), titlefont=dict(color="white")),
            opacity=0.88,
        ),
        text=[
            f"Strike: {row.strike:.0f}<br>DTE: {row.days_to_expiry}d<br>IV: {row.implied_vol*100:.1f}%<br>Type: {row.option_type}"
            for row in df.itertuples()
        ],
        hoverinfo="text",
    ))
    fig3d.update_layout(
        **dark_3d_layout(
            title=dict(text="Raw Implied Volatility Surface", font=dict(color="white", size=14)),
            scene_xaxis_title="Moneyness ln(K/S)",
            scene_yaxis_title="Days to Expiry",
            scene_zaxis_title="Implied Vol (%)",
            height=560,
        )
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # 2D smile slices
    st.subheader("IV Smile per Maturity")
    fig2d = go.Figure()
    dtes_sorted = sorted(df["days_to_expiry"].unique())
    colors = [f"hsl({int(220 + 140 * i / max(len(dtes_sorted)-1,1))},80%,60%)" for i in range(len(dtes_sorted))]

    for dte, col in zip(dtes_sorted, colors):
        sl = df[df["days_to_expiry"] == dte].sort_values("moneyness")
        if len(sl) < 2:
            continue
        fig2d.add_trace(go.Scatter(
            x=sl["moneyness"], y=sl["implied_vol"] * 100,
            mode="lines+markers", name=f"{dte}d",
            line=dict(color=col, width=1.8),
            marker=dict(size=5, color=col),
        ))

    fig2d.add_vline(x=0, line_color="#FFD700", line_dash="dash", line_width=1,
                    annotation_text="ATM", annotation_font_color="#FFD700")
    fig2d.update_layout(**dark_layout(
        title="IV Smile Slices",
        xaxis_title="Moneyness ln(K/S)",
        yaxis_title="Implied Vol (%)",
        legend=dict(bgcolor="#1a1d25", bordercolor="#444444"),
        height=420,
    ))
    st.plotly_chart(fig2d, use_container_width=True)

    with st.expander("Raw data table"):
        st.dataframe(
            df.style.format({
                "implied_vol": "{:.2%}", "moneyness": "{:.4f}",
                "market_price": "{:.4f}", "strike": "{:.0f}",
            }),
            use_container_width=True, height=320,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SVI CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("SVI Calibration per Maturity Slice")
    st.caption("Fits raw SVI  w(k) = a + b·(ρ(k−m) + √((k−m)²+σ²))  using differential evolution + L-BFGS-B polish.")

    # Ensure chain is loaded
    if df is None:
        st.warning("Load data in the Live Surface tab first.")
        st.stop()

    eligible = [d for d in sorted(df["days_to_expiry"].unique()) if len(df[df["days_to_expiry"] == d]) >= 5]
    st.info(f"Eligible slices (≥ 5 points): **{eligible}**  ·  Slices skipped: "
            f"{[d for d in sorted(df['days_to_expiry'].unique()) if d not in eligible]}")

    if st.button("Run SVI Calibration", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Starting calibration...")
        fits_tmp: dict = {}
        for i, dte in enumerate(eligible):
            progress_bar.progress((i) / len(eligible), text=f"Fitting DTE = {dte} days ({i+1}/{len(eligible)})...")
            sl = df[df["days_to_expiry"] == dte]
            T = dte / 365.0
            params, rmse = calibrate_slice(sl["moneyness"].values, sl["implied_vol"].values, T)
            fits_tmp[dte] = (params, rmse)
        progress_bar.progress(1.0, text="Calibration complete.")
        st.session_state.fits = fits_tmp
        st.session_state.arb_ran = False  # invalidate arb check cache

    if st.session_state.fits:
        fits = st.session_state.fits

        # Results table
        st.markdown("### Calibration Summary")
        rows = []
        for dte, (p, rmse) in sorted(fits.items()):
            n = len(df[df["days_to_expiry"] == dte])
            rows.append({
                "DTE (days)": dte,
                "Points": n,
                "RMSE (vol pts)": f"{rmse:.3f}%",
                "a": f"{p.a:.5f}",
                "b": f"{p.b:.5f}",
                "rho": f"{p.rho:.5f}",
                "m": f"{p.m:.5f}",
                "sigma": f"{p.sigma:.5f}",
            })
        results_df = pd.DataFrame(rows)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        col_dl, _ = st.columns([1, 3])
        with col_dl:
            csv_bytes = results_df.to_csv(index=False).encode()
            st.download_button(
                "Download calibration CSV",
                data=csv_bytes,
                file_name=f"{ticker}_svi_calibration.csv",
                mime="text/csv",
            )

        # Per-maturity chart with dropdown
        st.markdown("### Fitted Smile")
        selected_dte = st.selectbox(
            "Select maturity",
            options=sorted(fits.keys()),
            format_func=lambda d: f"{d} days",
        )

        p_sel, rmse_sel = fits[selected_dte]
        T_sel = selected_dte / 365.0
        sl_sel = df[df["days_to_expiry"] == selected_dte].sort_values("moneyness")

        k_lo = sl_sel["moneyness"].min()
        k_hi = sl_sel["moneyness"].max()
        buf = (k_hi - k_lo) * 0.12
        k_dense = np.linspace(k_lo - buf, k_hi + buf, 400)
        iv_dense = p_sel.implied_vol(k_dense, T_sel) * 100

        fig_svi = go.Figure()
        fig_svi.add_trace(go.Scatter(
            x=sl_sel["moneyness"], y=sl_sel["implied_vol"] * 100,
            mode="markers", name="Market IV",
            marker=dict(size=9, color="#FFD700", symbol="circle"),
        ))
        fig_svi.add_trace(go.Scatter(
            x=k_dense, y=iv_dense,
            mode="lines", name=f"SVI fit (RMSE {rmse_sel:.3f} vp)",
            line=dict(color="#00c3ff", width=2.2),
        ))
        fig_svi.add_vline(x=0, line_color="#FFD700", line_dash="dash", line_width=0.8)
        fig_svi.update_layout(**dark_layout(
            title=f"SVI Fit — DTE {selected_dte}d",
            xaxis_title="Moneyness ln(K/S)",
            yaxis_title="Implied Vol (%)",
            legend=dict(bgcolor="#1a1d25", bordercolor="#444444"),
            height=420,
        ))
        st.plotly_chart(fig_svi, use_container_width=True)
    else:
        st.info("Click **Run SVI Calibration** to fit the surface.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ARBITRAGE CHECK
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Static Arbitrage Diagnostics")

    if not st.session_state.fits:
        st.warning("Calibrate the SVI surface first (Tab 2).")
    else:
        fits = st.session_state.fits

        if not st.session_state.arb_ran:
            with st.spinner("Running arbitrage checks..."):
                st.session_state.cal_violations = check_calendar(fits)
                st.session_state.bfly_results = {
                    dte: check_butterfly(p, dte / 365.0)
                    for dte, (p, _) in fits.items()
                }
                st.session_state.arb_ran = True

        cal_viols = st.session_state.cal_violations
        bfly = st.session_state.bfly_results

        any_cal = len(cal_viols) > 0
        any_bfly = any(r["violated"] for r in bfly.values())

        # Status badges
        st.markdown("### Surface Status")
        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            if any_cal:
                st.markdown('<span class="badge-viol">CALENDAR VIOLATIONS</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-clean">CALENDAR CLEAN</span>', unsafe_allow_html=True)
        with c2:
            if any_bfly:
                st.markdown('<span class="badge-viol">BUTTERFLY VIOLATIONS</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-clean">BUTTERFLY CLEAN</span>', unsafe_allow_html=True)

        st.markdown("---")

        # Calendar table
        st.markdown("#### [1] Calendar Spread Arbitrage")
        st.caption("w(k,T₁) must be ≤ w(k,T₂) for all k when T₁ < T₂")
        if not cal_viols:
            st.success("No calendar spread violations.")
        else:
            cal_df = pd.DataFrame([{
                "DTE₁": v["dte1"], "DTE₂": v["dte2"],
                "Violation pts": v["n_violation_points"],
                "Max (variance)": f"{v['max_violation_var']:.5f}",
                "Max (vol pts)": f"{v['max_violation_vp']:.3f}%",
                "k worst": f"{v['k_worst']:.3f}",
                "Note": "boundary extrapolation" if abs(v["k_worst"]) > 2.5 else "",
            } for v in cal_viols])
            st.dataframe(cal_df, use_container_width=True, hide_index=True)

        st.markdown("#### [2] Butterfly Arbitrage  —  g(k) ≥ 0")
        st.caption("Risk-neutral density g(k) must stay non-negative. g(k) = (1 − k·w′/2w)² − w′²/4·(¼+1/w) + w″/2")
        bfly_df = pd.DataFrame([{
            "DTE": dte,
            "min g(k)": f"{r['min_g']:.6f}",
            "k @ min g": f"{r['k_min_g']:.4f}",
            "Status": "VIOLATION" if r["violated"] else "CLEAN",
        } for dte, r in sorted(bfly.items())])
        st.dataframe(
            bfly_df.style.applymap(
                lambda v: "color:#ff6e6e;font-weight:700" if v == "VIOLATION" else "color:#6eff9a;font-weight:700",
                subset=["Status"],
            ),
            use_container_width=True, hide_index=True,
        )

        # Variance curves plot
        st.markdown("#### Variance Term Structure")
        k_grid = np.linspace(-3.0, 1.0, 500)
        dtes_sorted = sorted(fits.keys())
        n_c = max(len(dtes_sorted) - 1, 1)

        fig_arb = go.Figure()
        pal = ["#9400d3", "#c2185b", "#ff6f00", "#ffd600", "#76ff03"]

        for i, dte in enumerate(dtes_sorted):
            p, _ = fits[dte]
            w = p.total_var(k_grid)
            col = pal[i % len(pal)]
            fig_arb.add_trace(go.Scatter(
                x=k_grid, y=w,
                mode="lines", name=f"{dte}d",
                line=dict(color=col, width=1.8),
            ))

        # Overlay red fill for calendar violations
        for v in cal_viols:
            p1, _ = fits[v["dte1"]]
            p2, _ = fits[v["dte2"]]
            w1 = p1.total_var(k_grid)
            w2 = p2.total_var(k_grid)
            mask = w1 > w2 + 1e-8
            if mask.any():
                k_viol = k_grid[mask]
                fig_arb.add_trace(go.Scatter(
                    x=np.concatenate([k_viol, k_viol[::-1]]),
                    y=np.concatenate([w1[mask], w2[mask][::-1]]),
                    fill="toself", fillcolor="rgba(255,68,68,0.25)",
                    line=dict(color="rgba(255,68,68,0.5)", width=0.5),
                    name=f"Cal viol {v['dte1']}/{v['dte2']}d",
                    showlegend=True,
                ))

        fig_arb.add_vline(x=0, line_color="#FFD700", line_dash="dash", line_width=0.8)
        fig_arb.update_layout(**dark_layout(
            title="Total Variance  w(k, T)  — red = calendar violation region",
            xaxis_title="Moneyness ln(K/S)",
            yaxis_title="Total Variance  w = IV² · T",
            legend=dict(bgcolor="#1a1d25", bordercolor="#444444"),
            height=450,
            yaxis=dict(**_AXIS, rangemode="tozero"),
        ))
        st.plotly_chart(fig_arb, use_container_width=True)

        with st.expander("Interpretation"):
            st.markdown(
                """
                **Calendar violations at k = −3** are boundary extrapolation artefacts —
                the 326d slice only has OTM *call* data (k > 0), so the SVI left wing is
                unconstrained and diverges outside the data domain. This is not a real
                market arbitrage. Fix: clip the variance check to the data range of each slice.

                **Butterfly violations** (g(k) < 0) indicate regions where the fitted smile
                is too concave, implying a negative risk-neutral density. For the 781d slice
                (36 points, balanced coverage) the surface is butterfly-clean.
                """
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BS vs SVI PRICER
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Black-Scholes vs SVI Pricer")

    in_col, res_col = st.columns([1, 2])

    with in_col:
        st.markdown("#### Inputs")
        S_inp  = st.number_input("Spot price S", value=float(spot) if spot else 500.0,
                                  min_value=1.0, step=1.0, format="%.2f")
        K_inp  = st.number_input("Strike K", value=float(spot) if spot else 500.0,
                                  min_value=1.0, step=1.0, format="%.2f")
        T_inp  = st.number_input("Time to expiry (days)", value=90, min_value=1, max_value=1000, step=1)
        r_inp  = st.number_input("Risk-free rate r", value=rf_rate, step=0.005, format="%.4f")
        sig_inp = st.slider("Flat vol (BS) σ", 0.01, 1.50, 0.20, step=0.005, format="%.3f")
        opt_inp = st.radio("Option type", ["call", "put"], horizontal=True)

    with res_col:
        T_yr = T_inp / 365.0
        k_inp = np.log(K_inp / S_inp)

        # BS price + Greeks
        greeks = bs_compute(S_inp, K_inp, T_yr, r_inp, sig_inp, opt_inp)

        # SVI vol
        svi_iv, svi_dte = svi_vol_at(st.session_state.fits, K_inp, S_inp, T_inp)
        svi_greeks = bs_compute(S_inp, K_inp, T_yr, r_inp, svi_iv, opt_inp) if svi_iv else None

        st.markdown("#### Black-Scholes (flat vol)")
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Price",  f"${greeks['price']:.4f}")
        g2.metric("Delta",  f"{greeks['delta']:.4f}")
        g3.metric("Gamma",  f"{greeks['gamma']:.6f}")
        g4.metric("Vega",   f"${greeks['vega']:.4f}")
        g5.metric("Theta",  f"${greeks['theta']:.4f}/day")

        st.markdown("#### SVI Smile-Adjusted")
        if svi_iv is None:
            st.info("Calibrate the SVI surface (Tab 2) to see smile-adjusted pricing.")
        else:
            diff_iv = (svi_iv - sig_inp) * 100
            diff_price = svi_greeks["price"] - greeks["price"]  # type: ignore[index]

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("SVI IV",  f"{svi_iv*100:.2f}%",
                      delta=f"{diff_iv:+.2f} vp vs flat")
            s2.metric("Price",   f"${svi_greeks['price']:.4f}",  # type: ignore[index]
                      delta=f"${diff_price:+.4f}")
            s3.metric("Delta",   f"{svi_greeks['delta']:.4f}")    # type: ignore[index]
            s4.metric("Gamma",   f"{svi_greeks['gamma']:.6f}")    # type: ignore[index]
            s5.metric("Vega",    f"${svi_greeks['vega']:.4f}")    # type: ignore[index]

            st.caption(f"SVI interpolated from DTE={svi_dte}d slice  ·  k = ln(K/S) = {k_inp:.4f}")

        # Vol smile context chart
        if st.session_state.fits:
            st.markdown("#### Position on the smile")
            # Find closest slice
            fits = st.session_state.fits
            dte_arr = np.array(sorted(fits.keys()))
            closest_dte = int(dte_arr[np.argmin(np.abs(dte_arr - T_inp))])
            p_ctx, _ = fits[closest_dte]
            T_ctx = closest_dte / 365.0

            sl_ctx = df[df["days_to_expiry"] == closest_dte].sort_values("moneyness")
            k_lo_ctx = min(sl_ctx["moneyness"].min() - 0.05, k_inp - 0.1)
            k_hi_ctx = max(sl_ctx["moneyness"].max() + 0.05, k_inp + 0.1)
            k_d = np.linspace(k_lo_ctx, k_hi_ctx, 400)
            iv_d = p_ctx.implied_vol(k_d, T_ctx) * 100

            fig_ctx = go.Figure()
            fig_ctx.add_trace(go.Scatter(x=k_d, y=iv_d, mode="lines",
                                         name=f"SVI {closest_dte}d", line=dict(color="#00c3ff", width=2)))
            fig_ctx.add_trace(go.Scatter(x=sl_ctx["moneyness"], y=sl_ctx["implied_vol"]*100,
                                         mode="markers", name="Market", marker=dict(color="#FFD700", size=7)))
            fig_ctx.add_vline(x=k_inp, line_color="#ff6e6e", line_dash="dash", line_width=1.5,
                              annotation_text="Your strike", annotation_font_color="#ff6e6e")
            fig_ctx.add_hline(y=sig_inp*100, line_color="gray", line_dash="dot",
                              annotation_text=f"Flat σ={sig_inp*100:.1f}%", annotation_font_color="gray")
            fig_ctx.update_layout(**dark_layout(
                title=f"Strike position on SVI smile (DTE {closest_dte}d)",
                xaxis_title="Moneyness ln(K/S)",
                yaxis_title="Implied Vol (%)",
                legend=dict(bgcolor="#1a1d25"),
                height=340,
            ))
            st.plotly_chart(fig_ctx, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DELTA HEDGE BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader("Delta-Hedge Backtest")
    st.markdown(
        "> **What this P&L represents:** We simulate buying an option and re-hedging its "
        "delta exposure daily. Positive P&L means the option was *cheap* — realized volatility "
        "exceeded the implied vol used to price the hedge, generating gamma profits that outpaced "
        "theta (time decay). Negative P&L means we overpaid for vol."
    )
    st.markdown("---")

    bt_col, res_bt_col = st.columns([1, 2])

    with bt_col:
        st.markdown("#### Backtest Parameters")
        bt_ticker = st.text_input("Ticker", value=ticker, key="bt_ticker").upper().strip()
        bt_K      = st.number_input("Strike K", value=float(spot) if spot else 500.0,
                                     min_value=1.0, step=5.0, format="%.0f", key="bt_K")
        bt_sigma  = st.slider("Implied vol (σ used for hedging)", 0.05, 1.5, 0.20, step=0.01,
                               format="%.2f", key="bt_sigma")
        bt_r      = st.number_input("Risk-free rate", value=rf_rate, step=0.005,
                                     format="%.4f", key="bt_r")
        bt_opt    = st.radio("Option type", ["call", "put"], horizontal=True, key="bt_opt")
        bt_start  = st.date_input("Start date", value=datetime.today() - timedelta(days=90),
                                   max_value=datetime.today() - timedelta(days=10), key="bt_start")
        bt_end    = st.date_input("End date", value=datetime.today(),
                                   max_value=datetime.today(), key="bt_end")
        bt_freq   = st.selectbox("Rebalance frequency", [1, 2, 5], index=0,
                                  format_func=lambda x: f"Every {x} day(s)", key="bt_freq")

        run_bt = st.button("Run Backtest", type="primary", use_container_width=True)

    with res_bt_col:
        if run_bt:
            with st.spinner("Fetching historical prices..."):
                try:
                    import yfinance as yf
                    raw = yf.Ticker(bt_ticker).history(
                        start=bt_start.strftime("%Y-%m-%d"),
                        end=bt_end.strftime("%Y-%m-%d"),
                    )
                    if raw.empty:
                        st.error(f"No historical data for {bt_ticker} in this date range.")
                        st.stop()
                    price_series = raw["Close"].dropna()
                except Exception as exc:
                    st.error(f"Failed to fetch prices: {exc}")
                    st.stop()

            if len(price_series) < 5:
                st.error("Need at least 5 trading days of data.")
                st.stop()

            T0_yr = (bt_end - bt_start).days / 365.0
            with st.spinner("Running delta-hedge simulation..."):
                try:
                    result = run_backtest(
                        price_series, bt_K, T0_yr, bt_sigma, bt_r, bt_opt, bt_freq
                    )
                    stats = summary_stats(result)
                    st.session_state.backtest_result = (result, stats)
                except Exception as exc:
                    st.error(f"Backtest error: {exc}")
                    st.stop()

        if st.session_state.backtest_result is not None:
            result, stats = st.session_state.backtest_result

            # Stats row
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total P&L",     f"${stats['total_pnl']:+.2f}")
            s2.metric("Sharpe (ann.)", f"{stats['sharpe']:.3f}")
            s3.metric("Max Drawdown",  f"${stats['max_drawdown']:.2f}")
            s4.metric("Win Rate",      f"{stats['win_rate']:.1%}")

            # P&L chart
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=result.index, y=result["cumulative_pnl"],
                mode="lines", name="Cumulative P&L",
                line=dict(color="#00c3ff", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,195,255,0.08)",
            ))
            fig_bt.add_trace(go.Scatter(
                x=result.index, y=result["total_pnl"],
                mode="lines", name="Daily P&L",
                line=dict(color="#FFD700", width=1, dash="dot"),
                opacity=0.7,
            ))
            fig_bt.add_hline(y=0, line_color="#444444", line_width=1)
            fig_bt.update_layout(**dark_layout(
                title=f"Delta-Hedge P&L — {bt_ticker} {bt_opt.upper()} K={bt_K:.0f}  σ={bt_sigma:.0%}",
                xaxis_title="Date",
                yaxis_title="P&L ($)",
                legend=dict(bgcolor="#1a1d25", bordercolor="#444444"),
                height=400,
            ))
            st.plotly_chart(fig_bt, use_container_width=True)

            # Delta path
            with st.expander("Delta path + option value"):
                fig_d = go.Figure()
                fig_d.add_trace(go.Scatter(
                    x=result.index, y=result["delta"],
                    mode="lines", name="Delta", line=dict(color="#9d4edd", width=1.5),
                ))
                fig_d.add_trace(go.Scatter(
                    x=result.index, y=result["spot"],
                    mode="lines", name="Spot price", line=dict(color="#FFD700", width=1.2),
                    yaxis="y2",
                ))
                fig_d.update_layout(**dark_layout(
                    title="Delta & Spot",
                    yaxis_title="Delta",
                    yaxis2=dict(title="Spot ($)", overlaying="y", side="right", **_AXIS),
                    legend=dict(bgcolor="#1a1d25"),
                    height=300,
                ))
                st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.info("Configure parameters and click **Run Backtest** to start.")

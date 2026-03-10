"""Page 01: Dashboard — KPI cards, waterfall chart, contract summary."""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from components.shared import init_page, load_csv, load_parquet, load_kpis, load_contract_summary
import streamlit as st, plotly.graph_objects as go
import pandas as pd

st.header("📊 Dashboard — Key Performance Indicators")

DATA_DIR = init_page()

kpis = load_kpis()
if not kpis:
    st.warning("KPI data not found. Run Layer 1 pipeline first.")
    st.stop()

# --- Live Price Banner ---
live_price = kpis.get("live_dam_latest_eur_mwh")
live_ts = kpis.get("live_dam_latest_timestamp", "")
if live_price:
    st.success(f"🔴 **LIVE DAM SPOT: €{live_price:.2f}/MWh** · {live_ts[:19]} · Source: EQ/Montel API")

# --- KPI Cards ---
st.subheader("Wholesale Market Snapshot")
c1, c2, c3, c4 = st.columns(4)
c1.metric("DAM Base (Latest Month)", f"€{kpis.get('dam_base_avg_latest_month',0):.2f}/MWh")
c2.metric("DAM Peak", f"€{kpis.get('dam_peak_avg_latest_month',0):.2f}/MWh")
c3.metric("Trailing 6M Avg", f"€{kpis.get('trailing_6m_avg',0):.2f}/MWh")
c4.metric("Trailing 12M Avg", f"€{kpis.get('trailing_12m_avg',0):.2f}/MWh")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Imbalance P50", f"€{kpis.get('imbalance_cost_p50',0):.2f}/MWh")
c6.metric("Imbalance P90", f"€{kpis.get('imbalance_cost_p90',0):.2f}/MWh")
c7.metric("EUR/RON", f"{kpis.get('eur_ron_latest',0):.4f}")
c8.metric("Off-Peak Avg", f"€{kpis.get('dam_offpeak_avg_latest_month',0):.2f}/MWh")

st.divider()

# --- Contract Register Summary (Layer 2 Integration) ---
contracts = load_contract_summary()
if contracts:
    st.subheader("📋 Contract Register Summary (Layer 2)")
    st.caption("Source: Excel CONTRACT_REGISTER sheet — summary-level export only")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Hedging Ratio", f"{contracts.get('hedging_ratio_pct', 0):.1f}%")
    k2.metric("Contracted Volume", f"{contracts.get('total_contracted_volume_mwh', 0):,.0f} MWh/mo")
    k3.metric("Wtd PPA Price", f"€{contracts.get('portfolio_weighted_price_eur', 0):.2f}/MWh")
    k4.metric("Active Contracts", f"{contracts.get('contract_count_active', 0)}")

    col_type, col_fuel = st.columns(2)
    with col_type:
        bt = contracts.get("breakdown_by_type", {})
        if bt:
            fig_bt = go.Figure(go.Pie(
                labels=list(bt.keys()), values=list(bt.values()),
                hole=0.5, textinfo="label+percent",
                marker=dict(colors=["#2c3e50", "#3498db", "#e67e22", "#27ae60", "#9b59b6"]),
            ))
            fig_bt.update_layout(height=300, title="By Contract Type", font=dict(family="DM Sans", size=10), margin=dict(t=40, b=10))
            st.plotly_chart(fig_bt, width='stretch')

    with col_fuel:
        bf = contracts.get("breakdown_by_fuel", {})
        if bf:
            fig_bf = go.Figure(go.Pie(
                labels=list(bf.keys()), values=list(bf.values()),
                hole=0.5, textinfo="label+percent",
                marker=dict(colors=["#004E98", "#FF6B35", "#06D6A0", "#FFD166", "#95a5a6"]),
            ))
            fig_bf.update_layout(height=300, title="By Fuel Source", font=dict(family="DM Sans", size=10), margin=dict(t=40, b=10))
            st.plotly_chart(fig_bf, width='stretch')

    residual = contracts.get("residual_unhedged_volume_mwh", 0)
    total = contracts.get("total_delivery_obligation_mwh", 1)
    if residual > 0:
        st.info(f"⚠️ **Residual unhedged exposure:** {residual:,.0f} MWh/month ({residual/total*100:.1f}% of delivery obligation) — priced at DAM spot forecast")

st.divider()

# --- Price Build-Up Waterfall ---
st.subheader("Supply Price Build-Up Waterfall (EUR/MWh)")
st.caption("Commercial MV customer · DEER 20kV · Mode A procurement weights")

eur_ron = kpis.get("eur_ron_latest", 4.977)
dam_base = kpis.get("trailing_6m_avg", 110)

components = {
    "Wholesale\n(blended)": round(dam_base * 0.95, 1),
    "Imbalance": 2.5,
    "Profile\ncost": 2.5,
    "Transmission\n(TG)": round(7.28 / eur_ron, 1),
    "Distribution\n(TDc)": round(42.1 / eur_ron, 1),
    "System\nservices": round(6.62 / eur_ron, 1),
    "Cogeneration": round(13.6 / eur_ron, 1),
    "CfD": round(0.206 / eur_ron, 2),
    "Green\ncerts": round(72.54 / eur_ron, 1),
    "Excise": round(2.5 / eur_ron, 1),
    "Risk\npremium": 11.5,
    "Supplier\nmargin": 4.5,
}

names = list(components.keys()) + ["Subtotal\n(ex-VAT)"]
values = list(components.values())
subtotal = sum(values)

fig = go.Figure(go.Waterfall(
    orientation="v",
    measure=["relative"] * len(values) + ["total"],
    x=names,
    y=values + [subtotal],
    connector={"line": {"color": "rgba(100,100,100,0.3)", "width": 1}},
    increasing={"marker": {"color": "#e74c3c"}},
    decreasing={"marker": {"color": "#27ae60"}},
    totals={"marker": {"color": "#2c3e50", "line": {"color": "#e74c3c", "width": 2}}},
    text=[f"€{v:.1f}" for v in values] + [f"€{subtotal:.1f}"],
    textposition="outside",
    textfont=dict(size=10),
))

vat = subtotal * 0.21
total_price = subtotal + vat

fig.update_layout(
    height=450, showlegend=False,
    yaxis_title="EUR/MWh",
    font=dict(family="DM Sans"),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=30, b=80),
    annotations=[
        dict(x=0.95, y=0.95, xref="paper", yref="paper", showarrow=False,
             text=f"<b>+ VAT 21%: €{vat:.1f}</b><br><b>TOTAL: €{total_price:.1f}/MWh</b><br>(~{total_price*eur_ron:.0f} RON/MWh)",
             font=dict(size=13, color="#e74c3c"), align="right",
             bordercolor="#e74c3c", borderwidth=1, borderpad=8, bgcolor="rgba(255,255,255,0.9)")
    ]
)
st.plotly_chart(fig, width='stretch')

# --- Monthly DAM Trend ---
st.subheader("DAM Monthly Price Trend")
monthly = load_csv("dam_monthly_summary.csv")
if not monthly.empty:
    last_36 = monthly.tail(36)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=last_36.index, y=last_36["base_avg"], name="Base", line=dict(color="#2c3e50", width=2)))
    fig2.add_trace(go.Scatter(x=last_36.index, y=last_36["peak_avg"], name="Peak", line=dict(color="#e74c3c", width=1.5, dash="dot")))
    fig2.add_trace(go.Scatter(x=last_36.index, y=last_36["offpeak_avg"], name="Off-Peak", line=dict(color="#3498db", width=1.5, dash="dot")))
    fig2.update_layout(height=350, yaxis_title="EUR/MWh", font=dict(family="DM Sans"),
                        legend=dict(orientation="h", y=-0.15), margin=dict(t=20))
    st.plotly_chart(fig2, width='stretch')

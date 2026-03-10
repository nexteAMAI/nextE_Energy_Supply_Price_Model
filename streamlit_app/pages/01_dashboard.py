"""Page 01: Dashboard — KPI cards, waterfall chart, contract summary."""
import streamlit as st, json, plotly.graph_objects as go
from pathlib import Path
import pandas as pd

st.header("📊 Dashboard — Key Performance Indicators")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
kpi_path = DATA_DIR / "streamlit_kpis.json"

if not kpi_path.exists():
    st.warning("KPI data not found. Run Layer 1 pipeline first.")
    st.stop()

with open(kpi_path) as f:
    kpis = json.load(f)

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
total = subtotal + vat

fig.update_layout(
    height=450, showlegend=False,
    yaxis_title="EUR/MWh",
    font=dict(family="DM Sans"),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=30, b=80),
    annotations=[
        dict(x=0.95, y=0.95, xref="paper", yref="paper", showarrow=False,
             text=f"<b>+ VAT 21%: €{vat:.1f}</b><br><b>TOTAL: €{total:.1f}/MWh</b><br>(~{total*eur_ron:.0f} RON/MWh)",
             font=dict(size=13, color="#e74c3c"), align="right",
             bordercolor="#e74c3c", borderwidth=1, borderpad=8, bgcolor="rgba(255,255,255,0.9)")
    ]
)
st.plotly_chart(fig, use_container_width=True)

# --- Monthly DAM Trend ---
st.subheader("DAM Monthly Price Trend")
monthly = pd.read_csv(DATA_DIR / "dam_monthly_summary.csv", index_col=0, parse_dates=True)
if not monthly.empty:
    last_36 = monthly.tail(36)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=last_36.index, y=last_36["base_avg"], name="Base", line=dict(color="#2c3e50", width=2)))
    fig2.add_trace(go.Scatter(x=last_36.index, y=last_36["peak_avg"], name="Peak", line=dict(color="#e74c3c", width=1.5, dash="dot")))
    fig2.add_trace(go.Scatter(x=last_36.index, y=last_36["offpeak_avg"], name="Off-Peak", line=dict(color="#3498db", width=1.5, dash="dot")))
    if "trailing_6m" in last_36.columns:
        fig2.add_trace(go.Scatter(x=last_36.index, y=last_36.get("trailing_6m", []), name="6M Trailing", line=dict(color="#f39c12", width=2, dash="dash")))
    fig2.update_layout(height=350, yaxis_title="EUR/MWh", font=dict(family="DM Sans"),
                        legend=dict(orientation="h", y=-0.15), margin=dict(t=20))
    st.plotly_chart(fig2, use_container_width=True)

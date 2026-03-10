"""Page 03: Forward Curve — RO power forward vs Aurora Central/Low/High."""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from components.shared import init_page, load_csv, load_parquet, load_kpis
import streamlit as st, pandas as pd, plotly.graph_objects as go

st.header("📉 Forward Curve & Aurora Forecast")
DATA_DIR = init_page()

aurora = load_csv("aurora_forecast.csv")
if aurora.empty:
    st.warning("Aurora forecast not found."); st.stop()

# --- Scenario selector ---
tab1, tab2 = st.tabs(["📈 Baseload Forecast", "☀️ Technology Captured Prices"])

with tab1:
    horizon = st.slider("Forecast horizon (years)", 2, 35, 10)
    end_date = pd.Timestamp("2026-01-01") + pd.DateOffset(years=horizon)
    aurora_f = aurora[aurora.index <= end_date]

    central = [c for c in aurora_f.columns if "Baseload_Central" in c and "Nominal" in c]
    low = [c for c in aurora_f.columns if "Baseload_Low" in c and "Nominal" in c]
    high = [c for c in aurora_f.columns if "Baseload_High" in c and "Nominal" in c]

    if central:
        fig = go.Figure()
        if high:
            fig.add_trace(go.Scatter(x=aurora_f.index, y=aurora_f[high[0]], name="High",
                                      line=dict(color="#e74c3c", width=1, dash="dot"),
                                      fill=None))
        fig.add_trace(go.Scatter(x=aurora_f.index, y=aurora_f[central[0]], name="Central",
                                  line=dict(color="#2c3e50", width=2.5)))
        if low:
            fig.add_trace(go.Scatter(x=aurora_f.index, y=aurora_f[low[0]], name="Low",
                                      line=dict(color="#27ae60", width=1, dash="dot"),
                                      fill="tonexty", fillcolor="rgba(39,174,96,0.1)"))

        # Add benchmark points
        benchmarks = {"Jan-26": 133.62, "Apr-26": 82.70, "Jul-26": 94.61, "Oct-26": 109.33, "Dec-26": 125.74}
        bm_dates = [pd.Timestamp(f"2026-{m.split('-')[0][:3]}-01") for m in benchmarks.keys()]

        fig.update_layout(
            height=500, yaxis_title="EUR/MWh (nominal)",
            title="Aurora Oct 2025 — Romania Baseload Power Price Forecast",
            font=dict(family="DM Sans"), legend=dict(orientation="h", y=-0.12),
            margin=dict(t=50),
            hovermode="x unified",
        )
        st.plotly_chart(fig, width='stretch')

        # Near-term table
        st.subheader("2026 Monthly Benchmarks (Central)")
        near = aurora_f[aurora_f.index.year == 2026]
        if not near.empty and central:
            display = near[["Month_Name"] + central + (low or []) + (high or [])].copy() if "Month_Name" in near.columns else near[central + (low or []) + (high or [])].copy()
            st.dataframe(display.style.format("{:.2f}", subset=[c for c in display.columns if "€" in c or "Nominal" in c]),
                          width='stretch')

with tab2:
    st.subheader("Technology-Specific Captured Prices (Central, Curtailed)")
    tech_cols = {
        "Onshore Wind": [c for c in aurora.columns if "Onshore_Wind_Curtailed_Central" in c],
        "Fixed Solar PV": [c for c in aurora.columns if "Fixed_Solar_PV_Curtailed_Central" in c],
        "Offshore Wind": [c for c in aurora.columns if "Offshore_Wind_Curtailed_Central" in c],
    }

    aurora_tech = aurora[aurora.index <= end_date]
    fig2 = go.Figure()
    colors = {"Onshore Wind": "#3498db", "Fixed Solar PV": "#f39c12", "Offshore Wind": "#9b59b6"}

    for tech, cols in tech_cols.items():
        if cols:
            fig2.add_trace(go.Scatter(x=aurora_tech.index, y=aurora_tech[cols[0]],
                                       name=tech, line=dict(color=colors[tech], width=2)))

    if central:
        fig2.add_trace(go.Scatter(x=aurora_tech.index, y=aurora_tech[central[0]],
                                   name="Baseload", line=dict(color="#2c3e50", width=2, dash="dash")))

    fig2.update_layout(height=450, yaxis_title="EUR/MWh (nominal)",
                        title="Captured Prices by Technology vs Baseload",
                        font=dict(family="DM Sans"), legend=dict(orientation="h", y=-0.12), margin=dict(t=50))
    st.plotly_chart(fig2, width='stretch')

    # Curtailment rates
    curt_cols = [c for c in aurora.columns if "Curtailment" in c and "Central" in c]
    if curt_cols:
        st.subheader("Curtailment Rates (Central Scenario)")
        curt_data = aurora_tech[curt_cols].tail(60)
        st.line_chart(curt_data)

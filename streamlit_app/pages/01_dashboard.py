"""
Page 01: Dashboard — KPI overview, waterfall chart, scenario toggle.
Reads: streamlit_kpis.json, contract_summary.json
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

st.header("📊 Dashboard — Key Performance Indicators")

# Load KPIs
kpi_path = DATA_DIR / "streamlit_kpis.json"
if kpi_path.exists():
    with open(kpi_path) as f:
        kpis = json.load(f)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("DAM Base Avg (Latest Month)", f"{kpis.get('dam_base_avg_latest_month', 0):.2f} €/MWh")
    col2.metric("DAM Peak Avg", f"{kpis.get('dam_peak_avg_latest_month', 0):.2f} €/MWh")
    col3.metric("Trailing 6M Avg", f"{kpis.get('trailing_6m_avg', 0):.2f} €/MWh")
    col4.metric("Imbalance Cost P50", f"{kpis.get('imbalance_cost_p50', 0):.2f} €/MWh")

    st.caption(f"Last updated: {kpis.get('last_updated', 'N/A')}")
else:
    st.warning("KPI data not found. Run Layer 1 pipeline first.")

# Contract summary
contract_path = DATA_DIR / "contract_summary.json"
if contract_path.exists():
    with open(contract_path) as f:
        contracts = json.load(f)
    st.subheader("Contract Register Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hedging Ratio", f"{contracts.get('hedging_ratio_pct', 0):.1f}%")
    c2.metric("Contracted Volume", f"{contracts.get('total_contracted_volume_mwh', 0):,.0f} MWh")
    c3.metric("Weighted PPA Price", f"{contracts.get('portfolio_weighted_price_eur', 0):.2f} €/MWh")

st.markdown("---")
st.info("Waterfall chart and scenario toggle will be rendered from Layer 1 processed data.")

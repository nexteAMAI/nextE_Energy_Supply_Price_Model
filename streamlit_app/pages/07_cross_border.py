"""
Page 07: Cross-Border — FBMC flow monitor, capacity utilization, price convergence.
Reads: streamlit_cross_border.parquet, cross_border_monthly.csv
"""
import streamlit as st
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

st.header("🌍 Cross-Border Flow Monitor")

cb_path = DATA_DIR / "cross_border_monthly.csv"
if cb_path.exists():
    cb = pd.read_csv(cb_path, index_col=0, parse_dates=True)
    border_cols = [c for c in cb.columns if "MW" in c]
    if border_cols:
        st.subheader("Monthly Net Imports by Border (MW avg)")
        st.line_chart(cb[border_cols].tail(36))
else:
    st.warning("Cross-border data not found. Run Layer 1 pipeline first.")

st.markdown("""
**CORE FBMC Context:**
Romania has been coupled via Single Day-Ahead Coupling (SDAC) under CORE Flow-Based Market
Coupling since 8 June 2022. Key borders: RO-HU, RO-BG.
""")

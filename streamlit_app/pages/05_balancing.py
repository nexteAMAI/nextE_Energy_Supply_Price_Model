"""
Page 05: Balancing — Imbalance cost rolling 30-day, Long/Short spread, volume distribution.
Reads: streamlit_imbalance.parquet, imbalance_monthly_stats.csv
"""
import streamlit as st
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

st.header("⚖️ Balancing & Imbalance Cost Tracker")

imb_path = DATA_DIR / "imbalance_monthly_stats.csv"
if imb_path.exists():
    imb = pd.read_csv(imb_path, index_col=0, parse_dates=True)
    cost_cols = [c for c in imb.columns if "cost" in c.lower()]
    if cost_cols:
        st.subheader("Monthly Imbalance Cost Adder (EUR/MWh)")
        st.line_chart(imb[cost_cols])

    spread_cols = [c for c in imb.columns if "spread" in c.lower()]
    if spread_cols:
        st.subheader("Long/Short Spread Statistics")
        st.dataframe(imb[spread_cols].tail(12).style.format("{:.2f}"), use_container_width=True)
else:
    st.warning("Imbalance data not found. Run Layer 1 pipeline first.")

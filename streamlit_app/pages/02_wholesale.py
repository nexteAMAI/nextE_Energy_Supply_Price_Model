"""
Page 02: Wholesale Analysis — DAM price history, IDM spread, bilateral benchmark.
Reads: streamlit_dam_timeseries.parquet, idm_monthly_spread.csv
"""

import streamlit as st
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

st.header("📈 Wholesale Energy Market Analysis")

# Load DAM data
dam_path = DATA_DIR / "streamlit_dam_timeseries.parquet"
if dam_path.exists():
    dam = pd.read_parquet(dam_path)
    dam.index = pd.to_datetime(dam.index)

    # Date range selector
    col1, col2 = st.columns(2)
    start_date = col1.date_input("From", dam.index.min().date())
    end_date = col2.date_input("To", dam.index.max().date())

    mask = (dam.index >= pd.Timestamp(start_date)) & (dam.index <= pd.Timestamp(end_date))
    filtered = dam[mask]

    price_col = [c for c in filtered.columns if "EUR/MWh" in c or "Value" in c]
    if price_col:
        st.subheader("Day-Ahead Market Price (EUR/MWh)")
        st.line_chart(filtered[price_col[0]])

        # Monthly summary
        monthly = filtered[price_col[0]].resample("MS").agg(["mean", "min", "max"])
        monthly.columns = ["Average", "Min", "Max"]
        st.subheader("Monthly DAM Statistics")
        st.dataframe(monthly.style.format("{:.2f}"), use_container_width=True)
else:
    st.warning("DAM data not found. Run Layer 1 pipeline first.")

# IDM spread
idm_path = DATA_DIR / "idm_monthly_spread.csv"
if idm_path.exists():
    idm = pd.read_csv(idm_path, index_col=0, parse_dates=True)
    st.subheader("IDM-DAM Spread (Monthly)")
    spread_cols = [c for c in idm.columns if "spread" in c.lower()]
    if spread_cols:
        st.bar_chart(idm[spread_cols[0]])

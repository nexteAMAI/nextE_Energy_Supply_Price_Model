"""
Page 03: Forward Curve — RO power forward vs Aurora Central/Low/High, SRMC overlay.
Reads: streamlit_forward_curve.parquet, aurora_forecast.csv
"""
import streamlit as st
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

st.header("📉 Forward Curve Analysis")

aurora_path = DATA_DIR / "aurora_forecast.csv"
if aurora_path.exists():
    aurora = pd.read_csv(aurora_path, index_col=0, parse_dates=True)
    central_cols = [c for c in aurora.columns if "Baseload_Central" in c]
    low_cols = [c for c in aurora.columns if "Baseload_Low" in c]
    high_cols = [c for c in aurora.columns if "Baseload_High" in c]

    if central_cols:
        chart_data = pd.DataFrame({
            "Central": aurora[central_cols[0]],
            "Low": aurora[low_cols[0]] if low_cols else None,
            "High": aurora[high_cols[0]] if high_cols else None,
        })
        # Filter to 2026–2035 for readability
        chart_data = chart_data.loc["2026":"2035"]
        st.subheader("Aurora Oct 2025 Baseload Forecast (EUR/MWh)")
        st.line_chart(chart_data)
else:
    st.warning("Aurora forecast data not found. Run Layer 1 pipeline first.")

fwd_path = DATA_DIR / "streamlit_forward_curve.parquet"
if fwd_path.exists():
    fwd = pd.read_parquet(fwd_path)
    st.subheader("EEX RO Power Forward Curve")
    st.dataframe(fwd.head(20), use_container_width=True)

st.info("SRMC overlay and contango/backwardation analysis will be rendered from Layer 1 SRMC data.")

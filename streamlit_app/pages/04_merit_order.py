"""
Page 04: Merit Order — Generation stack, residual demand, marginal price regime.
Reads: streamlit_generation_stack.parquet, generation_monthly.csv
"""
import streamlit as st
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

st.header("🏭 Generation Stack & Merit Order")

gen_path = DATA_DIR / "generation_monthly.csv"
if gen_path.exists():
    gen = pd.read_csv(gen_path, index_col=0, parse_dates=True)
    share_cols = [c for c in gen.columns if "share_pct" in c]
    if share_cols:
        st.subheader("Monthly Generation Mix (%)")
        st.area_chart(gen[share_cols].tail(36))

    avg_cols = [c for c in gen.columns if "avg_mw" in c]
    if avg_cols:
        st.subheader("Monthly Average Generation by Fuel Type (MW)")
        st.bar_chart(gen[avg_cols].tail(12))
else:
    st.warning("Generation data not found. Run Layer 1 pipeline first.")

st.info("Stacked area chart and residual demand visualization will use full 15-min data from Parquet.")

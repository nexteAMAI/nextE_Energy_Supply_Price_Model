"""RO Energy Supply Pricing Model v2.0 — Layer 3 Streamlit Dashboard."""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
from components.shared import init_page, load_csv, load_parquet, load_kpis
import streamlit as st

st.set_page_config(page_title="RO Energy Pricing Model", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

DATA_DIR = init_page()

st.markdown("# ⚡ Romania Energy Supply Pricing Model")
st.caption("**Layer 3 — Live Monitoring Dashboard** · v2.0 · Python Engine → Excel Workbook → Streamlit")
st.divider()

kpis = load_kpis()
if kpis:
    c1, c2, c3 = st.columns(3)
    c1.metric("Data Coverage", f"{kpis.get('data_start','')[:10]} → {kpis.get('data_end','')[:10]}")
    c2.metric("EUR/RON Rate", f"{kpis.get('eur_ron_latest',0):.4f}")
    c3.metric("Last Refreshed", kpis.get("last_updated","")[:16])

st.info("👈 Use the **sidebar** to navigate between dashboard pages.")
st.sidebar.markdown("---")
st.sidebar.caption("**Sources:** EQ · ENTSO-E · Balancing Services · JAO")
st.sidebar.caption("**Repo:** nexteAMAI/nextE_Energy_Supply_Price_Model")

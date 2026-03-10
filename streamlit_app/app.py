"""RO Energy Supply Pricing Model v2.0 — Layer 3 Streamlit Dashboard."""
import streamlit as st, json
from pathlib import Path

st.set_page_config(page_title="RO Energy Pricing Model", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
.main .block-container { padding-top: 1.5rem; max-width: 1400px; }
h1,h2,h3 { font-family: 'DM Sans', sans-serif !important; }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; }
footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)

st.markdown("# ⚡ Romania Energy Supply Pricing Model")
st.caption("**Layer 3 — Live Monitoring Dashboard** · v2.0 · Python Engine → Excel Workbook → Streamlit")
st.divider()

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
    st.error(f"No processed data in `{DATA_DIR}`. Run `make backtest` first.")
    st.stop()

kpi_path = DATA_DIR / "streamlit_kpis.json"
if kpi_path.exists():
    with open(kpi_path) as f: kpis = json.load(f)
    c1, c2, c3 = st.columns(3)
    c1.metric("Data Coverage", f"{kpis.get('data_start','')[:10]} → {kpis.get('data_end','')[:10]}")
    c2.metric("EUR/RON Rate", f"{kpis.get('eur_ron_latest',0):.4f}")
    c3.metric("Last Refreshed", kpis.get("last_updated","")[:16])

st.info("👈 Use the **sidebar** to navigate between dashboard pages.")
st.sidebar.markdown("---")
st.sidebar.caption("**Sources:** EQ · ENTSO-E · Balancing Services · JAO")
st.sidebar.caption("**Repo:** nexteAMAI/nextE_Energy_Supply_Price_Model")

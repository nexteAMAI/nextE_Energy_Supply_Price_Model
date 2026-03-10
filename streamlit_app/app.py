"""
RO Energy Supply Pricing Model — Layer 3 Streamlit Dashboard.

Multi-page application providing read-only monitoring for commercial team
and management. Reads processed data from Layer 1 outputs.

Pages:
  01 Dashboard:        KPI cards, waterfall chart, scenario toggle
  02 Wholesale:        DAM history (Base/Peak/Off-Peak), IDM spread, bilateral benchmark
  03 Forward Curve:    RO power forward vs Aurora Central/Low/High, SRMC overlay
  04 Merit Order:      Stacked area (generation by type), residual demand
  05 Balancing:        Imbalance cost rolling 30-day, Long/Short spread
  06 Scenarios:        3-panel comparison (Base/Stress/Low) with waterfall
  07 Cross-Border:     FBMC flow monitor (RO-HU, RO-BG), capacity utilization
"""

import streamlit as st

st.set_page_config(
    page_title="RO Energy Supply Pricing Model",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ Romania Energy Supply Pricing Model v2.0")
st.markdown("""
**Layer 3 — Monitoring Dashboard**

Navigate using the sidebar to explore:
- **Dashboard**: KPI overview and price waterfall
- **Wholesale Analysis**: DAM/IDM price history and statistics
- **Forward Curve**: EEX forwards vs Aurora forecast
- **Merit Order**: Generation stack and residual demand
- **Balancing**: Imbalance cost tracker
- **Scenarios**: 3-scenario comparison panel
- **Cross-Border**: FBMC flow and price convergence monitor

---
*Data source: Layer 1 Python Engine outputs (auto-refreshed daily)*
""")

# Sidebar metadata
st.sidebar.markdown("---")
st.sidebar.caption("Model v2.0 | Architecture: Python → Excel → Streamlit")
st.sidebar.caption("Data: EQ + ENTSO-E + Balancing Services + JAO")

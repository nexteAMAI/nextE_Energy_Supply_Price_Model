"""
Page 06: Scenarios — 3-panel comparison (Base/Stress/Low) with waterfall per scenario.
Reads: Layer 1 outputs + Layer 2 assumptions (summary export)
"""
import streamlit as st

st.header("🔀 Scenario Comparison")

st.markdown("""
### Scenario Definitions

| Scenario | DAM Price | Gas (TTF) | CO2 (EUA) | RES | Cross-Border |
|----------|-----------|-----------|-----------|-----|-------------|
| **Base Case** | Trailing 6M avg / Aurora Central | Current front-year | Current front-Dec | P50 | Normal capacity |
| **High-Price Stress** | P90 trailing 12M / Aurora High | +30% shock | +20% shock | P90 low | -30% capacity |
| **Low-Price / Oversupply** | P10 trailing 12M / Aurora Low | -20% | -15% | P10 high | Full + imports |
""")

st.info("Scenario waterfalls will be rendered when Layer 2 assumptions are exported to Layer 3.")

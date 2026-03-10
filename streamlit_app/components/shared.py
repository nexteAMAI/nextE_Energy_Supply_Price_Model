"""
Shared utilities for all Streamlit pages.
Every page MUST call init_page() at the top.
"""
import json, hmac
import pandas as pd
import streamlit as st
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
def _check_password() -> bool:
    if st.session_state.get("authenticated"):
        return True
    if "passwords" not in st.secrets:
        st.sidebar.warning("No access control configured.")
        return True
    def _validate():
        user = st.session_state.get("_auth_user", "")
        pwd = st.session_state.get("_auth_pass", "")
        if user in st.secrets["passwords"] and hmac.compare_digest(pwd, st.secrets["passwords"][user]):
            st.session_state["authenticated"] = True
            st.session_state["username"] = user
            del st.session_state["_auth_pass"]
        else:
            st.session_state["_auth_failed"] = True
    st.markdown('<div style="text-align:center;padding:3rem 0 1rem"><h1>⚡ RO Energy Pricing Model</h1><p style="color:#666;">Restricted access</p></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login"):
            st.text_input("Username", key="_auth_user")
            st.text_input("Password", type="password", key="_auth_pass")
            st.form_submit_button("Sign In", on_click=_validate, use_container_width=True)
        if st.session_state.get("_auth_failed"):
            st.error("Invalid username or password.")
    return False
def _show_user():
    if st.session_state.get("authenticated"):
        st.sidebar.markdown(f"**Signed in:** `{st.session_state.get('username','')}`")
        if st.sidebar.button("Sign Out"):
            st.session_state["authenticated"] = False
            st.session_state.pop("username", None)
            st.rerun()
def init_page():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
    .main .block-container{padding-top:1.5rem;max-width:1400px}
    h1,h2,h3{font-family:'DM Sans',sans-serif!important}
    [data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace!important;font-size:1.6rem!important}
    [data-testid="stMetricLabel"]{font-family:'DM Sans',sans-serif!important;font-weight:500!important}
    footer{visibility:hidden}
    </style>""", unsafe_allow_html=True)
    if not _check_password():
        st.stop()
    _show_user()
    return DATA_DIR
@st.cache_data(ttl=300)
def load_csv(name):
    p = DATA_DIR / name
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, index_col=0)
    try:
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("Europe/Bucharest")
    except Exception:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df
@st.cache_data(ttl=300)
def load_parquet(name):
    p = DATA_DIR / name
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df
@st.cache_data(ttl=300)
def load_kpis():
    p = DATA_DIR / "streamlit_kpis.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}
@st.cache_data(ttl=300)
def load_contract_summary():
    p = DATA_DIR / "contract_summary.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}

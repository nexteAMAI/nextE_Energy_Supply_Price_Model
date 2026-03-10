"""Shared data loading utilities for all Streamlit pages."""

import json, pandas as pd, streamlit as st
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

@st.cache_data(ttl=300)
def load_kpis():
    p = DATA_DIR / "streamlit_kpis.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return {}

@st.cache_data(ttl=300)
def load_csv(name):
    p = DATA_DIR / name
    if p.exists():
        return pd.read_csv(p, index_col=0, parse_dates=True)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_parquet(name):
    p = DATA_DIR / name
    if p.exists():
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        return df
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_aurora():
    p = DATA_DIR / "aurora_forecast.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0, parse_dates=True)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_tornado():
    p = DATA_DIR / "tornado_inputs.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

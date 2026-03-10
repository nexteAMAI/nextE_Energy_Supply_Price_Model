"""Date, Segment, and Scenario selector components."""
import streamlit as st
from datetime import date


def date_range_selector(default_start=None, default_end=None):
    st.sidebar.subheader("Date Range")
    start = st.sidebar.date_input("Start", value=default_start or date(2024, 1, 1))
    end = st.sidebar.date_input("End", value=default_end or date.today())
    return start, end


def scenario_selector():
    return st.sidebar.selectbox(
        "Scenario",
        ["Base Case", "High-Price Stress Test", "Low-Price / Oversupply"],
        index=0,
    )


def segment_selector():
    return st.sidebar.selectbox(
        "Customer Segment",
        ["Large Industrial (HV)", "Commercial (MV)", "Small Commercial", "Residential"],
        index=1,
    )

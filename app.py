"""Streamlit entry point of the nextE Energy Supply Bid Management Tool.

Run: streamlit run app.py
The twelve pages of the execution prompt (section 10.1) share one session model (app.state).
"""

from __future__ import annotations

import streamlit as st

from app import brand as B
from app import state as S
from app.pages import (
    audit,
    cashflow,
    data,
    engine_qh,
    exports,
    guarantees,
    overview,
    parameters,
    pnl,
    pricing,
    scenarios,
    sources,
)
from esb import __version__

st.set_page_config(page_title="nextE ESB", page_icon=None, layout="wide", initial_sidebar_state="expanded")
B.inject()
state = S.get()

with st.sidebar:
    st.markdown(f'<div class="esb-mark">{B.CONFIDENTIAL}</div>', unsafe_allow_html=True)
    st.markdown("**nextE Energy Supply Bid Management Tool**")
    st.markdown(f'<div class="esb-caption">Engine v{__version__} · scenario file {state.scenario.name} v{state.scenario.version} · '
                f'year {state.params.spine_year} · {state.params.scenario_active}</div>', unsafe_allow_html=True)
    if st.button("Run engine", type="primary", width="stretch"):
        with st.spinner("Running the engine"):
            state.run_engine()
        if state.last_error:
            st.markdown(f'<div class="esb-refusal">{state.last_error}</div>', unsafe_allow_html=True)
    status_slot = st.empty()

pages = [
    st.Page(overview.render, title="1 · Overview", default=True),
    st.Page(data.render, title="2 · Data", url_path="data"),
    st.Page(parameters.render, title="3 · Parameters", url_path="parameters"),
    st.Page(sources.render, title="4 · Sources and Contracts", url_path="sources"),
    st.Page(scenarios.render, title="5 · Scenarios", url_path="scenarios"),
    st.Page(engine_qh.render, title="6 · Engine (QH)", url_path="engine"),
    st.Page(pnl.render, title="7 · P&L", url_path="pnl"),
    st.Page(cashflow.render, title="8 · Cash Flow", url_path="cashflow"),
    st.Page(guarantees.render, title="9 · Guarantees", url_path="guarantees"),
    st.Page(pricing.render, title="10 · Pricing / Bid", url_path="pricing"),
    st.Page(exports.render, title="11 · Exports", url_path="exports"),
    st.Page(audit.render, title="12 · Audit and Log", url_path="audit"),
]
st.navigation(pages, position="sidebar").run()
with st.sidebar:
    status_slot.markdown(f"State: {B.status(state.result is not None and not state.dirty, 'RUN CURRENT', 'INPUTS CHANGED' if state.result else 'NO RUN')}",
                         unsafe_allow_html=True)

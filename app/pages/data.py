"""Page 2 - Data: upload, validate, reconcile; template downloads; provenance."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from app import brand as B
from app import state as S
from esb.importer import build_template
from esb.importer.contract import INPUT_CLASSES
from esb.scenarios import SCENARIO_PREFIX

CLASS_HELP = {
    "offtaker_load": "Off-taker consumption, metered and notified, one pair of series per off-taker code (DSO sign, k = -1)",
    "pv_generation": "PV forecast generation (uncurtailed), forecast deviation and imbalance deviation ratios (PV1)",
    "baseload_nomination": "Baseload product nominations per off-taker (BL24 / Peak / Off-Peak); informative - the engine takes strips from Parameters",
    "wholesale_prices": "DAM, IDCT VWAP15, Surplus and Deficit imbalance prices, system direction - one workbook per price scenario",
}


@st.cache_data(show_spinner=False)
def _template_bytes(input_class: str, year: int, codes: tuple[str, ...]) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"TPL_{input_class}_QH_{year}.xlsx"
        build_template(input_class, year, p, entity_codes=list(codes) if input_class in ("offtaker_load", "baseload_nomination") else None)
        return p.read_bytes()


def render() -> None:
    state = S.get()
    p = state.params
    B.page_title("Data", "Uploads in the standard quarter-hour format (ESB-STD-QH), the six reconciliation checks and the provenance of every series")

    st.markdown("## Series in use")
    c1, c2 = st.columns([2, 3])
    with c1:
        use_ref = st.toggle("Use the Reference Case fixture as the base layer (coded series of the frozen workbook, 2027)",
                            value=state.use_reference_fixture,
                            help="Uploads replace the fixture series by series. Switch off to run on uploads only.")
        if use_ref != state.use_reference_fixture:
            state.use_reference_fixture = use_ref
            state.mark_dirty("Reference Case fixture " + ("enabled" if use_ref else "disabled"))
            st.rerun()
    with c2:
        series = state.rebuild_series()
        cov = series.coverage
        st.markdown(f"Coverage: {B.status(cov.ok, 'COMPLETE', 'INCOMPLETE')}", unsafe_allow_html=True)
        if cov.problems:
            for pr in cov.problems:
                B.refusal(pr)
    rows = []
    for code, c in cov.offtakers.items():
        o = p.offtaker(code)
        rows.append((f"{o.label} ({code})", "Active" if o.active else "Inactive", "yes" if c["metered"] else "no", "yes" if c["notified"] else "no"))
    df = pd.DataFrame(rows, columns=["Off-taker", "State", "Metered series", "Notified series"]).set_index("Off-taker")
    B.table(df, index_label="Off-taker")
    pv = pd.DataFrame({"Loaded": ["yes" if v else "no" for v in cov.pv.values()]}, index=list(cov.pv))
    sc = pd.DataFrame({"Loaded": ["yes" if v else "no" for v in cov.scenarios.values()],
                       "Active": ["yes" if k == p.scenario_active else "" for k in cov.scenarios]}, index=list(cov.scenarios))
    c1, c2 = st.columns(2)
    with c1:
        B.eyebrow("PV series")
        B.table(pv, index_label="Series")
    with c2:
        B.eyebrow("Price scenarios")
        B.table(sc, index_label="Scenario")
    if cov.baseload_nomination:
        B.note("Baseload nomination series are loaded for information; the engine sources Baseload from the strips on the Parameters page (Reference Case method).")

    st.markdown("## Upload a delivery")
    B.note("A workbook is accepted only when its Std_Control, Series_Registry and RAW_EET_QH surface pass all six checks "
           "(spine, layout, surface, registry, energy, gaps). Blank cells stay blank; nothing is filled. The sign convention k is read "
           "from the registry, never inferred.")
    col1, col2 = st.columns([3, 2])
    with col1:
        up = st.file_uploader("Standard template workbook (.xlsx)", type=["xlsx"], accept_multiple_files=False, key="data_uploader")
    with col2:
        choice = st.selectbox("Price scenario (wholesale_prices deliveries only)", ["from the file's Std_Control", *SCENARIO_PREFIX.keys()], index=0)
    if up is not None and st.button("Validate and load", type="primary"):
        data = up.getvalue()
        if any(u.md5 == __import__("hashlib").md5(data).hexdigest() for u in state.uploads):
            B.note("This exact file is already loaded.")
        else:
            with st.spinner("Reading, standardising and checking"):
                rec = state.add_upload(up.name, data, None if choice.startswith("from") else choice)
            res = rec.result
            st.markdown(f"{'Accepted' if res.ok else 'Refused'}: {B.status(res.ok, 'ACCEPTED', 'REFUSED')} {up.name}", unsafe_allow_html=True)
            chk = pd.DataFrame([(c.code, c.name, c.status, c.detail) for c in res.checks], columns=["Code", "Check", "Status", "Detail"]).set_index("Code")
            B.table(chk, index_label="Code")
            if res.ok:
                st.rerun()

    st.markdown("## Loaded uploads")
    if not state.uploads:
        B.caption("No uploads in this session.")
    for u in list(state.uploads):
        pv_ = u.result.provenance
        with st.expander(f"{u.filename} · {pv_.input_class} · {pv_.time_basis} · md5 {pv_.md5[:12]}"):
            info = pd.DataFrame({"Value": [pv_.input_class, pv_.spine_year, pv_.time_basis, pv_.scenario or "–", u.scenario_choice or "–",
                                           pv_.provider or "–", pv_.delivery_date or "–", pv_.template_version, pv_.imported_at_utc, pv_.md5,
                                           f"{pv_.size_bytes:,}".replace(",", ".")]},
                                index=["Input class", "Spine year", "Time basis", "Scenario (Std_Control)", "Scenario choice (app)", "Provider",
                                       "Delivery date", "Template version", "Imported (UTC)", "md5", "Size (bytes)"])
            B.table(info, index_label="Field")
            chk = pd.DataFrame([(c["code"], c["name"], c["status"], c["detail"]) for c in pv_.checks], columns=["Code", "Check", "Status", "Detail"]).set_index("Code")
            B.eyebrow("Six checks")
            B.table(chk, index_label="Code")
            slots = pd.DataFrame(pv_.slots)
            if len(slots):
                B.eyebrow("Series registry (k per volume series)")
                cols = [c for c in ("slot", "name", "unit", "cls", "spring_rule", "autumn_rule", "k", "basis", "entity_code", "required") if c in slots.columns]
                B.table(slots[cols].set_index("slot"), index_label="Slot")
            if st.button("Remove this upload", key=f"rm_{u.md5}"):
                state.remove_upload(u.md5)
                st.rerun()

    st.markdown("## Template downloads")
    B.note("Templates carry Instructions, Std_Control, Series_Registry (slots pre-declared and locked), the RAW_EET_QH paste surface and the "
           "Recon_Check gate. Contract: docs/DATA_CONTRACT.md (ESB-STD-QH 1.0).")
    year = p.spine_year
    codes = tuple(o.code for o in p.offtakers)
    cols = st.columns(len(INPUT_CLASSES))
    for col, cls in zip(cols, INPUT_CLASSES, strict=True):
        with col:
            st.markdown(f"**{cls}**")
            B.caption(CLASS_HELP.get(cls, ""))
            st.download_button(f"Download TPL_{cls}_QH_{year}.xlsx", data=_template_bytes(cls, year, codes),
                               file_name=f"TPL_{cls}_QH_{year}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key=f"tpl_{cls}")

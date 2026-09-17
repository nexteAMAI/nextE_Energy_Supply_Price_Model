"""Page 12 - Audit and Log (replaces _CLAUDE_LOG / _AUDIT_*): run log, provenance, open findings,
decision register, calculation trace."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app import brand as B
from app import state as S
from esb import __version__

ROOT = Path(__file__).resolve().parents[2]


def _doc(name: str) -> str:
    p = ROOT / "docs" / name
    return p.read_text(encoding="utf-8") if p.exists() else f"{name} not found in this deployment."


def render() -> None:
    state = S.get()
    B.page_title("Audit and Log", "What ran, on which inputs, with which decisions - the institutional record of this session and of the engine")
    p = state.params
    B.kpi_row([
        ("Engine", f"v{__version__}", "", "nexte-esb; parity-proven against the frozen Reference Case (20.092 cells, 0 FAIL)"),
        ("Scenario file", f"{state.scenario.name}", "", f"v{state.scenario.version} · md5 {state.scenario.md5[:12] or '–'}"),
        ("Series", "Reference Case fixture" if state.use_reference_fixture else "uploads only", "", f"{len(state.uploads)} upload(s) accepted this session"),
        ("Run state", "current" if (state.result is not None and not state.dirty) else ("inputs changed" if state.result else "no run"), "",
         f"{len(state.log)} log entries"),
    ])

    st.markdown("## Run log (this session)")
    if state.log:
        df = pd.DataFrame([(e.at_utc, e.kind, e.text) for e in reversed(state.log)], columns=["Time (UTC)", "Kind", "Entry"]).set_index("Time (UTC)")
        B.table(df, index_label="Time (UTC)", scroll=len(df) > 15)
    else:
        B.caption("Empty.")

    st.markdown("## Provenance of the current series")
    if state.series is not None and state.series.sources:
        df = pd.DataFrame(state.series.sources).astype(str)
        df.index = [str(i + 1) for i in range(len(df))]
        B.table(df, index_label="#", scroll=True)
        B.caption(f"Assembled frame md5 {state.series_md5()[:12]} · {len(state.series.frame):,} rows · {len(state.series.frame.columns)} columns".replace(",", "."))
    else:
        B.caption("No series assembled yet.")
    if state.result is not None:
        B.eyebrow("Calculation-order trace of the last run (ruling G0-D6: fixed order, no iteration)")
        tr = pd.DataFrame({"Cumulative seconds": [t for _, t in state.result.trace]}, index=[n for n, _ in state.result.trace])
        B.table(tr, index_label="Stage", decimals=3, col_units={"Cumulative seconds": "s"})

    st.markdown("## Scenario file history")
    if state.scenario.history:
        hist = pd.DataFrame(state.scenario.history).set_index("version")
        hist["md5"] = hist["md5"].str[:12]
        hist["saved_at_utc"] = [B.dmy_hm(v) for v in hist["saved_at_utc"]]
        hist = hist.rename(columns={"saved_at_utc": "Saved (UTC)", "note": "Note"})
        B.table(hist, index_label="Version", decimals=0, scroll=True)
    B.eyebrow("Register verification status - the parameter catalogue (config/parameter_catalogue.yaml, ruling D-I)")
    from esb.catalogue import catalogue_for
    from esb.export import _flatten

    flat = _flatten(p.to_dict())
    cat = catalogue_for(flat.keys())
    rows = [(k, B.num(v, 4) if isinstance(v, float) else str(v), e.unit, e.standard, e.source, e.source_status)
            for k, v in flat.items() if (e := cat[k]).source_status]
    df = pd.DataFrame(rows, columns=["Parameter", "Value", "Unit", "Standard / default", "Source / vintage", "Status"]).set_index("Parameter")
    B.table(df, index_label="Parameter", scroll=len(df) > 16)
    B.caption("Status: verified = checked against the primary source; unverified = quoted from the Reference Case workbook, the verification pass "
              "(T12.7, CW-EMK-01) is owed; assumption = house assumption; to_verify = source named, not yet checked. Every export's Parameters "
              "sheet carries the same columns")

    st.markdown("## Registers")
    t1, t2, t3 = st.tabs(["Decisions", "Open items", "Methodology"])
    with t1:
        st.markdown(_doc("DECISIONS.md"))
    with t2:
        st.markdown(_doc("OPEN_ITEMS.md"))
    with t3:
        st.markdown(_doc("METHODOLOGY.md"))

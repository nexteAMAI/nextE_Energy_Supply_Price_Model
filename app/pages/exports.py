"""Page 11 - Exports: Excel, CSV, the case bundle (save and reproduce) and the parity report."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from app import brand as B
from app import state as S
from esb.bundle import CaseBundle, compare_summary, read_bundle
from esb.export import build_workbook, csv_bytes

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def render() -> None:
    state = S.get()
    B.page_title("Exports", "Excel and CSV of the run, the case bundle that reproduces it, and the parity report against the Reference Case")
    r = S.require_result(state)
    p = state.params
    stamp = f"{p.spine_year}_{state.scenario.name.replace(' ', '_')}_v{state.scenario.version:03d}"

    st.markdown("## Excel")
    B.note("Every sheet carries the CONFIDENTIAL · nextE marking; display names are allowed in exports; number formats render in the Romanian "
           "convention on a Romanian Excel. The full 35.040-row QH frame is added only on request.")
    if r is None:
        B.refusal(state.last_error or "No run available to export.")
    else:
        include_qh = st.checkbox("Include the full quarter-hour frame (large workbook)", value=False)
        if st.button("Build the Excel workbook", type="primary"):
            with st.spinner("Writing the workbook"):
                data = build_workbook(r, include_qh=include_qh, scenario_name=f"{state.scenario.name} v{state.scenario.version}",
                                      provenance=state.series.sources if state.series else None)
            st.session_state["xlsx_export"] = (stamp, data)
            state.add_log("export", f"Excel workbook built ({len(data):,} bytes)".replace(",", "."))
        x = st.session_state.get("xlsx_export")
        if x and x[0] == stamp:
            st.download_button("Download ESB_run_" + stamp + ".xlsx", data=x[1], file_name=f"ESB_run_{stamp}.xlsx", mime=XLSX)

        st.markdown("## CSV")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pf = r.pnl.portfolio.frame()
            pf.columns = [*B.MONTH_EN, "Year"]
            st.download_button("P&L portfolio (CSV)", data=csv_bytes(pf), file_name=f"pnl_portfolio_{stamp}.csv", mime="text/csv")
        with c2:
            cf = r.cashflow.frame()
            cf.columns = [*B.MONTH_EN, "Beyond Dec", "Year"]
            st.download_button("Cash flow monthly (CSV)", data=csv_bytes(cf), file_name=f"cashflow_monthly_{stamp}.csv", mime="text/csv")
        with c3:
            if r.cashflow.daily is not None:
                st.download_button("Daily ledger (CSV)", data=csv_bytes(r.cashflow.daily, index=False), file_name=f"cashflow_daily_{stamp}.csv", mime="text/csv")
        with c4:
            pr = r.pricing[r.selected_offtaker]
            df = pd.DataFrame({k: [v, *(list(pr.months[k]) if k in pr.months else [float("nan")] * 12)] for k, v in pr.year.items()}, index=["Year", *B.MONTH_EN]).T
            st.download_button(f"Pricing {r.selected_offtaker} (CSV)", data=csv_bytes(df), file_name=f"pricing_{r.selected_offtaker}_{stamp}.csv", mime="text/csv")
        B.caption("Semicolon-separated, decimal comma, UTF-8 with BOM - opens directly in a Romanian Excel")

    st.markdown("## Case bundle")
    B.note("A case bundle holds the scenario file, every accepted upload byte-identical, their provenance and the headline results. Re-uploading it "
           "re-imports the deliveries through the same checks and re-runs the engine; the headline values are compared at the parity tolerance. "
           "A number that cannot be reproduced is not put in front of an off-taker.")
    c1, c2 = st.columns(2)
    with c1:
        if r is not None:
            if st.button("Build the case bundle"):
                bundle = CaseBundle(scenario=state.scenario, uploads=list(state.uploads), selected_offtaker=state.selected_offtaker,
                                    use_reference_fixture=state.use_reference_fixture, summary=r.summary())
                data = bundle.to_bytes()
                st.session_state["case_bundle"] = (bundle.filename, data)
                state.add_log("export", f"case bundle built ({len(data):,} bytes, {len(state.uploads)} upload(s))".replace(",", "."))
            cb = st.session_state.get("case_bundle")
            if cb:
                st.download_button("Download " + cb[0], data=cb[1], file_name=cb[0], mime="application/zip")
    with c2:
        up = st.file_uploader("Reproduce a case bundle (.esb-case.zip)", type=["zip"], key="bundle_upload")
        if up is not None and st.button("Open and reproduce"):
            try:
                with st.spinner("Re-importing the deliveries and running the engine"):
                    b = read_bundle(up.getvalue())
                    refused = [u.filename for u in b.uploads if not u.result.ok]
                    if refused:
                        B.refusal("Uploads refused on re-import: " + ", ".join(refused))
                    state.scenario = b.scenario
                    state.uploads = [u for u in b.uploads if u.result.ok]
                    state.use_reference_fixture = b.use_reference_fixture
                    state.selected_offtaker = b.selected_offtaker
                    state.mark_dirty(f"case bundle {up.name} opened (scenario '{b.scenario.name}' v{b.scenario.version}, created {b.created_at_utc})")
                    fresh = state.run_engine()
                if fresh is None:
                    B.refusal(state.last_error)
                else:
                    rows = compare_summary(b.summary, fresh.summary())
                    ok = all(x["status"] == "PASS" for x in rows)
                    st.markdown(f"Reproduction: {B.status(ok, 'REPRODUCED', 'DIFFERS')}", unsafe_allow_html=True)
                    df = pd.DataFrame(rows).set_index("key")
                    keys = list(df.index)
                    df.index = [k.replace("_", " ") for k in df.index]
                    B.table(df, index_label="Headline value", decimals=2,
                            units=["MWh" if k in ("metered", "notified") else "EUR" for k in keys])
                    state.add_log("bundle", f"bundle reproduced: {'PASS' if ok else 'DIFFERS'}")
            except ValueError as e:
                B.refusal(str(e))

    st.markdown("## Parity report")
    B.note("The tie-out of the engine against the frozen Reference Case workbook (20.245 cached cells of the five result sheets, tolerance "
           "abs(py - xl) <= 1e-6 x max(abs(xl), 1)). It runs on the Reference Case fixture with the Reference Case register regardless of the "
           "session's inputs; the committed report is in the repository's audit folder.")
    if st.button("Run the parity tie-out now"):
        with st.spinner("Comparing 20.245 cells"):
            import sys

            root = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(root / "tools"))
            import parity_report  # noqa: E402

            with tempfile.TemporaryDirectory() as tmp:
                rc = parity_report.main(Path(tmp))
                md = next(Path(tmp).glob("PARITY_REPORT_*.md"))
                st.session_state["parity_out"] = (rc, md.name, md.read_bytes(), (Path(tmp) / "parity.json").read_bytes())
        state.add_log("parity", "parity tie-out run from the application: " + ("PASS" if st.session_state['parity_out'][0] == 0 else "FAIL"))
    po = st.session_state.get("parity_out")
    if po:
        st.markdown(f"Verdict: {B.status(po[0] == 0)}", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Download " + po[1], data=po[2], file_name=po[1], mime="text/markdown")
        with c2:
            st.download_button("Download parity.json", data=po[3], file_name="parity.json", mime="application/json")
        with st.expander("Report"):
            st.markdown(po[2].decode("utf-8"))

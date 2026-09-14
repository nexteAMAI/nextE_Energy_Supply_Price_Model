"""Page 3 - Parameters (replaces the Input sheet): sections A-E and B2, off-taker and counterparty
toggles, the admin layer for regulatory constants, and the versioned scenario file (PSTORE)."""

from __future__ import annotations

import copy
from datetime import date

import pandas as pd
import streamlit as st

from app import brand as B
from app import state as S
from config.schema import (
    BGL_FEE_TYPES,
    GUARANTEE_SIZINGS,
    GUARANTEE_TYPES,
    PREMIUM_COMPONENTS,
    PRODUCTS,
    Guarantee,
    Offtaker,
)
from esb.scenario_file import ScenarioFile

MONTHS = B.MONTH_EN
TARIFF_HELP = {
    "TL": "Transport tariff, extraction (TL) - EUR/MWh", "TG": "Transport tariff, injection (TG) - EUR/MWh", "SS": "System services - EUR/MWh",
    "T_HV": "Distribution HV - EUR/MWh", "T_MV": "Distribution MV - EUR/MWh", "T_LV": "Distribution LV - EUR/MWh",
    "cogeneration": "Cogeneration contribution - EUR/MWh", "cfd": "CfD contribution - EUR/MWh", "excise": "Excise duty - EUR/MWh",
}


def _guarantee_form(g: Guarantee, key: str, allow_fixed_none: bool = False) -> Guarantee:
    c1, c2, c3 = st.columns(3)
    with c1:
        typ = st.selectbox("Type", GUARANTEE_TYPES, index=GUARANTEE_TYPES.index(g.type), key=f"{key}_type")
        sizing = st.selectbox("Sizing", GUARANTEE_SIZINGS, index=GUARANTEE_SIZINGS.index(g.sizing), key=f"{key}_sizing")
        direction = st.text_input("Direction", g.direction, key=f"{key}_dir")
    with c2:
        fixed_default = float(g.fixed_amount) if g.fixed_amount is not None else 0.0
        derived = st.checkbox("Fixed amount derived by the engine", value=g.fixed_amount is None, key=f"{key}_derived") if allow_fixed_none else False
        fixed = st.number_input("Fixed amount (EUR)", value=fixed_default, step=1000.0, format="%.2f", key=f"{key}_fixed", disabled=derived)
        pctv = st.number_input("% of contract value (0,3 = 30 %)", value=float(g.pct_of_contract_value), step=0.01, format="%.4f", key=f"{key}_pct")
        cov = st.number_input("Coverage months", value=float(g.coverage_months), step=0.5, format="%.2f", key=f"{key}_cov")
    with c3:
        fee = st.number_input("BGL fee p.a. (0,015 = 1,5 %)", value=float(g.bgl_fee_pa), step=0.001, format="%.4f", key=f"{key}_fee")
        fee_type = st.selectbox("BGL fee type", BGL_FEE_TYPES, index=BGL_FEE_TYPES.index(g.bgl_fee_type), key=f"{key}_feetype")
        cash = st.number_input("Cash backing share (0..1)", value=float(g.cash_backing_pct), step=0.05, format="%.2f", key=f"{key}_cash")
        start = st.date_input("Window start", g.start, key=f"{key}_start", format="DD.MM.YYYY")
        end = st.date_input("Window end", g.end, key=f"{key}_end", format="DD.MM.YYYY")
    return Guarantee(type=typ, direction=direction, sizing=sizing, fixed_amount=None if derived else float(fixed), coverage_months=float(cov),
                     pct_of_contract_value=float(pctv), bgl_fee_pa=float(fee), bgl_fee_type=fee_type, cash_backing_pct=float(cash),
                     start=start, end=end)


def _month_table(title: str, data: dict[str, list[float]], key: str, unit: str) -> dict[str, list[float]]:
    B.eyebrow(f"{title} · {unit}")
    df = pd.DataFrame({p: data[p] for p in PRODUCTS}, index=MONTHS).T
    edited = st.data_editor(df, key=key, width="stretch", column_config={m: st.column_config.NumberColumn(m, format="%.2f") for m in MONTHS})
    return {p: [float(x) for x in edited.loc[p].tolist()] for p in PRODUCTS}


def _offtaker_form(o: Offtaker, key: str) -> Offtaker:
    n = copy.deepcopy(o)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n.name = st.text_input("Display name (kept in the scenario file only)", o.name, key=f"{key}_name")
        n.active = st.toggle("Active", value=o.active, key=f"{key}_active")
    with c2:
        n.contract_start = st.date_input("Contract start", o.contract_start, key=f"{key}_cs", format="DD.MM.YYYY")
        n.contract_end = st.date_input("Contract end", o.contract_end, key=f"{key}_ce", format="DD.MM.YYYY")
    with c3:
        n.payment_terms_days = int(st.number_input("Payment terms (days)", value=int(o.payment_terms_days), min_value=0, step=1, key=f"{key}_terms"))
        n.advance_pct = float(st.number_input("Advance share (0..1)", value=float(o.advance_pct), min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key=f"{key}_adv"))
    with c4:
        n.contract_price_eur_per_mwh = float(st.number_input("Contract price (EUR/MWh)", value=float(o.contract_price_eur_per_mwh), step=0.5, format="%.2f", key=f"{key}_cp"))
        n.pv_price_budget_eur_per_mwh = float(st.number_input("PV price budget (EUR/MWh)", value=float(o.pv_price_budget_eur_per_mwh), step=0.5, format="%.2f", key=f"{key}_pvb"))
        n.pv_price_forecast_eur_per_mwh = float(st.number_input("PV price forecast (EUR/MWh)", value=float(o.pv_price_forecast_eur_per_mwh), step=0.5, format="%.2f", key=f"{key}_pvf"))
    B.eyebrow("Risk premium components (B2) · EUR/MWh")
    prem = pd.DataFrame({"Budget": [o.premium_budget[c] for c in PREMIUM_COMPONENTS], "Forecast": [o.premium_forecast[c] for c in PREMIUM_COMPONENTS]},
                        index=list(PREMIUM_COMPONENTS)).T
    edited = st.data_editor(prem, key=f"{key}_prem", width="stretch",
                            column_config={c: st.column_config.NumberColumn(c, format="%.4f") for c in PREMIUM_COMPONENTS})
    n.premium_budget = {c: float(edited.loc["Budget", c]) for c in PREMIUM_COMPONENTS}
    n.premium_forecast = {c: float(edited.loc["Forecast", c]) for c in PREMIUM_COMPONENTS}
    c1, c2 = st.columns(2)
    with c1:
        n.target_gm_budget = float(st.number_input("Target gross margin budget (EUR/MWh)", value=float(o.target_gm_budget), step=0.25, format="%.2f", key=f"{key}_gmb"))
    with c2:
        n.target_gm_forecast = float(st.number_input("Target gross margin forecast (EUR/MWh)", value=float(o.target_gm_forecast), step=0.25, format="%.2f", key=f"{key}_gmf"))
    n.strip_mw = _month_table("Baseload strips", o.strip_mw, f"{key}_strip", "MW per product and month")
    n.product_price_budget = _month_table("Product prices budget", o.product_price_budget, f"{key}_ppb", "EUR/MWh")
    n.product_price_forecast = _month_table("Product prices forecast", o.product_price_forecast, f"{key}_ppf", "EUR/MWh")
    with st.expander("Own guarantee issued to this off-taker"):
        n.guarantee = _guarantee_form(o.guarantee, f"{key}_g")
    with st.expander("Regulated tariff components for this off-taker (override of the portfolio set)"):
        use_override = st.checkbox("Use an off-taker-specific tariff set", value=o.tariff_components is not None, key=f"{key}_tov")
        if use_override:
            base = o.tariff_components or S.get().params.tariff_components
            vals = {}
            cols = st.columns(3)
            for i, (k, help_) in enumerate(TARIFF_HELP.items()):
                with cols[i % 3]:
                    vals[k] = float(st.number_input(help_, value=float(base.get(k, 0.0)), step=0.01, format="%.6f", key=f"{key}_t_{k}"))
            n.tariff_components = vals
        else:
            n.tariff_components = None
    return n


def render() -> None:
    state = S.get()
    p = state.params
    B.page_title("Parameters", "Every parameter of the Input sheet, grouped as in the Reference Case; values apply on 'Apply changes' and persist in the scenario file")

    # ---- scenario file (PSTORE) ------------------------------------------------------------
    st.markdown("## Scenario file")
    B.note(f"Loaded: <b>{state.scenario.name}</b> · version {state.scenario.version} · saved {(state.scenario.saved_at_utc or '–').replace('T', ' ')[:16]} UTC · "
           f"md5 {state.scenario.md5[:12] or '–'}. Names live in this file only; the repository holds the coded Reference Case register.")
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        new_name = st.text_input("Scenario name", state.scenario.name, key="scn_name")
        note_ = st.text_input("Version note", "", key="scn_note")
        if st.button("Save as new version"):
            state.scenario.name = new_name
            state.scenario = state.scenario.stamped(note_)
            state.add_log("scenario", f"scenario '{state.scenario.name}' saved as v{state.scenario.version}")
            st.rerun()
        st.download_button("Download scenario file", data=state.scenario.to_bytes(), file_name=state.scenario.filename, mime="application/json")
    with c2:
        up = st.file_uploader("Load a scenario file (.esb-scn.json)", type=["json"], key="scn_upload")
        if up is not None and st.button("Load scenario"):
            try:
                scn = ScenarioFile.from_bytes(up.getvalue(), up.name)
                state.scenario = scn
                state.mark_dirty(f"scenario file '{scn.name}' v{scn.version} loaded ({up.name})")
                st.rerun()
            except ValueError as e:
                B.refusal(str(e))
    with c3:
        if st.button("Reset to the Reference Case register"):
            from esb.scenario_file import reference_case

            state.scenario = reference_case()
            state.mark_dirty("parameters reset to the Reference Case register")
            st.rerun()
        if state.scenario.history:
            hist = pd.DataFrame(state.scenario.history).set_index("version")
            hist["md5"] = hist["md5"].str[:12]
            hist["saved_at_utc"] = hist["saved_at_utc"].str.replace("T", " ").str[:16]
            B.table(hist, index_label="Version", decimals=0, scroll=True)

    tabs = st.tabs(["A · General", "B · Premium and margin", "C · Off-takers", "D · Counterparties", "E · Market guarantees", "Tariffs and GC (admin)"])

    # ---- A general -------------------------------------------------------------------------
    with tabs[0]:
        g = p.general
        with st.form("form_general"):
            c1, c2, c3 = st.columns(3)
            with c1:
                scen = st.selectbox("Active price scenario", p.scenario_names, index=p.scenario_names.index(p.scenario_active) if p.scenario_active in p.scenario_names else 0)
                year = st.number_input("Spine year", value=int(p.spine_year), min_value=2020, max_value=2060, step=1)
                cs = st.date_input("Case start", g.case_start, format="DD.MM.YYYY")
                ce = st.date_input("Case end", g.case_end, format="DD.MM.YYYY")
                fx = st.number_input("FX RON per EUR", value=float(g.fx_ron_per_eur), step=0.01, format="%.4f", help="Input!C16 - Forecast Q3 2026 basis (D54)")
            with c2:
                vat = st.number_input("VAT rate (0,21 = 21 %)", value=float(g.vat_rate), step=0.01, format="%.4f", help="Legea nr. 227/2015 per workbook; source_status: unverified")
                cit = st.number_input("CIT rate (0,16 = 16 %)", value=float(g.cit_rate), step=0.01, format="%.4f", help="Legea nr. 227/2015 art. 41 per workbook label; unverified")
                tax_day = st.number_input("Tax payment day of month", value=int(g.tax_payment_day), min_value=1, max_value=28, step=1)
                rc = st.toggle("Reverse charge VAT on source purchases", value=bool(g.reverse_charge_vat_on_sources), help="art. 331 alin. (2) lit. e) Codul fiscal per workbook; adviser confirmation outstanding")
                opening = st.number_input("Opening cash (EUR)", value=float(g.opening_cash_eur), step=1000.0, format="%.2f")
            with c3:
                opex = st.number_input("Portfolio OPEX (EUR per metered MWh)", value=float(g.portfolio_opex_eur_per_mwh_metered), step=0.01, format="%.4f")
                vopex = st.number_input("Variable OPEX / sales bonus (EUR per metered MWh)", value=float(g.variable_opex_eur_per_mwh_metered), step=0.01, format="%.4f")
                shl = st.number_input("Shareholder loan rate p.a. (0,07 = 7 %)", value=float(g.shareholder_loan_rate_pa), step=0.005, format="%.4f")
                rc_cost = st.number_input("PV resell cost factor vs curtailed DAM", value=float(p.resell_pv_cost_factor), step=0.01, format="%.4f", help="Input!C9")
                rc_rev = st.number_input("PV resell revenue factor vs curtailed DAM", value=float(p.resell_pv_revenue_factor), step=0.01, format="%.4f", help="Input!C10")
            if st.form_submit_button("Apply changes", type="primary"):
                p.scenario_active = scen
                p.meta["spine_year"] = int(year)
                g.case_start, g.case_end = cs, ce
                g.fx_ron_per_eur, g.vat_rate, g.cit_rate, g.tax_payment_day = float(fx), float(vat), float(cit), int(tax_day)
                g.reverse_charge_vat_on_sources, g.opening_cash_eur = bool(rc), float(opening)
                g.portfolio_opex_eur_per_mwh_metered, g.variable_opex_eur_per_mwh_metered, g.shareholder_loan_rate_pa = float(opex), float(vopex), float(shl)
                p.resell_pv_cost_factor, p.resell_pv_revenue_factor = float(rc_cost), float(rc_rev)
                state.mark_dirty("section A (general) applied")
                st.rerun()
        B.caption("The debt facility of Input!C28:C33 is present in the register and not modelled (workbook decision D62).")

    # ---- B premium standard ------------------------------------------------------------------
    with tabs[1]:
        B.note("Section B is the reference premium set of the workbook; the engine prices with each off-taker's own components (B2, tab C).")
        with st.form("form_premium"):
            cols = st.columns(4)
            vals = {}
            keys = list(p.premium_standard.keys())
            for i, k in enumerate(keys):
                with cols[i % 4]:
                    vals[k] = st.number_input(k, value=float(p.premium_standard[k]), step=0.25, format="%.4f", key=f"ps_{k}")
            kpi_t = st.number_input("KPI target: retail NM pre-tax (EUR/MWh)", value=float(p.meta.get("kpi_retail_nm_target_eur_per_mwh", 3.0)), step=0.25, format="%.2f",
                                    help="Execution prompt section 10.1 page 10; house benchmark (workflow EW-NFR-01)")
            if st.form_submit_button("Apply changes", type="primary"):
                p.premium_standard = {k: float(v) for k, v in vals.items()}
                p.meta["kpi_retail_nm_target_eur_per_mwh"] = float(kpi_t)
                state.mark_dirty("section B (premium standard) applied")
                st.rerun()

    # ---- C off-takers --------------------------------------------------------------------------
    with tabs[2]:
        B.note("Off-takers are coded by merit-order position (OT1 = first served). Display names stay in the scenario file. "
               "Inactive off-takers contribute nothing and their strips disappear (D77).")
        codes = [o.code for o in p.offtakers]
        sub = st.tabs([f"{o.label} ({o.code})" for o in p.offtakers] + ["+ add"])
        for i, o in enumerate(p.offtakers):
            with sub[i]:
                with st.form(f"form_ot_{o.code}"):
                    new = _offtaker_form(o, f"ot_{o.code}")
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        applied = st.form_submit_button("Apply changes", type="primary")
                    with c2:
                        removed = st.form_submit_button("Remove this off-taker") if len(p.offtakers) > 1 and i == len(p.offtakers) - 1 else False
                    if applied:
                        p.offtakers[i] = new
                        state.mark_dirty(f"off-taker {o.code} applied")
                        st.rerun()
                    if removed:
                        p.offtakers.pop(i)
                        state.mark_dirty(f"off-taker {o.code} removed")
                        st.rerun()
        with sub[-1]:
            B.caption("A new off-taker takes the last merit-order position and starts from a copy of the last one (inactive until switched on).")
            if st.button("Add off-taker"):
                base = copy.deepcopy(p.offtakers[-1])
                base.code = f"OT{len(codes) + 1}"
                base.name, base.active = "", False
                base.strip_mw = {k: [0.0] * 12 for k in PRODUCTS}
                p.offtakers.append(base)
                state.mark_dirty(f"off-taker {base.code} added (inactive)")
                st.rerun()

    # ---- D counterparties ------------------------------------------------------------------
    with tabs[3]:
        B.note("Forward Source 1 (PV), Forward Source 2 (Baseload), Spot (OPCOM), BRP, TSO and DSO: activity, payment terms, advances and guarantees.")
        labels = {"pv": "PV source", "baseload": "Baseload source", "spot": "Spot market", "brp": "BRP / imbalance", "tso": "TSO", "dso": "DSO"}
        sub = st.tabs([labels[k] for k in p.counterparties])
        for tab, (k, c) in zip(sub, p.counterparties.items(), strict=True):
            with tab, st.form(f"form_cp_{k}"):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    name = st.text_input("Display name", c.name, key=f"cp_{k}_name")
                    active = st.toggle("Active", value=c.active, key=f"cp_{k}_active")
                with c2:
                    terms = st.number_input("Payment terms (days)", value=int(c.payment_terms_days), min_value=0, step=1, key=f"cp_{k}_terms")
                    adv = st.number_input("Advance share (0..1)", value=float(c.advance_pct), min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key=f"cp_{k}_adv")
                with c3:
                    kk = st.selectbox("Sign convention k", ["not applicable", "-1 (DSO)", "+1 (BRP)"],
                                      index={None: 0, -1: 1, 1: 2}.get(c.k, 0), key=f"cp_{k}_k", help="Declared, never inferred (rule 3)")
                with c4:
                    dev = st.number_input("Deviation share (baseload)", value=float(c.deviation_pct), step=0.01, format="%.4f", key=f"cp_{k}_dev", disabled=k != "baseload")
                    idev = st.number_input("Imbalance deviation share (baseload)", value=float(c.imbalance_deviation_pct), step=0.01, format="%.4f", key=f"cp_{k}_idev", disabled=k != "baseload")
                B.eyebrow("Guarantee")
                gg = _guarantee_form(c.guarantee, f"cp_{k}_g", allow_fixed_none=(k == "pv"))
                if st.form_submit_button("Apply changes", type="primary"):
                    c.name, c.active, c.payment_terms_days, c.advance_pct = name, bool(active), int(terms), float(adv)
                    c.k = {"not applicable": None, "-1 (DSO)": -1, "+1 (BRP)": 1}[kk]
                    c.deviation_pct, c.imbalance_deviation_pct, c.guarantee = float(dev), float(idev), gg
                    state.mark_dirty(f"counterparty {k} applied")
                    st.rerun()

    # ---- E market guarantees -----------------------------------------------------------------
    with tabs[4]:
        mg = p.market_guarantees
        B.note("Regulatory sizing formulas. The citations quoted in the workbook (Transelectrica PO 01.13, ANRE Order 129/2015, BRP rule) are "
               "carried with source_status: unverified until checked against the primary source (docs/PARAMETERS.md section 3).")
        with st.form("form_mg"):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                buf = st.number_input("Spot buffer days", value=float(mg["spot"]["buffer_days"]), step=1.0, format="%.0f", help="Input!C111")
            with c2:
                rate = st.number_input("BRP rate (RON per MW)", value=float(mg["brp"]["rate_ron_per_mw"]), step=100.0, format="%.2f", help="Input!C116 - unverified")
                gen = st.number_input("Generation MW in the BRP", value=float(mg["brp"]["generation_mw_in_brp"]), step=1.0, format="%.2f", help="Input!C117")
            with c3:
                vtm = st.number_input("TSO multiplier Vtm", value=float(mg["tso"]["vtm_multiplier"]), step=0.5, format="%.2f", help="Input!C121 - unverified")
            with c4:
                vdm = st.number_input("DSO multiplier Vdm", value=float(mg["dso"]["vdm_multiplier"]), step=0.5, format="%.2f", help="Input!C125 - unverified")
                addon = st.number_input("DSO overdue add-on (EUR)", value=float(mg["dso"]["overdue_addon_eur"]), step=100.0, format="%.2f", help="Input!C126")
            if st.form_submit_button("Apply changes", type="primary"):
                mg["spot"]["buffer_days"] = float(buf)
                mg["brp"]["rate_ron_per_mw"], mg["brp"]["generation_mw_in_brp"] = float(rate), float(gen)
                mg["tso"]["vtm_multiplier"], mg["dso"]["vdm_multiplier"], mg["dso"]["overdue_addon_eur"] = float(vtm), float(vdm), float(addon)
                state.mark_dirty("section E (market guarantees) applied")
                st.rerun()

    # ---- admin: tariffs and GC -------------------------------------------------------------------
    with tabs[5]:
        B.note("Admin layer - regulated constants with their workbook origin and verification status. A change here is a change of the "
               "scenario, never of the code (rule 1). Vintage of the Reference Case values: Forecast Q3 2026 basis (D55 / D60); single DEER MV set (X-03).")
        with st.form("form_tariffs"):
            cols = st.columns(3)
            vals = {}
            for i, (k, help_) in enumerate(TARIFF_HELP.items()):
                with cols[i % 3]:
                    vals[k] = st.number_input(help_, value=float(p.tariff_components[k]), step=0.01, format="%.6f", key=f"tar_{k}")
            c1, c2, c3 = st.columns(3)
            with c1:
                quota = st.number_input("GC quota (GC per MWh)", value=float(p.gc_quota), step=0.001, format="%.6f", help="Cons_P&L!B6 - unverified")
            with c2:
                gcp = st.number_input("GC reference price (RON per GC)", value=float(p.gc_reference_price_ron), step=0.01, format="%.4f", help="Cons_P&L!B7")
            with c3:
                share = st.number_input("GC spot share (0..1)", value=float(p.gc_spot_share), step=0.05, format="%.2f", help="Cons_P&L!B9")
            if st.form_submit_button("Apply changes", type="primary"):
                p.tariff_components = {k: float(v) for k, v in vals.items()}
                p.gc_quota, p.gc_reference_price_ron, p.gc_spot_share = float(quota), float(gcp), float(share)
                state.mark_dirty("tariffs and GC (admin) applied")
                st.rerun()
        B.kpi_row([
            ("GC unit cost", p.gc_unit_cost, "EUR/MWh", "quota x reference price / FX (Input!C60)"),
            ("Regulated pass-through total", p.tariff_total, "EUR/MWh", "Input!C64 = sum of the components incl. GC"),
            ("TSO basis (TL + SS)", p.tso_tariff, "EUR/MWh", "Input!C122"),
            ("DSO basis (HV + MV + LV)", p.dso_tariff, "EUR/MWh", "Input!C127"),
        ])

    errs = p.validate()
    if errs:
        B.refusal("Register problems: " + "; ".join(errs))
    if isinstance(p.general.case_start, date) and p.general.case_start.year != p.spine_year:
        B.refusal(f"Case start {B.dmy(p.general.case_start)} is outside the spine year {p.spine_year}.")

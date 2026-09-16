"""Page 10 - Pricing / Bid (replaces Pricing_Calc): the offer build-up per off-taker and the manual
case for a new off-taker; re-pricing at the current forecast; the retail NM KPI against target."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import brand as B
from app import state as S
from config.schema import PREMIUM_COMPONENTS
from esb.labels import RETAIL, label, tagged, unit_of
from esb.layout import roles_for
from esb.pricing import PRICING_ROWS, YEAR_ONLY, manual_case

BUILDUP = [("purchase_price", "Purchase price"), ("imbalance_cost", "Imbalance cost"), ("premium_total", "Risk premium"), ("target_gm", "Target GM"),
           ("passthrough", "Regulated pass-through")]


def render() -> None:
    state = S.get()
    B.page_title("Pricing / Bid", "The offer build-up of the Pricing_Calc sheet for every off-taker, the manual case for a new one, and re-pricing at the current forecast")
    r = S.require_result(state)
    if r is None:
        B.refusal(state.last_error or "No run available.")
        return
    p = r.params
    codes = [o.code for o in p.offtakers]
    c1, c2 = st.columns([1, 3])
    with c1:
        code = st.selectbox("Off-taker", codes, index=codes.index(r.selected_offtaker) if r.selected_offtaker in codes else 0,
                            format_func=lambda c: f"{p.offtaker(c).label} ({c})")
        if code != state.selected_offtaker:
            state.selected_offtaker = code
            state.mark_dirty()
            st.rerun()
    pr = r.pricing[code]
    y = pr.year
    target = float(p.meta.get("kpi_retail_nm_target_eur_per_mwh", 3.0))
    T = r.pnl.sections[code]
    nm_spec = T.y("retail_nm_forecast_specific")
    with c2:
        B.kpi_row([
            (tagged("Energy price", RETAIL), y["energy_price"], "EUR/MWh", f"Physical cost {B.num(y['physical_cost'], 2)} + premium {B.num(y['premium_total'], 2)} + target GM {B.num(y['target_gm'], 2)}"),
            (tagged("Offer excl. VAT", RETAIL), y["offer_ex_vat"], "EUR/MWh", f"Incl. VAT {B.num(y['offer_incl_vat'], 2)} EUR/MWh"),
            (tagged("Contract minus offer", RETAIL), y["contract_minus_offer"], "EUR/MWh", f"Contract price {B.num(y['contract_price'], 2)}; implied GM {B.num(y['implied_gm'], 2)}"),
            (tagged("NM pre-tax forecasted (specific)", RETAIL), nm_spec, "EUR/MWh", f"Target {B.num(target, 2)} EUR/MWh · " + ("meets target" if nm_spec >= target else "below target")),
        ])
    st.markdown(f"KPI - retail NM pre-tax at least {B.num(target, 2)} EUR/MWh: {B.status(nm_spec >= target, 'MET', 'NOT MET')}", unsafe_allow_html=True)
    B.note("Cost to serve (C39) and the forecast premium (C46) use the corrected logic of decisions D87 / D88; the workbook's cached values are "
           "reproduced only for the parity tie-out. Only the position-3 view is cached in the workbook (D94); all views run the same code.")

    st.markdown("## Build-up · year · EUR/MWh")
    labels = [lbl for _, lbl in BUILDUP]
    values = [float(y[k]) for k, _ in BUILDUP]
    st.plotly_chart(B.waterfall_bars(labels, values), width="stretch", config={"displayModeBar": False})
    B.caption("Purchase price + imbalance cost = physical cost; + premium + target GM = energy price; + pass-through = offer excl. VAT. Source: Pricing_Calc column C")

    st.markdown("## Year and months")
    rows = {}
    for k in PRICING_ROWS:
        if k in y:
            months = pr.months.get(k)
            rows[k] = [y[k], *(list(months) if months is not None and k not in YEAR_ONLY else [float("nan")] * 12)]
    df = pd.DataFrame(rows, index=["Year", *B.MONTH_EN]).T
    units = [unit_of(k, "pricing") for k in df.index]
    roles = [roles_for("Pricing_<code>").get(k, "data") for k in df.index]
    df.index = [f"{label(k, leg=RETAIL)} [{PRICING_ROWS[k]}]" for k in df.index]
    B.table(df, index_label="Line [row]", scroll=True, units=units, roles=roles)
    B.caption("Unit per line in the Unit column; row numbers of the Reference Case sheet; year-only lines have no monthly values")

    st.markdown("## Re-pricing at the current forecast")
    B.kpi_row([
        (tagged("Forecast premium (C46, corrected)", RETAIL), y["forecast_premium"], "EUR/MWh", f"Derived forecast premium {B.num(y['derived_forecast_premium'], 2)} EUR/MWh"),
        (tagged("Re-priced energy price", RETAIL), y["repriced_energy_price"], "EUR/MWh", "Physical cost forecast + forecast premium + target GM forecast"),
        (tagged("Contract minus re-priced", RETAIL), y["contract_minus_repriced"], "EUR/MWh", "Negative: the contract price is below today's re-priced level"),
        (tagged("Reserve release (budget / forecast)", RETAIL), f"{B.num(y['reserve_release_budget'], 0)} / {B.num(y['reserve_release_forecast'], 0)}", "EUR", "Released at contract end"),
    ])

    st.markdown("## Manual case - a new off-taker or a what-if")
    B.caption("Column D of the sheet: overrides with blank fallback to the year column. Leave a field at the year value to keep it.")
    with st.form("manual_case"):
        c1, c2, c3 = st.columns(3)
        with c1:
            m_metered = B.num_input("Metered volume (MWh)", value=float(y["metered"]), decimals=2)
            m_pp = B.num_input("Purchase price (EUR/MWh)", value=float(y["purchase_price"]), decimals=4)
            m_imb = B.num_input("Imbalance cost (EUR/MWh)", value=float(y["imbalance_cost"]), decimals=4)
        with c2:
            prem = {}
            for c in PREMIUM_COMPONENTS[:4]:
                prem[c] = B.num_input(f"Premium {c}", value=float(y[f"premium_{c}"]), key=f"mc_{c}", decimals=4)
        with c3:
            for c in PREMIUM_COMPONENTS[4:]:
                prem[c] = B.num_input(f"Premium {c}", value=float(y[f"premium_{c}"]), key=f"mc_{c}", decimals=4)
            m_gm = B.num_input("Target GM (EUR/MWh)", value=float(y["target_gm"]), decimals=4)
            m_pt = B.num_input("Pass-through (EUR/MWh)", value=float(y["passthrough"]), decimals=4)
        if st.form_submit_button("Compute the manual case", type="primary"):
            manual = {"metered": m_metered, "purchase_price": m_pp, "imbalance_cost": m_imb, "target_gm": m_gm, "passthrough": m_pt,
                      **{f"premium_{c}": v for c, v in prem.items()}}
            d = manual_case(pr, manual)
            st.session_state["manual_result"] = (code, d)
            state.add_log("pricing", f"manual case computed for {code}: energy price {B.num(d['energy_price'], 2)} EUR/MWh")
    mr = st.session_state.get("manual_result")
    if mr and mr[0] == code:
        d = mr[1]
        B.kpi_row([
            (tagged("Energy price (manual)", RETAIL), d["energy_price"], "EUR/MWh", f"Physical {B.num(d['physical_cost'], 2)} + premium {B.num(d['premium_total'], 2)} + GM {B.num(d['target_gm'], 2)}"),
            (tagged("Offer excl. VAT (manual)", RETAIL), d["offer_ex_vat"], "EUR/MWh", f"Incl. VAT {B.num(d['offer_incl_vat'], 2)}"),
            (tagged("Indicative NM (manual)", RETAIL), d["indicative_nm"], "EUR/MWh", f"Cost to serve of the base case {B.num(y['cost_to_serve'], 2)} EUR/MWh"),
            (tagged("Annual revenue (manual)", RETAIL), d["annual_revenue"], "EUR", f"{B.num(d['metered'], 0)} MWh x energy price"),
        ])
        cmp = pd.DataFrame({"Year column": [y.get(k, float("nan")) for k in d], "Manual case": list(d.values())}, index=[label(k, leg=RETAIL) for k in d])
        cmp["Delta"] = cmp["Manual case"] - cmp["Year column"]
        B.table(cmp, index_label="Line", units=[unit_of(k, "pricing") for k in d])

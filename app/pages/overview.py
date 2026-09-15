"""Page 1 - Overview (replaces Portf Overview)."""

from __future__ import annotations

import numpy as np
import streamlit as st

from app import brand as B
from app import state as S
from esb import __version__
from esb.labels import RESELL, RETAIL, TOTAL, label, tagged, unit_of

OVERVIEW_ORDER = [
    "notified", "metered", "pv_buy_notified", "bl_buy_notified", "spot_buy_notified", "share_pv_bl", "price_pv", "price_bl", "price_spot",
    "price_total", "sell_price", "market_ref", "revenue", "resell_revenue", "total_revenue", "cost_pv", "cost_bl", "cost_spot", "cost_total",
    "resell_cost", "total_cost", "gm1", "gm1_pct", "gm1_specific", "resell_gm1", "resell_gm1_pct", "source_imb", "offtaker_imb",
    "resell_source_imb", "gm2", "gm2_pct", "gm2_specific", "resell_gm2", "resell_gm2_pct", "total_gm2", "total_gm2_pct", "total_gm2_specific",
    "premium", "reserve", "reserve_release", "passthrough", "opex", "variable_opex", "market_bgl", "offtaker_bgl", "guarantees", "interest",
    "unallocated", "nm", "nm_pct", "nm_specific", "retail_nm", "retail_nm_specific", "resell_nm", "cit", "nm_after_tax", "nm_after_tax_pct",
    "nm_after_tax_specific", "total_gm1", "total_gm1_pct", "total_gm1_specific",
]


def render() -> None:
    state = S.get()
    B.page_title("Overview", "Year KPIs, margins by leg and the engine's own checks - the Portf Overview sheet of the Reference Case")
    r = S.require_result(state)
    if r is None:
        B.refusal(state.last_error or "No run yet. Load series on the Data page and set parameters on the Parameters page.")
        return
    P = r.pnl.portfolio
    p = r.params
    B.note(f"Scenario file <b>{state.scenario.name}</b> v{state.scenario.version} · price scenario <b>{p.scenario_active}</b> · "
           f"year {p.spine_year} · series: {'Reference Case fixture' if state.use_reference_fixture else 'uploads only'}"
           f"{' + ' + str(len(state.uploads)) + ' upload(s)' if state.uploads else ''} · engine trace {B.num(r.trace[-1][1], 2)} s")

    B.eyebrow("Year - forecast case")
    B.kpi_row([
        (tagged("Revenue", RETAIL), P.y("revenue"), "EUR", "Metered volume x contract prices"),
        (tagged("GM2 forecasted", TOTAL), P.y("t_gm2_forecast"), "EUR", f"Budgeted {B.num(P.y('t_gm2_budget'), 0)} EUR"),
        (tagged("Net margin pre-tax forecasted", TOTAL), P.y("nm_forecast"), "EUR", f"Budgeted {B.num(P.y('nm_budget'), 0)} EUR"),
        (tagged("Net margin after tax forecasted", TOTAL), P.y("nm_forecast_after_tax"), "EUR", f"CIT {B.num(P.y('cit_forecast'), 0)} EUR"),
    ])
    st.write("")
    B.kpi_row([
        (tagged("Metered volume", RETAIL), P.y("metered"), "MWh", f"Notified {B.num(P.y('notified'), 0)} MWh"),
        (tagged("Guarantees outstanding (peak)", TOTAL), P.y("guarantees_outstanding"), "EUR", "Counterparties plus own guarantees"),
        (tagged("Peak funding (daily ledger)", TOTAL), r.cashflow.daily_summary.get("peak_funding", np.nan), "EUR",
         f"Monthly view {B.num(r.cashflow.y('peak_funding'), 0)} EUR"),
        (tagged("Financing interest", TOTAL), P.y("interest"), "EUR", f"Shareholder loan at {B.pct(p.general.shareholder_loan_rate_pa)} p.a."),
    ])

    st.markdown("## Margins by leg")
    B.eyebrow("Year values · budget vs forecast · unit per line")
    ov = r.overview.table.copy()
    keep = [k for k in OVERVIEW_ORDER if k in ov.index]
    ov = ov.loc[keep]
    ren = {"portfolio_budget": "Portfolio budget", "portfolio_forecast": "Portfolio forecast", "delta": "Delta", "resell": RESELL}
    for o in p.offtakers:
        ren[f"{o.code}_budget"] = f"{o.label} budget"
        ren[f"{o.code}_forecast"] = f"{o.label} forecast"
    ov = ov.rename(columns=ren)
    ov.index = [label(k) for k in ov.index]
    B.table(ov, decimals=2, index_label="Line", pct_rows={label(k) for k in keep if k.endswith("_pct")},
            total_rows={label("total_gm2"), label("nm")}, scroll=True, units=[unit_of(k) for k in keep])
    B.caption(f"Source: engine v{__version__} on the loaded series; delta = forecast - budget; percentages of revenue")

    c1, c2 = st.columns(2)
    with c1:
        B.eyebrow(f"GM2 by off-taker · forecast · EUR · {RETAIL}")
        names = [o.label for o in p.offtakers]
        vals = [r.pnl.sections[o.code].y("gm2_forecast") for o in p.offtakers]
        st.plotly_chart(B.bars(names, {"GM2 forecast": np.array(vals), "GM2 budget": np.array([r.pnl.sections[o.code].y("gm2_budget") for o in p.offtakers])}, y_title="EUR"),
                        width="stretch", config={"displayModeBar": False})
        B.caption("Retail GM2 per off-taker section, year; EUR. Source: Cons_P&L sections")
    with c2:
        B.eyebrow(f"Sourcing mix · notified buys · MWh · {RETAIL}")
        st.plotly_chart(B.donut(["PV", "Baseload", "Spot"], [P.y("pv_buy_notified"), P.y("bl_buy_notified"), P.y("spot_buy_notified")]),
                        width="stretch", config={"displayModeBar": False})
        B.caption("Share of notified retail buys by source, year; MWh. Source: QH_P&L merit order")

    st.markdown("## Cash flow")
    cf = r.overview.cashflow
    B.kpi_row([
        (tagged("Receipts", TOTAL), cf["inflows_year"], "EUR", f"Beyond December {B.num(cf['inflows_beyond'], 0)} EUR"),
        (tagged("Payments", TOTAL), cf["outflows_year"], "EUR", f"Beyond December {B.num(cf['outflows_beyond'], 0)} EUR"),
        (tagged("Injections", TOTAL), cf["injections_year"], "EUR", "Shareholder loan drawn in the year"),
        (tagged("Closing cash December", TOTAL), cf["closing_dec"], "EUR", f"Free cash after tax {B.num(cf['free_cash_after_tax_dec'], 0)} EUR"),
    ])

    st.markdown("## Pricing - selected off-taker")
    sel = p.offtaker(r.selected_offtaker)
    pr = r.overview.pricing
    B.kpi_row([
        (tagged(f"Energy price - {sel.label}", RETAIL), pr["energy_price"], "EUR/MWh", f"Physical cost {B.num(pr['physical_cost'], 2)} + premium and margin {B.num(pr['premium_plus_gm'], 2)}"),
        (tagged("Offer excl. VAT", RETAIL), pr["offer_ex_vat"], "EUR/MWh", "Energy price plus regulated pass-through"),
        (tagged("Contract minus offer", RETAIL), pr["contract_minus_offer"], "EUR/MWh", f"Contract price {B.num(pr['contract_price'], 2)} EUR/MWh"),
        (tagged("Re-priced energy price", RETAIL), pr["repriced_energy_price"], "EUR/MWh", "At the forecast premium (corrected C46, D87)"),
    ])

    st.markdown("## Checks and tripwires")
    B.eyebrow("Section 7 of the overview - every check is shown, none is hidden")
    checks = r.overview.checks
    rows = []
    for k, v in checks.items():
        if k == "label_self_check":
            ok, shown = True, "not applicable (rows addressed by key)"
        elif k == "strip_price_tripwire":
            ok, shown = v == 0, f"{int(v)} strip(s) with MW > 0 and a 0 product price"
        else:
            ok, shown = abs(v) < 1e-6, B.num(v, 6)
        rows.append((label(k, leg=""), shown, B.status(ok)))
    html = '<table class="esb"><thead><tr><th>Check</th><th class="txt">Unit</th><th>Value</th><th>State</th></tr></thead><tbody>'
    html += "".join(f'<tr><td>{a}</td><td class="unit">{"EUR" if a not in (label("label_self_check", leg=""), label("strip_price_tripwire", leg="")) else "–"}</td><td>{b}</td><td>{c}</td></tr>' for a, b, c in rows)
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)
    all_ok = all(abs(v) < 1e-6 for k, v in checks.items() if k not in ("label_self_check", "strip_price_tripwire")) and checks["strip_price_tripwire"] == 0
    st.markdown(f"Overall: {B.status(all_ok, 'ALL CHECKS PASS', 'A CHECK FAILS - do not use these figures')}", unsafe_allow_html=True)
    if state.dirty:
        B.note("Inputs changed since this run; the figures above are from the previous inputs until the engine runs again.")

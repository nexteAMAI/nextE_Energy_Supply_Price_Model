"""Page 7 - P&L (replaces Cons_P&L): monthly, by leg, by off-taker; cost-to-serve; CIT at portfolio level."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app import brand as B
from app import state as S
from esb.labels import RESELL, RETAIL, TOTAL, label, tagged
from esb.pnl import SECTION_ROWS, WORKBOOK_ROWS

GROUPS = {
    "Volumes and sourcing · Retail": ["notified", "metered", "pv_buy_notified", "bl_buy_notified", "spot_buy_notified", "buy_notified", "pv_buy_metered",
                             "bl_buy_metered", "spot_settlement", "buy_metered", "share_pv", "share_bl", "share_spot"],
    "Costs and prices · Retail": ["price_pv_budget", "price_bl_budget", "price_spot", "price_total_budget", "cost_pv_budget", "cost_bl_budget", "cost_spot", "cost_budget",
                         "price_pv_forecast", "price_bl_forecast", "price_spot_f", "price_total_forecast", "cost_pv_forecast", "cost_bl_forecast", "cost_spot_f",
                         "cost_forecast", "budget_minus_forecast_cost"],
    "Revenue and margins · Retail": ["sell_price", "revenue", "gm1_budget", "gm1_budget_pct", "gm1_budget_specific", "gm1_forecast", "gm1_forecast_pct",
                                   "gm1_forecast_specific", "source_imb", "offtaker_imb", "gm2_budget", "gm2_budget_pct", "gm2_budget_specific",
                                   "gm2_forecast", "gm2_forecast_pct", "gm2_forecast_specific"],
    "Wholesale spot resell": ["rs_volume", "rs_volume_mw", "rs_cost_forecast", "rs_cost_budget", "rs_revenue", "rs_source_imb", "rs_gm1_budget",
                         "rs_gm1_forecast", "rs_gm2_budget", "rs_gm2_forecast", "rs_pv_volume", "rs_pv_cost", "rs_pv_revenue", "rs_bl_volume",
                         "rs_bl_cost_forecast", "rs_bl_cost_budget", "rs_bl_revenue"],
    "Totals · Total": ["t_buy_notified", "t_buy_metered", "t_cost_budget", "t_cost_forecast", "t_revenue", "t_imb", "t_gm2_budget", "t_gm2_budget_pct",
               "t_gm2_budget_specific", "t_gm2_forecast", "t_gm2_forecast_pct", "t_gm2_forecast_specific"],
    "Cost to serve and financing · Total": ["passthrough_revenue", "passthrough_cost", "reserve", "opex", "variable_opex", "offtaker_bgl_fees",
                                    "market_bgl_fees", "guarantees_outstanding", "interest", "unallocated", "reserve_release_budget",
                                    "reserve_release_forecast", "reserve_balance"],
    "Net margin and tax · Total": ["nm_budget", "nm_budget_pct", "nm_budget_specific", "nm_forecast", "nm_forecast_pct", "nm_forecast_specific", "cit_budget",
                           "nm_budget_after_tax", "nm_budget_after_tax_pct", "nm_budget_after_tax_specific", "cit_forecast", "nm_forecast_after_tax",
                           "nm_forecast_after_tax_pct", "nm_forecast_after_tax_specific"],
    "Green certificates (memo) · Retail": ["gc_count", "gc_spot", "gc_bilateral", "gc_value", "gc_unit"],
    "Checks": ["check_demand", "check_pv", "check_bl", "check_imb", "check_sections_buy", "check_sections_cost", "check_legs_budget",
               "check_legs_forecast", "check_sections_nm_budget", "check_sections_nm_forecast"],
}
SECTION_GROUPS = {
    "Volumes and sourcing": ["notified", "metered", "net_imbalance_volume", "pv_buy_notified", "bl_buy_notified", "spot_buy_notified", "buy_notified",
                             "pv_buy_metered", "bl_buy_metered", "spot_settlement", "buy_metered", "share_pv", "share_bl", "share_spot"],
    "Costs and prices": ["price_pv_budget", "price_bl_budget", "price_spot", "price_total_budget", "cost_pv_budget", "cost_bl_budget", "cost_spot",
                         "cost_budget", "price_pv_forecast", "price_bl_forecast", "price_spot_f", "price_total_forecast", "cost_pv_forecast",
                         "cost_bl_forecast", "cost_spot_f", "cost_forecast", "budget_minus_forecast_cost", "budget_minus_forecast_price"],
    "Risk premium and reserve": ["premium_monthly", "premium_cumulative", "premium_settlement", "premium_settlement_cumulative", "premium_deviation",
                                 "premium_budget_input", "premium_forecast_input", "premium_forecast_derived", "reserve_release_budget", "reserve_release_forecast"],
    "Revenue and margins": ["sell_price", "revenue", "gm1_budget", "gm1_budget_pct", "gm1_budget_specific", "gm1_forecast", "gm1_forecast_pct",
                            "gm1_forecast_specific", "source_imb", "offtaker_imb", "gm2_budget", "gm2_budget_pct", "gm2_budget_specific", "gm2_forecast",
                            "gm2_forecast_pct", "gm2_forecast_specific"],
    "Cost to serve and net margin": ["passthrough_revenue", "passthrough_cost", "reserve", "opex", "variable_opex", "own_guarantee", "own_bgl_fee",
                                     "memo_bgl_share", "memo_interest_share", "retail_nm_budget", "retail_nm_budget_pct", "retail_nm_budget_specific",
                                     "retail_nm_forecast", "retail_nm_forecast_pct", "retail_nm_forecast_specific"],
    "Green certificates and remaining sources": ["gc_count", "gc_spot", "gc_bilateral", "gc_value", "gc_unit", "pv_remaining", "bl_remaining"],
}


def _frame(T, keys, rowmap: dict | None = None, position: int | None = None) -> pd.DataFrame:
    rows = {}
    for k in keys:
        if k in T:
            rows[k] = T[k]
    df = pd.DataFrame(rows, index=[*B.MONTH_EN, "Year"]).T
    labels = []
    for k in df.index:
        ref = ""
        if rowmap is not None and k in rowmap:
            ref = f" [{rowmap[k]}]"
        elif position is not None and k in SECTION_ROWS:
            from esb.pnl import section_row

            ref = f" [{section_row(position, k)}]"
        labels.append((label(k, leg=RETAIL) if position is not None else label(k)) + ref)
    df.index = labels
    return df


def render() -> None:
    state = S.get()
    B.page_title("P&L", "Monthly consolidated P&L: portfolio block, wholesale resell, totals, cost to serve, net margin and CIT; sections per off-taker")
    r = S.require_result(state)
    if r is None:
        B.refusal(state.last_error or "No run available.")
        return
    P = r.pnl.portfolio
    p = r.params
    B.kpi_row([
        (tagged("GM2 forecasted", TOTAL), P.y("t_gm2_forecast"), "EUR", f"{B.num(P.y('t_gm2_forecast_specific'), 2)} EUR/MWh bought"),
        (tagged("Cost to serve", TOTAL), P.y("opex") + P.y("variable_opex") + P.y("offtaker_bgl_fees") + P.y("market_bgl_fees") + P.y("interest"), "EUR",
         "OPEX + variable OPEX + BGL fees + interest"),
        (tagged("Net margin pre-tax forecasted", TOTAL), P.y("nm_forecast"), "EUR", f"{B.pct(P.y('nm_forecast_pct'))} of revenue · {B.num(P.y('nm_forecast_specific'), 2)} EUR/MWh"),
        (tagged("CIT forecasted", TOTAL), P.y("cit_forecast"), "EUR", f"Rate {B.pct(p.general.cit_rate, 0)} on cumulative year-to-date NM, floored at 0, quarterly"),
    ])

    st.markdown("## Margin build-up by month · forecast case · EUR")
    m = {tagged("GM1", RETAIL): P.m("gm1_forecast"), tagged("GM2", RETAIL): P.m("gm2_forecast"), tagged("GM2", RESELL): P.m("rs_gm2_forecast"),
         tagged("NM pre-tax", TOTAL): P.m("nm_forecast")}
    st.plotly_chart(B.bars(B.MONTH_EN, m, y_title="EUR"), width="stretch", config={"displayModeBar": False})
    B.caption("Cons_P&L rows 85, 96, 156 and 228 by month; EUR")

    st.markdown("## Portfolio")
    which = st.multiselect("Blocks", list(GROUPS), default=["Revenue and margins · Retail", "Totals · Total", "Cost to serve and financing · Total", "Net margin and tax · Total"])
    for name in which:
        B.eyebrow(f"{name} · workbook row in brackets")
        df = _frame(P, GROUPS[name], WORKBOOK_ROWS)
        B.table(df, index_label="Line", scroll=len(df) > 14, pct_rows={i for i in df.index if "%" in i or "pct" in i})
    B.caption("EUR unless the label says MWh, MW, EUR/MWh or %; year column per the workbook rule (sum, last, max or mean)")

    st.markdown("## Off-taker sections")
    code = st.selectbox("Off-taker", [o.code for o in p.offtakers], format_func=lambda c: f"{p.offtaker(c).label} ({c})")
    T = r.pnl.sections[code]
    pos = int(code.replace("OT", "")) if code.startswith("OT") and code[2:].isdigit() else None
    o = p.offtaker(code)
    B.kpi_row([
        (tagged("Metered volume", RETAIL), T.y("metered"), "MWh", f"Contract price {B.num(o.contract_price_eur_per_mwh, 2)} EUR/MWh"),
        (tagged("GM2 forecasted", RETAIL), T.y("gm2_forecast"), "EUR", f"{B.num(T.y('gm2_forecast_specific'), 2)} EUR/MWh"),
        (tagged("NM pre-tax forecasted", RETAIL), T.y("retail_nm_forecast"), "EUR", f"{B.num(T.y('retail_nm_forecast_specific'), 2)} EUR/MWh"),
        (tagged("Own guarantee (peak)", RETAIL), float(np.max(T.m("own_guarantee"))) if "own_guarantee" in T else 0.0, "EUR", f"BGL fee {B.num(T.y('own_bgl_fee'), 0)} EUR"),
    ])
    for name, keys in SECTION_GROUPS.items():
        with st.expander(name, expanded=name == "Revenue and margins"):
            df = _frame(T, keys, position=pos)
            B.table(df, index_label="Line", pct_rows={i for i in df.index if "%" in i})
    B.caption("Section rows are addressed relative to the position-1 anchor (row 266, pitch 136); references in brackets are the Reference Case layout")

    st.markdown("## Checks")
    df = _frame(P, GROUPS["Checks"], WORKBOOK_ROWS)
    ok = bool((df["Year"].abs() < 1e-6 * max(1.0, abs(P.y("t_revenue")))).all())
    st.markdown(f"All P&L checks: {B.status(ok)}", unsafe_allow_html=True)
    B.table(df[["Year"]], index_label="Check", decimals=6)

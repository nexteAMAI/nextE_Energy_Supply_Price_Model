"""Page 8 - Cash Flow (replaces CF_Mth + CF_Daily_Ledger): monthly and daily, working capital,
shareholder loan, restricted cash; the calculation-order trace in place of an iteration count (G0-D6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app import brand as B
from app import state as S
from esb.cashflow import CF_ROWS
from esb.labels import TOTAL, label, tagged

ROWMAP = {v: k for k, v in CF_ROWS.items()}
GROUPS = {
    "Accruals": ["acc_resell_revenue", "acc_imbalance", "acc_pv_purchases", "acc_bl_purchases", "acc_spot_purchases", "acc_grid_cost", "acc_opex",
                 "acc_variable_opex", "acc_bgl_fees", "acc_reserve", "acc_cit", "acc_guarantees"],
    "Receipts": ["in_resell", "in_imbalance", "in_total"],
    "Payments": ["out_pv", "out_bl", "out_spot", "out_grid", "out_opex", "out_variable_opex", "out_bgl", "out_cit", "out_total"],
    "VAT": ["vat_output", "vat_input", "vat_net_position", "vat_paid", "vat_credit", "vat_cash"],
    "Financing": ["net_cf_before_tax", "opening", "restricted_reserve", "restricted_collateral", "injection", "closing", "free_cash",
                  "loan_outstanding", "interest", "interest_cumulative", "tax_paid_cumulative", "free_cash_after_tax", "peak_funding"],
}


def _frame(cf, keys) -> pd.DataFrame:
    rows = {k: cf.rows[k] for k in keys if k in cf.rows}
    df = pd.DataFrame(rows, index=[*B.MONTH_EN, "Beyond Dec", "Year"]).T
    df.index = [label(k, fallback=TOTAL) + (f" [{ROWMAP[k]}]" if k in ROWMAP else "") for k in df.index]
    return df


def render() -> None:
    state = S.get()
    B.page_title("Cash Flow", "Monthly settlement of every P&L line on its payment terms, VAT, restricted cash and the shareholder loan; then the daily ledger")
    r = S.require_result(state)
    if r is None:
        B.refusal(state.last_error or "No run available.")
        return
    cf = r.cashflow
    p = r.params
    ds = cf.daily_summary
    B.kpi_row([
        (tagged("Peak funding - monthly view", TOTAL), cf.y("peak_funding"), "EUR", "Maximum shareholder loan outstanding, month ends"),
        (tagged("Peak funding - daily ledger", TOTAL), ds.get("peak_funding", np.nan), "EUR", f"Daily minus monthly {B.num(ds.get('peak_vs_monthly', np.nan), 0)} EUR"),
        (tagged("Financing interest", TOTAL), cf.y("interest"), "EUR", f"Daily basis {B.num(ds.get('interest_daily_basis', np.nan), 0)} EUR · rate {B.pct(p.general.shareholder_loan_rate_pa)}"),
        (tagged("Cash trough", TOTAL), ds.get("cash_trough", np.nan), "EUR", f"Minimum free cash after tax {B.num(ds.get('min_free_cash_after_tax', np.nan), 0)} EUR"),
    ])
    B.note("Calculation order (ruling G0-D6): the monthly cash flow reads the stage-1 P&L, its interest enters the net margin, and the tax "
           "outflows return to the cash flow the month after each quarter. There is no iteration; the trace of this run: "
           + " → ".join(f"{n} ({B.num(t, 2)} s)" for n, t in r.trace) + ". Iteration count: 0 by construction.")

    st.markdown("## Monthly position")
    c1, c2 = st.columns(2)
    with c1:
        B.eyebrow("Closing cash, loan outstanding, restricted cash · EUR")
        st.plotly_chart(B.lines(B.MONTH_EN, {"Closing cash": cf.m("closing"), "Loan outstanding": cf.m("loan_outstanding"),
                                            "Restricted (reserve + collateral)": cf.m("restricted_reserve") + cf.m("restricted_collateral"),
                                            "Free cash after tax": cf.m("free_cash_after_tax")}, y_title="EUR"),
                        width="stretch", config={"displayModeBar": False})
        B.caption("CF_Mth rows 86, 88, 83 + 84, 92; EUR")
    with c2:
        B.eyebrow("Receipts and payments · EUR")
        st.plotly_chart(B.bars(B.MONTH_EN, {"Receipts": cf.m("in_total"), "Payments": -cf.m("out_total"), "VAT cash": cf.m("vat_cash"),
                                           "Net before tax": cf.m("net_cf_before_tax")}, y_title="EUR"),
                        width="stretch", config={"displayModeBar": False})
        B.caption("CF_Mth rows 56, 68 (shown negative), 77, 81; EUR")

    which = st.multiselect("Blocks", list(GROUPS), default=["Receipts", "Payments", "VAT", "Financing"])
    for name in which:
        B.eyebrow(f"{name} · workbook row in brackets")
        B.table(_frame(cf, GROUPS[name]), index_label="Line", scroll=len(GROUPS[name]) > 12)
    B.eyebrow("Per off-taker receipts")
    keys = [k for k in cf.rows if k.endswith("_in_energy") or k.endswith("_in_passthrough") or k.endswith("_acc_revenue")]
    df = pd.DataFrame({k: cf.rows[k] for k in keys}, index=[*B.MONTH_EN, "Beyond Dec", "Year"]).T
    df.index = [f"{p.offtaker(k.split('_')[0]).label} - {label(k)}" for k in df.index]
    B.table(df, index_label="Line", scroll=True)
    B.caption("Settlement key of each line: the month containing month end + payment terms; December with terms > 0 settles beyond December (column N)")

    st.markdown("## Daily ledger")
    d = cf.daily
    if d is not None:
        B.eyebrow("Loan outstanding and free cash by day · EUR")
        x = [B.dmy(v) for v in d["date"]]
        st.plotly_chart(B.lines(x, {"Loan outstanding": d["loan"].values, "Free cash after tax": d["free_cash_after_tax"].values,
                                    "Restricted floor": d["floor"].values}, y_title="EUR", height=340),
                        width="stretch", config={"displayModeBar": False})
        B.caption("CF_Daily_Ledger columns X, AB, V; EUR; settlement days = month end + terms, taxes on the payment day")
        summ = pd.DataFrame({"Value": list(ds.values())}, index=[label(k, fallback=TOTAL) for k in ds])
        c1, c2 = st.columns([1, 2])
        with c1:
            B.eyebrow("Ledger summary")
            B.table(summ, index_label="Item")
            ok = abs(ds.get("check_net_cf", 0.0)) < 1e-4 and abs(ds.get("check_receipts", 0.0)) < 1e-4
            st.markdown(f"Ledger reconciles to the monthly table: {B.status(ok)}", unsafe_allow_html=True)
        with c2:
            B.eyebrow("Daily rows (first 60 shown; full ledger in the exports)")
            show = d.copy()
            show["date"] = [B.dmy(v) for v in show["date"]]
            show = show.set_index("date")
            show.columns = [label(c, fallback=TOTAL) for c in show.columns]
            B.table(show, index_label="Date", decimals=0, scroll=True, max_rows=60)

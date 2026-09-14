"""Page 6 - Engine (QH) (replaces QH_P&L): merit-order result, sourcing stack, resell and LIFO
attribution, imbalance; aggregates by default, the quarter-hour frame on demand."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from app import brand as B
from app import state as S
from esb.labels import label
from esb.merit_order import OFFTAKER_COLUMNS, WORKBOOK_COLUMNS

STACK = [("pv_delivered", "PV delivered"), ("bl_delivered", "Baseload delivered"), ("spot_buy_notified", "Spot bought")]
RESELL = [("resell_pv_volume", "PV resold"), ("resell_bl_volume", "Baseload resold")]


def render() -> None:
    state = S.get()
    B.page_title("Engine (QH)", "The quarter-hour merit order: PV, then Baseload, then Spot, cascaded through the off-takers by position; surplus resold")
    r = S.require_result(state)
    if r is None:
        B.refusal(state.last_error or "No run available.")
        return
    q = r.qh.qh
    g = r.grid
    p = r.params
    months = g["date"].dt.month.values
    dates = g["date"].dt.date.values

    B.kpi_row([
        ("Retail demand notified", float(q["retail_buy_notified"].sum()), "MWh", f"Metered {B.num(float(q['retail_buy_metered'].sum()), 0)} MWh"),
        ("PV available (notified)", float(q["pv_avail_notified"].sum()), "MWh", f"Delivered to off-takers {B.num(float(q['pv_delivered'].sum()), 0)} MWh"),
        ("Baseload available (notified)", float(q["bl_avail_notified"].sum()), "MWh", f"Delivered {B.num(float(q['bl_delivered'].sum()), 0)} MWh"),
        ("Spot bought (notified)", float(q["spot_buy_notified"].sum()), "MWh", f"Resold PV {B.num(float(q['resell_pv_volume'].sum()), 0)} · Baseload {B.num(float(q['resell_bl_volume'].sum()), 0)} MWh"),
    ])

    st.markdown("## Sourcing stack by month")
    B.eyebrow("Delivered to the retail book and resold · MWh")
    agg = pd.DataFrame({k: q[k].values for k, _ in STACK + RESELL}).groupby(months).sum()
    st.plotly_chart(B.bars(B.MONTH_EN[: len(agg)], {lbl: agg[k].values for k, lbl in STACK + RESELL}, y_title="MWh", stacked=True, height=340),
                    width="stretch", config={"displayModeBar": False})
    B.caption("Monthly sums of QH_P&L columns CG, CQ, CU (delivered / bought) and CX, DD (resold); MWh")

    c1, c2 = st.columns(2)
    with c1:
        B.eyebrow("Retail margin by month · EUR")
        m = pd.DataFrame({"GM2 budget": q["retail_gm2_budget"].values, "GM2 forecast": q["retail_gm2_forecast"].values,
                          "Resell GM2": q["resell_gm2"].values}).groupby(months).sum()
        st.plotly_chart(B.bars(B.MONTH_EN[: len(m)], {c: m[c].values for c in m.columns}, y_title="EUR"), width="stretch", config={"displayModeBar": False})
        B.caption("Sums of BD, BE (retail GM2) and BK (resell GM2); EUR")
    with c2:
        B.eyebrow("Imbalance by month · EUR")
        m = pd.DataFrame({"Source imbalance": q["retail_source_imb"].values, "Off-taker imbalance": q["retail_offtaker_imb"].values,
                          "Resell source imbalance": q["resell_source_imb"].values}).groupby(months).sum()
        st.plotly_chart(B.bars(B.MONTH_EN[: len(m)], {c: m[c].values for c in m.columns}, y_title="EUR"), width="stretch", config={"displayModeBar": False})
        B.caption("Sums of AZ, BA, BJ; EUR (positive = revenue to the BRP)")

    st.markdown("## Off-takers in the cascade")
    rows = []
    for o in p.offtakers:
        c = o.code
        rows.append({"Off-taker": f"{o.label} ({c})", "Notified": q[f"{c}_demand_notified"].sum(), "Metered": q[f"{c}_demand_metered"].sum(),
                     "PV": q[f"{c}_pv_buy_notified"].sum(), "Baseload": q[f"{c}_bl_buy_notified"].sum(), "Spot": q[f"{c}_spot_buy_notified"].sum(),
                     "Strip notified": q[f"{c}_strip_notified"].sum(), "Resell attributed (LIFO)": q[f"{c}_resell_attributed"].sum(),
                     "GM2 forecast": q[f"{c}_gm2_forecast"].sum(), "Off-taker imbalance": q[f"{c}_offtaker_imb"].sum()})
    df = pd.DataFrame(rows).set_index("Off-taker")
    B.table(df, index_label="Off-taker", decimals=0)
    B.caption("Year sums; MWh for volumes, EUR for margins. LIFO: Baseload surplus is attributed to the last position first, position 1 takes the remainder")

    st.markdown("## Checks")
    chk = {k: float(q[k].abs().sum()) for k in ("check_demand", "check_pv", "check_bl", "check_imb", "check_origin")}
    html = '<table class="esb"><thead><tr><th>Check</th><th>Sum of absolute residuals</th><th>State</th></tr></thead><tbody>'
    html += "".join(f"<tr><td>{label(k)}</td><td>{B.num(v, 6)}</td><td>{B.status(v < 1e-6)}</td></tr>" for k, v in chk.items())
    st.markdown(html + "</tbody></table>", unsafe_allow_html=True)

    st.markdown("## Drill-down to the quarter-hour")
    B.caption("Aggregates are plotted by default; the 35.040-row frame is loaded only for the day or the download requested here.")
    c1, c2 = st.columns([1, 3])
    with c1:
        d0 = date(p.spine_year, 1, 1)
        d1 = date(p.spine_year, 12, 31)
        day = st.date_input("Day", value=d0, min_value=d0, max_value=d1, format="DD.MM.YYYY")
        view = st.selectbox("Columns", ["Sourcing", "Prices", "Imbalance", "Resell", "Off-taker block"])
        code = st.selectbox("Off-taker (for the block view)", [o.code for o in p.offtakers], format_func=lambda c: f"{p.offtaker(c).label} ({c})")
    with c2:
        mask = dates == day
        sub = q[mask]
        x = [f"{(i - 1) // 4:02d}:{((i - 1) % 4) * 15:02d}" for i in g["interval"].values[mask]]
        if view == "Sourcing":
            cols = {"PV delivered": sub["pv_delivered"].values, "Baseload delivered": sub["bl_delivered"].values, "Spot bought": sub["spot_buy_notified"].values}
            st.plotly_chart(B.bars(x, cols, y_title="MWh per QH", stacked=True), width="stretch", config={"displayModeBar": False})
            keys = ["retail_buy_notified", "pv_avail_notified", "pv_delivered", "pv_remaining", "bl_avail_notified", "bl_delivered", "bl_remaining", "spot_buy_notified"]
        elif view == "Prices":
            cols = {"DAM": sub["dam"].values, "IDCT": sub["idct"].values, "Surplus": sub["surplus_price"].values, "Deficit": sub["deficit_price"].values}
            st.plotly_chart(B.lines(x, cols, y_title="EUR/MWh"), width="stretch", config={"displayModeBar": False})
            keys = ["dam", "idct", "dam_curtailed", "idct_curtailed", "surplus_price", "deficit_price", "direction"]
        elif view == "Imbalance":
            cols = {"Source imbalance": sub["retail_source_imb"].values, "Off-taker imbalance": sub["retail_offtaker_imb"].values}
            st.plotly_chart(B.bars(x, cols, y_title="EUR"), width="stretch", config={"displayModeBar": False})
            keys = ["pv_metered", "pv_specific_imb", "bl_specific_imb", "retail_source_imb", "retail_offtaker_imb", "total_imb_all_legs"]
        elif view == "Resell":
            cols = {"PV resold": sub["resell_pv_volume"].values, "Baseload resold": sub["resell_bl_volume"].values}
            st.plotly_chart(B.bars(x, cols, y_title="MWh per QH", stacked=True), width="stretch", config={"displayModeBar": False})
            keys = ["resell_pv_volume", "resell_pv_cost", "resell_pv_revenue", "resell_pv_source_imb", "resell_bl_volume", "resell_bl_cost_forecast",
                    "resell_bl_revenue", "resell_bl_source_imb", "resell_total_gm2"]
        else:
            keys = [f"{code}_{k}" for k in ("demand_notified", "demand_metered", "pv_buy_notified", "bl_buy_notified", "spot_buy_notified", "buy_notified",
                                            "cost_forecast", "revenue", "source_imb", "offtaker_imb", "gm2_forecast", "strip_notified", "resell_attributed")]
            cols = {"PV": sub[f"{code}_pv_buy_notified"].values, "Baseload": sub[f"{code}_bl_buy_notified"].values, "Spot": sub[f"{code}_spot_buy_notified"].values}
            st.plotly_chart(B.bars(x, cols, y_title="MWh per QH", stacked=True), width="stretch", config={"displayModeBar": False})
        B.caption(f"{B.dmy(day)} · 96 quarter-hours EET · Peak intervals 37-84")
        tbl = sub[keys].copy()
        tbl.index = x
        tbl.columns = [_col_label(c) for c in keys]
        B.table(tbl, index_label="QH", decimals=4, scroll=True)

    with st.expander("Full quarter-hour frame (35.040 rows) - load on demand"):
        if st.checkbox("Show the frame (first 500 rows) and enable the CSV download"):
            full = q.copy()
            full.insert(0, "interval", g["interval"].values)
            full.insert(0, "date", dates)
            st.dataframe(full.head(500), width="stretch", height=400)
            from esb.export import csv_bytes

            st.download_button("Download QH frame (CSV, ';' separated, decimal comma)", data=csv_bytes(full, index=False),
                               file_name=f"QH_frame_{p.spine_year}.csv", mime="text/csv")


def _col_label(c: str) -> str:
    for code_prefix in ("OT",):
        if c.startswith(code_prefix) and "_" in c:
            code, rest = c.split("_", 1)
            letter = OFFTAKER_COLUMNS.get(rest, "")
            return f"{code} {label(rest)}" + (f" [{letter}]" if letter else "")
    letter = WORKBOOK_COLUMNS.get(c, "")
    return label(c) + (f" [{letter}]" if letter else "")


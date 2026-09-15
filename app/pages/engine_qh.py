"""Page 6 - Engine (QH) (replaces QH_P&L): merit-order result, sourcing stack, resell and LIFO
attribution, imbalance; aggregates by default, the quarter-hour frame on demand."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from app import brand as B
from app import state as S
from esb.labels import RESELL, RETAIL, label, tagged, unit_of
from esb.merit_order import OFFTAKER_COLUMNS, WORKBOOK_COLUMNS

STACK = [("pv_delivered", "PV delivered"), ("bl_delivered", "Baseload delivered"), ("spot_buy_notified", "Spot bought")]
RESOLD = [("resell_pv_volume", "PV resold"), ("resell_bl_volume", "Baseload resold")]


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

    demand_notified = sum(q[f"{o.code}_demand_notified"].sum() for o in p.offtakers)
    demand_metered = sum(q[f"{o.code}_demand_metered"].sum() for o in p.offtakers)
    B.kpi_row([
        (tagged("Demand notified", RETAIL), float(demand_notified), "MWh",
         f"Demand metered {B.num(float(demand_metered), 0)} MWh · bought on the metered basis {B.num(float(q['retail_buy_metered'].sum()), 0)} MWh"),
        (tagged("PV available (notified)", "Sources"), float(q["pv_avail_notified"].sum()), "MWh", f"Delivered to the retail book {B.num(float(q['pv_delivered'].sum()), 0)} MWh"),
        (tagged("Baseload available (notified)", "Sources"), float(q["bl_avail_notified"].sum()), "MWh", f"Delivered to the retail book {B.num(float(q['bl_delivered'].sum()), 0)} MWh"),
        (tagged("Spot bought (notified)", RETAIL), float(q["spot_buy_notified"].sum()), "MWh", f"Resold to the market: PV {B.num(float(q['resell_pv_volume'].sum()), 0)} · Baseload {B.num(float(q['resell_bl_volume'].sum()), 0)} MWh"),
    ])

    st.markdown("## Sourcing stack by month")
    B.eyebrow(f"Delivered to the retail book ({RETAIL}) and resold ({RESELL}) · MWh")
    agg = pd.DataFrame({k: q[k].values for k, _ in STACK + RESOLD}).groupby(months).sum()
    st.plotly_chart(B.bars(B.MONTH_EN[: len(agg)], {lbl: agg[k].values for k, lbl in STACK + RESOLD}, y_title="MWh", stacked=True, height=340),
                    width="stretch", config={"displayModeBar": False})
    B.caption("Monthly sums of QH_P&L columns CG, CQ, CU (delivered / bought) and CX, DD (resold); MWh")

    c1, c2 = st.columns(2)
    with c1:
        B.eyebrow("GM2 by month and leg · EUR")
        m = pd.DataFrame({tagged("GM2 budget", RETAIL): q["retail_gm2_budget"].values, tagged("GM2 forecast", RETAIL): q["retail_gm2_forecast"].values,
                          tagged("GM2", RESELL): q["resell_gm2"].values}).groupby(months).sum()
        st.plotly_chart(B.bars(B.MONTH_EN[: len(m)], {c: m[c].values for c in m.columns}, y_title="EUR"), width="stretch", config={"displayModeBar": False})
        B.caption("Sums of BD, BE (retail GM2) and BK (resell GM2); EUR")
    with c2:
        B.eyebrow("Imbalance by month · EUR")
        m = pd.DataFrame({tagged("Source imbalance", RETAIL): q["retail_source_imb"].values, tagged("Off-taker imbalance", RETAIL): q["retail_offtaker_imb"].values,
                          tagged("Source imbalance", RESELL): q["resell_source_imb"].values}).groupby(months).sum()
        st.plotly_chart(B.bars(B.MONTH_EN[: len(m)], {c: m[c].values for c in m.columns}, y_title="EUR"), width="stretch", config={"displayModeBar": False})
        B.caption("Sums of AZ, BA, BJ; EUR (positive = revenue to the BRP)")

    st.markdown("## Off-takers in the cascade")
    rows = []
    for o in p.offtakers:
        c = o.code
        rows.append({"Off-taker": f"{o.label} ({c})", "Notified": q[f"{c}_demand_notified"].sum(), "Metered": q[f"{c}_demand_metered"].sum(),
                     "PV": q[f"{c}_pv_buy_notified"].sum(), "Baseload": q[f"{c}_bl_buy_notified"].sum(), "Spot": q[f"{c}_spot_buy_notified"].sum(),
                     tagged("Strip notified", RETAIL): q[f"{c}_strip_notified"].sum(), tagged("Attributed (LIFO)", RESELL): q[f"{c}_resell_attributed"].sum(),
                     tagged("GM2 forecast", RETAIL): q[f"{c}_gm2_forecast"].sum(), tagged("Off-taker imbalance", RETAIL): q[f"{c}_offtaker_imb"].sum()})
    df = pd.DataFrame(rows).set_index("Off-taker")
    B.table(df, index_label="Off-taker", decimals=0,
            col_units={"Notified": "MWh", "Metered": "MWh", "PV": "MWh", "Baseload": "MWh", "Spot": "MWh", tagged("Strip notified", RETAIL): "MWh",
                       tagged("Attributed (LIFO)", RESELL): "MWh", tagged("GM2 forecast", RETAIL): "EUR", tagged("Off-taker imbalance", RETAIL): "EUR"})
    B.caption("Year sums; unit in every column header. LIFO: Baseload surplus is attributed to the last position first, position 1 takes the remainder")

    st.markdown("## Checks")
    chk = {k: float(q[k].abs().sum()) for k in ("check_demand", "check_pv", "check_bl", "check_imb", "check_origin")}
    html = '<table class="esb"><thead><tr><th>Check</th><th class="txt">Unit</th><th>Sum of absolute residuals</th><th>State</th></tr></thead><tbody>'
    html += "".join(f"<tr><td>{label(k, leg='')}</td><td class=\"unit\">{'MWh' if k in ('check_demand', 'check_pv', 'check_bl', 'check_origin') else 'EUR'}</td><td>{B.num(v, 6)}</td><td>{B.status(v < 1e-6)}</td></tr>" for k, v in chk.items())
    st.markdown(html + "</tbody></table>", unsafe_allow_html=True)

    st.markdown("## Drill-down to the quarter-hour")
    B.caption("Aggregates are plotted by default; the 35.040-row frame is loaded only for the day or the download requested here.")
    c1, c2 = st.columns([1, 3])
    with c1:
        d0 = date(p.spine_year, 1, 1)
        d1 = date(p.spine_year, 12, 31)
        day = st.date_input("Day", value=d0, min_value=d0, max_value=d1, format="DD.MM.YYYY")
        view = st.selectbox("Columns", ["Sourcing", "Prices", "Imbalance", "Wholesale spot resell", "Off-taker block"])
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
        elif view == "Wholesale spot resell":
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
        B.table(tbl, index_label="QH", decimals=4, scroll=True, col_units={_col_label(c): _qh_unit(c) for c in keys})

    with st.expander("Full quarter-hour frame (35.040 rows) - load on demand"):
        if st.checkbox("Show the frame (first day, 96 rows) and enable the CSV download of all rows"):
            full = q.copy()
            for col, vals in (("interval", g["interval"].values), ("date", dates)):
                if col in full.columns:
                    full = full.drop(columns=col)
                full.insert(0, col, vals)
            show = full.head(96).copy()
            show["date"] = [B.dmy(v) for v in show["date"]]
            show = show.set_index("date")
            cu = {_col_label(c): _qh_unit(c) for c in show.columns if c != "interval"}
            show.columns = [_col_label(c) for c in show.columns]
            B.table(show, index_label="Date", decimals=4, scroll=True, decimals_by_col={"Interval": 0}, col_units=cu)
            from esb.export import csv_bytes

            st.download_button("Download QH frame (CSV, ';' separated, decimal comma)", data=csv_bytes(full, index=False),
                               file_name=f"QH_frame_{p.spine_year}.csv", mime="text/csv")


def _qh_unit(c: str) -> str:
    """Quarter-hour frame units: energy per interval is MWh, MW columns MW, prices EUR/MWh, money EUR."""
    return unit_of(c)


def _col_label(c: str) -> str:
    for code_prefix in ("OT",):
        if c.startswith(code_prefix) and "_" in c:
            code, rest = c.split("_", 1)
            letter = OFFTAKER_COLUMNS.get(rest, "")
            return f"{code} {label(rest)}" + (f" [{letter}]" if letter else "")
    letter = WORKBOOK_COLUMNS.get(c, "")
    return label(c) + (f" [{letter}]" if letter else "")


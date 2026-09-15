"""Page 5 - Scenarios (replaces Whol_Sport_Imb_Fcst): the loaded price scenarios, their statistics,
the active selection and a side-by-side comparison of the engine on each loaded scenario."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app import brand as B
from app import state as S
from esb.engine import run
from esb.scenarios import PRICE_COLUMNS, SCENARIO_PREFIX, annual_mean_nonzero, scenario_block

HEADLINE = [("revenue", "Retail revenue"), ("cost_forecast", "Sourcing cost forecast"), ("t_gm2_forecast", "Total GM2 forecast"),
            ("nm_forecast", "NM pre-tax forecast"), ("cit_forecast", "CIT forecast"), ("guarantees_outstanding", "Guarantees outstanding (peak)"),
            ("interest", "Financing interest")]


def render() -> None:
    state = S.get()
    p = state.params
    B.page_title("Scenarios", "Aurora Central, Aurora Low and User Forecast price scenarios for the case year; the active one drives the engine")
    B.note("Scenario libraries for 2026-2031 and multi-year runs are Phase 7 scope. This release runs one spine year; each price scenario "
           "is a wholesale_prices upload in the standard format (Data page) or the Reference Case fixture.")

    series = state.rebuild_series()
    loaded = {name: ok for name, ok in series.coverage.scenarios.items()}
    c1, c2 = st.columns([1, 2])
    with c1:
        B.eyebrow("Active scenario")
        options = list(SCENARIO_PREFIX)
        sel = st.selectbox("Price scenario", options, index=options.index(p.scenario_active) if p.scenario_active in options else 0,
                           format_func=lambda n: f"{n}" + ("" if loaded.get(n) else " (not loaded)"))
        if sel != p.scenario_active and st.button("Set active", type="primary"):
            p.scenario_active = sel
            state.mark_dirty(f"active scenario set to '{sel}'")
            st.rerun()
    with c2:
        B.eyebrow("Loaded scenarios · annual statistics (EUR/MWh)")
        rows = []
        for name, prefix in SCENARIO_PREFIX.items():
            if not loaded.get(name):
                rows.append({"Scenario": name, "State": "not loaded"})
                continue
            blk = scenario_block(series.frame, prefix)
            rows.append({
                "Scenario": name, "State": "active" if name == p.scenario_active else "loaded",
                "DAM mean (non-zero)": annual_mean_nonzero(blk["dam"]), "IDCT mean (non-zero)": annual_mean_nonzero(blk["idct"]),
                "Surplus mean": float(np.nanmean(blk["surplus_price"])), "Deficit mean": float(np.nanmean(blk["deficit_price"])),
                "DAM min": float(np.nanmin(blk["dam"])), "DAM max": float(np.nanmax(blk["dam"])),
                "Negative DAM hours": float((blk["dam"] < 0).sum() / 4.0),
                "Long share": float(np.mean(blk["direction"] == "Positive (Long)")),
            })
        df = pd.DataFrame(rows).set_index("Scenario")
        B.table(df, index_label="Scenario", pct_rows=set(), decimals=2, decimals_by_col={"Negative DAM hours": 0},
                col_units={"DAM mean (non-zero)": "EUR/MWh", "IDCT mean (non-zero)": "EUR/MWh", "Surplus mean": "EUR/MWh", "Deficit mean": "EUR/MWh", "DAM min": "EUR/MWh",
                           "DAM max": "EUR/MWh", "Negative DAM hours": "h", "Long share": "share"})
        B.caption("Non-zero means as the workbook's row 3 (IFERROR(AVERAGEIF(range, '<>0'), 0)); the loaded Reference Case scenarios carry Surplus = Deficit (X-19)")

    active_prefix = SCENARIO_PREFIX.get(p.scenario_active)
    if loaded.get(p.scenario_active) and active_prefix:
        blk = scenario_block(series.frame, active_prefix)
        g = state.series.frame
        months = pd.to_datetime(g["date"]).dt.month.values
        st.markdown(f"## {p.scenario_active} - monthly profile")
        mdf = pd.DataFrame({"DAM": blk["dam"].values, "IDCT": blk["idct"].values, "Surplus": blk["surplus_price"].values, "Deficit": blk["deficit_price"].values, "m": months})
        mm = mdf.groupby("m").mean()
        st.plotly_chart(B.lines(B.MONTH_EN, {"DAM": mm["DAM"].values, "IDCT VWAP15": mm["IDCT"].values, "Surplus imbalance": mm["Surplus"].values,
                                             "Deficit imbalance": mm["Deficit"].values}, y_title="EUR/MWh"),
                        width="stretch", config={"displayModeBar": False})
        B.caption("Monthly means of the quarter-hour prices; EUR/MWh. Source: active scenario block (Whol_Sport_Imb_Fcst AJ:AP)")
        hour = ((g["interval"].values - 1) // 4).astype(int)
        hdf = pd.DataFrame({"DAM": blk["dam"].values, "IDCT": blk["idct"].values, "h": hour}).groupby("h").mean()
        st.plotly_chart(B.lines([f"{h:02d}:00" for h in hdf.index], {"DAM": hdf["DAM"].values, "IDCT VWAP15": hdf["IDCT"].values}, y_title="EUR/MWh", height=260),
                        width="stretch", config={"displayModeBar": False})
        B.caption("Average daily shape by hour (EET); EUR/MWh")

    st.markdown("## Scenario comparison")
    B.caption("Runs the engine once per loaded scenario with the current parameters; year values of the forecast case.")
    if st.button("Compare loaded scenarios"):
        results = {}
        with st.spinner("Running the engine per scenario"):
            for name, ok in loaded.items():
                if not ok:
                    continue
                pp = p.copy()
                pp.scenario_active = name
                try:
                    results[name] = run(series.frame, pp, selected_offtaker=state.selected_offtaker or None)
                except Exception as e:  # shown, never raised to the user
                    B.refusal(f"{name}: run refused - {type(e).__name__}: {e}")
        if results:
            table = {}
            for name, r in results.items():
                P = r.pnl.portfolio
                table[name] = [P.y(k) for k, _ in HEADLINE] + [r.cashflow.daily_summary.get("peak_funding", np.nan)]
            df = pd.DataFrame(table, index=[lbl for _, lbl in HEADLINE] + ["Peak funding (daily)"])
            if len(df.columns) >= 2:
                first = df.columns[0]
                for c in df.columns[1:]:
                    df[f"{c} - {first}"] = df[c] - df[first]
            B.table(df, index_label="Year value", decimals=0, units=["EUR"] * len(df))
            state.add_log("run", f"scenario comparison on {', '.join(results)}")
    st.markdown("## Price columns of the contract")
    B.table(pd.DataFrame({"Unit": ["EUR/MWh"] * 4 + ["flag"], "Role": ["Day-ahead reference; curtailed twin MAX(p, 0)", "Intraday continuous VWAP 15 min; curtailed twin",
                                                                    "Price of BRP surplus", "Price of BRP deficit", "Positive (Long) | Negative (Short) | Balanced; derived when absent"]},
                         index=[*PRICE_COLUMNS, "System_imbalance_direction"]), index_label="Series")

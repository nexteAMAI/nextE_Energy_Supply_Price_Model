"""Page 9 - Guarantees (Input section E and Cons_P&L rows 204-218): required amounts per
counterparty under each sizing method, BGL fees, outstanding windows, regulatory inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app import brand as B
from app import state as S
from config.schema import GUARANTEE_SIZINGS
from esb.guarantees import sizing_comparison
from esb.labels import TOTAL, tagged

LABELS = {"pv": "PV source", "baseload": "Baseload source", "spot": "Spot (OPCOM)", "brp": "BRP", "tso": "TSO", "dso": "DSO"}


def render() -> None:
    state = S.get()
    B.page_title("Guarantees", "Guarantees issued to sources and market operators and received from off-takers; sizing, windows, fees")
    r = S.require_result(state)
    if r is None:
        B.refusal(state.last_error or "No run available.")
        return
    P = r.pnl.portfolio
    p = r.params
    reg = r.pnl.regulatory
    inp = r.pnl.reg_inputs
    B.kpi_row([
        (tagged("Guarantees outstanding (peak)", TOTAL), P.y("guarantees_outstanding"), "EUR", "Counterparties plus own guarantees, maximum month"),
        (tagged("Market BGL fees (year)", TOTAL), P.y("market_bgl_fees"), "EUR", f"Off-taker BGL fees {B.num(P.y('offtaker_bgl_fees'), 0)} EUR"),
        (tagged("PV fixed amount (derived)", TOTAL), r.pnl.pv_fixed_guarantee, "EUR", "mean(PV cost budget) + mean(PV resell cost), Input!C85"),
        (tagged("Regulatory total (spot + BRP + TSO + DSO)", TOTAL), reg.spot + reg.brp + reg.tso + reg.dso, "EUR", "Section E formulas on this run's inputs"),
    ])

    st.markdown("## Outstanding by month · EUR")
    series = {LABELS[k]: P.m(f"g_out_{k}") for k in LABELS}
    own = {f"{o.label} (own)": r.pnl.sections[o.code].m("own_guarantee") for o in p.offtakers if "own_guarantee" in r.pnl.sections[o.code]}
    st.plotly_chart(B.bars(B.MONTH_EN, {**series, **own}, y_title="EUR", stacked=True, height=340), width="stretch", config={"displayModeBar": False})
    B.caption("Cons_P&L rows 204-209 (counterparties) and the sections' own guarantee rows; EUR; a guarantee is outstanding in the months whose first day lies in its window")

    st.markdown("## Per counterparty")
    rows = []
    for k, lbl in LABELS.items():
        c = p.counterparties[k]
        g = c.guarantee
        out = P.m(f"g_out_{k}")
        fee = P.m(f"g_fee_{k}")
        rows.append({"Counterparty": lbl + (f" - {c.name}" if c.name else ""), "Active": "Active" if c.active else "Inactive", "Type": g.type, "Sizing": g.sizing,
                     "Peak outstanding": float(np.max(out)), "Months outstanding": int((out > 0).sum()), "Window": f"{B.dmy(g.start)} - {B.dmy(g.end)}",
                     "BGL fee p.a.": B.pct(g.bgl_fee_pa, 2), "Fee type": g.bgl_fee_type, "Fee (year)": float(fee.sum()), "Cash backing": B.pct(g.cash_backing_pct, 0)})
    df = pd.DataFrame(rows).set_index("Counterparty")
    B.table(df, index_label="Counterparty", decimals=0, col_units={"Peak outstanding": "EUR", "Months outstanding": "months", "Fee (year)": "EUR"})

    st.markdown("## Required amount under each sizing method · EUR (peak month)")
    cmp = sizing_comparison(r.pnl, p)
    df = pd.DataFrame(cmp).T
    df.index = [LABELS[k] for k in df.index]
    df = df[list(GUARANTEE_SIZINGS)]
    B.table(df, index_label="Counterparty", decimals=0, col_units={c: "EUR" for c in df.columns})
    B.caption("Same bases as the P&L: annual contract value (revenue for PV, spot, BRP, TSO, DSO; forecast Baseload cost incl. resell for Baseload), "
              "the regulatory amount where a formula exists, the fixed amount of the register (PV: the user's input, or the workbook derivation of Input!C85 when none is set - D110). The register's own sizing is the one in force")

    st.markdown("## Regulatory formulas · inputs of this run")
    mg = p.market_guarantees
    fx = p.general.fx_ron_per_eur
    from esb.catalogue import entry_for

    def origin(key: str) -> str:
        """Source and status of the driving register entry, read from the parameter catalogue (G5-10, D-I)."""
        e = entry_for(key)
        return f"{e.source} [{e.source_status}]" if e.source else "–"

    delegated = str(mg["brp"].get("method", "rate_per_mw")) == "pre_delegated"
    spot_vat_txt = f" x (1 + VAT {B.pct(p.general.vat_rate, 0)})" if bool(mg["spot"].get("vat_inclusive", False)) else ""
    if delegated:
        vals = np.abs(np.asarray(inp.imbalance_value_monthly if inp.imbalance_value_monthly is not None else [], dtype=float))
        avg = float(vals.mean()) if vals.size else 0.0
        vat_txt = f" x (1 + VAT {B.pct(p.general.vat_rate, 0)})" if bool(mg["brp"].get("pre_vat_inclusive", True)) else ""
        brp_formula = (f"max({B.num(mg['brp'].get('pre_initial_ron', 0.0), 0)} RON / {B.num(fx, 2)}; "
                       f"{B.num(mg['brp'].get('pre_months_of_imbalance', 0.0), 2)} months x mean |monthly imbalance value| {B.num(avg, 0)} EUR{vat_txt})")
        brp_origin = f"Delegated PRE (D119, D121): floor {origin('market_guarantees.brp.pre_initial_ron')}; months {origin('market_guarantees.brp.pre_months_of_imbalance')}"
    else:
        brp_formula = f"{B.num(mg['brp']['rate_ron_per_mw'], 0)} RON/MW x ({B.num(mg['brp']['generation_mw_in_brp'], 0)} + {B.num(inp.peak_retail_buy_mw, 2)}) MW / {B.num(fx, 2)}"
        brp_origin = f"Workbook rule, kept for parity: {origin('market_guarantees.brp.rate_ron_per_mw')}"
    rows = [
        ("Spot (OPCOM)", f"{B.num(mg['spot']['buffer_days'], 0)} days x {B.num(inp.peak_daily_spot_buy_mwh, 2)} MWh x {B.num(inp.peak_dam_price, 2)} EUR/MWh{spot_vat_txt}", reg.spot,
         origin("market_guarantees.spot.buffer_days")),
        ("BRP", brp_formula, reg.brp, brp_origin),
        ("TSO", f"Vtm {B.num(mg['tso']['vtm_multiplier'], 1)} x 4 x (TL + SS) x metered / 12 = annual value {B.num(reg.tso_annual_value, 0)} EUR", reg.tso,
         origin("market_guarantees.tso.vtm_multiplier")),
        ("DSO", f"Vdm {B.num(mg['dso']['vdm_multiplier'], 1)} x 4 x (T_HV + T_MV + T_LV) x metered / 12 + add-on = annual value {B.num(reg.dso_annual_value, 0)} EUR", reg.dso,
         origin("market_guarantees.dso.vdm_multiplier")),
    ]
    df = pd.DataFrame(rows, columns=["Counterparty", "Formula on this run", "Amount (EUR)", "Origin / status"]).set_index("Counterparty")
    B.table(df, index_label="Counterparty", decimals=0)
    B.caption("Origin and status are read from the parameter catalogue (config/parameter_catalogue.yaml, docs/PARAMETERS.md section 3): "
              "verified = read on the primary source; contradicted = the source says otherwise, value kept for the Reference Case parity; "
              "to_verify = source named, pending the signed supply contract; assumption = house proxy. The numbers are the engine's.")

    st.markdown("## Own guarantees received from off-takers")
    rows = []
    for o in p.offtakers:
        T = r.pnl.sections[o.code]
        g = o.guarantee
        out = T.m("own_guarantee") if "own_guarantee" in T else np.zeros(12)
        rows.append({"Off-taker": f"{o.label} ({o.code})", "Type": g.type, "Sizing": g.sizing, "Peak outstanding": float(np.max(out)),
                     "BGL fee (year)": T.y("own_bgl_fee") if "own_bgl_fee" in T else 0.0, "Window": f"{B.dmy(g.start)} - {B.dmy(g.end)}" if g.type != "None" else "–"})
    B.table(pd.DataFrame(rows).set_index("Off-taker"), index_label="Off-taker", decimals=0, col_units={"Peak outstanding": "EUR", "BGL fee (year)": "EUR"})
    B.caption("Dynamic sizing is monthly (X-26): pct x monthly revenue; Regulatory formula is not defined for off-takers and yields 0")

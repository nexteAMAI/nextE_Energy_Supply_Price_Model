"""Page 4 - Sources and Contracts (replaces Dashboard_Input): the contractual position of every
source and off-taker as the engine models it. The C3 price types (Cap+Excess, DAM-indexed,
IDM-indexed, floors, caps, index deltas, invoice FX) are Phase 7 scope and are stated as such."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app import brand as B
from app import state as S

SOURCE_LABELS = {"pv": "Forward Source 1 - PV", "baseload": "Forward Source 2 - Baseload", "spot": "Wholesale spot (OPCOM)",
                 "brp": "BRP / imbalance settlement", "tso": "Transmission (TSO)", "dso": "Distribution (DSO)"}


def render() -> None:
    state = S.get()
    p = state.params
    B.page_title("Sources and Contracts", "Nomination and pricing basis of each source, contract terms of each off-taker, cover flags - as modelled in the Reference Case")
    B.note("Price type in this release: <b>Fixed</b> for PV (per off-taker PV price) and Baseload (monthly product prices per off-taker), "
           "<b>MIN(DAM, IDCT) on notified volume</b> for Spot, surplus / deficit prices for imbalance. The Cap+Excess, DAM-indexed and "
           "IDM-indexed price types, floors, caps, index deltas, contract currency and invoice FX of the C3 dashboard are Phase 7 "
           "enhancements (execution prompt section 11) and are not modelled yet.")

    st.markdown("## Sources and market counterparties")
    rows = []
    for k, c in p.counterparties.items():
        g = c.guarantee
        rows.append({
            "Counterparty": f"{SOURCE_LABELS[k]}" + (f" - {c.name}" if c.name else ""),
            "Active": "Active" if c.active else "Inactive",
            "Nomination basis": {"pv": "PV forecast (uncurtailed) with deviation ratios", "baseload": "Strips per off-taker (BL24 / Peak / Off-Peak)",
                                 "spot": "Residual notified demand", "brp": "Metered minus notified (k declared)", "tso": "Metered volume",
                                 "dso": "Metered volume"}[k],
            "Price basis": {"pv": "Fixed per off-taker (budget / forecast)", "baseload": "Monthly product prices per off-taker (MW-weighted)",
                            "spot": "MIN(DAM, IDCT) per quarter-hour", "brp": "Surplus / Deficit imbalance prices", "tso": "TL + SS (EUR/MWh)",
                            "dso": "T_HV + T_MV + T_LV (EUR/MWh)"}[k],
            "Settlement volume": {"pv": "Metered", "baseload": "Metered", "spot": "Notified", "brp": "Imbalance", "tso": "Metered", "dso": "Metered"}[k],
            "Terms (days)": c.payment_terms_days,
            "Advance": B.pct(c.advance_pct, 0),
            "k": "–" if c.k is None else str(c.k),
            "Guarantee": f"{g.type} · {g.sizing}" if g.type != "None" else "None",
            "Window": f"{B.dmy(g.start)} - {B.dmy(g.end)}" if g.type != "None" else "–",
            "BGL fee": f"{B.pct(g.bgl_fee_pa, 2)} {g.bgl_fee_type}" if g.type == "Bank Guarantee Letter" else "–",
        })
    df = pd.DataFrame(rows).set_index("Counterparty")
    B.table(df, index_label="Counterparty", decimals=0)
    B.caption("Source: parameter register sections D and E; k per volume series is declared in the upload registry (docs/DATA_CONTRACT.md)")

    st.markdown("## Off-taker contracts")
    rows = []
    for o in p.offtakers:
        strips = np.array(o.strip_mw["BL24"]) + np.array(o.strip_mw["Peak"]) + np.array(o.strip_mw["OffPeak"])
        rows.append({
            "Off-taker": f"{o.label} ({o.code})",
            "Position": o.code.replace("OT", ""),
            "Active": "Active" if o.active else "Inactive",
            "Contract": f"{B.dmy(o.contract_start)} - {B.dmy(o.contract_end)}",
            "Contract price": B.num(o.contract_price_eur_per_mwh, 2, "EUR/MWh"),
            "PV price budget / forecast": f"{B.num(o.pv_price_budget_eur_per_mwh, 2)} / {B.num(o.pv_price_forecast_eur_per_mwh, 2)}",
            "Strip BL24 (MW, avg)": B.num(float(np.mean(o.strip_mw["BL24"])), 2),
            "Strip total (MW, max)": B.num(float(strips.max()), 2),
            "BL24 price budget (avg)": B.num(float(np.mean(o.product_price_budget["BL24"])), 2),
            "BL24 price forecast (avg)": B.num(float(np.mean(o.product_price_forecast["BL24"])), 2),
            "Premium budget / forecast": f"{B.num(o.premium_budget_total, 2)} / {B.num(o.premium_forecast_total, 2)}",
            "Target GM budget / forecast": f"{B.num(o.target_gm_budget, 2)} / {B.num(o.target_gm_forecast, 2)}",
            "Terms (days)": o.payment_terms_days,
            "Advance": B.pct(o.advance_pct, 0),
            "Own guarantee": f"{o.guarantee.type} · {o.guarantee.sizing}" if o.guarantee.type != "None" else "None",
            "Tariff set": "own" if o.tariff_components else "portfolio",
        })
    df = pd.DataFrame(rows).set_index("Off-taker")
    B.table(df, index_label="Off-taker", decimals=0, scroll=True)
    B.caption("EUR/MWh unless stated; strips in MW; the Baseload availability is the sum of the active off-takers' strips (D77)")

    st.markdown("## Strips by month")
    for o in p.offtakers:
        with st.expander(f"{o.label} ({o.code}) - strips and product prices by month"):
            df = pd.DataFrame({
                "BL24 MW": o.strip_mw["BL24"], "Peak MW": o.strip_mw["Peak"], "Off-Peak MW": o.strip_mw["OffPeak"],
                "BL24 budget": o.product_price_budget["BL24"], "Peak budget": o.product_price_budget["Peak"], "Off-Peak budget": o.product_price_budget["OffPeak"],
                "BL24 forecast": o.product_price_forecast["BL24"], "Peak forecast": o.product_price_forecast["Peak"], "Off-Peak forecast": o.product_price_forecast["OffPeak"],
            }, index=B.MONTH_EN)
            B.table(df, index_label="Month")
    B.caption("Cover flags: a strip with MW > 0 and a 0 product price trips the overview tripwire (Portf Overview!E148). Edit on the Parameters page.")

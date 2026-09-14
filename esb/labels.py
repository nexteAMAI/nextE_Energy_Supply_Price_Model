"""esb.labels - user-facing labels for engine keys (tables, exports, charts).

Keys are the engine's stable identifiers (docs/METHODOLOGY.md); labels are for people. Unknown
keys are humanised deterministically so every row of every table carries a readable label.
"""

from __future__ import annotations

EXPLICIT: dict[str, str] = {
    # P&L portfolio
    "metered": "Metered volume", "notified": "Notified volume", "metered_mw": "Metered volume (avg MW)",
    "notified_mw": "Notified volume (avg MW)", "revenue": "Retail revenue", "sell_price": "Average sell price",
    "gm1_budget": "GM1 budgeted", "gm1_forecast": "GM1 forecasted", "gm2_budget": "GM2 budgeted", "gm2_forecast": "GM2 forecasted",
    "source_imb": "Source imbalance", "offtaker_imb": "Off-taker imbalance", "cost_budget": "Sourcing cost budgeted",
    "cost_forecast": "Sourcing cost forecasted", "cost_pv_budget": "PV cost budgeted", "cost_pv_forecast": "PV cost forecasted",
    "cost_bl_budget": "Baseload cost budgeted", "cost_bl_forecast": "Baseload cost forecasted", "cost_spot": "Spot cost",
    "rs_volume": "Resell volume", "rs_revenue": "Resell revenue", "rs_cost_budget": "Resell cost budgeted",
    "rs_cost_forecast": "Resell cost forecasted", "rs_gm1_budget": "Resell GM1 budgeted", "rs_gm1_forecast": "Resell GM1 forecasted",
    "rs_gm2_budget": "Resell GM2 budgeted", "rs_gm2_forecast": "Resell GM2 forecasted", "rs_source_imb": "Resell source imbalance",
    "t_revenue": "Total revenue", "t_gm2_budget": "Total GM2 budgeted", "t_gm2_forecast": "Total GM2 forecasted",
    "t_imb": "Total imbalance (all legs)", "t_buy_metered": "Total bought (metered)",
    "nm_budget": "Net margin pre-tax budgeted", "nm_forecast": "Net margin pre-tax forecasted",
    "nm_budget_after_tax": "Net margin after tax budgeted", "nm_forecast_after_tax": "Net margin after tax forecasted",
    "cit_budget": "Corporate income tax budgeted", "cit_forecast": "Corporate income tax forecasted",
    "guarantees_outstanding": "Guarantees outstanding", "interest": "Financing interest", "unallocated": "Unallocated costs",
    "market_bgl_fees": "Market BGL fees", "opex": "Portfolio OPEX", "variable_opex": "Variable OPEX (sales bonus)",
    "reserve": "Risk reserve", "reserve_balance": "Risk reserve balance", "reserve_release_budget": "Reserve release budgeted",
    "reserve_release_forecast": "Reserve release forecasted", "passthrough_revenue": "Pass-through revenue",
    "passthrough_cost": "Pass-through cost", "own_guarantee": "Own guarantee outstanding", "own_bgl_fee": "Own BGL fee",
    "memo_bgl_share": "Memo: market BGL fee share", "memo_interest_share": "Memo: interest share",
    "retail_nm_budget": "Retail NM pre-tax budgeted", "retail_nm_forecast": "Retail NM pre-tax forecasted",
    "premium_monthly": "Risk premium accrued", "premium_cumulative": "Risk premium cumulative",
    "gc_count": "Green certificates", "gc_value": "GC value", "gc_unit": "GC unit cost",
    # cash flow
    "in_total": "Receipts total", "out_total": "Payments total", "vat_cash": "VAT cash", "net_cf_before_tax": "Net cash flow before tax",
    "opening": "Opening cash", "closing": "Closing cash", "injection": "Shareholder loan injection", "loan_outstanding": "Loan outstanding",
    "free_cash": "Free cash", "free_cash_after_tax": "Free cash after tax", "peak_funding": "Peak funding",
    "restricted_reserve": "Restricted: reserve", "restricted_collateral": "Restricted: collateral", "interest_cumulative": "Interest cumulative",
    "tax_paid_cumulative": "Tax paid cumulative", "vat_output": "VAT output", "vat_input": "VAT input", "vat_net_position": "VAT net position",
    "vat_paid": "VAT paid", "vat_credit": "VAT credit carried", "out_cit": "CIT paid",
    # pricing
    "purchase_cost": "Purchase cost", "purchase_price": "Purchase price", "imbalance_cost": "Imbalance cost (specific)",
    "physical_cost": "Physical cost", "premium_total": "Risk premium total", "target_gm": "Target gross margin",
    "energy_price": "Energy price", "passthrough": "Regulated pass-through", "offer_ex_vat": "Offer excl. VAT", "vat": "VAT",
    "offer_incl_vat": "Offer incl. VAT", "contract_price": "Contract price", "contract_minus_offer": "Contract minus offer",
    "implied_gm": "Implied gross margin", "cost_to_serve": "Cost to serve", "forecast_nm_specific": "Forecast NM (specific)",
    "indicative_nm": "Indicative net margin", "annual_revenue": "Annual revenue", "forecast_premium": "Forecast premium",
    "repriced_energy_price": "Re-priced energy price", "contract_minus_repriced": "Contract minus re-priced",
    "derived_forecast_premium": "Derived forecast premium",
    # overview / checks
    "qh_checks": "Quarter-hour checks (sum)", "pnl_checks": "P&L checks (sum)", "cf_checks": "Cash-flow checks (sum)",
    "ledger_checks": "Daily ledger checks (sum)", "label_self_check": "Label self-check", "strip_price_tripwire": "Strip price tripwire",
    "origin_layering_check": "Origin layering check", "total_gm1": "Total GM1", "total_gm1_pct": "Total GM1 %",
    "total_gm1_specific": "Total GM1 specific", "total_cost_revenue": "Total cost and revenue",
}
for _k, _n in (("pv", "PV source"), ("baseload", "Baseload source"), ("spot", "Spot"), ("brp", "BRP"), ("tso", "TSO"), ("dso", "DSO")):
    EXPLICIT[f"g_out_{_k}"] = f"Guarantee outstanding - {_n}"
    EXPLICIT[f"g_fee_{_k}"] = f"BGL fee - {_n}"


ABBREVIATIONS = {
    "gm1": "GM1", "gm2": "GM2", "nm": "NM", "pv": "PV", "bl": "Baseload", "rs": "Resell", "cit": "CIT", "vat": "VAT", "bgl": "BGL",
    "imb": "imbalance", "pct": "%", "mw": "MW", "mwh": "MWh", "cf": "cash flow", "acc": "accrual", "in": "receipts", "out": "payments",
    "key": "settlement key", "t": "total", "f": "forecast", "spec": "specific", "fx": "FX", "gc": "GC", "opex": "OPEX", "qh": "QH",
    "eur": "EUR", "dam": "DAM", "idct": "IDCT", "tso": "TSO", "dso": "DSO", "brp": "BRP",
}


def label(key: str) -> str:
    if key in EXPLICIT:
        return EXPLICIT[key]
    parts = key.split("_")
    words = [ABBREVIATIONS.get(p, p) for p in parts if p != ""]
    text = " ".join(words)
    return text[:1].upper() + text[1:]


def label_series(keys) -> list[str]:
    return [label(k) for k in keys]

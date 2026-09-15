"""esb.labels - user-facing labels for engine keys (tables, exports, charts).

Keys are the engine's stable identifiers (docs/METHODOLOGY.md); labels are for people. Unknown
keys are humanised deterministically so every row of every table carries a readable label.

Every metric carries the leg it is allocated to (G5 finding C1): Retail (the off-taker book),
Wholesale spot resell (surplus PV and Baseload resold to the market) or Total (both legs, and the
cost-to-serve, financing and tax lines deducted below Total GM2). Checks, market series, source
availability before allocation and parameters carry no leg tag.
"""

from __future__ import annotations

EXPLICIT: dict[str, str] = {
    # P&L portfolio
    "metered": "Metered volume", "notified": "Notified volume", "metered_mw": "Metered volume (avg MW)",
    "notified_mw": "Notified volume (avg MW)", "revenue": "Revenue", "sell_price": "Average sell price",
    "gm1_budget": "GM1 budgeted", "gm1_forecast": "GM1 forecasted", "gm2_budget": "GM2 budgeted", "gm2_forecast": "GM2 forecasted",
    "source_imb": "Source imbalance", "offtaker_imb": "Off-taker imbalance", "cost_budget": "Sourcing cost budgeted",
    "cost_forecast": "Sourcing cost forecasted", "cost_pv_budget": "PV cost budgeted", "cost_pv_forecast": "PV cost forecasted",
    "cost_bl_budget": "Baseload cost budgeted", "cost_bl_forecast": "Baseload cost forecasted", "cost_spot": "Spot cost",
    "rs_volume": "Volume resold", "rs_revenue": "Revenue", "rs_cost_budget": "Cost budgeted",
    "rs_cost_forecast": "Cost forecasted", "rs_gm1_budget": "GM1 budgeted", "rs_gm1_forecast": "GM1 forecasted",
    "rs_gm2_budget": "GM2 budgeted", "rs_gm2_forecast": "GM2 forecasted", "rs_source_imb": "Source imbalance",
    "t_revenue": "Revenue", "t_gm2_budget": "GM2 budgeted", "t_gm2_forecast": "GM2 forecasted",
    "t_imb": "Imbalance (all legs)", "t_buy_metered": "Bought (metered)",
    "nm_budget": "Net margin pre-tax budgeted", "nm_forecast": "Net margin pre-tax forecasted",
    "nm_budget_after_tax": "Net margin after tax budgeted", "nm_forecast_after_tax": "Net margin after tax forecasted",
    "cit_budget": "Corporate income tax budgeted", "cit_forecast": "Corporate income tax forecasted",
    "guarantees_outstanding": "Guarantees outstanding", "interest": "Financing interest", "unallocated": "Unallocated costs",
    "market_bgl_fees": "Market BGL fees", "opex": "Portfolio OPEX", "variable_opex": "Variable OPEX (sales bonus)",
    "reserve": "Risk reserve", "reserve_balance": "Risk reserve balance", "reserve_release_budget": "Reserve release budgeted",
    "reserve_release_forecast": "Reserve release forecasted", "passthrough_revenue": "Pass-through revenue",
    "passthrough_cost": "Pass-through cost", "own_guarantee": "Own guarantee outstanding", "own_bgl_fee": "Own BGL fee",
    "memo_bgl_share": "Memo: market BGL fee share", "memo_interest_share": "Memo: interest share",
    "retail_nm_budget": "NM pre-tax budgeted", "retail_nm_forecast": "NM pre-tax forecasted",
    "premium_monthly": "Risk premium accrued", "premium_cumulative": "Risk premium cumulative",
    "gc_count": "Green certificates", "gc_value": "GC value", "gc_unit": "GC unit cost",
    # cash flow
    "days_in_month": "Days in month", "month": "Month", "date": "Date", "interval": "Interval", "seq": "Sequence",
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
    "origin_layering_check": "Origin layering check", "total_gm1": "GM1", "total_gm1_pct": "GM1 %",
    "total_gm1_specific": "GM1 specific", "total_cost_revenue": "Cost and revenue",
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


# ---- legs (C1) -------------------------------------------------------------------------------
RETAIL = "Retail"
RESELL = "Wholesale spot resell"
TOTAL = "Total"
LEG_SEP = " · "
_LEG_TOKENS = frozenset({"rs", "resell", "retail", "t", "total"})

# portfolio-level lines that sit between Total GM2 and the net margin: deducted at Total level
_TOTAL_KEYS = frozenset({
    "opex", "variable_opex", "market_bgl_fees", "market_bgl", "guarantees_outstanding", "guarantees", "interest", "interest_cumulative",
    "unallocated", "cit", "tax_paid_cumulative", "budget_minus_forecast",
})
_RETAIL_KEYS = frozenset({
    "reserve", "reserve_balance", "reserve_release", "reserve_release_budget", "reserve_release_forecast", "passthrough", "passthrough_revenue",
    "passthrough_cost", "offtaker_bgl_fees", "offtaker_bgl", "own_guarantee", "own_bgl_fee", "memo_bgl_share", "memo_interest_share",
    "premium", "settlement_cumulative", "budget_minus_forecast_cost", "budget_minus_forecast_price", "net_imbalance_volume",
    "gc_count", "gc_spot", "gc_bilateral", "gc_value", "gc_unit", "pv_remaining", "bl_remaining", "pv_remaining_mw", "bl_remaining_mw",
    "sell_price", "market_ref_retail",
})
_RETAIL_PREFIXES = ("notified", "metered", "pv_buy_", "bl_buy_", "spot_buy_", "spot_settlement", "buy_notified", "buy_metered", "share_",
                    "cost_", "price_", "premium_", "revenue", "gm1", "gm2", "source_imb", "offtaker_imb", "retail_", "strip_",
                    "purchase_", "imbalance_cost", "physical_cost", "target_gm", "energy_price", "offer_", "vat", "contract_", "implied_gm",
                    "cost_to_serve", "forecast_", "indicative_nm", "annual_revenue", "repriced_", "derived_", "demand_", "sell_", "ratio")
_RESELL_PREFIXES = ("rs_", "resell_", "acc_resell", "in_resell", "receipts_resell")
_TOTAL_PREFIXES = ("t_", "total_", "nm_", "cit_", "g_out_", "g_fee_")
_UNTAGGED_PREFIXES = ("check", "qh_checks", "pnl_checks", "cf_checks", "ledger_checks", "label_self_check", "strip_price_tripwire",
                      "origin_layering", "dam", "idct", "surplus_price", "deficit_price", "direction", "seq", "date", "interval", "month",
                      "peak", "days_in_month", "pv_avail", "pv_metered", "pv_ratio", "pv_specific", "bl_avail", "bl_metered", "bl_ratio",
                      "bl_specific", "pv_delivered", "bl_delivered", "available", "market_ref", "gc_quota", "gc_price", "fx", "gc_spot_share",
                      "gc_bilateral_share", "gc_price_eur")


NO_LEG = ""


def leg_of(key: str) -> str | None:
    """The leg a metric key is allocated to; NO_LEG ("") for keys that are known not to be leg metrics
    (checks, market series, source availability, parameters); None for keys the module does not know."""
    k = key
    if k.startswith("OT") and "_" in k and k[2 : k.index("_")].isdigit():
        rest = k[k.index("_") + 1 :]
        return RESELL if rest.startswith("resell") else RETAIL
    if k.startswith(_UNTAGGED_PREFIXES) or k.endswith("_check"):
        return NO_LEG
    if k.startswith(_RESELL_PREFIXES):
        return RESELL
    if k in _TOTAL_KEYS or k.startswith(_TOTAL_PREFIXES):
        return TOTAL
    if k in _RETAIL_KEYS or k.startswith(_RETAIL_PREFIXES):
        return RETAIL
    return None


def label(key: str, leg: str | None = None, fallback: str | None = None) -> str:
    """Readable label with its leg tag. leg="" suppresses the tag, leg="Retail" forces it,
    fallback names the leg for keys leg_of() does not recognise (cash flow: Total)."""
    known = leg_of(key)
    tag = leg if leg is not None else (fallback if known is None else known)
    if key in EXPLICIT:
        text = EXPLICIT[key]
    else:
        parts = key.split("_")
        if tag and len(parts) > 1 and parts[0] in _LEG_TOKENS:  # the leg is in the tag; do not repeat it in the text
            parts = parts[1:]
        words = [ABBREVIATIONS.get(p, p) for p in parts if p != ""]
        text = " ".join(words)
        text = text[:1].upper() + text[1:]
    return f"{text}{LEG_SEP}{tag}" if tag else text


def plain(key: str) -> str:
    return label(key, leg="")


def tagged(text: str, leg: str) -> str:
    """Attach a leg to a free-text title (KPI cards)."""
    return f"{text}{LEG_SEP}{leg}" if leg else text


def label_series(keys, **kw) -> list[str]:
    return [label(k, **kw) for k in keys]


# ---- units (G5 request 2) --------------------------------------------------------------------
EUR, MWH, MW, EURMWH, PCT = "EUR", "MWh", "MW", "EUR/MWh", "%"
_UNIT_EXPLICIT: dict[str, str] = {
    "gc_quota": "GC/MWh", "gc_price_ron": "RON/GC", "fx": "RON/EUR", "gc_spot_share": PCT, "gc_bilateral_share": PCT, "gc_price_eur": "EUR/GC",
    "gc_count": "GC", "gc_spot": "GC", "gc_bilateral": "GC", "gc_value": EUR, "gc_unit": EURMWH,
    "sell_price": EURMWH, "market_ref": EURMWH, "dam": EURMWH, "idct": EURMWH, "dam_curtailed": EURMWH, "idct_curtailed": EURMWH,
    "surplus_price": EURMWH, "deficit_price": EURMWH, "dam_monthly": EURMWH, "idm_monthly": EURMWH, "direction": "sign",
    "pv_ratio": "ratio", "bl_ratio": "ratio", "ratio": "ratio", "peak": "flag", "seq": "#", "interval": "#", "month": "#", "days_in_month": "days",
    "premium_deviation": EURMWH, "budget_minus_forecast": EUR, "budget_minus_forecast_price": EURMWH, "budget_minus_forecast_cost": EUR,
    "days_to_settle": "days", "spot_buy_mwh": MWH, "strip_price_tripwire": "#", "label_self_check": "", "peak_dam_price": EURMWH,
    "peak_retail_buy_mw": MW, "peak_daily_spot_buy_mwh": MWH, "cumulative_s": "s", "net_imbalance_volume": MWH, "date": "",
    "vat_cash_year": EUR, "vat_collected": EUR, "vat_paid": EUR, "vat_input": EUR, "strip_notified": MWH,
}
_VOLUME_TOKENS = ("notified", "metered", "volume", "delivered", "avail", "remaining", "settlement", "buy", "resold", "attributed", "count")
_PRICE_TOKENS = ("price", "specific", "purchase_price", "imbalance_cost", "physical_cost", "premium_", "target_gm", "energy_price", "offer_",
                 "contract_", "implied_gm", "cost_to_serve", "forecast_nm_specific", "indicative_nm", "repriced", "derived_forecast_premium",
                 "forecast_premium", "passthrough", "tariff", "premium", "vat")


def unit_of(key: str) -> str:
    """The unit of an engine key: EUR unless the key names a volume (MWh), a rate (EUR/MWh), a share (%) or a special case."""
    k = key
    if k.startswith("OT") and "_" in k and k[2 : k.index("_")].isdigit():
        k = k[k.index("_") + 1 :]
    if k in _UNIT_EXPLICIT:
        return _UNIT_EXPLICIT[k]
    if k.startswith("check") or k.endswith("_check") or k.endswith("_checks"):
        return EUR
    if k.endswith("_pct") or k.startswith("share_") or k.endswith("_share"):
        return PCT
    if k.endswith("_mw") or k.startswith("strip_"):
        return MW
    if k == "premium_volume" or k.startswith("premium_") and k not in ("premium_monthly", "premium_cumulative", "premium_settlement",
                                                                         "premium_settlement_cumulative", "premium_total_eur"):
        return EURMWH if k in ("premium_volume", "premium_price", "premium_profile", "premium_credit", "premium_regulatory", "premium_fx",
                               "premium_collateral", "premium_total", "premium_budget_input", "premium_forecast_input", "premium_forecast_derived") else EUR
    if k in ("premium_monthly", "premium_cumulative", "premium_settlement", "premium_settlement_cumulative", "premium", "reserve",
             "reserve_balance", "reserve_release", "reserve_release_budget", "reserve_release_forecast", "settlement_cumulative", "passthrough_revenue",
             "passthrough_cost", "annual_revenue", "vat_cash", "vat_output", "vat_input", "vat_net_position", "vat_paid", "vat_credit"):
        return EUR
    if any(t in k for t in ("_specific", "price_", "_price")) or k.startswith(_PRICE_TOKENS) or k in ("passthrough", "vat"):
        return EURMWH
    if any(t in k for t in _VOLUME_TOKENS) and not any(t in k for t in ("cost", "revenue", "imb", "gm", "fee", "acc_", "in_", "out_", "pay_", "receipts")):
        return MWH
    return EUR


def unit_series(keys) -> list[str]:
    return [unit_of(k) for k in keys]

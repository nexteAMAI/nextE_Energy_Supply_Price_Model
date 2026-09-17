"""esb.reporting - the portfolio overview (workbook sheet `Portf Overview`) and the run-level
reconciliation (section 7 tripwires).

Sections 1-4 are year values of the P&L: portfolio Budgeted / Forecasted / delta, one Budgeted
and Forecasted column per off-taker, and the wholesale resell column. Section 5 reads the
monthly cash flow and the daily ledger, section 6 the pricing calculator of the selected
off-taker, section 7 the reconciliation checks.

`OVERVIEW_ROWS` maps overview keys to (workbook row, portfolio budget key, portfolio forecast
key, section budget key, section forecast key, resell key); a None entry means the workbook cell
is empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.schema import Parameters
from esb.cashflow import CashflowResult
from esb.merit_order import QHResult
from esb.pnl import PnLResult
from esb.pricing import PricingResult

# key: (row, portfolio budget, portfolio forecast, section budget, section forecast, resell)
OVERVIEW_ROWS: dict[str, tuple] = {
    "notified": (15, "notified", "notified", "notified", "notified", "rs_volume"),
    "metered": (16, "metered", "metered", "metered", "metered", None),
    "pv_buy_notified": (17, "pv_buy_notified", "pv_buy_notified", "pv_buy_notified", "pv_buy_notified", "rs_pv_volume"),
    "bl_buy_notified": (18, "bl_buy_notified", "bl_buy_notified", "bl_buy_notified", "bl_buy_notified", "rs_bl_volume"),
    "spot_buy_notified": (19, "spot_buy_notified", "spot_buy_notified", "spot_buy_notified", "spot_buy_notified", None),
    "pv_buy_metered": (20, "pv_buy_metered", "pv_buy_metered", "pv_buy_metered", "pv_buy_metered", None),
    "bl_buy_metered": (21, "bl_buy_metered", "bl_buy_metered", "bl_buy_metered", "bl_buy_metered", None),
    "buy_metered": (22, "buy_metered", "buy_metered", "buy_metered", "buy_metered", None),
    "share_pv_bl": (23, "share_pv", "share_bl", "share_pv", "share_bl", None),
    "available": (24, "rs_pv_available", "rs_bl_available", None, None, None),
    "price_pv": (28, "price_pv_budget", "price_pv_forecast", "price_pv_budget", "price_pv_forecast", None),
    "price_bl": (29, "price_bl_budget", "price_bl_forecast", "price_bl_budget", "price_bl_forecast", None),
    "price_spot": (30, "price_spot", "price_spot", "price_spot", "price_spot", None),
    "price_total": (31, "price_total_budget", "price_total_forecast", "price_total_budget", "price_total_forecast", None),
    "sell_price": (32, "sell_price", "sell_price", "sell_price", "sell_price", None),
    "market_ref": (33, "dam_monthly", "idm_monthly", None, None, None),
    "revenue": (37, "revenue", "revenue", "revenue", "revenue", "rs_revenue"),
    "resell_revenue": (38, "rs_revenue", "rs_revenue", None, None, None),
    "total_revenue": (39, "t_revenue", "t_revenue", None, None, None),
    "cost_pv": (40, "cost_pv_budget", "cost_pv_forecast", "cost_pv_budget", "cost_pv_forecast", "rs_pv_cost"),
    "cost_bl": (41, "cost_bl_budget", "cost_bl_forecast", "cost_bl_budget", "cost_bl_forecast", None),
    "cost_spot": (42, "cost_spot", "cost_spot", "cost_spot", "cost_spot", None),
    "cost_total": (43, "cost_budget", "cost_forecast", "cost_budget", "cost_forecast", "rs_cost_forecast"),
    "resell_cost": (44, "rs_cost_budget", "rs_cost_forecast", None, None, None),
    "total_cost": (45, "t_cost_budget", "t_cost_forecast", None, None, None),
    "gm1": (47, "gm1_budget", "gm1_forecast", "gm1_budget", "gm1_forecast", None),
    "gm1_pct": (48, "gm1_budget_pct", "gm1_forecast_pct", "gm1_budget_pct", "gm1_forecast_pct", None),
    "gm1_specific": (49, "gm1_budget_specific", "gm1_forecast_specific", "gm1_budget_specific", "gm1_forecast_specific", None),
    "resell_gm1": (51, "rs_gm1_budget", "rs_gm1_forecast", None, None, None),
    "resell_gm1_pct": (52, "rs_gm1_budget_pct", "rs_gm1_forecast_pct", None, None, None),
    "resell_gm1_specific": (53, "rs_gm1_budget_specific", "rs_gm1_forecast_specific", None, None, None),
    "source_imb": (59, None, "source_imb", None, "source_imb", "rs_source_imb"),
    "offtaker_imb": (60, None, "offtaker_imb", None, "offtaker_imb", None),
    "resell_source_imb": (61, None, "rs_source_imb", None, None, None),
    "gm2": (63, "gm2_budget", "gm2_forecast", "gm2_budget", "gm2_forecast", "rs_gm2_forecast"),
    "gm2_pct": (64, "gm2_budget_pct", "gm2_forecast_pct", "gm2_budget_pct", "gm2_forecast_pct", "rs_gm2_forecast_pct"),
    "gm2_specific": (65, "gm2_budget_specific", "gm2_forecast_specific", "gm2_budget_specific", "gm2_forecast_specific", "rs_gm2_forecast_specific"),
    "resell_gm2": (67, "rs_gm2_budget", "rs_gm2_forecast", None, None, None),
    "resell_gm2_pct": (68, "rs_gm2_budget_pct", "rs_gm2_forecast_pct", None, None, None),
    "resell_gm2_specific": (69, "rs_gm2_budget_specific", "rs_gm2_forecast_specific", None, None, None),
    "total_gm2": (71, "t_gm2_budget", "t_gm2_forecast", None, None, None),
    "total_gm2_pct": (72, "t_gm2_budget_pct", "t_gm2_forecast_pct", None, None, None),
    "total_gm2_specific": (73, "t_gm2_budget_specific", "t_gm2_forecast_specific", None, None, None),
    "total_cost_revenue": (74, "t_cost_budget", "t_cost_forecast", None, None, None),
    "budget_minus_forecast": (78, None, "budget_minus_forecast_cost", None, "budget_minus_forecast_cost", None),
    "premium": (79, None, None, "premium_budget_input", "premium_forecast_derived", None),
    "reserve": (80, "reserve", "reserve", "reserve", "reserve", None),
    "settlement_cumulative": (81, None, "premium_settlement_cumulative", None, "premium_settlement_cumulative", None),
    "reserve_release": (82, "reserve_release_budget", "reserve_release_budget", "reserve_release_budget", "reserve_release_budget", None),
    "passthrough": (83, "passthrough_revenue", "passthrough_revenue", "passthrough_revenue", "passthrough_revenue", None),
    "opex": (84, "opex", "opex", "opex", "opex", None),
    "variable_opex": (85, "variable_opex", "variable_opex", "variable_opex", "variable_opex", None),
    "market_bgl": (86, "market_bgl_fees", "market_bgl_fees", "memo_bgl_share", "memo_bgl_share", None),
    "offtaker_bgl": (87, "offtaker_bgl_fees", "offtaker_bgl_fees", "own_bgl_fee", "own_bgl_fee", None),
    "guarantees": (88, "guarantees_outstanding", "guarantees_outstanding", "own_guarantee", "own_guarantee", None),
    "interest": (89, "interest", "interest", "memo_interest_share", "memo_interest_share", None),
    "pre_service_fee": (None, "pre_service_fee", "pre_service_fee", None, None, None),  # D120 - no workbook row
    "pre_aggregation_gain": (None, "pre_aggregation_gain", "pre_aggregation_gain", None, None, None),  # D120 - no workbook row
    "nm": (91, "nm_budget", "nm_forecast", None, None, None),
    "nm_pct": (92, "nm_budget_pct", "nm_forecast_pct", None, None, None),
    "nm_specific": (93, "nm_budget_specific", "nm_forecast_specific", None, None, None),
    "retail_nm": (95, "retail_nm_budget", "retail_nm_forecast", "retail_nm_budget", "retail_nm_forecast", None),
    "retail_nm_pct": (96, "retail_nm_budget_pct", "retail_nm_forecast_pct", "retail_nm_budget_pct", "retail_nm_forecast_pct", None),
    "retail_nm_specific": (97, "retail_nm_budget_specific", "retail_nm_forecast_specific", "retail_nm_budget_specific", "retail_nm_forecast_specific", None),
    "resell_nm": (99, "resell_nm_budget", "resell_nm_forecast", None, None, None),
    "resell_nm_pct": (100, "resell_nm_budget_pct", "resell_nm_forecast_pct", None, None, None),
    "resell_nm_specific": (101, "resell_nm_budget_specific", "resell_nm_forecast_specific", None, None, None),
    "unallocated": (103, "unallocated", "unallocated", None, None, None),
    "check_legs": (104, "check_legs_budget", "check_legs_forecast", None, None, None),
    "cit": (106, "cit_budget", "cit_forecast", None, None, None),
    "nm_after_tax": (108, "nm_budget_after_tax", "nm_forecast_after_tax", None, None, None),
    "nm_after_tax_pct": (109, "nm_budget_after_tax_pct", "nm_forecast_after_tax_pct", None, None, None),
    "nm_after_tax_specific": (110, "nm_budget_after_tax_specific", "nm_forecast_after_tax_specific", None, None, None),
}
# overview columns for off-taker position p (1-based): budget = H + 3(p-1), forecast = I + 3(p-1)
OFFTAKER_COL_BASE = 8  # H
RESELL_COL = 20  # T


@dataclass
class OverviewResult:
    table: pd.DataFrame  # index: overview keys; columns: portfolio_budget, portfolio_forecast, delta, <code>_budget, <code>_forecast, resell
    cashflow: dict[str, float] = field(default_factory=dict)  # section 5 (rows 114-128)
    pricing: dict[str, float] = field(default_factory=dict)  # section 6 (rows 133-139)
    checks: dict[str, float] = field(default_factory=dict)  # section 7 (rows 143-149)
    selected_offtaker: str = ""


def build_overview(qh: QHResult, pnl: PnLResult, cf: CashflowResult, pricing: PricingResult, params: Parameters) -> OverviewResult:
    P = pnl.portfolio
    cols = ["portfolio_budget", "portfolio_forecast", "delta"]
    for c in pnl.codes:
        cols += [f"{c}_budget", f"{c}_forecast"]
    cols.append("resell")
    data: dict[str, list] = {}
    for key, (_row, pb, pf, sb, sf, rs) in OVERVIEW_ROWS.items():
        vals: list = []
        b = P.y(pb) if pb else np.nan
        f = P.y(pf) if pf else np.nan
        vals += [b, f, (f - b) if (pb and pf and pb != pf) else np.nan]
        for c in pnl.codes:
            T = pnl.sections[c]
            vals.append(T.y(sb) if sb and sb in T else np.nan)
            vals.append(T.y(sf) if sf and sf in T else np.nan)
        vals.append(P.y(rs) if rs else np.nan)
        data[key] = vals
    table = pd.DataFrame(data, index=cols).T
    table.loc["total_cost_revenue", "delta"] = P.y("t_revenue")  # workbook row 74: F = total sell revenue
    # rows 55-57: total GM1 = retail GM1 + resell GM1 (per view)
    g1 = table.loc["gm1", ["portfolio_budget", "portfolio_forecast"]].to_numpy() + table.loc["resell_gm1", ["portfolio_budget", "portfolio_forecast"]].to_numpy()
    rev = P.y("t_revenue")
    sold = P.y("metered") + P.y("rs_volume")
    table.loc["total_gm1"] = np.nan
    table.loc["total_gm1", ["portfolio_budget", "portfolio_forecast", "delta"]] = [g1[0], g1[1], g1[1] - g1[0]]
    table.loc["total_gm1_pct"] = np.nan
    table.loc["total_gm1_pct", ["portfolio_budget", "portfolio_forecast", "delta"]] = [g1[0] / rev if rev else 0, g1[1] / rev if rev else 0, (g1[1] - g1[0]) / rev if rev else 0]
    table.loc["total_gm1_specific"] = np.nan
    table.loc["total_gm1_specific", ["portfolio_budget", "portfolio_forecast", "delta"]] = [g1[0] / sold if sold else 0, g1[1] / sold if sold else 0, (g1[1] - g1[0]) / sold if sold else 0]

    cash = {
        "inflows_year": cf.y("in_total"), "outflows_year": cf.y("out_total"), "vat_cash_year": cf.y("vat_cash"),
        "net_cf_year": cf.y("net_cf_before_tax"), "inflows_beyond": float(cf.m13("in_total")[12]), "outflows_beyond": float(cf.m13("out_total")[12]),
        "injections_year": cf.y("injection"), "peak_funding_monthly": cf.y("peak_funding"),
        "peak_funding_daily": cf.daily_summary.get("peak_funding", np.nan), "daily_vs_monthly": cf.daily_summary.get("peak_vs_monthly", np.nan),
        "interest_year": cf.y("interest"), "tax_paid_year": -float(cf.m("out_cit").sum()), "closing_dec": float(cf.m("closing")[11]),
        "restricted_dec": float(cf.m("restricted_reserve")[11] + cf.m("restricted_collateral")[11]), "free_cash_after_tax_dec": float(cf.m("free_cash_after_tax")[11]),
    }
    y = pricing.year
    pr = {"physical_cost": y["physical_cost"], "premium_plus_gm": y["premium_total"] + y["target_gm"], "energy_price": y["energy_price"],
          "offer_ex_vat": y["offer_ex_vat"], "contract_price": y["contract_price"], "contract_minus_offer": y["contract_minus_offer"],
          "repriced_energy_price": y["repriced_energy_price"]}
    q = qh.qh
    checks = {
        "qh_checks": float(q["check_demand"].sum() + q["check_pv"].sum() + q["check_bl"].sum() + q["check_imb"].sum()),
        "pnl_checks": float(sum(P.y(k) for k in ("check_demand", "check_pv", "check_bl", "check_imb", "check_sections_buy", "check_sections_cost",
                                                  "check_legs_budget", "check_legs_forecast", "check_sections_nm_budget", "check_sections_nm_forecast"))),
        "cf_checks": float(cf.y("check_revenue") + cf.y("check_purchases") + cf.y("check_closing")),
        "ledger_checks": float(cf.daily_summary.get("check_net_cf", 0.0) + cf.daily_summary.get("check_receipts", 0.0)),
        "label_self_check": 0.0,  # not applicable: the engine addresses rows by key, never by label
        "strip_price_tripwire": strip_price_tripwire(params),
        "origin_layering_check": float(np.round(q["check_origin"].sum(), 2)),
    }
    return OverviewResult(table=table, cashflow=cash, pricing=pr, checks=checks, selected_offtaker=pricing.code)


def strip_price_tripwire(params: Parameters) -> float:
    """Portf Overview!E148: count of (off-taker, month, product) with strip MW > 0 and a 0 product price
    in the budgeted or the forecast table."""
    n = 0
    for o in params.offtakers:
        for p in ("BL24", "Peak", "OffPeak"):
            for i in range(12):
                if o.strip_mw[p][i] > 0 and (o.product_price_budget[p][i] == 0 or o.product_price_forecast[p][i] == 0):
                    n += 1
    return float(n)

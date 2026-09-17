"""esb.pnl - the monthly P&L (workbook sheet `Cons_P&L`): portfolio block, wholesale resell block,
total block, cost-to-serve, guarantees, net margin, corporate income tax, margin by leg, and one
section per off-taker.

Two stages, because the financing interest (CF_Mth row 89) enters the net margin (row 220):
    stage 1  everything down to the total guarantees outstanding (row 218) and the sections'
             retail net margins (rows 385 / 388 do not need interest);
    stage 2  after the monthly cash flow: interest (220), net margins (225 / 228), corporate
             income tax (232 / 237), after-tax rows and the leg checks.

Row keys are semantic
WORKBOOK_ROWS / SECTION_ROWS map them to the Reference Case rows for
the parity tie-out (SPEC section 8). Section rows are relative to the section anchor
(OT1 = row 266)
positions 3 and 4 carry one spacer row after relative row 55 (X-27 note).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config.schema import Parameters
from esb import guarantees as gr
from esb.merit_order import QHResult
from esb.monthly import MonthlyTable, QHAggregator, ratio_row, safe_div

SECTION_ANCHOR = 266
SECTION_PITCH = 136

# portfolio rows (Cons_P&L column A row numbers)
WORKBOOK_ROWS = {
    "gc_quota": 6, "gc_price_ron": 7, "fx": 8, "gc_spot_share": 9, "gc_bilateral_share": 10, "gc_price_eur": 11,
    "notified": 17, "notified_mw": 18, "metered": 19, "metered_mw": 20, "net_imbalance_volume": 21,
    "pv_buy_notified": 22, "pv_buy_notified_mw": 23, "bl_buy_notified": 24, "bl_buy_notified_mw": 25,
    "spot_buy_notified": 26, "spot_buy_notified_mw": 27, "buy_notified": 28, "buy_notified_mw": 29,
    "pv_buy_metered": 30, "pv_buy_metered_mw": 31, "bl_buy_metered": 32, "bl_buy_metered_mw": 33,
    "spot_settlement": 34, "spot_settlement_mw": 35, "buy_metered": 36, "buy_metered_mw": 37,
    "share_pv": 39, "share_bl": 40, "share_spot": 41,
    "price_pv_budget": 45, "price_bl_budget": 46, "price_spot": 47, "price_total_budget": 48,
    "cost_pv_budget": 49, "cost_bl_budget": 50, "cost_spot": 51, "cost_budget": 52,
    "price_pv_forecast": 56, "price_bl_forecast": 57, "price_spot_f": 58, "price_total_forecast": 59,
    "cost_pv_forecast": 60, "cost_bl_forecast": 61, "cost_spot_f": 62, "cost_forecast": 63,
    "budget_minus_forecast_cost": 67, "budget_minus_forecast_price": 68,
    "premium_monthly": 70, "premium_cumulative": 71, "premium_settlement": 72, "premium_settlement_cumulative": 73,
    "premium_deviation": 74,
    "sell_price": 78, "revenue": 79, "gm1_budget": 81, "gm1_budget_pct": 82, "gm1_budget_specific": 83,
    "gm1_forecast": 85, "gm1_forecast_pct": 86, "gm1_forecast_specific": 87,
    "source_imb": 89, "offtaker_imb": 90, "gm2_budget": 92, "gm2_budget_pct": 93, "gm2_budget_specific": 94,
    "gm2_forecast": 96, "gm2_forecast_pct": 97, "gm2_forecast_specific": 98,
    "gc_count": 102, "gc_spot": 103, "gc_bilateral": 104, "gc_value": 105, "gc_unit": 106,
    "rs_pv_available": 110, "rs_pv_available_mw": 111, "rs_pv_delivered": 112, "rs_pv_delivered_mw": 113,
    "rs_pv_volume": 114, "rs_pv_volume_mw": 115, "rs_pv_cost": 116, "rs_pv_revenue": 117, "rs_pv_source_imb": 118,
    "rs_pv_gm2": 119, "rs_pv_gm2_pct": 120, "rs_pv_gm2_specific": 121,
    "rs_bl_available": 123, "rs_bl_available_mw": 124, "rs_bl_delivered": 125, "rs_bl_delivered_mw": 126,
    "rs_bl_volume": 127, "rs_bl_volume_mw": 128, "rs_bl_cost_forecast": 129, "rs_bl_cost_budget": 130,
    "rs_bl_revenue": 131, "rs_bl_source_imb": 132, "rs_bl_gm2": 133, "rs_bl_gm2_pct": 134, "rs_bl_gm2_specific": 135,
    "rs_volume": 137, "rs_volume_mw": 138, "rs_cost_forecast": 139, "rs_cost_budget": 140, "rs_revenue": 141,
    "rs_source_imb": 142, "rs_gm1_budget": 144, "rs_gm1_budget_pct": 145, "rs_gm1_budget_specific": 146,
    "rs_gm1_forecast": 148, "rs_gm1_forecast_pct": 149, "rs_gm1_forecast_specific": 150,
    "rs_gm2_forecast": 152, "rs_gm2_forecast_pct": 153, "rs_gm2_forecast_specific": 154,
    "rs_gm2_budget": 156, "rs_gm2_budget_pct": 157, "rs_gm2_budget_specific": 158,
    "t_buy_notified": 162, "t_buy_notified_mw": 163, "t_buy_metered": 164, "t_buy_metered_mw": 165,
    "t_cost_budget": 166, "t_cost_forecast": 167, "t_revenue": 168, "t_imb": 169,
    "t_gm2_budget": 171, "t_gm2_budget_pct": 172, "t_gm2_budget_specific": 173,
    "t_gm2_forecast": 175, "t_gm2_forecast_pct": 176, "t_gm2_forecast_specific": 177, "t_gm2_forecast_specific_bought": 178,
    "dam_monthly": 182, "idm_monthly": 183,
    "check_demand": 187, "check_pv": 188, "check_bl": 189, "check_imb": 190, "check_sections_buy": 191, "check_sections_cost": 192,
    "passthrough_revenue": 196, "passthrough_cost": 197, "reserve": 199, "opex": 200, "variable_opex": 201, "offtaker_bgl_fees": 202,
    "g_out_pv": 204, "g_out_baseload": 205, "g_out_spot": 206, "g_out_brp": 207, "g_out_tso": 208, "g_out_dso": 209,
    "g_fee_pv": 211, "g_fee_baseload": 212, "g_fee_spot": 213, "g_fee_brp": 214, "g_fee_tso": 215, "g_fee_dso": 216,
    "market_bgl_fees": 217, "guarantees_outstanding": 218, "interest": 220, "reserve_release_budget": 221,
    "reserve_release_forecast": 222, "reserve_balance": 223,
    "nm_budget": 225, "nm_budget_pct": 226, "nm_budget_specific": 227, "nm_forecast": 228, "nm_forecast_pct": 229,
    "nm_forecast_specific": 230, "cit_budget": 232, "nm_budget_after_tax": 234, "nm_budget_after_tax_pct": 235,
    "nm_budget_after_tax_specific": 236, "cit_forecast": 237, "nm_forecast_after_tax": 238, "nm_forecast_after_tax_pct": 239,
    "nm_forecast_after_tax_specific": 240,
    "retail_nm_budget": 244, "retail_nm_budget_pct": 245, "retail_nm_budget_specific": 246,
    "retail_nm_forecast": 247, "retail_nm_forecast_pct": 248, "retail_nm_forecast_specific": 249,
    "resell_nm_budget": 252, "resell_nm_budget_pct": 253, "resell_nm_budget_specific": 254,
    "resell_nm_forecast": 255, "resell_nm_forecast_pct": 256, "resell_nm_forecast_specific": 257,
    "unallocated": 259, "check_legs_budget": 261, "check_legs_forecast": 262, "check_sections_nm_budget": 263, "check_sections_nm_forecast": 264,
}
# section rows relative to the anchor (OT1 row 266 -> 0)
SECTION_ROWS = {
    "notified": 4, "notified_mw": 5, "metered": 6, "metered_mw": 7, "net_imbalance_volume": 8,
    "pv_buy_notified": 9, "pv_buy_notified_mw": 10, "bl_buy_notified": 11, "bl_buy_notified_mw": 12,
    "spot_buy_notified": 13, "spot_buy_notified_mw": 14, "buy_notified": 15, "buy_notified_mw": 16,
    "pv_buy_metered": 17, "pv_buy_metered_mw": 18, "bl_buy_metered": 19, "bl_buy_metered_mw": 20,
    "spot_settlement": 21, "spot_settlement_mw": 22, "buy_metered": 23, "buy_metered_mw": 24,
    "share_pv": 26, "share_bl": 27, "share_spot": 28,
    "price_pv_budget": 32, "price_bl_budget": 33, "price_spot": 34, "price_total_budget": 35,
    "cost_pv_budget": 36, "cost_bl_budget": 37, "cost_spot": 38, "cost_budget": 39,
    "price_pv_forecast": 43, "price_bl_forecast": 44, "price_spot_f": 45, "price_total_forecast": 46,
    "cost_pv_forecast": 47, "cost_bl_forecast": 48, "cost_spot_f": 49, "cost_forecast": 50,
    "budget_minus_forecast_cost": 54, "budget_minus_forecast_price": 55,
    "premium_monthly": 56, "premium_cumulative": 57, "premium_settlement": 58, "premium_settlement_cumulative": 59,
    "premium_deviation": 61, "premium_budget_input": 62, "premium_forecast_input": 63, "premium_forecast_derived": 64,
    "reserve_release_budget": 65, "reserve_release_forecast": 66,
    "sell_price": 70, "revenue": 71, "gm1_budget": 73, "gm1_budget_pct": 74, "gm1_budget_specific": 75,
    "gm1_forecast": 77, "gm1_forecast_pct": 78, "gm1_forecast_specific": 79, "source_imb": 81, "offtaker_imb": 82,
    "gm2_budget": 84, "gm2_budget_pct": 85, "gm2_budget_specific": 86, "gm2_forecast": 88, "gm2_forecast_pct": 89,
    "gm2_forecast_specific": 90, "gc_count": 94, "gc_spot": 95, "gc_bilateral": 96, "gc_value": 97, "gc_unit": 98,
    "pv_remaining": 102, "pv_remaining_mw": 103, "bl_remaining": 104, "bl_remaining_mw": 105,
    "passthrough_revenue": 109, "passthrough_cost": 110, "reserve": 111, "opex": 112, "variable_opex": 113,
    "own_guarantee": 114, "own_bgl_fee": 115, "memo_bgl_share": 116, "memo_interest_share": 117,
    "retail_nm_budget": 119, "retail_nm_budget_pct": 120, "retail_nm_budget_specific": 121,
    "retail_nm_forecast": 122, "retail_nm_forecast_pct": 123, "retail_nm_forecast_specific": 124,
}
SPACER_AFTER_REL = 55  # positions 3+ have one extra row after relative row 55


def section_row(position: int, key: str) -> int:
    """Workbook row of a section key for merit-order position 1..n (Reference Case layout)."""
    rel = SECTION_ROWS[key]
    row = SECTION_ANCHOR + (position - 1) * SECTION_PITCH + rel
    if position >= 3 and rel > SPACER_AFTER_REL:
        row += 1
    return row


@dataclass
class PnLResult:
    portfolio: MonthlyTable
    sections: dict[str, MonthlyTable]
    regulatory: gr.RegulatoryAmounts
    reg_inputs: gr.RegulatoryInputs
    pv_fixed_guarantee: float  # Input!C85 as derived
    codes: list[str] = field(default_factory=list)
    stage: int = 1


def _month_key_of_date(dates: np.ndarray) -> np.ndarray:
    return dates.astype("datetime64[M]").astype(int) % 12 + 1


def build_pnl(qh: QHResult, params: Parameters) -> PnLResult:
    """Stage 1 of the monthly P&L."""
    q = qh.qh
    agg = QHAggregator(q["month"].to_numpy())
    year = params.spine_year
    P = MonthlyTable()
    codes = qh.codes

    def sums(col: str) -> np.ndarray:
        return agg.sum(q[col].to_numpy())

    def spec_col(code: str, name: str) -> np.ndarray:
        return q[f"{code}_{name}"].to_numpy()

    # ---- GC block ----------------------------------------------------------------------
    for key, val in (("gc_quota", params.gc_quota), ("gc_price_ron", params.gc_reference_price_ron), ("fx", params.general.fx_ron_per_eur),
                     ("gc_spot_share", params.gc_spot_share), ("gc_bilateral_share", 1 - params.gc_spot_share),
                     ("gc_price_eur", params.gc_reference_price_eur)):
        P.put(key, np.full(12, np.nan), val)  # single cells in the workbook (column B); the year slot carries the value

    # ---- sections first (the portfolio rows 22-34, 49-51, 60-62 sum the sections) ------------
    S: dict[str, MonthlyTable] = {}
    for o in params.offtakers:
        c = o.code
        T = MonthlyTable()
        T.put("notified", sums(f"{c}_demand_notified"))
        m, y = agg.mean_positive(spec_col(c, "demand_notified") * 4)
        T.put("notified_mw", m, y)
        T.put("metered", sums(f"{c}_demand_metered"))
        m, y = agg.mean_positive(spec_col(c, "demand_metered") * 4)
        T.put("metered_mw", m, y)
        T.put("net_imbalance_volume", T.m("metered") - T.m("notified"), T.y("metered") - T.y("notified"))
        for key, col in (("pv_buy_notified", "pv_buy_notified"), ("bl_buy_notified", "bl_buy_notified"), ("spot_buy_notified", "spot_buy_notified"),
                         ("buy_notified", "buy_notified"), ("pv_buy_metered", "pv_buy_metered"), ("bl_buy_metered", "bl_buy_metered"),
                         ("spot_settlement", "spot_buy_notified"), ("buy_metered", "buy_metered")):
            T.put(key, sums(f"{c}_{col}"))
            m, y = agg.mean_positive(spec_col(c, col) * 4)
            T.put(key + "_mw", m, y)
        ratio_row(T, "share_pv", "pv_buy_notified", "buy_notified")
        ratio_row(T, "share_bl", "bl_buy_notified", "buy_notified")
        ratio_row(T, "share_spot", "spot_buy_notified", "buy_notified")
        T.put("cost_pv_budget", sums(f"{c}_pv_cost_budget"))
        T.put("cost_bl_budget", sums(f"{c}_bl_cost_budget"))
        T.put("cost_spot", sums(f"{c}_spot_cost"))
        T.put("cost_budget", sums(f"{c}_cost_budget"))
        ratio_row(T, "price_pv_budget", "cost_pv_budget", "pv_buy_metered")
        ratio_row(T, "price_bl_budget", "cost_bl_budget", "bl_buy_metered")
        ratio_row(T, "price_spot", "cost_spot", "spot_settlement")
        ratio_row(T, "price_total_budget", "cost_budget", "buy_metered")
        T.put("cost_pv_forecast", sums(f"{c}_pv_cost_forecast"))
        T.put("cost_bl_forecast", sums(f"{c}_bl_cost_forecast"))
        T.put("cost_spot_f", sums(f"{c}_spot_cost"))
        T.put("cost_forecast", sums(f"{c}_cost_forecast"))
        ratio_row(T, "price_pv_forecast", "cost_pv_forecast", "pv_buy_metered")
        ratio_row(T, "price_bl_forecast", "cost_bl_forecast", "bl_buy_metered")
        ratio_row(T, "price_spot_f", "cost_spot", "spot_settlement")
        ratio_row(T, "price_total_forecast", "cost_forecast", "buy_metered")
        T.put("budget_minus_forecast_cost", T.m("cost_budget") - T.m("cost_forecast"), T.y("cost_budget") - T.y("cost_forecast"))
        T.put("budget_minus_forecast_price", T.m("price_total_budget") - T.m("price_total_forecast"), T.y("price_total_budget") - T.y("price_total_forecast"))
        prem = T.m("metered") * o.premium_budget_total
        T.put("premium_monthly", prem)
        T.put("premium_cumulative", np.cumsum(prem), year_rule="last")
        settle = T.m("budget_minus_forecast_cost") + prem
        T.put("premium_settlement", settle, T.y("budget_minus_forecast_cost") + T.y("premium_monthly"))
        T.put("premium_settlement_cumulative", np.cumsum(settle), year_rule="last")
        T.put("premium_deviation", T.m("premium_cumulative") - T.m("premium_settlement_cumulative"), year_rule="last")
        T.put("premium_budget_input", np.full(12, o.premium_budget_total), o.premium_budget_total)
        T.put("premium_forecast_input", np.full(12, o.premium_forecast_total), o.premium_forecast_total)
        cum_met = np.cumsum(T.m("metered"))
        T.put("premium_forecast_derived", safe_div(T.m("premium_settlement_cumulative"), cum_met),
              float(safe_div(np.array([T.m("premium_settlement_cumulative")[-1]]), np.array([T.y("metered")]))[0]))
        # reserve release in the month containing the contract end date (both views use the budgeted cumulative)
        release = np.zeros(12)
        end = o.contract_end
        if end.year == year:
            release[end.month - 1] = T.m("premium_cumulative")[end.month - 1]
        T.put("reserve_release_budget", release)
        T.put("reserve_release_forecast", release)
        T.put("revenue", sums(f"{c}_revenue"))
        ratio_row(T, "sell_price", "revenue", "metered")
        for key, col in (("gm1_budget", "gm1_budget"), ("gm1_forecast", "gm1_forecast"), ("source_imb", "source_imb"),
                         ("offtaker_imb", "offtaker_imb"), ("gm2_budget", "gm2_budget"), ("gm2_forecast", "gm2_forecast")):
            T.put(key, sums(f"{c}_{col}"))
        for key in ("gm1_budget", "gm1_forecast", "gm2_budget", "gm2_forecast"):
            ratio_row(T, key + "_pct", key, "revenue")
            ratio_row(T, key + "_specific", key, "metered")
        T.put("gc_count", params.gc_quota * T.m("metered"), params.gc_quota * T.y("metered"))
        T.put("gc_spot", params.gc_spot_share * T.m("gc_count"), params.gc_spot_share * T.y("gc_count"))
        T.put("gc_bilateral", (1 - params.gc_spot_share) * T.m("gc_count"), (1 - params.gc_spot_share) * T.y("gc_count"))
        T.put("gc_value", params.gc_reference_price_eur * T.m("gc_count"), params.gc_reference_price_eur * T.y("gc_count"))
        ratio_row(T, "gc_unit", "gc_value", "metered")
        T.put("pv_remaining", sums(f"{c}_pv_remaining_after"))
        m, y = agg.mean_positive(spec_col(c, "pv_remaining_after") * 4)
        T.put("pv_remaining_mw", m, y)
        T.put("bl_remaining", sums(f"{c}_bl_remaining_after"))
        m, y = agg.mean_positive(spec_col(c, "bl_remaining_after") * 4)
        T.put("bl_remaining_mw", m, y)
        tariff = params.tariff_total_for(o)
        T.put("passthrough_revenue", T.m("metered") * tariff)
        T.put("passthrough_cost", T.m("passthrough_revenue"))
        T.put("reserve", T.m("premium_monthly"))
        T.put("opex", T.m("metered") * params.general.portfolio_opex_eur_per_mwh_metered)
        T.put("variable_opex", T.m("metered") * params.general.variable_opex_eur_per_mwh_metered)
        own = gr.offtaker_outstanding(o.guarantee, o.active, T.y("revenue"), T.m("revenue"), year)
        T.put("own_guarantee", own, year_rule="max")
        T.put("own_bgl_fee", gr.bgl_fee(o.guarantee, own, year))
        T.put("retail_nm_budget", T.m("gm2_budget") - T.m("reserve") - T.m("opex") - T.m("variable_opex") - T.m("own_bgl_fee") + T.m("reserve_release_budget"))
        T.put("retail_nm_forecast", T.m("gm2_forecast") - T.m("reserve") - T.m("opex") - T.m("variable_opex") - T.m("own_bgl_fee") + T.m("reserve_release_forecast"))
        for key in ("retail_nm_budget", "retail_nm_forecast"):
            ratio_row(T, key + "_pct", key, "revenue")
            ratio_row(T, key + "_specific", key, "metered")
        S[c] = T

    def sum_sections(key: str) -> np.ndarray:
        return np.sum([S[c][key] for c in codes], axis=0) if codes else np.zeros(13)

    # ---- portfolio retail block -------------------------------------------------------------
    P.put("notified", sums("retail_sell_notified"))
    m, y = agg.mean_positive(q["retail_sell_notified"].to_numpy() * 4)
    P.put("notified_mw", m, y)
    P.put("metered", sums("retail_sell_metered"))
    m, y = agg.mean_positive(q["retail_sell_metered"].to_numpy() * 4)
    P.put("metered_mw", m, y)
    P.put("net_imbalance_volume", P.m("metered") - P.m("notified"), P.y("metered") - P.y("notified"))
    for key, qcol in (("pv_buy_notified", "pv_delivered"), ("bl_buy_notified", "bl_delivered"), ("spot_buy_notified", "spot_buy_notified"),
                      ("pv_buy_metered", "retail_pv_buy_metered"), ("bl_buy_metered", "retail_bl_buy_metered"), ("spot_settlement", "retail_spot_settlement")):
        row = sum_sections(key)
        P.put(key, row[:12], float(row[12]))
        m, y = agg.mean_positive(q[qcol].to_numpy() * 4)
        P.put(key + "_mw", m, y)
    P.put("buy_notified", sums("retail_buy_notified"))
    m, y = agg.mean_positive(q["retail_buy_notified"].to_numpy() * 4)
    P.put("buy_notified_mw", m, y)
    P.put("buy_metered", sums("retail_buy_metered"))
    m, y = agg.mean_positive(q["retail_buy_metered"].to_numpy() * 4)
    P.put("buy_metered_mw", m, y)
    ratio_row(P, "share_pv", "pv_buy_notified", "buy_notified")
    ratio_row(P, "share_bl", "bl_buy_notified", "buy_notified")
    ratio_row(P, "share_spot", "spot_buy_notified", "buy_notified")
    for key in ("cost_pv_budget", "cost_bl_budget", "cost_spot", "cost_pv_forecast", "cost_bl_forecast"):
        row = sum_sections(key)
        P.put(key, row[:12], float(row[12]))
    P.put("cost_budget", sums("retail_cost_budget"))
    P.put("cost_spot_f", P.m("cost_spot"), P.y("cost_spot"))
    P.put("cost_forecast", sums("retail_cost_forecast"))
    ratio_row(P, "price_pv_budget", "cost_pv_budget", "pv_buy_metered")
    ratio_row(P, "price_bl_budget", "cost_bl_budget", "bl_buy_metered")
    ratio_row(P, "price_spot", "cost_spot", "spot_settlement")
    ratio_row(P, "price_total_budget", "cost_budget", "buy_metered")
    ratio_row(P, "price_pv_forecast", "cost_pv_forecast", "pv_buy_metered")
    ratio_row(P, "price_bl_forecast", "cost_bl_forecast", "bl_buy_metered")
    ratio_row(P, "price_spot_f", "cost_spot", "spot_settlement")
    ratio_row(P, "price_total_forecast", "cost_forecast", "buy_metered")
    P.put("budget_minus_forecast_cost", P.m("cost_budget") - P.m("cost_forecast"), P.y("cost_budget") - P.y("cost_forecast"))
    P.put("budget_minus_forecast_price", P.m("price_total_budget") - P.m("price_total_forecast"), P.y("price_total_budget") - P.y("price_total_forecast"))
    prem = sum_sections("premium_monthly")
    P.put("premium_monthly", prem[:12], float(prem[12]))
    P.put("premium_cumulative", np.cumsum(P.m("premium_monthly")), year_rule="last")
    P.put("premium_settlement", P.m("budget_minus_forecast_cost") + P.m("premium_monthly"), P.y("budget_minus_forecast_cost") + P.y("premium_monthly"))
    P.put("premium_settlement_cumulative", np.cumsum(P.m("premium_settlement")), year_rule="last")
    P.put("premium_deviation", np.full(12, np.nan), P.y("premium_monthly") - P.y("premium_settlement"))
    P.put("revenue", sums("retail_revenue"))
    ratio_row(P, "sell_price", "revenue", "metered")
    for key, col in (("gm1_budget", "retail_gm1_budget"), ("gm1_forecast", "retail_gm1_forecast"), ("source_imb", "retail_source_imb"),
                     ("offtaker_imb", "retail_offtaker_imb"), ("gm2_budget", "retail_gm2_budget"), ("gm2_forecast", "retail_gm2_forecast")):
        P.put(key, sums(col))
    for key in ("gm1_budget", "gm1_forecast", "gm2_budget", "gm2_forecast"):
        ratio_row(P, key + "_pct", key, "revenue")
        ratio_row(P, key + "_specific", key, "metered")
    P.put("gc_count", params.gc_quota * P.m("metered"), params.gc_quota * P.y("metered"))
    P.put("gc_spot", params.gc_spot_share * P.m("gc_count"), params.gc_spot_share * P.y("gc_count"))
    P.put("gc_bilateral", (1 - params.gc_spot_share) * P.m("gc_count"), (1 - params.gc_spot_share) * P.y("gc_count"))
    P.put("gc_value", params.gc_reference_price_eur * P.m("gc_count"), params.gc_reference_price_eur * P.y("gc_count"))
    ratio_row(P, "gc_unit", "gc_value", "metered")

    # ---- wholesale resell block -------------------------------------------------------------
    for key, col in (("rs_pv_available", "pv_avail_notified"), ("rs_pv_delivered", "pv_delivered"), ("rs_pv_volume", "resell_pv_volume"),
                     ("rs_bl_available", "bl_avail_notified"), ("rs_bl_delivered", "bl_delivered"), ("rs_bl_volume", "resell_bl_volume"),
                     ("rs_volume", "resell_total_volume")):
        P.put(key, sums(col))
        m, y = agg.mean_positive(q[col].to_numpy() * 4)
        P.put(key + "_mw", m, y)
    for key, col in (("rs_pv_cost", "resell_pv_cost"), ("rs_pv_revenue", "resell_pv_revenue"), ("rs_pv_source_imb", "resell_pv_source_imb"),
                     ("rs_pv_gm2", "resell_pv_gm2"), ("rs_bl_cost_forecast", "resell_bl_cost_forecast"), ("rs_bl_cost_budget", "resell_bl_cost_budget"),
                     ("rs_bl_revenue", "resell_bl_revenue"), ("rs_bl_source_imb", "resell_bl_source_imb"), ("rs_bl_gm2", "resell_bl_gm2"),
                     ("rs_cost_forecast", "resell_total_cost"), ("rs_revenue", "resell_total_revenue"), ("rs_source_imb", "resell_total_source_imb"),
                     ("rs_gm2_forecast", "resell_total_gm2")):
        P.put(key, sums(col))
    ratio_row(P, "rs_pv_gm2_pct", "rs_pv_gm2", "rs_pv_revenue")
    ratio_row(P, "rs_pv_gm2_specific", "rs_pv_gm2", "rs_pv_volume")
    ratio_row(P, "rs_bl_gm2_pct", "rs_bl_gm2", "rs_bl_revenue")
    ratio_row(P, "rs_bl_gm2_specific", "rs_bl_gm2", "rs_bl_volume")
    P.put("rs_cost_budget", P.m("rs_pv_cost") + P.m("rs_bl_cost_budget"))
    P.put("rs_gm1_budget", P.m("rs_revenue") - P.m("rs_cost_budget"))
    P.put("rs_gm1_forecast", P.m("rs_revenue") - P.m("rs_cost_forecast"))
    P.put("rs_gm2_budget", P.m("rs_revenue") - P.m("rs_cost_budget") + P.m("rs_source_imb"))
    for key in ("rs_gm1_budget", "rs_gm1_forecast", "rs_gm2_forecast", "rs_gm2_budget"):
        ratio_row(P, key + "_pct", key, "rs_revenue")
        ratio_row(P, key + "_specific", key, "rs_volume")

    # ---- total block -------------------------------------------------------------------------
    P.put("t_buy_notified", sums("total_buy_notified"))
    m, y = agg.mean_positive(q["total_buy_notified"].to_numpy() * 4)
    P.put("t_buy_notified_mw", m, y)
    P.put("t_buy_metered", sums("total_buy_metered"))
    m, y = agg.mean_positive(q["total_buy_metered"].to_numpy() * 4)
    P.put("t_buy_metered_mw", m, y)
    for key, col in (("t_cost_budget", "total_cost_budget"), ("t_cost_forecast", "total_cost_forecast"), ("t_revenue", "total_revenue"),
                     ("t_imb", "total_imb_all_legs"), ("t_gm2_budget", "total_gm2_budget"), ("t_gm2_forecast", "total_gm2_forecast")):
        P.put(key, sums(col))
    sold = P["metered"] + P["rs_volume"]
    for key in ("t_gm2_budget", "t_gm2_forecast"):
        ratio_row(P, key + "_pct", key, "t_revenue")
        r = safe_div(P[key], sold)
        P.put(key + "_specific", r[:12], float(r[12]))
    r = safe_div(P["t_gm2_forecast"], P["t_buy_metered"])
    P.put("t_gm2_forecast_specific_bought", r[:12], float(r[12]))
    P.put("dam_monthly", agg.mean(q["dam"].to_numpy()), year_rule="mean")
    P.put("idm_monthly", agg.mean(q["idct"].to_numpy()), year_rule="mean")
    for key, col in (("check_demand", "check_demand"), ("check_pv", "check_pv"), ("check_bl", "check_bl"), ("check_imb", "check_imb")):
        P.put(key, sums(col))
    P.put("check_sections_buy", P.m("buy_notified") - P.m("pv_buy_notified") - P.m("bl_buy_notified") - P.m("spot_buy_notified"),
          P.y("buy_notified") - P.y("pv_buy_notified") - P.y("bl_buy_notified") - P.y("spot_buy_notified"))
    P.put("check_sections_cost", P.m("cost_forecast") - P.m("cost_pv_forecast") - P.m("cost_bl_forecast") - P.m("cost_spot_f"),
          P.y("cost_forecast") - P.y("cost_pv_forecast") - P.y("cost_bl_forecast") - P.y("cost_spot_f"))

    # ---- cost-to-serve and guarantees ------------------------------------------------------
    for key in ("passthrough_revenue", "passthrough_cost", "reserve", "variable_opex"):
        row = sum_sections(key)
        P.put(key, row[:12], float(row[12]))
    P.put("opex", P.m("metered") * params.general.portfolio_opex_eur_per_mwh_metered)
    row = sum_sections("own_bgl_fee")
    P.put("offtaker_bgl_fees", row[:12], float(row[12]))

    # regulatory inputs (Input section E)
    daily_spot = q.groupby(q["date"].values)["spot_buy_notified"].sum().to_numpy()
    reg_inp = gr.RegulatoryInputs(
        peak_daily_spot_buy_mwh=float(daily_spot.max()) if len(daily_spot) else 0.0,
        peak_dam_price=float(np.nanmax(q["dam"].to_numpy())),
        peak_retail_buy_mw=float(q["retail_buy_notified_mw"].max()),
        metered_year_by_offtaker={c: S[c].y("metered") for c in codes},
        imbalance_value_monthly=P.m("source_imb") + P.m("offtaker_imb") + P.m("rs_source_imb"),
    )
    reg = gr.regulatory_amounts(params, reg_inp)
    cp = params.counterparties
    pv_derived = float(P.m("cost_pv_budget").mean() + P.m("rs_pv_cost").mean())  # Input!C85 as the workbook derives it
    # D110: the PV fixed amount is a user input; the workbook derivation is the default when the register holds none
    pv_fixed = float(cp["pv"].guarantee.fixed_amount) if cp["pv"].guarantee.fixed_amount is not None else pv_derived
    bases = {"pv": P.y("t_revenue"), "baseload": P.y("cost_bl_forecast") + P.y("rs_bl_cost_forecast"), "spot": P.y("t_revenue"),
             "brp": P.y("t_revenue"), "tso": P.y("t_revenue"), "dso": P.y("t_revenue")}
    required = {"pv": None, "baseload": None, **reg.by_counterparty()}
    fixed = {k: (pv_fixed if k == "pv" else float(cp[k].guarantee.fixed_amount or 0.0)) for k in cp}
    total_out = np.zeros(12)
    total_fee = np.zeros(12)
    for k in ("pv", "baseload", "spot", "brp", "tso", "dso"):
        out = gr.counterparty_outstanding(cp[k].guarantee, cp[k].active, bases[k], required[k], fixed[k], year)
        fee = gr.bgl_fee(cp[k].guarantee, out, year)
        P.put(f"g_out_{k}", out, year_rule="max")
        P.put(f"g_fee_{k}", fee)
        total_out = total_out + out
        total_fee = total_fee + fee
    P.put("market_bgl_fees", total_fee)
    own = sum_sections("own_guarantee")[:12]
    P.put("guarantees_outstanding", total_out + own, year_rule="max")
    rel_b = sum_sections("reserve_release_budget")
    P.put("reserve_release_budget", rel_b[:12], float(rel_b[12]))
    rel_f = sum_sections("reserve_release_forecast")
    P.put("reserve_release_forecast", rel_f[:12], float(rel_f[12]))
    P.put("reserve_balance", np.cumsum(P.m("reserve")) - np.cumsum(P.m("reserve_release_budget")), year_rule="last")

    # retail leg (does not need interest) and resell leg
    P.put("retail_nm_budget", P.m("gm2_budget") - P.m("reserve") - P.m("opex") - P.m("variable_opex") - P.m("offtaker_bgl_fees") + P.m("reserve_release_budget"))
    P.put("retail_nm_forecast", P.m("gm2_forecast") - P.m("reserve") - P.m("opex") - P.m("variable_opex") - P.m("offtaker_bgl_fees") + P.m("reserve_release_forecast"))
    for key in ("retail_nm_budget", "retail_nm_forecast"):
        ratio_row(P, key + "_pct", key, "revenue")
        ratio_row(P, key + "_specific", key, "metered")
    P.put("resell_nm_budget", P.m("rs_gm2_budget"))
    P.put("resell_nm_forecast", P.m("rs_gm2_forecast"))
    for key in ("resell_nm_budget", "resell_nm_forecast"):
        ratio_row(P, key + "_pct", key, "rs_revenue")
        ratio_row(P, key + "_specific", key, "rs_volume")
    return PnLResult(portfolio=P, sections=S, regulatory=reg, reg_inputs=reg_inp, pv_fixed_guarantee=pv_fixed, codes=codes, stage=1)


def finalize_pnl(pnl: PnLResult, interest_monthly: np.ndarray, params: Parameters) -> PnLResult:
    """Stage 2: interest, net margins, corporate income tax, after-tax rows, legs and checks."""
    P = pnl.portfolio
    P.put("interest", np.asarray(interest_monthly, dtype=float))
    P.put("unallocated", P.m("market_bgl_fees") + P.m("interest"))
    cost = P.m("reserve") + P.m("opex") + P.m("variable_opex") + P.m("offtaker_bgl_fees") + P.m("market_bgl_fees") + P.m("interest")
    P.put("nm_budget", P.m("t_gm2_budget") - cost + P.m("reserve_release_budget"))
    P.put("nm_forecast", P.m("t_gm2_forecast") - cost + P.m("reserve_release_forecast"))
    sold = P["metered"] + P["rs_pv_volume"] + P["rs_bl_volume"]
    rate = params.general.cit_rate
    for view in ("budget", "forecast"):
        nm = P.m(f"nm_{view}")
        cit = np.zeros(12)
        paid = 0.0
        for i in range(12):
            if (i + 1) % 3 == 0:
                cit[i] = max(0.0, max(0.0, nm[: i + 1].sum()) * rate - paid)
                paid += cit[i]
        P.put(f"cit_{view}", cit)
        P.put(f"nm_{view}_after_tax", nm - cit)
        for key in (f"nm_{view}", f"nm_{view}_after_tax"):
            ratio_row(P, key + "_pct", key, "t_revenue")
            r = safe_div(P[key], sold)
            P.put(key + "_specific", r[:12], float(r[12]))
    P.put("check_legs_budget", P.m("retail_nm_budget") + P.m("resell_nm_budget") - P.m("unallocated") - P.m("nm_budget"))
    P.put("check_legs_forecast", P.m("retail_nm_forecast") + P.m("resell_nm_forecast") - P.m("unallocated") - P.m("nm_forecast"))
    sec_b = np.sum([pnl.sections[c].m("retail_nm_budget") for c in pnl.codes], axis=0) if pnl.codes else np.zeros(12)
    sec_f = np.sum([pnl.sections[c].m("retail_nm_forecast") for c in pnl.codes], axis=0) if pnl.codes else np.zeros(12)
    P.put("check_sections_nm_budget", sec_b - P.m("retail_nm_budget"))
    P.put("check_sections_nm_forecast", sec_f - P.m("retail_nm_forecast"))
    for c in pnl.codes:
        T = pnl.sections[c]
        share = safe_div(T.m("metered"), P.m("metered"))
        T.put("memo_bgl_share", P.m("market_bgl_fees") * share)
        T.put("memo_interest_share", P.m("interest") * share)
    pnl.stage = 2
    return pnl

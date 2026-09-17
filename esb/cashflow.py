"""esb.cashflow - monthly cash flow (workbook sheet `CF_Mth`) and daily ledger (`CF_Daily_Ledger`).

Monthly (columns: 12 months, "beyond" = settles after December, "year" = total / max):
    accruals from the P&L -> settlement month keys (month containing month-end + payment terms)
    -> inflows / outflows net of VAT with advance shares -> VAT (output on retail receipts, input on
    grid payments and, without reverse charge, on PV / Baseload payments; net position on accruals;
    payment to ANAF the month after; credit carried) -> net operating cash flow before tax ->
    restricted cash (risk reserve balance + collateral backing of guarantees) -> shareholder loan
    injection MAX(0, restricted - (opening + net CF)) -> closing cash, loan outstanding, interest
    (outstanding x rate / 12) -> corporate tax payments the month after each quarter.

The interest feeds the P&L net margin (Cons_P&L row 220); the tax lines read the P&L's forecast
CIT accrual (row 237). The injection never reads the tax, so the chain is acyclic: stage A
(everything but the tax lines) -> P&L stage 2 -> stage B (tax lines, cumulative tax, free cash
after tax).

Daily ledger: settlements on the exact day (month end + terms; advances on day 1), spot and
resell spread evenly over the days of the settlement month, fixed OPEX on the last day, BGL
fees on day 1, VAT and CIT on the payment day, a daily restricted floor (previous month's
reserve balance + this month's reserve pro rata by day - release at month end + collateral),
daily loan injection MAX(0, floor - (cumulative CF + loan so far)), daily interest x rate / 365.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from config.schema import Parameters
from esb.merit_order import QHResult
from esb.monthly import MONTHS
from esb.pnl import PnLResult

BEYOND = 12  # index of the "beyond December" column
YEAR = 13

# workbook rows of CF_Mth for the parity tie-out (row -> key)
CF_ROWS = {
    17: "acc_resell_revenue", 18: "acc_imbalance", 20: "acc_pv_purchases", 21: "acc_bl_purchases", 22: "acc_spot_purchases",
    23: "acc_grid_cost", 25: "acc_opex", 26: "acc_variable_opex", 27: "acc_bgl_fees", 28: "acc_reserve", 29: "acc_cit", 30: "acc_guarantees",
    38: "key_pv", 39: "key_bl", 40: "key_spot", 41: "key_imbalance", 42: "key_grid",
    54: "in_resell", 55: "in_imbalance", 56: "in_total",
    60: "out_pv", 61: "out_bl", 62: "out_spot", 63: "out_grid", 64: "out_opex", 65: "out_variable_opex", 66: "out_bgl", 67: "out_cit", 68: "out_total",
    72: "vat_output", 73: "vat_input", 74: "vat_net_position", 75: "vat_paid", 76: "vat_credit", 77: "vat_cash",
    81: "net_cf_before_tax", 82: "opening", 83: "restricted_reserve", 84: "restricted_collateral", 85: "injection", 86: "closing",
    87: "free_cash", 88: "loan_outstanding", 89: "interest", 90: "interest_cumulative", 91: "tax_paid_cumulative", 92: "free_cash_after_tax",
    96: "peak_funding",
}
# per off-taker rows: revenue accrual 7.., pass-through accrual 12.., receipt keys 34.., energy receipts 46.., pass-through receipts 50..
CF_OFFTAKER_ROWS = {"acc_revenue": 7, "acc_passthrough": 12, "key_receipts": 34, "in_energy": 46, "in_passthrough": 50}


def _eom(d: date) -> date:
    return date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def settlement_keys(year: int, terms_days: int) -> list[date]:
    """CF_Mth rows 34-42: EOMONTH(EOMONTH(month start, 0) + terms, -1) + 1 for each month."""
    return [_month_start(_eom(date(year, m, 1)) + timedelta(days=int(terms_days))) for m in MONTHS]


def _settle(accrual: np.ndarray, keys: list[date], year: int, advance: float) -> np.ndarray:
    """13 values: 12 months + beyond. (1 - adv) x sum of accruals whose key is the month + adv x own accrual;
    beyond = (1 - adv) x sum of accruals whose key lies after December."""
    out = np.zeros(13)
    starts = [date(year, m, 1) for m in MONTHS]
    for i, s in enumerate(starts):
        due = sum(accrual[j] for j in range(12) if keys[j] == s)
        out[i] = (1 - advance) * due + advance * accrual[i]
    out[BEYOND] = (1 - advance) * sum(accrual[j] for j in range(12) if keys[j] > starts[-1])
    return out


@dataclass
class CashflowResult:
    rows: dict[str, np.ndarray] = field(default_factory=dict)  # 14 slots: months, beyond, year
    codes: list[str] = field(default_factory=list)
    keys: dict[str, list[date]] = field(default_factory=dict)
    daily: pd.DataFrame | None = None
    daily_summary: dict[str, float] = field(default_factory=dict)
    daily_monthly: pd.DataFrame | None = None
    stage: str = "A"

    def put(self, key: str, values13: np.ndarray, year: float | None = None, year_rule: str = "sum") -> np.ndarray:
        v = np.asarray(values13, dtype=float)
        if v.shape == (12,):
            v = np.concatenate([v, [0.0]])
        if v.shape != (13,):
            raise ValueError(key)
        if year is None:
            year = float(v.sum()) if year_rule == "sum" else (float(v[:12].max()) if year_rule == "max" else float(v[11]))
        self.rows[key] = np.concatenate([v, [float(year)]])
        return self.rows[key]

    def m(self, key: str) -> np.ndarray:
        return self.rows[key][:12]

    def m13(self, key: str) -> np.ndarray:
        return self.rows[key][:13]

    def y(self, key: str) -> float:
        return float(self.rows[key][YEAR])

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, index=[*MONTHS, "beyond", "year"]).T


def build_cashflow(pnl: PnLResult, params: Parameters) -> CashflowResult:
    """Stage A of the monthly cash flow (no tax lines)."""
    P = pnl.portfolio
    S = pnl.sections
    g = params.general
    year = params.spine_year
    cf = CashflowResult(codes=list(pnl.codes))
    zero13 = np.zeros(13)

    # ---- accruals (rows 7-30) --------------------------------------------------------------
    for c in pnl.codes:
        cf.put(f"{c}_acc_revenue", S[c].m("revenue"))
        cf.put(f"{c}_acc_passthrough", S[c].m("passthrough_revenue"))
    cf.put("acc_resell_revenue", P.m("rs_pv_revenue") + P.m("rs_bl_revenue"))
    cf.put("acc_imbalance", P.m("t_imb"))
    cf.put("acc_pv_purchases", P.m("cost_pv_forecast") + P.m("rs_pv_cost"))
    cf.put("acc_bl_purchases", P.m("cost_bl_forecast") + P.m("rs_bl_cost_forecast"))
    cf.put("acc_spot_purchases", P.m("cost_spot"))
    cf.put("acc_grid_cost", P.m("passthrough_cost"))
    cf.put("acc_opex", P.m("opex"))
    cf.put("acc_variable_opex", P.m("variable_opex"))
    cf.put("acc_bgl_fees", P.m("offtaker_bgl_fees") + P.m("market_bgl_fees"))
    cf.put("acc_pre_service", P.m("pre_service_fee") - P.m("pre_aggregation_gain"))  # D120: net PRE service, invoiced at month end
    cf.put("acc_reserve", P.m("reserve"))
    cf.put("acc_guarantees", P.m("guarantees_outstanding"), year_rule="max")

    # ---- settlement keys (rows 34-42) --------------------------------------------------------
    cp = params.counterparties
    for o in params.offtakers:
        cf.keys[o.code] = settlement_keys(year, o.payment_terms_days)
    cf.keys["pv"] = settlement_keys(year, cp["pv"].payment_terms_days)
    cf.keys["bl"] = settlement_keys(year, cp["baseload"].payment_terms_days)
    cf.keys["spot"] = settlement_keys(year, cp["spot"].payment_terms_days)
    cf.keys["imbalance"] = settlement_keys(year, cp["brp"].payment_terms_days)
    cf.keys["grid"] = settlement_keys(year, cp["tso"].payment_terms_days)  # Input!C23 = G80 (TSO terms)

    # ---- inflows (rows 46-56) ----------------------------------------------------------------
    total_in = zero13.copy()
    for o in params.offtakers:
        c = o.code
        e = _settle(cf.m(f"{c}_acc_revenue"), cf.keys[c], year, o.advance_pct)
        pt = _settle(cf.m(f"{c}_acc_passthrough"), cf.keys[c], year, o.advance_pct)
        cf.put(f"{c}_in_energy", e)
        cf.put(f"{c}_in_passthrough", pt)
        total_in = total_in + e + pt
    rs = _settle(cf.m("acc_resell_revenue"), cf.keys["spot"], year, 0.0)
    imb = _settle(cf.m("acc_imbalance"), cf.keys["imbalance"], year, 0.0)
    cf.put("in_resell", rs)
    cf.put("in_imbalance", imb)
    cf.put("in_total", total_in + rs + imb)

    # ---- outflows (rows 60-66; 67 in stage B) -----------------------------------------------
    cf.put("out_pv", -_settle(cf.m("acc_pv_purchases"), cf.keys["pv"], year, cp["pv"].advance_pct))
    cf.put("out_bl", -_settle(cf.m("acc_bl_purchases"), cf.keys["bl"], year, cp["baseload"].advance_pct))
    cf.put("out_spot", -_settle(cf.m("acc_spot_purchases"), cf.keys["spot"], year, cp["spot"].advance_pct))
    cf.put("out_grid", -_settle(cf.m("acc_grid_cost"), cf.keys["grid"], year, 0.0))
    cf.put("out_opex", np.concatenate([-cf.m("acc_opex"), [0.0]]))
    vo = zero13.copy()
    vo[BEYOND] = -cf.y("acc_variable_opex")
    cf.put("out_variable_opex", vo)
    cf.put("out_bgl", np.concatenate([-cf.m("acc_bgl_fees"), [0.0]]))
    cf.put("out_pre_service", np.concatenate([[0.0], -cf.m("acc_pre_service")]))  # paid within 4 working days of the month-end invoice
    cf.put("out_cit", zero13.copy())
    cf.put("out_total", cf.m13("out_pv") + cf.m13("out_bl") + cf.m13("out_spot") + cf.m13("out_grid") + cf.m13("out_opex")
           + cf.m13("out_variable_opex") + cf.m13("out_bgl") + cf.m13("out_pre_service") + cf.m13("out_cit"))

    # ---- VAT (rows 72-77) --------------------------------------------------------------------
    vat = g.vat_rate
    rc = g.reverse_charge_vat_on_sources
    retail_receipts = sum(cf.m13(f"{c}_in_energy") + cf.m13(f"{c}_in_passthrough") for c in pnl.codes) if pnl.codes else zero13
    cf.put("vat_output", retail_receipts * vat, year=float((retail_receipts * vat)[:12].sum()))  # O72 = SUM(B:M)
    src_pay = zero13 if rc else (cf.m13("out_pv") + cf.m13("out_bl"))
    vin = (cf.m13("out_grid") + src_pay + cf.m13("out_pre_service")) * vat  # the PRE service invoices carry VAT (contract art. 8.12, A2.1)
    cf.put("vat_input", vin, year=float(vin[:12].sum()))  # O73 = SUM(B:M)
    retail_acc = sum(cf.m(f"{c}_acc_revenue") + cf.m(f"{c}_acc_passthrough") for c in pnl.codes) if pnl.codes else np.zeros(12)
    src_acc = np.zeros(12) if rc else (cf.m("acc_pv_purchases") + cf.m("acc_bl_purchases"))
    net_pos = (retail_acc - cf.m("acc_grid_cost") - src_acc - cf.m("acc_pre_service")) * vat
    cf.put("vat_net_position", np.concatenate([net_pos, [0.0]]))
    paid = np.zeros(13)
    credit = np.zeros(12)
    credit[0] = min(0.0, net_pos[0])
    for i in range(1, 12):
        paid[i] = -max(0.0, net_pos[i - 1] + min(0.0, credit[i - 1]))
        credit[i] = min(0.0, credit[i - 1] + net_pos[i])
    paid[BEYOND] = -max(0.0, net_pos[11] + min(0.0, credit[10]))
    cf.put("vat_paid", paid)
    cf.put("vat_credit", np.concatenate([credit, [0.0]]))
    vat_cash = cf.m13("vat_output") + cf.m13("vat_input") + cf.m13("vat_paid")
    vat_cash[BEYOND] = paid[BEYOND]
    cf.put("vat_cash", vat_cash)

    # ---- funding (rows 81-90) ----------------------------------------------------------------
    net = cf.m13("in_total") + cf.m13("out_total") - cf.m13("out_cit") + cf.m13("vat_cash")
    cf.put("net_cf_before_tax", net)
    reserve_bal = P.m("reserve_balance")
    coll = np.zeros(12)
    for k in ("pv", "baseload", "spot", "brp", "tso", "dso"):
        coll = coll + P.m(f"g_out_{k}") * cp[k].guarantee.cash_backing_pct
    for o in params.offtakers:
        coll = coll + S[o.code].m("own_guarantee") * o.guarantee.cash_backing_pct
    opening = np.zeros(12)
    injection = np.zeros(12)
    closing = np.zeros(12)
    prev_close = g.opening_cash_eur
    for i in range(12):
        opening[i] = prev_close
        injection[i] = max(0.0, reserve_bal[i] + coll[i] - (opening[i] + net[i]))
        closing[i] = opening[i] + net[i] + injection[i]
        prev_close = closing[i]
    cf.put("opening", opening, year=float("nan"))
    cf.put("restricted_reserve", reserve_bal, year=float("nan"))
    cf.put("restricted_collateral", coll, year=float("nan"))
    cf.put("injection", injection)
    cf.put("closing", closing, year=float("nan"))
    cf.put("free_cash", closing - reserve_bal - coll, year=float("nan"))
    loan = np.cumsum(injection)
    cf.put("loan_outstanding", loan, year=float("nan"))
    interest = loan * g.shareholder_loan_rate_pa / 12.0
    cf.put("interest", interest)
    cf.put("interest_cumulative", np.cumsum(interest), year=float("nan"))
    cf.put("peak_funding", loan, year_rule="max")
    cf.stage = "A"
    return cf


def finalize_cashflow(cf: CashflowResult, pnl: PnLResult, params: Parameters) -> CashflowResult:
    """Stage B: tax accrual and payments, cumulative tax, free cash after tax, checks."""
    P = pnl.portfolio
    cit = P.m("cit_forecast")
    cf.put("acc_cit", cit)
    out_cit = np.zeros(13)
    for i in range(1, 12):
        if (i + 1) % 3 == 1:
            out_cit[i] = -cit[i - 1]
    out_cit[BEYOND] = -cit[11]
    cf.put("out_cit", out_cit)
    cf.put("out_total", cf.m13("out_pv") + cf.m13("out_bl") + cf.m13("out_spot") + cf.m13("out_grid") + cf.m13("out_opex")
           + cf.m13("out_variable_opex") + cf.m13("out_bgl") + cf.m13("out_pre_service") + out_cit)
    cf.put("tax_paid_cumulative", np.cumsum(out_cit[:12]), year=float("nan"))
    cf.put("free_cash_after_tax", cf.m("free_cash") + cf.m("tax_paid_cumulative"), year=float("nan"))
    checks = {
        "check_revenue": float(sum(cf.y(f"{c}_acc_revenue") + cf.y(f"{c}_acc_passthrough") - cf.y(f"{c}_in_energy") - cf.y(f"{c}_in_passthrough") for c in cf.codes)),
        "check_purchases": float(cf.y("acc_pv_purchases") + cf.y("acc_bl_purchases") + cf.y("acc_spot_purchases") + cf.y("acc_grid_cost")
                                 + cf.y("out_pv") + cf.y("out_bl") + cf.y("out_spot") + cf.y("out_grid")),
        "check_closing": float(cf.m("closing")[11] - (params.general.opening_cash_eur + cf.m("net_cf_before_tax").sum() + cf.m("injection").sum())),
    }
    for k, v in checks.items():
        cf.put(k, np.zeros(13), year=v)
    cf.stage = "B"
    return cf


# ---- daily ledger ------------------------------------------------------------------------------
LEDGER_COLUMNS = {
    "receipts_resell": "H", "imbalance": "I", "pay_pv": "J", "pay_bl": "K", "pay_spot": "L", "pay_grid": "M", "opex": "N", "bgl": "O",
    "vat_paid": "P", "cit_paid": "Q", "vat_collected": "R", "vat_input": "S", "net_cf": "T", "cum_cf": "U", "floor": "V", "injection": "W",
    "loan": "X", "funded_balance": "Y", "free_cash": "Z", "cum_tax": "AA", "free_cash_after_tax": "AB", "interest": "AC", "spot_buy_mwh": "AJ",
}


def build_daily_ledger(cf: CashflowResult, pnl: PnLResult, qh: QHResult, params: Parameters) -> CashflowResult:
    P = pnl.portfolio
    g = params.general
    year = params.spine_year
    cp = params.counterparties
    days = [date(year, 1, 1) + timedelta(days=i) for i in range((date(year, 12, 31) - date(year, 1, 1)).days + 1)]
    n = len(days)
    month_starts = [date(year, m, 1) for m in MONTHS]
    month_ends = [_eom(s) for s in month_starts]
    mi = np.array([d.month - 1 for d in days])
    dim = np.array([calendar.monthrange(year, d.month)[1] for d in days])
    is_eom = np.array([d == _eom(d) for d in days])
    is_first = np.array([d.day == 1 for d in days])
    is_payday = np.array([d.day == g.tax_payment_day for d in days])
    L = pd.DataFrame({"date": pd.to_datetime(days), "month": mi + 1, "days_in_month": dim})

    def on_day(monthly_acc: np.ndarray, terms: int, advance: float, month_key_amount: np.ndarray | None = None) -> np.ndarray:
        """SUMPRODUCT((month end + terms = day) x accrual) x (1 - adv) + SUMPRODUCT((month start = day) x accrual) x adv."""
        out = np.zeros(n)
        for j in range(12):
            due = month_ends[j] + timedelta(days=int(terms))
            if due.year == year:
                out[(due - date(year, 1, 1)).days] += (1 - advance) * monthly_acc[j]
            if advance:
                out[(month_starts[j] - date(year, 1, 1)).days] += advance * monthly_acc[j]
        return out

    def spread(monthly: np.ndarray, terms: int) -> np.ndarray:
        """INDEX(monthly, month of (day - terms)) / days in that month, 0 outside the year."""
        out = np.zeros(n)
        for i, d in enumerate(days):
            ref = d - timedelta(days=int(terms))
            if ref.year == year:
                out[i] = monthly[ref.month - 1] / calendar.monthrange(year, ref.month)[1]
        return out

    receipts_total = np.zeros(n)
    for o in params.offtakers:
        c = o.code
        r = on_day(cf.m(f"{c}_acc_revenue") + cf.m(f"{c}_acc_passthrough"), o.payment_terms_days, o.advance_pct)
        L[f"{c}_receipts"] = r
        receipts_total += r
    L["receipts_resell"] = spread(cf.m("acc_resell_revenue"), cp["spot"].payment_terms_days)
    L["imbalance"] = on_day(cf.m("acc_imbalance"), cp["brp"].payment_terms_days, 0.0)
    L["pay_pv"] = -on_day(cf.m("acc_pv_purchases"), cp["pv"].payment_terms_days, cp["pv"].advance_pct)
    L["pay_bl"] = -on_day(cf.m("acc_bl_purchases"), cp["baseload"].payment_terms_days, cp["baseload"].advance_pct)
    L["pay_spot"] = -spread(cf.m("acc_spot_purchases"), cp["spot"].payment_terms_days)
    L["pay_grid"] = -on_day(cf.m("acc_grid_cost"), cp["tso"].payment_terms_days, 0.0)
    L["opex"] = np.where(is_eom, -cf.m("acc_opex")[mi], 0.0)
    L["bgl"] = np.where(is_first, -cf.m("acc_bgl_fees")[mi], 0.0)
    L["pre_service"] = np.where(is_first & (mi > 0), -cf.m("acc_pre_service")[np.maximum(mi - 1, 0)], 0.0)
    L["vat_paid"] = np.where(is_payday, cf.m("vat_paid")[mi], 0.0)
    L["cit_paid"] = np.where(is_payday, cf.m("out_cit")[mi], 0.0)
    L["vat_collected"] = receipts_total * g.vat_rate
    src = 0.0 if g.reverse_charge_vat_on_sources else (L["pay_pv"].to_numpy() + L["pay_bl"].to_numpy())
    L["vat_input"] = (L["pay_grid"].to_numpy() + src + L["pre_service"].to_numpy()) * g.vat_rate
    flow_cols = [f"{c}_receipts" for c in cf.codes] + ["receipts_resell", "imbalance", "pay_pv", "pay_bl", "pay_spot", "pay_grid", "opex", "bgl",
                 "pre_service", "vat_paid"]
    L["net_cf"] = L[flow_cols].sum(axis=1) + L["vat_collected"] + L["vat_input"]
    L["cum_cf"] = g.opening_cash_eur + L["net_cf"].cumsum()
    reserve_bal = P.m("reserve_balance")
    reserve_acc = cf.m("acc_reserve")
    release = P.m("reserve_release_budget")
    coll = cf.m("restricted_collateral")
    day_no = np.array([d.day for d in days])
    prev_bal = np.where(mi > 0, reserve_bal[np.maximum(mi - 1, 0)], 0.0)
    L["floor"] = prev_bal + reserve_acc[mi] * day_no / dim - np.where(is_eom, release[mi], 0.0) + coll[mi]
    inj = np.zeros(n)
    loan = 0.0
    cum = L["cum_cf"].to_numpy()
    floor = L["floor"].to_numpy()
    for i in range(n):
        inj[i] = max(0.0, floor[i] - (cum[i] + loan))
        loan += inj[i]
    L["injection"] = inj
    L["loan"] = np.cumsum(inj)
    L["funded_balance"] = L["cum_cf"] + L["loan"]
    L["free_cash"] = L["funded_balance"] - L["floor"]
    L["cum_tax"] = L["cit_paid"].cumsum()
    L["free_cash_after_tax"] = L["free_cash"] + L["cum_tax"]
    L["interest"] = L["loan"] * g.shareholder_loan_rate_pa / 365.0
    daily_spot = qh.qh.groupby(qh.qh["date"].values)["spot_buy_notified"].sum()
    L["spot_buy_mwh"] = daily_spot.reindex(pd.to_datetime(days)).fillna(0.0).to_numpy()
    cf.daily = L
    cf.daily_summary = {
        "peak_funding": float(L["loan"].max()),
        "cash_trough": float(L["cum_cf"].min()),
        "min_free_cash": float(L["free_cash"].min()),
        "min_free_cash_after_tax": float(L["free_cash_after_tax"].min()),
        "interest_daily_basis": float(L["interest"].sum()),
        "check_net_cf": float(L["net_cf"].sum() - cf.m("net_cf_before_tax").sum()),
        "peak_vs_monthly": float(L["loan"].max() - cf.y("peak_funding")),
        "check_receipts": float(receipts_total.sum() + L["receipts_resell"].sum() + L["imbalance"].sum() + L["vat_collected"].sum()
                                - cf.m("in_total").sum() - cf.m("vat_output").sum()),
    }
    rows = []
    for m in MONTHS:
        sel = L[L["month"] == m]
        rows.append({"month": m, "min_cum_cf": sel["cum_cf"].min(), "loan_eom": sel["loan"].max(), "min_free_cash": sel["free_cash"].min(), "net_cf": sel["net_cf"].sum()})
    cf.daily_monthly = pd.DataFrame(rows)
    return cf

"""esb.guarantees - guarantee sizing and bank-guarantee-letter (BGL) fees.

Workbook origin: Input section D/E (counterparties), Input section C rows 65-75 (own guarantees
to off-takers), Cons_P&L rows 204-217 (sources and market) and 380-381 (per off-taker).

Sizing methods (Input!C84 / C67), CHOOSE(MATCH(method, {...})):
    counterparties (Cons_P&L!204..209):
        Fixed                -> fixed amount
        % of Contract Value  -> pct x annual base
        Coverage Months      -> months x annual base / 12
        Dynamic              -> pct x annual base / 12            (a constant; X-26)
        Regulatory formula   -> required amount of Input section E
      annual base = Cons_P&L!O168 (total sell revenue, year), except Baseload: O61 + O129
      (forecast Baseload retail cost + resell cost, year).
    own guarantee to an off-taker (Cons_P&L!380):
        Fixed                -> fixed amount
        % of Contract Value  -> pct x section revenue (year)
        Coverage Months      -> months x section revenue (year) / 12
        Dynamic              -> pct x section revenue of the month (genuinely monthly; X-26)
        Regulatory formula   -> not defined (IFERROR -> 0)
The amount applies in the months whose start lies within [month start of guarantee start,
guarantee end]; an inactive counterparty / off-taker or guarantee type "None" gives 0.

BGL fee (Cons_P&L!211..216 / 381), only for type "Bank Guarantee Letter":
    Monthly               -> outstanding x fee / 12 every month
    One-time at issuance  -> outstanding x fee x (end - start + 1) / 365 in the month of the
                             guarantee start; 0 otherwise

Regulatory formulas (Input section E):
    OPCOM spot      = buffer days x max daily notified spot buy x max DAM (active scenario, uncurtailed)
    BRP / imbalance = rate RON/MW x (generation MW in BRP + max quarter-hour notified retail buy MW) / FX
                      (method "rate_per_mw", the workbook rule) or, with method "pre_delegated" (D119, the
                      balancing responsibility delegated to a PRE service provider):
                      max(initial guarantee RON / FX, months x average monthly imbalance value x (1 + VAT if
                      the basis is VAT-inclusive)) where the monthly imbalance value is |source + off-taker +
                      resell imbalance| of the engine's own ledger at the imbalance prices
    TSO             = Vtm x sum_offtakers (TL + SS) x active x metered year volume / 12
    DSO             = Vdm x sum_offtakers (T_HV + T_MV + T_LV) x active x metered year volume / 12 + add-on
Each x 1 if the counterparty is Active else 0.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date

import numpy as np

from config.schema import Guarantee, Parameters
from esb.monthly import MONTHS


def month_starts(year: int) -> list[date]:
    return [date(year, m, 1) for m in MONTHS]


def month_end(d: date) -> date:
    """EOMONTH(d, 0)."""
    return date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])


@dataclass
class RegulatoryInputs:
    """Derived quantities the regulatory formulas need (workbook Input!C112, C113, C118, and the
    per-off-taker metered year volumes)."""

    peak_daily_spot_buy_mwh: float  # Input!C112 = MAX over days of sum(QH_P&L!CU)
    peak_dam_price: float  # Input!C113 = MAX(Whol_Sport_Imb_Fcst!AJ)
    peak_retail_buy_mw: float  # Input!C118 = MAX(QH_P&L!AJ)
    metered_year_by_offtaker: dict[str, float]  # Cons_P&L section "Metered Volume", year
    imbalance_value_monthly: np.ndarray | None = None  # EUR per month, |source + off-taker + resell imbalance| (D119)


@dataclass
class RegulatoryAmounts:
    spot: float
    brp: float
    tso: float
    dso: float
    tso_annual_value: float  # Input!C122
    dso_annual_value: float  # Input!C127

    def by_counterparty(self) -> dict[str, float]:
        return {"spot": self.spot, "brp": self.brp, "tso": self.tso, "dso": self.dso}


def regulatory_amounts(params: Parameters, inp: RegulatoryInputs) -> RegulatoryAmounts:
    mg = params.market_guarantees
    cp = params.counterparties
    on = {k: (1.0 if cp[k].active else 0.0) for k in ("spot", "brp", "tso", "dso")}
    spot_vat = 1.0 + float(params.general.vat_rate) if bool(mg["spot"].get("vat_inclusive", False)) else 1.0  # D121
    spot = float(mg["spot"]["buffer_days"]) * inp.peak_daily_spot_buy_mwh * inp.peak_dam_price * spot_vat * on["spot"]
    brp = brp_required(params, inp) * on["brp"]
    tso_val = 0.0
    dso_val = 0.0
    for o in params.offtakers:
        a = 1.0 if o.active else 0.0
        v = inp.metered_year_by_offtaker.get(o.code, 0.0)
        tso_val += params.tso_tariff_for(o) * a * v
        dso_val += params.dso_tariff_for(o) * a * v
    tso = float(mg["tso"]["vtm_multiplier"]) * tso_val / 12.0 * on["tso"]
    dso = (float(mg["dso"]["vdm_multiplier"]) * dso_val / 12.0 + float(mg["dso"]["overdue_addon_eur"])) * on["dso"]
    return RegulatoryAmounts(spot=spot, brp=brp, tso=tso, dso=dso, tso_annual_value=tso_val, dso_annual_value=dso_val)


def brp_required(params: Parameters, inp: RegulatoryInputs) -> float:
    """The BRP / imbalance guarantee before the activity switch: the workbook rule (rate per MW) or the
    delegated-PRE rule of D119 (initial amount, then months x average monthly imbalance value)."""
    mg = params.market_guarantees["brp"]
    fx = float(params.general.fx_ron_per_eur)
    if str(mg.get("method", "rate_per_mw")) != "pre_delegated":
        return float(mg["rate_ron_per_mw"]) * (float(mg["generation_mw_in_brp"]) + inp.peak_retail_buy_mw) / fx
    values = np.abs(np.asarray(inp.imbalance_value_monthly if inp.imbalance_value_monthly is not None else [], dtype=float))
    avg = float(values.mean()) if values.size else 0.0
    vat = 1.0 + float(params.general.vat_rate) if bool(mg.get("pre_vat_inclusive", True)) else 1.0
    return max(float(mg.get("pre_initial_ron", 0.0)) / fx, float(mg.get("pre_months_of_imbalance", 0.0)) * avg * vat)


def pre_service_monthly(params: Parameters, imbalance_value_monthly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """D120: the delegated-PRE service flows per month (EUR, positive amounts): the aggregation gain the PRE's internal
    redistribution returns to the member (an assumed share of |imbalance value|; 0 = not modelled) and the service fee
    = fixed monthly fee (contract Anexa 2, Tf) + gain share x gain (Tv). Both zero under the workbook BRP rule."""
    mg = params.market_guarantees["brp"]
    zero = np.zeros(12)
    if str(mg.get("method", "rate_per_mw")) != "pre_delegated":
        return zero, zero
    fx = float(params.general.fx_ron_per_eur)
    gain = np.abs(np.asarray(imbalance_value_monthly, dtype=float)) * float(mg.get("pre_aggregation_gain_pct_of_imbalance", 0.0))
    fee = np.full(12, float(mg.get("pre_fee_fixed_ron_per_month", 0.0)) / fx) + gain * float(mg.get("pre_fee_gain_share_pct", 0.0))
    return gain, fee


def _window(g: Guarantee, starts: list[date]) -> np.ndarray:
    lo = date(g.start.year, g.start.month, 1)
    return np.array([1.0 if (lo <= s <= g.end) else 0.0 for s in starts])


def counterparty_outstanding(g: Guarantee, active: bool, annual_base: float, required: float | None,
                             fixed_amount: float, year: int) -> np.ndarray:
    """Cons_P&L rows 204-209 for one counterparty: 12 monthly outstanding amounts."""
    if not active or g.type in ("None", ""):
        return np.zeros(12)
    if g.sizing == "Fixed":
        amt = fixed_amount
    elif g.sizing == "% of Contract Value":
        amt = g.pct_of_contract_value * annual_base
    elif g.sizing == "Coverage Months":
        amt = g.coverage_months * annual_base / 12.0
    elif g.sizing == "Dynamic":
        amt = g.pct_of_contract_value * annual_base / 12.0
    else:  # Regulatory formula
        amt = 0.0 if required is None else required
    return _window(g, month_starts(year)) * float(amt)


def offtaker_outstanding(g: Guarantee, active: bool, revenue_year: float, revenue_month: np.ndarray, year: int) -> np.ndarray:
    """Cons_P&L row 380 for one off-taker."""
    if not active or g.type in ("None", ""):
        return np.zeros(12)
    if g.sizing == "Fixed":
        amt = np.full(12, float(g.fixed_amount or 0.0))
    elif g.sizing == "% of Contract Value":
        amt = np.full(12, g.pct_of_contract_value * revenue_year)
    elif g.sizing == "Coverage Months":
        amt = np.full(12, g.coverage_months * revenue_year / 12.0)
    elif g.sizing == "Dynamic":
        amt = g.pct_of_contract_value * np.asarray(revenue_month, dtype=float)
    else:  # Regulatory formula is not defined for off-takers -> IFERROR 0
        amt = np.zeros(12)
    return _window(g, month_starts(year)) * amt


def bgl_fee(g: Guarantee, outstanding: np.ndarray, year: int) -> np.ndarray:
    """Cons_P&L rows 211-216 / 381."""
    if g.type != "Bank Guarantee Letter":
        return np.zeros(12)
    out = np.asarray(outstanding, dtype=float)
    if g.bgl_fee_type == "Monthly":
        return out * g.bgl_fee_pa / 12.0
    fee = np.zeros(12)
    for i, s in enumerate(month_starts(year)):
        e = _eom(s)
        if s <= g.start <= e:
            fee[i] = out[i] * g.bgl_fee_pa * ((g.end - g.start).days + 1) / 365.0
    return fee


def _eom(s: date) -> date:
    return month_end(s)


def sizing_comparison(pnl, params: Parameters) -> dict[str, dict[str, float]]:
    """Application page 9: the peak outstanding amount every counterparty would need under each
    sizing method, with the same bases the P&L used (annual base, regulatory amount, fixed amount)."""
    import copy

    from config.schema import GUARANTEE_SIZINGS

    P = pnl.portfolio
    cp = params.counterparties
    year = params.spine_year
    bases = {"pv": P.y("t_revenue"), "baseload": P.y("cost_bl_forecast") + P.y("rs_bl_cost_forecast"), "spot": P.y("t_revenue"),
             "brp": P.y("t_revenue"), "tso": P.y("t_revenue"), "dso": P.y("t_revenue")}
    required = {"pv": None, "baseload": None, **pnl.regulatory.by_counterparty()}
    fixed = {k: (pnl.pv_fixed_guarantee if k == "pv" else float(cp[k].guarantee.fixed_amount or 0.0)) for k in cp}
    out: dict[str, dict[str, float]] = {}
    for k in ("pv", "baseload", "spot", "brp", "tso", "dso"):
        row: dict[str, float] = {}
        for sizing in GUARANTEE_SIZINGS:
            g = copy.deepcopy(cp[k].guarantee)
            g.sizing = sizing
            if g.type in ("None", ""):
                g.type = "Bank Guarantee Letter"
            if sizing == "Regulatory formula" and required[k] is None:
                row[sizing] = float("nan")
                continue
            amounts = counterparty_outstanding(g, True, bases[k], required[k], fixed[k], year)
            row[sizing] = float(np.max(amounts)) if len(amounts) else 0.0
        out[k] = row
    return out

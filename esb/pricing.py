"""esb.pricing - the offer-price calculator (workbook sheet `Pricing_Calc`) for one off-taker.

Sections (workbook rows; C = year, F:Q = months, D = manual case with blank-fallback to C):
    1  volumes and physical cost: metered (8), notified (9), forecast purchase cost (10), source
       imbalance allocated (11), off-taker imbalance (12), weighted purchase price = 10/8 (13),
       imbalance cost per MWh = -(11 + 12)/8 (14), physical delivery cost = 13 + 14 (15)
    2  budgeted risk premium components of the off-taker (18-24), total (25), target GM (26)
    3  energy supply price = 15 + 25 + 26 (29), pass-through total (30), offer ex VAT (31),
       VAT (32), offer incl. VAT (33)
    4  contracted sell price (36), contract - offer (37), implied GM (38), cost-to-serve beyond
       physical (39), forecast specific retail NM (40), indicative NM at offer = 29 - 15 - 25 - 39
       (41), annual energy revenue = 8 x 29 (42)
    5  forecast premium (46), re-priced energy price = 15 + 46 + 26 (47), contract - re-priced
       (48), derived forecast premium (49), reserve released budget / forecast (50 / 51)

Two known defects of the frozen workbook are corrected here (register X-15, X-16; decisions
D87 / D88): row 46 returns the off-taker's *total* forecast premium (the workbook's +94 row
offset lands on the Collateral component), and row 39 returns
(OPEX + variable OPEX + own BGL fee + allocated market BGL fees + allocated interest) / metered
(the workbook's lookups use two retired labels and fall to 0). `as_cached=True` reproduces the
workbook's defective values instead, for the parity tie-out of the cached cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config.schema import PREMIUM_COMPONENTS, Parameters
from esb.monthly import safe_div
from esb.pnl import PnLResult

PRICING_ROWS = {
    "metered": 8, "notified": 9, "purchase_cost": 10, "source_imb": 11, "offtaker_imb": 12, "purchase_price": 13,
    "imbalance_cost": 14, "physical_cost": 15, "premium_volume": 18, "premium_price": 19, "premium_profile": 20,
    "premium_credit": 21, "premium_regulatory": 22, "premium_fx": 23, "premium_collateral": 24, "premium_total": 25,
    "target_gm": 26, "energy_price": 29, "passthrough": 30, "offer_ex_vat": 31, "vat": 32, "offer_incl_vat": 33,
    "contract_price": 36, "contract_minus_offer": 37, "implied_gm": 38, "cost_to_serve": 39, "forecast_nm_specific": 40,
    "indicative_nm": 41, "annual_revenue": 42, "forecast_premium": 46, "repriced_energy_price": 47,
    "contract_minus_repriced": 48, "derived_forecast_premium": 49, "reserve_release_budget": 50, "reserve_release_forecast": 51,
}
YEAR_ONLY = {"forecast_premium", "repriced_energy_price", "contract_minus_repriced", "derived_forecast_premium",
             "reserve_release_budget", "reserve_release_forecast"}


@dataclass
class PricingResult:
    code: str
    year: dict[str, float] = field(default_factory=dict)  # column C
    months: dict[str, np.ndarray] = field(default_factory=dict)  # columns F:Q
    manual: dict[str, float] = field(default_factory=dict)  # column D (manual case)
    as_cached: bool = False


def price_offtaker(pnl: PnLResult, params: Parameters, code: str, manual: dict[str, float] | None = None, as_cached: bool = False) -> PricingResult:
    o = params.offtaker(code)
    T = pnl.sections[code]
    vat = params.general.vat_rate
    tariff = params.tariff_total_for(o)
    y: dict[str, float] = {}
    m: dict[str, np.ndarray] = {}

    def both(key: str, sec_key: str) -> None:
        y[key] = T.y(sec_key)
        m[key] = T.m(sec_key)

    both("metered", "metered")
    both("notified", "notified")
    both("purchase_cost", "cost_forecast")
    both("source_imb", "source_imb")
    both("offtaker_imb", "offtaker_imb")
    y["purchase_price"] = float(safe_div(np.array([y["purchase_cost"]]), np.array([y["metered"]]))[0])
    m["purchase_price"] = safe_div(m["purchase_cost"], m["metered"])
    y["imbalance_cost"] = float(safe_div(np.array([-(y["source_imb"] + y["offtaker_imb"])]), np.array([y["metered"]]))[0])
    m["imbalance_cost"] = safe_div(-(m["source_imb"] + m["offtaker_imb"]), m["metered"])
    y["physical_cost"] = y["purchase_price"] + y["imbalance_cost"]
    m["physical_cost"] = m["purchase_price"] + m["imbalance_cost"]
    for comp in PREMIUM_COMPONENTS:
        y[f"premium_{comp}"] = float(o.premium_budget[comp])
        m[f"premium_{comp}"] = np.full(12, float(o.premium_budget[comp]))
    y["premium_total"] = o.premium_budget_total
    m["premium_total"] = np.full(12, o.premium_budget_total)
    y["target_gm"] = float(o.target_gm_budget)
    m["target_gm"] = np.full(12, float(o.target_gm_budget))
    y["energy_price"] = y["physical_cost"] + y["premium_total"] + y["target_gm"]
    m["energy_price"] = m["physical_cost"] + m["premium_total"] + m["target_gm"]
    y["passthrough"] = tariff
    m["passthrough"] = np.full(12, tariff)
    y["offer_ex_vat"] = y["energy_price"] + y["passthrough"]
    m["offer_ex_vat"] = m["energy_price"] + m["passthrough"]
    y["vat"] = y["offer_ex_vat"] * vat
    m["vat"] = m["offer_ex_vat"] * vat
    y["offer_incl_vat"] = y["offer_ex_vat"] + y["vat"]
    m["offer_incl_vat"] = m["offer_ex_vat"] + m["vat"]
    both("contract_price", "sell_price")
    y["contract_minus_offer"] = y["contract_price"] - y["energy_price"]
    m["contract_minus_offer"] = m["contract_price"] - m["energy_price"]
    y["implied_gm"] = y["contract_price"] - y["physical_cost"] - y["premium_total"]
    m["implied_gm"] = m["contract_price"] - m["physical_cost"] - m["premium_total"]
    if as_cached:
        y["cost_to_serve"] = 0.0
        m["cost_to_serve"] = np.zeros(12)
    else:
        memo_b = T["memo_bgl_share"] if "memo_bgl_share" in T else np.zeros(13)
        memo_i = T["memo_interest_share"] if "memo_interest_share" in T else np.zeros(13)
        cts = T["opex"] + T["variable_opex"] + T["own_bgl_fee"] + memo_b + memo_i
        r = safe_div(cts, T["metered"])
        y["cost_to_serve"] = float(r[12])
        m["cost_to_serve"] = r[:12]
    both("forecast_nm_specific", "retail_nm_forecast_specific")
    y["indicative_nm"] = y["energy_price"] - y["physical_cost"] - y["premium_total"] - y["cost_to_serve"]
    m["indicative_nm"] = m["energy_price"] - m["physical_cost"] - m["premium_total"] - m["cost_to_serve"]
    y["annual_revenue"] = y["metered"] * y["energy_price"]
    m["annual_revenue"] = m["metered"] * m["energy_price"]
    # section 5 (year only)
    if as_cached:
        # workbook: MATCH("Total risk premium", Input!B96:B105) + 94 -> row 103 (Collateral) of the forecast column
        y["forecast_premium"] = float(o.premium_forecast["collateral"])
    else:
        y["forecast_premium"] = o.premium_forecast_total
    y["repriced_energy_price"] = y["physical_cost"] + y["forecast_premium"] + y["target_gm"]
    y["contract_minus_repriced"] = y["contract_price"] - y["repriced_energy_price"]
    y["derived_forecast_premium"] = T.y("premium_forecast_derived")
    y["reserve_release_budget"] = T.y("reserve_release_budget")
    y["reserve_release_forecast"] = T.y("reserve_release_forecast")
    res = PricingResult(code=code, year=y, months=m, as_cached=as_cached)
    if manual:
        res.manual = manual_case(res, manual)
    return res


def manual_case(base: PricingResult, manual: dict[str, float]) -> dict[str, float]:
    """Column D: user overrides with blank-fallback to the year column (rows 8, 13, 14, 18-24, 26, 30)."""
    y = base.year
    g = lambda k: float(manual[k]) if k in manual and manual[k] is not None else y[k]  # noqa: E731
    d: dict[str, float] = {}
    d["metered"] = g("metered")
    d["physical_cost"] = g("purchase_price") + g("imbalance_cost")
    d["premium_total"] = sum(g(f"premium_{c}") for c in PREMIUM_COMPONENTS)
    d["target_gm"] = g("target_gm")
    d["energy_price"] = d["physical_cost"] + d["premium_total"] + d["target_gm"]
    d["passthrough"] = g("passthrough")
    d["offer_ex_vat"] = d["energy_price"] + d["passthrough"]
    d["vat"] = d["offer_ex_vat"] * (y["vat"] / y["offer_ex_vat"] if y["offer_ex_vat"] else 0.0)
    d["offer_incl_vat"] = d["offer_ex_vat"] + d["vat"]
    d["indicative_nm"] = d["energy_price"] - d["physical_cost"] - d["premium_total"] - y["cost_to_serve"]
    d["annual_revenue"] = d["metered"] * d["energy_price"]
    return d

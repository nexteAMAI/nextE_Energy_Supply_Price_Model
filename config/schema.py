"""Typed parameter model for the engine, loaded from config/parameters.yaml (Reference Case) or
from the application's parameter store. Every value the workbook's Input sheet holds is here;
nothing is a code constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

PRODUCTS = ("BL24", "Peak", "OffPeak")
PREMIUM_COMPONENTS = ("volume", "price", "profile", "credit", "regulatory", "fx", "collateral")
GUARANTEE_SIZINGS = ("Fixed", "% of Contract Value", "Coverage Months", "Dynamic", "Regulatory formula")
GUARANTEE_TYPES = ("None", "Bank Guarantee Letter", "Cash Collateral", "Parent Company Guarantee")
BGL_FEE_TYPES = ("Monthly", "One-time at issuance")
COUNTERPARTIES = ("pv", "baseload", "spot", "brp", "tso", "dso")


def _d(v) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v))


@dataclass
class Guarantee:
    type: str = "None"
    direction: str = ""
    sizing: str = "Fixed"
    fixed_amount: float | None = 0.0
    coverage_months: float = 0.0
    pct_of_contract_value: float = 0.0
    bgl_fee_pa: float = 0.0
    bgl_fee_type: str = "Monthly"
    cash_backing_pct: float = 0.0
    start: date = date(2027, 1, 1)
    end: date = date(2027, 12, 31)

    @classmethod
    def from_dict(cls, d: dict) -> Guarantee:
        d = dict(d)
        d["start"] = _d(d.get("start", "2027-01-01"))
        d["end"] = _d(d.get("end", "2027-12-31"))
        d["type"] = str(d.get("type", "None"))
        g = cls(**d)
        if g.sizing not in GUARANTEE_SIZINGS:
            raise ValueError(f"guarantee sizing '{g.sizing}' not in {GUARANTEE_SIZINGS}")
        if g.type not in GUARANTEE_TYPES:
            raise ValueError(f"guarantee type '{g.type}' not in {GUARANTEE_TYPES}")
        if g.bgl_fee_type not in BGL_FEE_TYPES:
            raise ValueError(f"bgl_fee_type '{g.bgl_fee_type}' not in {BGL_FEE_TYPES}")
        return g


@dataclass
class Offtaker:
    code: str
    active: bool
    contract_start: date
    contract_end: date
    payment_terms_days: int
    advance_pct: float
    contract_price_eur_per_mwh: float
    pv_price_budget_eur_per_mwh: float
    pv_price_forecast_eur_per_mwh: float
    premium_budget: dict[str, float]
    premium_forecast: dict[str, float]
    target_gm_budget: float
    target_gm_forecast: float
    guarantee: Guarantee
    strip_mw: dict[str, list[float]]
    product_price_budget: dict[str, list[float]]
    product_price_forecast: dict[str, list[float]]
    tariff_components: dict[str, float] | None = None  # per-off-taker override of the portfolio tariff set (Input!C54:F63)

    @property
    def premium_budget_total(self) -> float:
        return float(sum(self.premium_budget[c] for c in PREMIUM_COMPONENTS))

    @property
    def premium_forecast_total(self) -> float:
        return float(sum(self.premium_forecast[c] for c in PREMIUM_COMPONENTS))

    @classmethod
    def from_dict(cls, d: dict) -> Offtaker:
        d = dict(d)
        d["contract_start"] = _d(d["contract_start"])
        d["contract_end"] = _d(d["contract_end"])
        d["guarantee"] = Guarantee.from_dict(d.get("guarantee", {}))
        for key in ("strip_mw", "product_price_budget", "product_price_forecast"):
            tbl = d[key]
            for p in PRODUCTS:
                if p not in tbl or len(tbl[p]) != 12:
                    raise ValueError(f"{d['code']}.{key}.{p} must hold 12 monthly values")
                tbl[p] = [float(x) for x in tbl[p]]
        for key in ("premium_budget", "premium_forecast"):
            miss = [c for c in PREMIUM_COMPONENTS if c not in d[key]]
            if miss:
                raise ValueError(f"{d['code']}.{key} lacks {miss}")
        return cls(**d)


@dataclass
class Counterparty:
    active: bool
    payment_terms_days: int
    advance_pct: float
    guarantee: Guarantee
    k: int | None = None
    deviation_pct: float = 0.0
    imbalance_deviation_pct: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> Counterparty:
        d = dict(d)
        d["guarantee"] = Guarantee.from_dict(d.get("guarantee", {}))
        return cls(**d)


@dataclass
class General:
    case_start: date
    case_end: date
    fx_ron_per_eur: float
    vat_rate: float
    tax_payment_day: int
    cit_rate: float
    opening_cash_eur: float
    opening_cash_ron: float
    portfolio_opex_eur_per_mwh_metered: float
    variable_opex_eur_per_mwh_metered: float
    shareholder_loan_rate_pa: float
    reverse_charge_vat_on_sources: bool
    debt_facility: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> General:
        d = dict(d)
        d["case_start"] = _d(d["case_start"])
        d["case_end"] = _d(d["case_end"])
        return cls(**d)


@dataclass
class Parameters:
    meta: dict
    scenario_active: str
    scenario_names: list[str]
    resell_pv_cost_factor: float
    resell_pv_revenue_factor: float
    general: General
    premium_standard: dict[str, float]
    gc_quota: float
    gc_reference_price_ron: float
    gc_spot_share: float
    tariff_components: dict[str, float]
    offtakers: list[Offtaker]
    counterparties: dict[str, Counterparty]
    market_guarantees: dict

    # ---- derived ----------------------------------------------------------------------
    @property
    def spine_year(self) -> int:
        return int(self.meta["spine_year"])

    @property
    def scenario_index(self) -> int:
        """1-based, as Input!C7 (MATCH with fallback 1)."""
        try:
            return self.scenario_names.index(self.scenario_active) + 1
        except ValueError:
            return 1

    @property
    def gc_reference_price_eur(self) -> float:
        return self.gc_reference_price_ron / self.general.fx_ron_per_eur  # Cons_P&L!B11

    @property
    def gc_unit_cost(self) -> float:
        return self.gc_quota * self.gc_reference_price_eur  # Input!C60

    @property
    def tariff_total(self) -> float:
        """Input!C64 = SUM(C54:C63): TL + TG + SS + T_HV + T_MV + T_LV + GC + cogeneration + CfD + excise."""
        t = self.tariff_components
        return (
            t["TL"] + t["TG"] + t["SS"] + t["T_HV"] + t["T_MV"] + t["T_LV"]
            + self.gc_unit_cost + t["cogeneration"] + t["cfd"] + t["excise"]
        )

    @property
    def tso_tariff(self) -> float:
        return self.tariff_components["TL"] + self.tariff_components["SS"]  # Input!C122 basis (rows 54 + 56)

    @property
    def dso_tariff(self) -> float:
        t = self.tariff_components
        return t["T_HV"] + t["T_MV"] + t["T_LV"]  # Input!C127 basis (rows 57..59)

    def tariffs_for(self, o: Offtaker) -> dict[str, float]:
        """The regulated component set of one off-taker (its own column of Input rows 54-63 or the portfolio set)."""
        t = dict(self.tariff_components)
        if o.tariff_components:
            t.update({k: float(v) for k, v in o.tariff_components.items()})
        return t

    def tariff_total_for(self, o: Offtaker) -> float:
        t = self.tariffs_for(o)
        return t["TL"] + t["TG"] + t["SS"] + t["T_HV"] + t["T_MV"] + t["T_LV"] + self.gc_unit_cost + t["cogeneration"] + t["cfd"] + t["excise"]

    def tso_tariff_for(self, o: Offtaker) -> float:
        t = self.tariffs_for(o)
        return t["TL"] + t["SS"]

    def dso_tariff_for(self, o: Offtaker) -> float:
        t = self.tariffs_for(o)
        return t["T_HV"] + t["T_MV"] + t["T_LV"]

    def offtaker(self, code: str) -> Offtaker:
        for o in self.offtakers:
            if o.code == code:
                return o
        raise KeyError(code)

    @classmethod
    def from_dict(cls, d: dict) -> Parameters:
        gc = d["green_certificates"]
        return cls(
            meta=d["meta"],
            scenario_active=d["scenario"]["active"],
            scenario_names=list(d["scenario"]["names"]),
            resell_pv_cost_factor=float(d["resell"]["pv_cost_factor_vs_dam_curtailed"]),
            resell_pv_revenue_factor=float(d["resell"]["pv_revenue_factor_vs_dam_curtailed"]),
            general=General.from_dict(d["general"]),
            premium_standard=dict(d["premium_standard"]),
            gc_quota=float(gc["quota_gc_per_mwh"]),
            gc_reference_price_ron=float(gc["reference_price_ron_per_gc"]),
            gc_spot_share=float(gc["spot_share"]),
            tariff_components={k: float(v) for k, v in d["tariff_components_eur_per_mwh"].items()},
            offtakers=[Offtaker.from_dict(o) for o in d["offtakers"]],
            counterparties={k: Counterparty.from_dict(v) for k, v in d["counterparties"].items()},
            market_guarantees=d["market_guarantees"],
        )

    def validate(self) -> list[str]:
        errs: list[str] = []
        codes = [o.code for o in self.offtakers]
        if len(set(codes)) != len(codes):
            errs.append("duplicate off-taker codes")
        for c in COUNTERPARTIES:
            if c not in self.counterparties:
                errs.append(f"counterparty '{c}' missing")
        if not (0 <= self.general.vat_rate < 1 and 0 <= self.general.cit_rate < 1):
            errs.append("vat_rate / cit_rate out of range")
        return errs


def load_parameters(path: str | Path | None = None) -> Parameters:
    path = Path(path) if path else Path(__file__).with_name("parameters.yaml")
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    p = Parameters.from_dict(d)
    errs = p.validate()
    if errs:
        raise ValueError("; ".join(errs))
    return p

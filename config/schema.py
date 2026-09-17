"""Typed parameter model for the engine, loaded from config/parameters.yaml (Reference Case) or
from the application's parameter store. Every value the workbook's Input sheet holds is here;
nothing is a code constant.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import yaml

PRODUCTS = ("BL24", "Peak", "OffPeak")
PREMIUM_COMPONENTS = ("volume", "price", "profile", "credit", "regulatory", "fx", "collateral")
GUARANTEE_SIZINGS = ("Fixed", "% of Contract Value", "Coverage Months", "Dynamic", "Regulatory formula")
GUARANTEE_TYPES = ("None", "Bank Guarantee Letter", "Cash Collateral", "Parent Company Guarantee")
BGL_FEE_TYPES = ("Monthly", "One-time at issuance")
COUNTERPARTIES = ("pv", "baseload", "spot", "brp", "tso", "dso")
TARIFF_KEYS = ("TL", "TG", "SS", "T_HV", "T_MV", "T_LV", "cogeneration", "cfd", "excise")
VOLTAGE_LEVELS = ("HV (>=110 kV) TSO", "HV (110 kV) DSO", "MV (6-20 kV) DSO", "LV (0,4 kV) DSO")


def load_grid_tariffs(path: str | Path | None = None) -> dict:
    """config/tariffs_ro.yaml: the RON/MWh grid tariff table with its operators, levels and provenance (D109)."""
    path = Path(path) if path else Path(__file__).with_name("tariffs_ro.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def distribution_operators(rows: list[dict] | None = None) -> list[str]:
    cfg = load_grid_tariffs()
    listed = list(cfg.get("distribution_operators", []))
    if rows:
        for r in rows:
            if r["component"] in ("T_HV", "T_MV", "T_LV") and r["owner"] not in listed:
                listed.append(r["owner"])
    return listed


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
    dso: str | None = None  # distribution operator serving the metering point (grid tariff table, D109)
    voltage_level: str | None = None  # one of VOLTAGE_LEVELS; with dso set, the components are derived by the cascading rule
    name: str = ""  # display name, set in the application or a scenario file; never in the repository (F-035)
    position: int = 0  # merit-order position (1-based); 0 = by list order

    @property
    def label(self) -> str:
        return self.name or self.code

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
    name: str = ""  # display name (application / scenario file only)

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
    grid_tariffs: list[dict] = field(default_factory=list)  # RON/MWh rows by owner and component (config/tariffs_ro.yaml, D109)

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
        """The regulated component set of one off-taker: derived from the grid tariff table when the off-taker
        names its DSO and voltage level (D109, cascading rule of the v1.1 template), else its own override
        column (Input rows 54-63), else the portfolio set."""
        if o.dso and o.voltage_level:
            return self.tariffs_by_grid(o.dso, o.voltage_level)
        t = dict(self.tariff_components)
        if o.tariff_components:
            t.update({k: float(v) for k, v in o.tariff_components.items()})
        return t

    def tariffs_by_grid(self, dso: str, voltage_level: str) -> dict[str, float]:
        """EUR/MWh components at the register FX from the RON/MWh grid tariff table (v1.1 Dashboard_Input rows 304-313):
        TL, TG, SS, cogeneration, CfD and excise per their applicability flags at the voltage level; the distribution
        tariffs of the named DSO cascade (T_HV from HV DSO down, T_MV from MV DSO down, T_LV at LV DSO only).
        A DSO without distribution rows is refused (blank is not zero)."""
        if voltage_level not in VOLTAGE_LEVELS:
            raise ValueError(f"voltage level '{voltage_level}' not in {VOLTAGE_LEVELS}")
        lvl = VOLTAGE_LEVELS.index(voltage_level)  # 0 = HV TSO ... 3 = LV DSO
        fx = float(self.general.fx_ron_per_eur)
        out = {k: 0.0 for k in TARIFF_KEYS}
        dso_rows = {r["component"]: r for r in self.grid_tariffs if r.get("owner") == dso and r["component"] in ("T_HV", "T_MV", "T_LV")}
        for r in self.grid_tariffs:
            comp = r["component"]
            if comp in ("T_HV", "T_MV", "T_LV"):
                continue
            flags = r.get("applies") or [True, True, True, True]
            if flags[lvl]:
                out[comp] = float(r["ron_per_mwh"]) / fx
        cascade = {"T_HV": lvl >= 1, "T_MV": lvl >= 2, "T_LV": lvl >= 3}
        if any(cascade.values()) and not dso_rows:
            raise ValueError(f"no distribution tariff rows for '{dso}' in the grid tariff table")
        for comp, applies in cascade.items():
            if applies:
                if comp not in dso_rows:
                    raise ValueError(f"grid tariff table lacks {comp} for '{dso}'")
                out[comp] = float(dso_rows[comp]["ron_per_mwh"]) / fx
        return out

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
            grid_tariffs=[dict(r) for r in (d.get("grid_tariffs") if d.get("grid_tariffs") is not None else load_grid_tariffs()["tariffs"])],
        )

    def to_dict(self) -> dict:
        """The register as a plain dict in the layout of config/parameters.yaml (round-trips through from_dict)."""

        def g(x: Guarantee) -> dict:
            d = asdict(x)
            d["start"], d["end"] = x.start.isoformat(), x.end.isoformat()
            return d

        offtakers = []
        for o in self.offtakers:
            d = asdict(o)
            d["contract_start"], d["contract_end"] = o.contract_start.isoformat(), o.contract_end.isoformat()
            d["guarantee"] = g(o.guarantee)
            offtakers.append(d)
        counterparties = {}
        for k, c in self.counterparties.items():
            d = asdict(c)
            d["guarantee"] = g(c.guarantee)
            counterparties[k] = d
        general = asdict(self.general)
        general["case_start"], general["case_end"] = self.general.case_start.isoformat(), self.general.case_end.isoformat()
        return {
            "meta": dict(self.meta),
            "scenario": {"active": self.scenario_active, "names": list(self.scenario_names)},
            "resell": {"pv_cost_factor_vs_dam_curtailed": self.resell_pv_cost_factor,
                       "pv_revenue_factor_vs_dam_curtailed": self.resell_pv_revenue_factor},
            "general": general,
            "premium_standard": dict(self.premium_standard),
            "green_certificates": {"quota_gc_per_mwh": self.gc_quota, "reference_price_ron_per_gc": self.gc_reference_price_ron,
                                   "spot_share": self.gc_spot_share},
            "tariff_components_eur_per_mwh": dict(self.tariff_components),
            "offtakers": offtakers,
            "counterparties": counterparties,
            "market_guarantees": copy.deepcopy(self.market_guarantees),
            "grid_tariffs": copy.deepcopy(self.grid_tariffs),
        }

    def copy(self) -> Parameters:
        return Parameters.from_dict(copy.deepcopy(self.to_dict()))

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
        for o in self.offtakers:
            if o.contract_end < o.contract_start:
                errs.append(f"{o.code}: contract_end before contract_start")
            if o.payment_terms_days < 0 or not (0 <= o.advance_pct <= 1):
                errs.append(f"{o.code}: payment terms or advance out of range")
            if (o.dso is None) != (o.voltage_level is None):
                errs.append(f"{o.code}: DSO and voltage level must be set together")
            if o.dso and o.voltage_level:
                try:
                    self.tariffs_by_grid(o.dso, o.voltage_level)
                except ValueError as e:
                    errs.append(f"{o.code}: {e}")
        for r in self.grid_tariffs:
            if r.get("component") not in TARIFF_KEYS:
                errs.append(f"grid tariff table: unknown component '{r.get('component')}'")
            if r.get("ron_per_mwh") is None:
                errs.append(f"grid tariff table: {r.get('owner')} {r.get('component')} has no value (blank is not zero)")
        return errs

    # ---- D119: bid-scenario defaults and regulatory warnings -----------------------------------------------------
    def apply_bid_defaults(self) -> list[str]:
        """The rulings of 17.09.2026 for bid scenarios (not for the frozen Reference Case): reverse charge off for a
        spine year after 2026 (RC-2027), BRP guarantee by the delegated-PRE rule (BRP-GF), the shipped ANRE grid
        tariff table (TAR-2026). Returns the changes made."""
        done: list[str] = []
        if self.spine_year > 2026 and self.general.reverse_charge_vat_on_sources:
            self.general.reverse_charge_vat_on_sources = False
            done.append("reverse charge on source purchases switched off (measure ends 31.12.2026)")
        if str(self.market_guarantees["brp"].get("method", "rate_per_mw")) != "pre_delegated":
            self.market_guarantees["brp"]["method"] = "pre_delegated"
            done.append("BRP guarantee by the delegated-PRE rule")
        if not bool(self.market_guarantees["spot"].get("vat_inclusive", False)):
            self.market_guarantees["spot"]["vat_inclusive"] = True
            done.append("spot collateral proxy VAT-inclusive (OPCOM PO garantii PZU & PI pct. 6.8)")
        shipped = load_grid_tariffs()["tariffs"]
        key = lambda rows: [(r["owner"], r["component"], float(r["ron_per_mwh"])) for r in rows]  # noqa: E731
        if key(self.grid_tariffs) != key(shipped):
            self.grid_tariffs = [dict(r) for r in shipped]
            done.append("grid tariff table reset to the shipped ANRE 2026 table")
        return done

    def regulatory_warnings(self) -> list[str]:
        """Non-blocking: settings that contradict the verified regulatory state (docs/PARAMETERS.md section 3)."""
        warns: list[str] = []
        if self.spine_year > 2026 and self.general.reverse_charge_vat_on_sources:
            warns.append("RC-2027: the reverse charge on electricity purchases ends 31.12.2026 (Codul fiscal art. 331 alin. (6)); "
                         f"the spine year {self.spine_year} applies it - keep only with the adviser's confirmation of an extension")
        if str(self.market_guarantees["brp"].get("method", "rate_per_mw")) == "rate_per_mw":
            warns.append("BRP-GF: the BRP guarantee uses the workbook rate-per-MW rule, which no procedure supports; bid scenarios "
                         "use the delegated-PRE rule")
        return warns


def load_parameters(path: str | Path | None = None) -> Parameters:
    path = Path(path) if path else Path(__file__).with_name("parameters.yaml")
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    p = Parameters.from_dict(d)
    errs = p.validate()
    if errs:
        raise ValueError("; ".join(errs))
    return p

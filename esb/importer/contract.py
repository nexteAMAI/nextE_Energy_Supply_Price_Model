"""Standard time-series upload contract (D-TPL): what a delivered workbook must declare.

The contract is the Python form of the `STD_EET_QH` template (nextE data time-series standard,
template v1.0 of 25.08.2026): a control block, a series registry and a raw paste surface.
Every upload declares itself completely; the importer never guesses a unit, a class, a sign
convention or a time basis.

Workbook layout produced by esb.importer.template and read by esb.importer.reader:

* `Instructions`  - text for the person filling the file
* `Std_Control`   - key / value block (see CONTROL_KEYS)
* `Series_Registry` - one row per slot E01.. (numeric) and C01.. (categorical); any width of the number (E01, E001, E0001)
* `RAW_EET_QH`    - row 1 headers `Date_EET | Start_EET | End_EET | E01 | ... | C01 | ...`;
                    data from row 2; values only, no formulas
* `Recon_Check`   - convenience gate for the person filling the file (counts); the binding
                    checks are the importer's (esb.importer.checks)
"""

from __future__ import annotations

from dataclasses import dataclass, field

TEMPLATE_VERSION = "ESB-STD-QH 1.1"  # written into new templates
TEMPLATE_VERSIONS = ("ESB-STD-QH 1.0", "ESB-STD-QH 1.1")
# 1.1 (D112, 16.09.2026): optional registry columns scenario_name (wholesale: several scenario blocks per file,
# series keyed by scenario_name + name) and entity_name (reference only, never read into the engine); a declared
# slot whose RAW column is entirely blank counts as NOT DELIVERED (listed in provenance), not as a gap.

# ---- vocabulary --------------------------------------------------------------------------
CLASSES = ("Extensive", "Intensive", "Categorical")
SPRING_RULES = {
    "Extensive": ("inject_zero",),
    "Intensive": ("interpolate", "locf"),
    "Categorical": ("carry_forward",),
}
AUTUMN_RULES = {
    "Extensive": ("sum",),
    "Intensive": ("mean", "volume_weighted"),
    "Categorical": ("first",),
}
TIME_BASES = ("local_clock", "fixed_96")
INPUT_CLASSES = ("offtaker_load", "pv_generation", "baseload_nomination", "wholesale_prices")
K_VALUES = (-1, 1)
VOLUME_UNITS = ("MWh",)
BASES = ("metered", "notified", "forecast", "nomination", "price", "ratio", "flag")

CONTROL_KEYS = (
    "template_version",
    "input_class",
    "spine_year",
    "intervals_per_day",
    "time_basis",
    "offset_eet_cet_h",
    "scenario",
    "provider",
    "delivery_date",
    "notes",
)

REGISTRY_COLUMNS = (
    "slot",
    "name",
    "unit",
    "class",
    "spring_rule",
    "autumn_rule",
    "paired_volume_slot",
    "k",
    "basis",
    "entity_code",
    "required",
    "notes",
)
OPTIONAL_REGISTRY_COLUMNS = ("scenario_name", "entity_name")

RAW_FIXED_COLUMNS = ("Date_EET", "Start_EET", "End_EET")
MAX_E_SLOTS = 40  # of the 1.0 template; 1.1 files declare as many slots as they carry (D112)
MAX_C_SLOTS = 10


@dataclass
class SlotSpec:
    slot: str  # E01.. or C01.. (any digit width)
    name: str
    unit: str
    cls: str  # Extensive | Intensive | Categorical
    spring_rule: str
    autumn_rule: str
    paired_volume_slot: str | None = None
    k: int | None = None
    basis: str | None = None
    scenario_name: str | None = None  # 1.1: wholesale scenario block this slot belongs to
    entity_name: str | None = None  # 1.1: reference only (names never enter the engine or the repository)
    entity_code: str | None = None
    required: bool = True
    notes: str = ""

    @property
    def is_volume(self) -> bool:
        return self.unit in VOLUME_UNITS and self.basis in ("metered", "notified", "forecast", "nomination")

    @property
    def frame_name(self) -> str:
        """Column name in the standardised frame: the series name, prefixed by the scenario block (1.1)."""
        return f"{self.scenario_name}__{self.name}" if self.scenario_name else self.name

    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.cls not in CLASSES:
            errs.append(f"{self.slot}: class '{self.cls}' not in {CLASSES}")
            return errs
        if self.slot.startswith("C") != (self.cls == "Categorical"):
            errs.append(f"{self.slot}: C-slots are Categorical and E-slots are numeric")
        if self.spring_rule not in SPRING_RULES[self.cls]:
            errs.append(f"{self.slot}: spring_rule '{self.spring_rule}' not allowed for {self.cls}")
        if self.autumn_rule not in AUTUMN_RULES[self.cls]:
            errs.append(f"{self.slot}: autumn_rule '{self.autumn_rule}' not allowed for {self.cls}")
        if self.autumn_rule == "volume_weighted" and not self.paired_volume_slot:
            errs.append(f"{self.slot}: volume_weighted needs paired_volume_slot")
        if self.is_volume and self.k not in K_VALUES:
            errs.append(f"{self.slot}: volume series must declare k = -1 (DSO) or +1 (BRP)")
        if self.k is not None and self.k not in K_VALUES:
            errs.append(f"{self.slot}: k must be -1 or +1")
        if self.basis is not None and self.basis not in BASES:
            errs.append(f"{self.slot}: basis '{self.basis}' not in {BASES}")
        if not self.name:
            errs.append(f"{self.slot}: name missing")
        return errs


@dataclass
class Control:
    input_class: str
    spine_year: int
    time_basis: str = "local_clock"
    intervals_per_day: int = 96
    offset_eet_cet_h: int = 1
    template_version: str = TEMPLATE_VERSION
    scenario: str = ""
    provider: str = ""
    delivery_date: str = ""
    notes: str = ""

    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.input_class not in INPUT_CLASSES:
            errs.append(f"input_class '{self.input_class}' not in {INPUT_CLASSES}")
        if not (2020 <= int(self.spine_year) <= 2060):
            errs.append(f"spine_year {self.spine_year} out of range")
        if self.time_basis not in TIME_BASES:
            errs.append(f"time_basis '{self.time_basis}' not in {TIME_BASES}")
        if int(self.intervals_per_day) != 96:
            errs.append("intervals_per_day must be 96")
        if int(self.offset_eet_cet_h) != 1:
            errs.append("offset_eet_cet_h must be 1")
        if str(self.template_version).strip() not in TEMPLATE_VERSIONS:
            errs.append(f"template_version '{self.template_version}' not in {TEMPLATE_VERSIONS}")
        return errs

    @property
    def contract_11(self) -> bool:
        return str(self.template_version).strip() == "ESB-STD-QH 1.1"


@dataclass
class Registry:
    slots: list[SlotSpec] = field(default_factory=list)

    def by_slot(self) -> dict[str, SlotSpec]:
        return {s.slot: s for s in self.slots}

    def validate(self) -> list[str]:
        errs: list[str] = []
        seen: set[str] = set()
        names: set[str] = set()
        for s in self.slots:
            if s.slot in seen:
                errs.append(f"{s.slot}: declared twice")
            seen.add(s.slot)
            key = (s.scenario_name or "", s.name)
            if key in names:
                errs.append(f"{s.slot}: name '{s.name}' used twice" + (f" in scenario '{s.scenario_name}'" if s.scenario_name else ""))
            names.add(key)
            errs.extend(s.validate())
        by = self.by_slot()
        for s in self.slots:
            if s.paired_volume_slot:
                p = by.get(s.paired_volume_slot)
                if p is None:
                    errs.append(f"{s.slot}: paired_volume_slot {s.paired_volume_slot} not declared")
                elif p.cls != "Extensive":
                    errs.append(f"{s.slot}: paired_volume_slot {s.paired_volume_slot} is not Extensive")
        return errs


# ---- presets per input class ----------------------------------------------------------------
def _c(i: int, width: int = 2) -> str:
    return f"C{i:0{width}d}"


def _e(i: int) -> str:
    return f"E{i:02d}"


def offtaker_load_registry(entity_codes: list[str]) -> Registry:
    """Two Extensive slots per off-taker: DSO metered and DSO notified consumption (+), k = -1."""
    slots: list[SlotSpec] = []
    i = 1
    for code in entity_codes:
        for basis in ("metered", "notified"):
            slots.append(
                SlotSpec(
                    slot=_e(i),
                    name=f"{code}_{basis}_consumption_MWh",
                    unit="MWh",
                    cls="Extensive",
                    spring_rule="inject_zero",
                    autumn_rule="sum",
                    k=-1,
                    basis=basis,
                    entity_code=code,
                    required=True,
                    notes="DSO metering sign convention: consumption (+); k = -1 declared once per series",
                )
            )
            i += 1
    return Registry(slots)


def pv_generation_registry(entity_code: str = "PV1") -> Registry:
    return Registry(
        [
            SlotSpec(
                _e(1),
                f"{entity_code}_forecast_generation_uncurtailed_MWh",
                "MWh",
                "Extensive",
                "inject_zero",
                "sum",
                k=-1,
                basis="forecast",
                entity_code=entity_code,
                notes="Positive generation forecast; the engine derives DSO metered/notified (sign -) "
                "with k = -1 from this series and the two deviation series",
            ),
            SlotSpec(
                _e(2),
                f"{entity_code}_forecast_generation_deviation_pct",
                "ratio",
                "Intensive",
                "interpolate",
                "volume_weighted",
                paired_volume_slot=_e(1),
                basis="ratio",
                entity_code=entity_code,
                notes="Metered = -(forecast + deviation x |forecast|); dimensionless (0,05 = 5 %)",
            ),
            SlotSpec(
                _e(3),
                f"{entity_code}_imbalance_deviation_pct",
                "ratio",
                "Intensive",
                "interpolate",
                "volume_weighted",
                paired_volume_slot=_e(1),
                basis="ratio",
                entity_code=entity_code,
                notes="IMB_% in the rule-book sense (negative = deficit); notified = metered / (1 + k x IMB_% x sign(metered))",
            ),
        ]
    )


def baseload_nomination_registry(entity_codes: list[str]) -> Registry:
    slots: list[SlotSpec] = []
    i = 1
    for code in entity_codes:
        for product in ("BL24", "Peak", "OffPeak"):
            slots.append(
                SlotSpec(
                    _e(i),
                    f"{code}_{product}_nomination_MWh",
                    "MWh",
                    "Extensive",
                    "inject_zero",
                    "sum",
                    k=1,
                    basis="nomination",
                    entity_code=code,
                    required=False,
                    notes="Energy per quarter-hour (MW / 4); BRP sign convention (+ = delivered to the portfolio)",
                )
            )
            i += 1
    return Registry(slots)


def wholesale_prices_registry(scenarios: list[str] | None = None) -> Registry:
    """One block of five series per scenario (1.1, D112): DAM, IDCT VWAP15, surplus and deficit imbalance prices and the
    system direction flag. Without scenarios: the single unnamed block of contract 1.0."""
    blocks = scenarios or [None]
    slots: list[SlotSpec] = []
    e = c = 1
    for sc in blocks:
        slots += [
            SlotSpec(_e(e), "DAM_price_EUR_MWh", "EUR/MWh", "Intensive", "interpolate", "mean", basis="price", scenario_name=sc),
            SlotSpec(_e(e + 1), "IDCT_VWAP15_price_EUR_MWh", "EUR/MWh", "Intensive", "interpolate", "mean", basis="price", scenario_name=sc),
            SlotSpec(_e(e + 2), "Surplus_imbalance_price_EUR_MWh", "EUR/MWh", "Intensive", "interpolate", "mean", basis="price", scenario_name=sc),
            SlotSpec(_e(e + 3), "Deficit_imbalance_price_EUR_MWh", "EUR/MWh", "Intensive", "interpolate", "mean", basis="price", scenario_name=sc),
            SlotSpec(_c(c), "System_imbalance_direction", "flag", "Categorical", "carry_forward", "first", basis="flag", required=False,
                     scenario_name=sc, notes="'Positive (Long)' | 'Negative (Short)' | 'Balanced'"),
        ]
        e += 4
        c += 1
    return Registry(slots)


def preset_registry(input_class: str, entity_codes: list[str] | None = None, scenarios: list[str] | None = None) -> Registry:
    codes = entity_codes or ["OT1", "OT2", "OT3", "OT4"]
    if input_class == "offtaker_load":
        return offtaker_load_registry(codes)
    if input_class == "pv_generation":
        return pv_generation_registry((entity_codes or ["PV1"])[0])
    if input_class == "baseload_nomination":
        return baseload_nomination_registry(codes)
    if input_class == "wholesale_prices":
        return wholesale_prices_registry(scenarios)
    raise ValueError(f"unknown input_class {input_class}")

"""The six binding checks of the upload contract (D-IMP). Any FAIL refuses the delivery.

1 spine     - control block valid: input class, spine year, 96 intervals, time basis, offset
2 layout    - every (date, interval) of the year delivered exactly as many times as the time
              basis requires (once; local_clock: 0 x on the four spring intervals, 2 x on the
              four autumn intervals); no rows outside the year; no unreadable date/time
3 surface   - no formulas in RAW_EET_QH; no undeclared column carrying data; header intact;
              no non-numeric value in a numeric slot
4 registry  - every declared slot consistent (class / rules / unit / k / pairing); required
              slots present in the raw surface
5 energy    - for every Extensive slot: sum over the year and per day of the standardised
              series equals the sum of the raw rows (tolerance 1e-6 relative, floor 1e-6)
6 gaps      - no blank cell inside a required series after standardisation (the importer
              never fills a gap); blanks in optional series are counted and reported
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from esb.importer.contract import Control, Registry
from esb.importer.reader import RawDelivery
from esb.importer.standardise import LayoutReport

TOL_REL = 1e-6
TOL_FLOOR = 1.0


@dataclass
class CheckResult:
    code: str
    name: str
    passed: bool
    detail: str = ""
    items: list = field(default_factory=list)  # up to ~50 concrete positions or messages

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


def _within(a: float, b: float) -> bool:
    return abs(a - b) <= TOL_REL * max(abs(b), TOL_FLOOR)


def check_spine(control: Control, problems: list[str]) -> CheckResult:
    errs = list(problems) + control.validate()
    return CheckResult("1", "spine", not errs, "control block valid" if not errs else "; ".join(errs[:10]), errs[:50])


def check_layout(layout: LayoutReport) -> CheckResult:
    items: list[str] = []
    if layout.unparseable_rows:
        items.append(f"{layout.unparseable_rows} rows with unreadable date or start time")
    if layout.out_of_year:
        items.append(f"{layout.out_of_year} rows outside the spine year")
    for d, iv in layout.missing[:20]:
        items.append(f"missing {d:%d.%m.%Y} interval {iv}")
    for d, iv in layout.duplicated[:20]:
        items.append(f"duplicated {d:%d.%m.%Y} interval {iv}")
    ok = layout.ok
    detail = (
        f"{layout.delivered_rows} rows delivered, {layout.expected_rows} expected"
        + ("" if ok else f"; {len(layout.missing)} missing, {len(layout.duplicated)} duplicated")
        + ("; rows were not in chronological order (accepted, re-sorted)" if layout.non_monotonic else "")
    )
    return CheckResult("2", "layout", ok, detail, items)


def check_surface(delivery: RawDelivery, non_numeric: dict[str, int]) -> CheckResult:
    items: list[str] = []
    if delivery.formula_cells:
        items.append(f"formulas found at {', '.join(delivery.formula_cells[:10])}" + (" ..." if len(delivery.formula_cells) > 10 else ""))
    if delivery.undeclared_columns:
        items.append(f"undeclared columns with data: {', '.join(delivery.undeclared_columns)}")
    for slot, n in non_numeric.items():
        items.append(f"{slot}: {n} non-numeric values")
    return CheckResult("3", "surface", not items, "values only, declared columns only" if not items else "; ".join(items), items)


def check_registry(registry: Registry, raw_columns: list[str]) -> CheckResult:
    errs = registry.validate()
    if not registry.slots:
        errs.append("no slot declared")
    for s in registry.slots:
        if s.required and s.slot not in raw_columns:
            errs.append(f"{s.slot} ({s.name}) declared as required but absent from RAW_EET_QH")
    return CheckResult("4", "registry", not errs, f"{len(registry.slots)} slots declared" if not errs else "; ".join(errs[:10]), errs[:50])


def check_energy(labelled: pd.DataFrame, std: pd.DataFrame, registry: Registry) -> CheckResult:
    items: list[str] = []
    n = 0
    for s in registry.slots:
        col = s.slot if s.slot in labelled.columns else (s.name if s.name in labelled.columns else None)
        if s.cls != "Extensive" or col is None:
            continue
        n += 1
        raw_total = float(pd.to_numeric(labelled[col], errors="coerce").sum())
        std_total = float(std[s.frame_name].sum())
        if not _within(std_total, raw_total):
            items.append(f"{s.name}: year {std_total:.6f} vs raw {raw_total:.6f}")
        raw_day = pd.to_numeric(labelled[col], errors="coerce").groupby(labelled["date"].values).sum()
        std_day = std.groupby(std["date"].dt.date.values)[s.frame_name].sum()
        diff = (std_day.reindex(raw_day.index).fillna(0) - raw_day).abs()
        tol = TOL_REL * np.maximum(raw_day.abs(), TOL_FLOOR)
        bad = diff[diff > tol]
        for d in list(bad.index)[:5]:
            items.append(f"{s.name}: {d:%d.%m.%Y} std {std_day.get(d, float('nan')):.6f} vs raw {raw_day[d]:.6f}")
    return CheckResult("5", "energy", not items, f"{n} extensive slots reconciled (year and day)" if not items else "; ".join(items[:5]), items)


def check_gaps(std: pd.DataFrame, registry: Registry) -> CheckResult:
    items: list[str] = []
    optional_blanks: dict[str, int] = {}
    for s in registry.slots:
        if s.frame_name not in std.columns:
            continue
        col = std[s.frame_name]
        blanks = col.isna() if s.cls != "Categorical" else col.isna() | (col.astype(object) == "")
        nb = int(blanks.sum())
        if nb == 0:
            continue
        if s.required:
            first = std.loc[blanks, ["date", "interval"]].head(3)
            where = ", ".join(f"{d:%d.%m.%Y}/{iv}" for d, iv in zip(first["date"], first["interval"], strict=True))
            items.append(f"{s.name}: {nb} blank cells (first at {where})")
        else:
            optional_blanks[s.name] = nb
    detail = "no blank inside a required series" if not items else "; ".join(items[:5])
    if optional_blanks:
        detail += "; optional series with blanks: " + ", ".join(f"{k} ({v})" for k, v in optional_blanks.items())
    return CheckResult("6", "gaps", not items, detail, items)

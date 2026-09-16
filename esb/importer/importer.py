"""Orchestrator: workbook -> validated, standardised frame with provenance (D-IMP)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from esb.importer.checks import (
    CheckResult,
    check_energy,
    check_gaps,
    check_layout,
    check_registry,
    check_spine,
    check_surface,
)
from esb.importer.contract import Control, Registry
from esb.importer.reader import RawDelivery, read_workbook
from esb.importer.standardise import label_rows, standardise


@dataclass
class Provenance:
    filename: str
    md5: str
    size_bytes: int
    template_version: str
    input_class: str
    spine_year: int
    time_basis: str
    scenario: str
    provider: str
    delivery_date: str
    slots: list[dict]
    k: dict[str, int]
    imported_at_utc: str
    checks: list[dict] = field(default_factory=list)
    not_delivered: list[str] = field(default_factory=list)  # 1.1 (D112): declared slots with an entirely blank column
    scenario_blocks: list[str] = field(default_factory=list)  # 1.1: scenario names found in the registry (wholesale)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


@dataclass
class ImportResult:
    ok: bool
    checks: list[CheckResult]
    frame: pd.DataFrame | None  # grid columns + one column per slot name; None when refused
    provenance: Provenance
    control: Control
    registry: Registry

    def summary(self) -> str:
        lines = [f"{'ACCEPTED' if self.ok else 'REFUSED'}: {self.provenance.filename} ({self.provenance.input_class}, {self.provenance.spine_year}, {self.provenance.time_basis})"]
        for c in self.checks:
            lines.append(f"  {c.code} {c.name:9s} {c.status}  {c.detail}")
        return "\n".join(lines)


def run_checks(delivery: RawDelivery) -> tuple[list[CheckResult], pd.DataFrame | None]:
    checks: list[CheckResult] = []
    c1 = check_spine(delivery.control, delivery.problems)
    checks.append(c1)
    c4 = check_registry(delivery.registry, list(delivery.raw.columns))
    checks.append(c4)
    c3 = check_surface(delivery, delivery.non_numeric)
    checks.append(c3)
    if not (c1.passed and c4.passed) or delivery.raw.empty:
        if delivery.raw.empty:
            checks.append(CheckResult("2", "layout", False, "RAW_EET_QH has no data rows"))
        return sorted(checks, key=lambda c: c.code), None
    labelled, layout = label_rows(delivery.raw, delivery.control)
    checks.append(check_layout(layout))
    std = standardise(labelled, delivery.control, delivery.registry)
    checks.append(check_energy(labelled, std, delivery.registry))
    checks.append(check_gaps(std, delivery.registry))
    return sorted(checks, key=lambda c: c.code), std


def import_workbook(path: str | Path) -> ImportResult:
    delivery = read_workbook(path)
    checks, std = run_checks(delivery)
    ok = all(c.passed for c in checks)
    prov = Provenance(
        filename=delivery.path.name,
        md5=delivery.md5,
        size_bytes=delivery.size,
        template_version=delivery.control.template_version,
        input_class=delivery.control.input_class,
        spine_year=int(delivery.control.spine_year),
        time_basis=delivery.control.time_basis,
        scenario=delivery.control.scenario,
        provider=delivery.control.provider,
        delivery_date=delivery.control.delivery_date,
        slots=[asdict(s) for s in delivery.registry.slots],
        k={s.name: s.k for s in delivery.registry.slots if s.k is not None},
        imported_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        checks=[{"code": c.code, "name": c.name, "status": c.status, "detail": c.detail} for c in checks],
        not_delivered=list(delivery.not_delivered),
        scenario_blocks=sorted({s.scenario_name for s in delivery.registry.slots if s.scenario_name}),
    )
    return ImportResult(ok=ok, checks=checks, frame=std if ok else None, provenance=prov, control=delivery.control, registry=delivery.registry)

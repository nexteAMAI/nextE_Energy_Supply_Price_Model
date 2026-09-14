"""Reader for a delivered upload workbook: control block, registry and the raw paste surface.

The workbook is opened with formulas visible (data_only=False) so that any formula in
RAW_EET_QH is detected and refused; values are taken as stored. Nothing is transformed here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from esb.grid import excel_serial_to_date
from esb.importer.contract import (
    RAW_FIXED_COLUMNS,
    Control,
    Registry,
    SlotSpec,
)

REQUIRED_SHEETS = ("Std_Control", "Series_Registry", "RAW_EET_QH")


@dataclass
class RawDelivery:
    path: Path
    md5: str
    size: int
    control: Control
    registry: Registry
    raw: pd.DataFrame  # columns: Date_EET (date), Start_EET (time), End_EET (time), <slot>...
    header: list[str]
    problems: list[str] = field(default_factory=list)  # structural problems found while reading
    formula_cells: list[str] = field(default_factory=list)
    undeclared_columns: list[str] = field(default_factory=list)
    non_numeric: dict[str, int] = field(default_factory=dict)  # slot -> count of non-numeric cells


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_date(v) -> date | None:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, (int, float)):
        return excel_serial_to_date(v)
    if isinstance(v, str):
        s = v.strip()
        for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                pass
    raise ValueError(f"unreadable date {v!r}")


def _to_time(v) -> time | None:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v.time()
    if isinstance(v, time):
        return v
    if isinstance(v, timedelta):
        secs = int(v.total_seconds()) % 86400
        return time(secs // 3600, (secs % 3600) // 60)
    if isinstance(v, (int, float)):
        secs = int(round((float(v) % 1) * 86400))
        return time(secs // 3600, (secs % 3600) // 60)
    if isinstance(v, str):
        s = v.strip()
        for fmt in ("%H:%M", "%H:%M:%S"):
            try:
                return datetime.strptime(s, fmt).time()
            except ValueError:
                pass
    raise ValueError(f"unreadable time {v!r}")


def _read_control(ws) -> tuple[Control, list[str]]:
    kv: dict[str, object] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row and row[0]:
            kv[str(row[0]).strip()] = row[1] if len(row) > 1 else None
    problems: list[str] = []

    def get(k, default=None):
        v = kv.get(k, default)
        return default if v is None or v == "" else v

    try:
        control = Control(
            input_class=str(get("input_class", "")),
            spine_year=int(get("spine_year", 0)),
            time_basis=str(get("time_basis", "local_clock")),
            intervals_per_day=int(get("intervals_per_day", 96)),
            offset_eet_cet_h=int(get("offset_eet_cet_h", 1)),
            template_version=str(get("template_version", "")),
            scenario=str(get("scenario", "")),
            provider=str(get("provider", "")),
            delivery_date=str(get("delivery_date", "")),
            notes=str(get("notes", "")),
        )
    except (TypeError, ValueError) as e:
        problems.append(f"Std_Control unreadable: {e}")
        control = Control(input_class="", spine_year=0)
    return control, problems


def _read_registry(ws) -> tuple[Registry, list[str]]:
    header = [str(c).strip() if c is not None else "" for c in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
    idx = {h: i for i, h in enumerate(header)}
    problems: list[str] = []
    needed = ("slot", "name", "unit", "class", "spring_rule", "autumn_rule")
    for n in needed:
        if n not in idx:
            problems.append(f"Series_Registry lacks column '{n}'")
    slots: list[SlotSpec] = []
    if not problems:
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row or not row[idx["slot"]]:
                continue

            def cell(name, _row=row):
                i = idx.get(name)
                if i is None or i >= len(_row):
                    return None
                v = _row[i]
                return None if v is None or v == "" else v

            name = cell("name")
            if name is None:
                continue  # an undeclared slot row (template placeholder)
            k = cell("k")
            slots.append(
                SlotSpec(
                    slot=str(cell("slot")).strip(),
                    name=str(name).strip(),
                    unit=str(cell("unit") or "").strip(),
                    cls=str(cell("class") or "").strip(),
                    spring_rule=str(cell("spring_rule") or "").strip(),
                    autumn_rule=str(cell("autumn_rule") or "").strip(),
                    paired_volume_slot=(str(cell("paired_volume_slot")).strip() or None) if cell("paired_volume_slot") else None,
                    k=int(k) if k is not None else None,
                    basis=(str(cell("basis")).strip() or None) if cell("basis") else None,
                    entity_code=(str(cell("entity_code")).strip() or None) if cell("entity_code") else None,
                    required=str(cell("required") or "Y").strip().upper() != "N",
                    notes=str(cell("notes") or ""),
                )
            )
    return Registry(slots), problems


def read_workbook(path: str | Path) -> RawDelivery:
    """Read an upload workbook. Structural problems are collected, not raised, so that the
    importer can report every finding at once."""
    path = Path(path)
    md5 = file_md5(path)
    size = path.stat().st_size
    wb = load_workbook(path, read_only=True, data_only=False)
    problems: list[str] = []
    for s in REQUIRED_SHEETS:
        if s not in wb.sheetnames:
            problems.append(f"sheet '{s}' missing")
    if problems:
        wb.close()
        return RawDelivery(path, md5, size, Control("", 0), Registry(), pd.DataFrame(), [], problems)

    control, p = _read_control(wb["Std_Control"])
    problems += p
    registry, p = _read_registry(wb["Series_Registry"])
    problems += p
    declared = registry.by_slot()

    ws = wb["RAW_EET_QH"]
    rows = ws.iter_rows(values_only=True)
    header = [str(c).strip() if c is not None else "" for c in next(rows)]
    if tuple(header[:3]) != RAW_FIXED_COLUMNS:
        problems.append(f"RAW_EET_QH header must start with {RAW_FIXED_COLUMNS}, found {header[:3]}")
    slot_cols = header[3:]
    undeclared_seen: dict[str, bool] = {}
    formula_cells: list[str] = []
    records: list[list] = []
    for r, row in enumerate(rows, start=2):
        if row is None:
            continue
        row = list(row) + [None] * (len(header) - len(row))
        if all(v is None or v == "" for v in row[: len(header)]):
            continue
        rec: list = []
        for c, v in enumerate(row[: len(header)]):
            if isinstance(v, str) and v.startswith("="):
                if len(formula_cells) < 50:
                    formula_cells.append(f"{header[c] or c}{r}")
                v = None
            if c >= 3:
                col = slot_cols[c - 3]
                if col not in declared and v is not None and v != "":
                    undeclared_seen[col or f"col{c + 1}"] = True
            rec.append(v)
        records.append(rec)
    wb.close()

    df = pd.DataFrame(records, columns=header) if records else pd.DataFrame(columns=header)
    if not df.empty:
        try:
            df["Date_EET"] = [_to_date(v) for v in df["Date_EET"]]
            df["Start_EET"] = [_to_time(v) for v in df["Start_EET"]]
            df["End_EET"] = [_to_time(v) for v in df["End_EET"]]
        except ValueError as e:
            problems.append(f"RAW_EET_QH date/time column unreadable: {e}")
    keep = list(RAW_FIXED_COLUMNS) + [c for c in slot_cols if c in declared]
    df = df[[c for c in keep if c in df.columns]]
    non_numeric: dict[str, int] = {}
    for spec in registry.slots:
        if spec.slot in df.columns:
            if spec.cls == "Categorical":
                df[spec.slot] = df[spec.slot].map(lambda v: None if v is None or v == "" or (isinstance(v, float) and v != v) else str(v))
            else:
                present = df[spec.slot].map(lambda v: v is not None and v != "")
                num = pd.to_numeric(df[spec.slot], errors="coerce")
                bad = int((present & num.isna()).sum())
                if bad:
                    non_numeric[spec.slot] = bad
                df[spec.slot] = num
        else:
            df[spec.slot] = float("nan") if spec.cls != "Categorical" else None
    return RawDelivery(
        path=path,
        md5=md5,
        size=size,
        control=control,
        registry=registry,
        raw=df,
        header=header,
        problems=problems,
        formula_cells=formula_cells,
        undeclared_columns=sorted(undeclared_seen),
        non_numeric=non_numeric,
    )

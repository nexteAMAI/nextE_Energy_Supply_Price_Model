"""Writer for the standard upload template (D-TPL) and for filled fixture workbooks.

`build_template(...)` writes a blank template for one input class; `write_delivery(...)` writes
a filled workbook (used for the Reference Case fixtures and for tests). Both produce exactly
the layout that esb.importer.reader expects.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from esb import grid
from esb.importer.contract import (
    CONTROL_KEYS,
    RAW_FIXED_COLUMNS,
    REGISTRY_COLUMNS,
    TEMPLATE_VERSION,
    Control,
    Registry,
    preset_registry,
)

NAVY = "1F3E66"  # nexte-brand navy (app.brand.NAVY)
INK = "0E1C2E"
MUTED = "6A6A6A"
GREY = "F7F6F3"  # brand paper: the cells the deliverer fills
HEAD_FONT = Font(name="Montserrat", size=9, bold=True, color="FFFFFF")
BODY_FONT = Font(name="Montserrat", size=9, color=INK)
TITLE_FONT = Font(name="Montserrat", size=12, bold=True, color=NAVY)
EYEBROW_FONT = Font(name="Montserrat", size=8, bold=True, color=MUTED)
NUM_FORMAT = "#,##0.000;(#,##0.000);0.000"  # locale-aware tokens: a Romanian Excel renders 1.234,500
HEAD_FILL = PatternFill("solid", fgColor=NAVY)
INPUT_FILL = PatternFill("solid", fgColor=GREY)

INSTRUCTIONS = [
    "nextE Energy Supply Bid Management Tool - standard quarter-hourly upload template",
    f"Template version {TEMPLATE_VERSION}. Sheets: Std_Control, Series_Registry, RAW_EET_QH, Recon_Check.",
    "",
    "01 What to deliver",
    "One workbook per input class and spine year. Paste values only into RAW_EET_QH (no formulas, no links).",
    "Columns A:C carry the local date (EET, Romanian clock), the interval start and the interval end.",
    "Columns D onward are the series slots E01.. (numbers) and C01.. (categories) exactly as declared in Series_Registry.",
    "",
    "02 Time basis (Std_Control!time_basis)",
    "local_clock: rows follow the real local clock. The last Sunday of March has 92 rows (03:00-03:45 absent);",
    "  the last Sunday of October has 100 rows (03:00-03:45 twice). The importer injects / merges by the declared rules.",
    "fixed_96: rows are already 96 per day for every day (positional grid). No daylight-saving rule is applied.",
    "",
    "03 Series registry",
    "Every slot with data must be declared: name, unit, class (Extensive = energy per interval; Intensive = price, ratio, power;",
    "Categorical = flag), the spring and autumn rules, and for volume series the sign convention k (-1 DSO metering,",
    "+1 BRP imbalance). Declared once per series, never inferred from the values.",
    "",
    "04 Rules applied at import (local_clock only)",
    "Extensive: injected interval = 0; merged interval = sum. Intensive: injected = linear interpolation or last value",
    "(spring_rule); merged = mean or volume-weighted mean over paired_volume_slot (autumn_rule). Categorical: carry forward / first.",
    "",
    "05 What the importer refuses",
    "Formulas in RAW_EET_QH; undeclared columns with data; blanks inside a required series (gaps are never filled);",
    "missing or duplicated intervals other than the daylight-saving pattern; dates outside the spine year;",
    "a volume series without k; a registry that contradicts itself.",
    "",
    "06 Formats",
    "Dates dd.mm.yyyy, times hh:mm. Decimal separator follows your Excel locale; the stored values are what count.",
    "Off-takers are referenced by code (OT1, OT2, ...); the name-to-code mapping is kept in the application, not in this file.",
]


def _style_header(ws, ncols: int) -> None:
    for c in range(1, ncols + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = HEAD_FONT
        cell.fill = HEAD_FILL
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 30
    ws.freeze_panes = "A2"
    _style_body(ws)


def _style_body(ws) -> None:
    """Montserrat on every written cell below the header (brand: one typeface on every surface)."""
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            if cell.value is not None and cell.font.name != "Montserrat":
                cell.font = BODY_FONT


def _write_instructions(wb: Workbook) -> None:
    ws = wb.active
    ws.title = "Instructions"
    ws["A1"] = "CONFIDENTIAL - nextE"
    ws["A1"].font = EYEBROW_FONT
    for i, line in enumerate(INSTRUCTIONS, start=2):
        cell = ws.cell(row=i, column=1, value=line)
        cell.font = TITLE_FONT if i == 2 else (Font(name="Montserrat", size=9, bold=True, color=NAVY) if line[:2].isdigit() else BODY_FONT)
    ws.column_dimensions["A"].width = 120


def _write_control(wb: Workbook, control: Control) -> None:
    ws = wb.create_sheet("Std_Control")
    ws.append(["key", "value", "note"])
    notes = {
        "template_version": "do not change",
        "input_class": "offtaker_load | pv_generation | baseload_nomination | wholesale_prices",
        "spine_year": "calendar year of every row in RAW_EET_QH",
        "intervals_per_day": "96 (quarter-hourly)",
        "time_basis": "local_clock | fixed_96 (see Instructions 02)",
        "offset_eet_cet_h": "1 (fixed)",
        "scenario": "free text, e.g. Aurora Central Q2/2026 (wholesale_prices only)",
        "provider": "who produced the data",
        "delivery_date": "dd.mm.yyyy",
        "notes": "free text",
    }
    values = {
        "template_version": control.template_version,
        "input_class": control.input_class,
        "spine_year": control.spine_year,
        "intervals_per_day": control.intervals_per_day,
        "time_basis": control.time_basis,
        "offset_eet_cet_h": control.offset_eet_cet_h,
        "scenario": control.scenario,
        "provider": control.provider,
        "delivery_date": control.delivery_date,
        "notes": control.notes,
    }
    for k in CONTROL_KEYS:
        ws.append([k, values[k], notes[k]])
        ws.cell(row=ws.max_row, column=2).fill = INPUT_FILL
    _style_header(ws, 3)
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 28
    ws.column_dimensions["C"].width = 70


def _write_registry(wb: Workbook, registry: Registry) -> None:
    ws = wb.create_sheet("Series_Registry")
    ws.append(list(REGISTRY_COLUMNS))
    for s in registry.slots:
        ws.append(
            [
                s.slot,
                s.name,
                s.unit,
                s.cls,
                s.spring_rule,
                s.autumn_rule,
                s.paired_volume_slot or "",
                s.k if s.k is not None else "",
                s.basis or "",
                s.entity_code or "",
                "Y" if s.required else "N",
                s.notes,
            ]
        )
    _style_header(ws, len(REGISTRY_COLUMNS))
    widths = [7, 44, 10, 12, 14, 16, 18, 4, 12, 12, 9, 80]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _write_raw(wb: Workbook, registry: Registry, data: pd.DataFrame | None) -> int:
    ws = wb.create_sheet("RAW_EET_QH")
    slots = [s.slot for s in registry.slots]
    ws.append(list(RAW_FIXED_COLUMNS) + slots)
    n = 0
    if data is not None:
        cols = ["Date_EET", "Start_EET", "End_EET"] + [
            s.name for s in registry.slots
        ]
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise ValueError(f"data lacks columns {missing}")
        for row in data[cols].itertuples(index=False):
            vals = list(row)
            out = [vals[0], vals[1], vals[2]] + [
                (None if (isinstance(v, float) and v != v) else v) for v in vals[3:]
            ]
            ws.append(out)
            n += 1
    _style_header(ws, 3 + len(slots))
    ws.column_dimensions["A"].width = 12
    for j, s in enumerate(registry.slots, start=4):  # column-level formats so typed values render 1.234,500 in a Romanian Excel
        col = ws.column_dimensions[get_column_letter(j)]
        col.width = 16
        col.number_format = "@" if s.cls == "Categorical" else NUM_FORMAT
        col.font = BODY_FONT
    for letter, fmt in (("A", "dd.mm.yyyy"), ("B", "hh:mm"), ("C", "hh:mm")):
        ws.column_dimensions[letter].number_format = fmt
        ws.column_dimensions[letter].font = BODY_FONT
    for r in range(2, ws.max_row + 1):
        ws.cell(row=r, column=1).number_format = "dd.mm.yyyy"
        ws.cell(row=r, column=2).number_format = "hh:mm"
        ws.cell(row=r, column=3).number_format = "hh:mm"
        for j, s in enumerate(registry.slots, start=4):
            ws.cell(row=r, column=j).number_format = "@" if s.cls == "Categorical" else NUM_FORMAT
    return n


def _write_recon(wb: Workbook, control: Control) -> None:
    ws = wb.create_sheet("Recon_Check")
    g = grid.grid_info(int(control.spine_year))
    required = g.rows  # local_clock delivers -4 +4 rows, the same count
    ws.append(["check", "value", "required", "status"])
    ws.append(["rows delivered (dates in RAW_EET_QH)", "=COUNT(RAW_EET_QH!A:A)", required, '=IF(B2=C2,"PASS","FAIL")'])
    ws.append(["first date", "=MIN(RAW_EET_QH!A:A)", date(g.year, 1, 1), '=IF(B3=C3,"PASS","FAIL")'])
    ws.append(["last date", "=MAX(RAW_EET_QH!A:A)", date(g.year, 12, 31), '=IF(B4=C4,"PASS","FAIL")'])
    ws.append(["note", "The binding checks (six) run in the application at upload; this sheet is a convenience.", "", ""])
    for r in (3, 4):
        ws.cell(row=r, column=2).number_format = "dd.mm.yyyy"
        ws.cell(row=r, column=3).number_format = "dd.mm.yyyy"
    _style_header(ws, 4)
    ws.column_dimensions["A"].width = 40
    ws.column_dimensions["B"].width = 60
    ws.column_dimensions["C"].width = 14


def write_delivery(
    path: str | Path,
    control: Control,
    registry: Registry,
    data: pd.DataFrame | None = None,
) -> Path:
    """Write a template (data=None) or a filled delivery.

    `data` columns: Date_EET (date/datetime), Start_EET (time), End_EET (time) and one column per
    registry slot *name*. NaN is written as an empty cell.
    """
    errs = control.validate() + registry.validate()
    if errs:
        raise ValueError("; ".join(errs))
    wb = Workbook()
    _write_instructions(wb)
    _write_control(wb, control)
    _write_registry(wb, registry)
    _write_raw(wb, registry, data)
    _write_recon(wb, control)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


def build_template(
    input_class: str,
    spine_year: int,
    path: str | Path,
    entity_codes: list[str] | None = None,
    time_basis: str = "local_clock",
) -> Path:
    control = Control(input_class=input_class, spine_year=spine_year, time_basis=time_basis)
    return write_delivery(path, control, preset_registry(input_class, entity_codes))


def raw_frame_fixed_96(year: int, series: dict[str, pd.Series | list]) -> pd.DataFrame:
    """Helper: build a fixed_96 RAW frame (days x 96 rows) from per-row series keyed by slot name."""
    g = grid.make_grid(year)
    out = pd.DataFrame(
        {
            "Date_EET": [d.date() for d in g["date"]],
            "Start_EET": [t.time() for t in g["start_eet"]],
            "End_EET": [t.time() for t in g["end_eet"]],
        }
    )
    for name, s in series.items():
        out[name] = list(s)
    return out

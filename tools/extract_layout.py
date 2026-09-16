"""Extract the canonical output layout (D113) from the CEO's formatted template into data/layout/run_export.json.

Usage: python tools/extract_layout.py <TPL_output_run_export_ESB.xlsx> [out.json]

The workbook is streamed part by part (never opened with openpyxl in normal mode - it is tens of MB of styled
empty cells). Per sheet the tool records the metadata band, the header row, the used columns (value / spacer),
every row with its label, engine key (resolved through esb.labels against a Reference Case run), role (from the
designed signals: fill and font, never label prose), the CEO's unit and the engine's unit, blank separators and
section headers; for the wide grids the block titles, the column groups and the calendar fields. The renderer
(esb.layout) rebuilds the workbook from this specification with the run's values, so the CEO's files
stay the visual reference and never enter the repository or the deployed application.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import warnings
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from xml.etree import ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

SECTION_FILL, TOTAL_FILL, GROUP_FILL = "FF1F3E66", "FF0E1C2E", "FFF4F4F4"
UNIT_EQ = {"€": "EUR", "€/MWh": "EUR/MWh", "RON/€": "RON/EUR", "€/GC": "EUR/GC"}


# ---- streaming reader ---------------------------------------------------------------------------
def col_to_num(c: str) -> int:
    n = 0
    for ch in c:
        n = n * 26 + (ord(ch) - 64)
    return n


def num_to_col(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def split_ref(ref: str) -> tuple[str, int]:
    m = re.match(r"([A-Z]+)(\d+)", ref)
    return m.group(1), int(m.group(2))


class Book:
    def __init__(self, path: Path):
        self.z = zipfile.ZipFile(path)
        self.md5 = hashlib.md5(path.read_bytes()).hexdigest()
        self.xfs = self._styles()
        self.sst = self._sst()
        self.sheets = self._sheets()

    def _styles(self):
        t = ET.fromstring(self.z.read("xl/styles.xml"))
        numfmts = {0: "General", 1: "0", 2: "0.00", 3: "#,##0", 4: "#,##0.00", 9: "0%", 10: "0.00%", 14: "m/d/yyyy", 49: "@"}
        for nf in t.iter(NS + "numFmt"):
            numfmts[int(nf.get("numFmtId"))] = nf.get("formatCode")
        fonts = []
        for f in t.find(NS + "fonts"):
            d = {}
            for ch in f:
                tag = ch.tag.replace(NS, "")
                if tag == "name":
                    d["name"] = ch.get("val")
                elif tag == "sz":
                    d["sz"] = float(ch.get("val"))
                elif tag in ("b", "i"):
                    d[tag] = True
                elif tag == "color":
                    d["color"] = ch.get("rgb")
            fonts.append(d)
        fills = []
        for f in t.find(NS + "fills"):
            pf = f.find(NS + "patternFill")
            fg = pf.find(NS + "fgColor") if pf is not None else None
            fills.append(fg.get("rgb") if fg is not None else None)
        xfs = []
        for xf in t.find(NS + "cellXfs"):
            al = xf.find(NS + "alignment")
            xfs.append({"font": fonts[int(xf.get("fontId", 0))], "fill": fills[int(xf.get("fillId", 0))],
                        "nf": numfmts.get(int(xf.get("numFmtId", 0)), "General"), "h": al.get("horizontal") if al is not None else None})
        return xfs

    def _sst(self):
        try:
            t = ET.fromstring(self.z.read("xl/sharedStrings.xml"))
        except KeyError:
            return []
        return ["".join(x.text or "" for x in si.iter(NS + "t")) for si in t.iter(NS + "si")]

    def _sheets(self):
        wb = ET.fromstring(self.z.read("xl/workbook.xml"))
        rels = ET.fromstring(self.z.read("xl/_rels/workbook.xml.rels"))
        rmap = {r.get("Id"): r.get("Target") for r in rels}
        out = []
        for s in wb.find(NS + "sheets"):
            tgt = rmap[s.get(RID)].replace("/xl/", "")
            out.append((s.get("name"), "xl/" + tgt))
        return out

    def rows(self, part: str, max_rows: int | None = None):
        """Yield (row number, {col: (value, xf)}) plus a final dict of sheet properties."""
        props: dict = {"cols": [], "merges": [], "row_heights": {}}
        with self.z.open(part) as fh:
            for _ev, el in ET.iterparse(fh):
                tag = el.tag
                if tag == NS + "col":
                    props["cols"].append((int(el.get("min")), int(el.get("max")), float(el.get("width", 0)), el.get("hidden") == "1"))
                elif tag == NS + "mergeCell":
                    props["merges"].append(el.get("ref"))
                elif tag == NS + "sheetView":
                    p = el.find(NS + "pane")
                    props["freeze"] = p.get("topLeftCell") if p is not None and p.get("state") == "frozen" else None
                    props["gridlines"] = el.get("showGridLines") != "0"
                elif tag == NS + "row":
                    r = int(el.get("r"))
                    if el.get("ht"):
                        props["row_heights"][r] = float(el.get("ht"))
                    cells = {}
                    for c in el.findall(NS + "c"):
                        col, _ = split_ref(c.get("r"))
                        v = c.find(NS + "v")
                        isel = c.find(NS + "is")
                        val = None
                        if v is not None and v.text is not None:
                            val = self.sst[int(v.text)] if c.get("t") == "s" else v.text
                        elif isel is not None:
                            val = "".join(x.text or "" for x in isel.iter(NS + "t"))
                        cells[col] = (val, self.xfs[int(c.get("s", 0))])
                    yield r, cells
                    el.clear()
                    if max_rows and r >= max_rows:
                        break
        yield -1, props


# ---- resolution against the engine ------------------------------------------------------------------
def engine_context():
    import pandas as pd

    from config.schema import load_parameters
    from esb.engine import run
    from esb.export import _flatten
    from esb.labels import RETAIL, TOTAL, label, unit_of

    r = run(pd.read_parquet(ROOT / "data" / "reference" / "rc_v03_series.parquet"), load_parameters())
    first = r.pnl.codes[0]
    blocks = {
        "overview": {label(k): k for k in r.overview.table.index},
        "overview_cf": {label(k, leg=TOTAL): k for k in r.overview.cashflow},
        "overview_checks": {label(k, leg=""): k for k in r.overview.checks},
        "portfolio": {label(k): k for k in r.pnl.portfolio.order},
        "section": {label(k, leg=RETAIL): k for k in r.pnl.sections[first].order},
        "cashflow": {label(k, fallback=TOTAL): k for k in r.cashflow.rows},
        "pricing": {label(k, leg=RETAIL): k for k in r.pricing[first].year},
        "ledger_summary": {label(k, fallback=TOTAL): k for k in r.cashflow.daily_summary},
        "ledger": {label(c, fallback=TOTAL): c for c in r.cashflow.daily.columns},
        "parameters": {k: k for k in _flatten(r.params.to_dict())},
    }
    qh_keys = set(r.qh.qh.columns)
    return blocks, qh_keys, unit_of


def measure(text: str) -> str:
    return text.split(" · ")[0].strip()


def resolve(text: str, table: dict[str, str]) -> str | None:
    if text in table:
        return table[text]
    by_measure = {measure(k): v for k, v in table.items()}
    return by_measure.get(measure(text))


def role_of(cell, key: str | None, unit_of) -> str:
    val, xf = cell
    if xf["fill"] == SECTION_FILL:
        return "section"
    if xf["fill"] == TOTAL_FILL:
        return "total"
    if xf["fill"] == GROUP_FILL:
        return "group"
    if key and unit_of(key) == "check":
        return "check"
    if xf["font"].get("i"):
        return "memo"
    if xf["font"].get("b"):
        return "subtotal"
    return "data"


def widths_of(cols, upto: int) -> dict[str, float]:
    out = {}
    for lo, hi, w, _hidden in cols:
        for n in range(lo, min(hi, upto) + 1):
            out[num_to_col(n)] = round(w, 2)
    return out


# ---- per family ----------------------------------------------------------------------------------------
def label_sheet(book: Book, part: str, table_blocks: list[tuple[str, str]], tables: dict[str, dict], unit_of, ctx: str | None = None):
    """Label-column family: header row 6, label A, unit B, value columns from the header row; rows resolved
    block by block in reading order (the block changes at each section header the caller names)."""
    rows_out, band, header, props = [], {}, {}, {}
    block_i = 0
    section_starts = {b[0]: b[1] for b in table_blocks}  # section text -> block name
    current = table_blocks[0][1] if table_blocks else None
    for r, cells in book.rows(part):
        if r == -1:
            props = cells
            break
        if r <= 4:
            band[f"A{r}"] = cells.get("A", (None, None))[0]
            continue
        if r == 6:
            header = {c: v[0] for c, v in cells.items() if v[0] is not None}
            continue
        if r == 5:
            continue  # the empty row between the band and the header row
        a = cells.get("A", (None, book.xfs[0]))
        text = (a[0] or "").strip() if a[0] else ""
        b = cells.get("B", (None, book.xfs[0]))[0]
        values = {c: v[0] for c, v in cells.items() if v[0] is not None and c not in ("A", "B")}
        if not text and not values:
            rows_out.append({"r": r, "kind": "blank"})
            continue
        if a[1]["fill"] == SECTION_FILL:
            if text in section_starts:
                current = section_starts[text]
            rows_out.append({"r": r, "kind": "section", "text": text, "block": current})
            continue
        if a[1]["fill"] == TOTAL_FILL and (b == "Unit" or values.get("D") in ("Year", "Portfolio budget")):
            rows_out.append({"r": r, "kind": "header", "text": text, "block": current, "values": values})
            continue
        key = resolve(text, tables.get(current, {})) if current else None
        rows_out.append({"r": r, "kind": "line", "block": current, "label": text, "key": key, "role": role_of(a, key, unit_of),
                         "unit_template": b, "unit": unit_of(key, ctx) if key else UNIT_EQ.get(b, b),
                         "values": values or None})
        block_i += 1
    return {"family": "label", "band": band, "header": header, "rows": rows_out, "widths": widths_of(props["cols"], 30),
            "freeze": props.get("freeze"), "row_heights": {str(k): v for k, v in props["row_heights"].items() if v != 24.95}}


def grid_sheet(book: Book, part: str, qh_keys: set[str], unit_of):
    """Wide-grid family (QH_full): block titles row 5, KPI rows 7-10, share row 11, key row 12, unit row 13, label row 14."""
    band, titles, keys, units, labels, props = {}, {}, {}, {}, {}, {}
    for r, cells in book.rows(part, max_rows=14):
        if r == -1:
            props = cells
            break
        if r <= 4:
            band[f"A{r}"] = cells.get("A", (None, None))[0]
        elif r == 5:
            titles = {c: v[0] for c, v in cells.items() if v[0]}
        elif r == 12:
            keys = {c: v[0] for c, v in cells.items() if v[0]}
        elif r == 13:
            units = {c: v[0] for c, v in cells.items() if v[0] and c != "A"}
        elif r == 14:
            labels = {c: v[0] for c, v in cells.items() if v[0]}
    # props arrive after the break only if the loop ran to the end; fetch them separately
    for r, cells in book.rows(part, max_rows=1):
        if r == -1:
            props = cells
    cols = sorted(set(keys) | set(labels), key=lambda c: (len(c), c))
    columns = []
    for c in cols:
        k = keys.get(c)
        lab = labels.get(c)
        kind = "engine" if k in qh_keys else ("grid" if k in ("date", "interval", "seq", "month", "peak") else "calendar")
        columns.append({"col": c, "kind": kind, "key": k, "label": lab, "unit_template": units.get(c),
                        "unit": unit_of(k) if k in qh_keys else None})
    # block titles: merged ranges give the extent; without merges the title spans to the next spacer
    blocks = []
    merges = {split_ref(m.split(":")[0])[0]: m for m in props.get("merges", []) if m.split(":")[0][-1:].isdigit() and split_ref(m.split(":")[0])[1] == 5}
    for c, t in sorted(titles.items(), key=lambda kv: col_to_num(kv[0])):
        span = merges.get(c)
        blocks.append({"col": c, "title": t, "span_to": split_ref(span.split(":")[1])[0] if span else None})
    used = max(col_to_num(c) for c in cols) if cols else 0
    return {"family": "grid", "band": band, "blocks": blocks, "columns": columns, "widths": widths_of(props["cols"], used),
            "freeze": props.get("freeze"), "merged_titles": bool(merges)}


def parameters_sheet(book: Book, part: str, table: dict):
    band, header, rows_out, props = {}, {}, [], {}
    for r, cells in book.rows(part):
        if r == -1:
            props = cells
            break
        if r <= 4:
            band[f"A{r}"] = cells.get("A", (None, None))[0]
        elif r == 6:
            header = {c: v[0] for c, v in cells.items() if v[0]}
        elif r >= 7:
            a = cells.get("A", (None, book.xfs[0]))
            if a[0]:
                rows_out.append({"r": r, "text": a[0], "key": a[0] if a[0] in table else None, "fill": a[1]["fill"], "role": role_of(a, None, lambda k: "")})
    return {"family": "parameters", "band": band, "header": header, "rows": rows_out, "widths": widths_of(props["cols"], 6), "freeze": props.get("freeze")}


def provenance_sheet(book: Book, part: str):
    band, header, trace_rows, props = {}, {}, [], {}
    for r, cells in book.rows(part):
        if r == -1:
            props = cells
            break
        if r <= 4:
            band[f"A{r}"] = cells.get("A", (None, None))[0]
        elif r == 6:
            header = {c: v[0] for c, v in cells.items() if v[0]}
        elif r >= 7:
            a = cells.get("A", (None, book.xfs[0]))
            if a[0]:
                trace_rows.append({"r": r, "text": a[0], "fill": a[1]["fill"]})
    return {"family": "provenance", "band": band, "header": header, "rows": trace_rows, "widths": widths_of(props["cols"], 12), "freeze": props.get("freeze")}


def ledger_sheet(book: Book, part: str, tables: dict, unit_of):
    band, header, summary, props, first_day_row, n_day_rows = {}, {}, [], {}, None, 0
    in_summary = False
    for r, cells in book.rows(part):
        if r == -1:
            props = cells
            break
        if r <= 4:
            band[f"A{r}"] = cells.get("A", (None, None))[0]
        elif r == 6:
            header = {c: v[0] for c, v in cells.items() if v[0]}
        elif r >= 7:
            a = cells.get("A", (None, book.xfs[0]))
            if a[0] == "Summary":
                in_summary = True
                summary.append({"r": r, "kind": "section", "text": "Summary"})
            elif in_summary:
                if a[0]:
                    key = resolve(a[0], tables["ledger_summary"])
                    summary.append({"r": r, "kind": "header" if a[0] == "Item" else "line", "label": a[0], "key": key, "role": role_of(a, key, unit_of),
                                    "unit": unit_of(key) if key else None})
            else:
                if first_day_row is None:
                    first_day_row = r
                n_day_rows += 1
    columns = [{"col": c, "label": h, "key": resolve(h, tables["ledger"]), "unit": unit_of(resolve(h, tables["ledger"])) if resolve(h, tables["ledger"]) else None}
               for c, h in sorted(header.items(), key=lambda kv: col_to_num(kv[0]))]
    return {"family": "ledger", "band": band, "columns": columns, "first_row": first_day_row, "day_rows": n_day_rows, "summary": summary,
            "widths": widths_of(props["cols"], 31), "freeze": props.get("freeze")}


def main(src: str, out: str | None = None) -> None:
    book = Book(Path(src))
    blocks, qh_keys, unit_of = engine_context()
    spec = {"source": {"file": Path(src).name, "md5": book.md5, "extracted": date.today().strftime("%d.%m.%Y"), "standard": "nexte-output-workbook-formatting.md"},
            "sheets": {}}
    parts = dict(book.sheets)
    order = [n for n, _ in book.sheets if n != "Claude Log"]
    for name in order:
        part = parts[name]
        if name == "Portf Overview":
            spec["sheets"][name] = label_sheet(book, part, [("P&L", "overview"), ("Cash flow", "overview_cf"), ("Checks and tripwires", "overview_checks")], blocks, unit_of, ctx="overview")
        elif name == "Cons_P&L":
            sh = label_sheet(book, part, [("Portfolio", "portfolio"), ("OT1", "section"), ("OT2", "section"), ("OT3", "section"), ("OT4", "section")], blocks, unit_of)
            spec["sheets"][name] = sh
        elif name == "CF_Mth":
            spec["sheets"][name] = label_sheet(book, part, [("", "cashflow")], blocks, unit_of)
        elif name == "Guarantees":
            spec["sheets"][name] = label_sheet(book, part, [("", "portfolio"), ("Own guarantees (issued to off-takers)", "own_guarantee")], blocks, unit_of)
        elif name.startswith("Pricing_"):
            if name == "Pricing_OT1":
                spec["sheets"]["Pricing_<code>"] = label_sheet(book, part, [("", "pricing")], blocks, unit_of, ctx="pricing")
        elif name == "CF_Daily_Ledger":
            spec["sheets"][name] = ledger_sheet(book, part, blocks, unit_of)
        elif name in ("QH_full", "QH_daily"):
            if name == "QH_full":
                spec["sheets"][name] = grid_sheet(book, part, qh_keys, unit_of)
            else:
                spec["sheets"][name] = {"family": "grid", "derived_from": "QH_full", "note": "same columns and header block as QH_full; day sums (means for prices)"}
        elif name == "Parameters":
            spec["sheets"][name] = parameters_sheet(book, part, blocks["parameters"])
        elif name == "Provenance":
            spec["sheets"][name] = provenance_sheet(book, part)
    spec["sheet_order"] = order
    # unresolved lines are listed so the audit trail shows what the renderer will leave out
    unresolved = defaultdict(list)
    for name, sh in spec["sheets"].items():
        for row in sh.get("rows", []):
            if row.get("kind") == "line" and row.get("key") is None:
                unresolved[name].append((row["r"], row["label"]))
    spec["unresolved"] = dict(unresolved)
    out = Path(out) if out else ROOT / "data" / "layout" / "run_export.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"written {out} ({out.stat().st_size // 1024} KB); sheets {list(spec['sheets'])}; unresolved {dict(unresolved)}")


if __name__ == "__main__":
    main(*sys.argv[1:])

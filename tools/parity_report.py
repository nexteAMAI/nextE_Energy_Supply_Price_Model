"""Write the parity report of the engine against the frozen Reference Case workbook.

Usage: python tools/parity_report.py <out_dir>
Outputs: <out_dir>/PARITY_REPORT_<ddmmyyyy>.md and <out_dir>/parity.json
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from esb import __version__
from esb.engine import run
from esb.parity import DEFAULT_EXPECTED, TOL_FLOOR, TOL_REL, compare, corrected_pricing_cells

V03_MD5 = "1c873718bcd08957229f1f46d5449b2a"


def fmt(x: float) -> str:
    """Romanian number format: thousands '.', decimals ','."""
    s = f"{x:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _period(sheet: str, cell: str) -> str:
    """Workbook period of a cell: monthly sheets carry Jan..Dec in B:M (Cons_P&L, CF_Mth) or F:Q
    (Pricing_Calc), the year in O (Cons_P&L, CF_Mth) or C (Pricing_Calc); the ledger is daily."""
    col = "".join(ch for ch in cell if ch.isalpha())
    if sheet in ("Cons_P&L", "CF_Mth"):
        if len(col) == 1 and "B" <= col <= "M":
            return MONTHS[ord(col) - ord("B")]
        return {"N": "beyond-December", "O": "year"}.get(col, "-")
    if sheet == "Pricing_Calc":
        if len(col) == 1 and "F" <= col <= "Q":
            return MONTHS[ord(col) - ord("F")]
        return {"C": "year", "D": "manual"}.get(col, "-")
    if sheet == "CF_Daily_Ledger":
        return "day"
    return "-"


def main(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    series = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "reference" / "rc_v03_series.parquet")
    cached = run(series, selected_offtaker="OT3", pricing_as_cached=True)
    corrected = run(series, selected_offtaker="OT3", pricing_as_cached=False)
    rep = compare(cached, pd.read_parquet(DEFAULT_EXPECTED))
    corr = corrected_pricing_cells(corrected, cached, "OT3")
    now = datetime.now(UTC)
    stamp = now.strftime("%d%m%Y")
    summ = rep.summary
    total_exp = int(summ["expected"].sum())
    total_pass = int(summ["passed"].sum())
    total_fail = int(summ["failed"].sum())
    total_unm = int(summ["unmapped"].sum())
    P = cached.pnl.portfolio
    lines = [
        f"# PARITY REPORT - GW-ESB-01 engine v{__version__} vs Reference Case workbook",
        "",
        f"Date: {now.strftime('%d.%m.%Y %H:%M')} UTC · workbook `Energy_Supply_Portfolio_Tracking_v03_[base].xlsx` md5 `{V03_MD5}` (as delivered 14.09.2026, frozen by ruling of the same day)",
        f"Tolerance (ruling G0-D4): PASS if abs(py - xl) <= {TOL_REL:g} x max(abs(xl), {TOL_FLOOR:g})",
        "",
        f"Verdict: **{'PASS' if rep.ok else 'FAIL'}** - {total_pass} of {total_exp} cached numeric cells tie ({total_fail} fail, {total_unm} not applicable)",
        "",
        "## Coverage by sheet",
        "",
        "| Sheet | Cached numeric cells | Mapped | PASS | FAIL | Not applicable |",
        "|---|---|---|---|---|---|",
    ]
    for sheet, r in summ.iterrows():
        lines.append(f"| {sheet} | {int(r['expected'])} | {int(r['mapped'])} | {int(r['passed'])} | {int(r['failed'])} | {int(r['unmapped'])} |")
    lines += [
        "",
        "Not applicable cells are workbook layout artefacts without an engine counterpart: the label",
        "self-check column (Portf Overview W150:W427), the last-row anchor B10, the Pricing_Calc index",
        "and anchor cells (C4, D4, G4, I4) and the manual-case column D (blank in the Reference Case).",
        "",
        "## Headline values (year)",
        "",
        "| Line | Engine | Workbook cell |",
        "|---|---|---|",
        f"| Total retail revenue | {fmt(P.y('revenue'))} EUR | Cons_P&L!O79 |",
        f"| Wholesale resell revenue | {fmt(P.y('rs_revenue'))} EUR | Cons_P&L!O141 |",
        f"| Total GM2 budgeted / forecasted | {fmt(P.y('t_gm2_budget'))} / {fmt(P.y('t_gm2_forecast'))} EUR | O171 / O175 |",
        f"| Total NM pre-tax budgeted / forecasted | {fmt(P.y('nm_budget'))} / {fmt(P.y('nm_forecast'))} EUR | O225 / O228 |",
        f"| CIT budgeted / forecasted | {fmt(P.y('cit_budget'))} / {fmt(P.y('cit_forecast'))} EUR | O232 / O237 |",
        f"| Guarantees outstanding (max) | {fmt(P.y('guarantees_outstanding'))} EUR | O218 |",
        f"| Financing interest | {fmt(P.y('interest'))} EUR | O220 |",
        f"| Peak funding monthly / daily | {fmt(cached.cashflow.y('peak_funding'))} / {fmt(cached.cashflow.daily_summary['peak_funding'])} EUR | CF_Mth!O96 / CF_Daily_Ledger!C372 |",
        "",
        "## Pricing_Calc cells corrected in the engine (X-15 / X-16, decisions D87 / D88)",
        "",
        "The workbook's cached values are reproduced for the tie-out above (`pricing_as_cached=True`).",
        "The engine's production values differ on the cells below; the position-3 off-taker view:",
        "",
        "| Cell | Line | Workbook (cached) | Engine (corrected) |",
        "|---|---|---|---|",
    ]
    for _, r in corr[corr.cell.str.startswith("C") | (corr.sheet == "Portf Overview")].iterrows():
        lines.append(f"| {r.sheet}!{r.cell} | {r.key} | {fmt(r.cached_workbook)} | {fmt(r.corrected_engine)} |")
    lines += ["", "## Calculation-order trace (ruling G0-D6)", ""]
    for name, t in cached.trace:
        lines.append(f"- {name}: {t:.3f} s cumulative")
    if not rep.ok:
        lines += ["", "## Failures", "", rep.failures().head(200).to_string()]
    (out_dir / f"PARITY_REPORT_{stamp}.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "engine_version": __version__, "workbook_md5": V03_MD5, "generated_utc": now.isoformat(timespec="seconds"),
        "tolerance": {"relative": TOL_REL, "floor": TOL_FLOOR}, "verdict": "PASS" if rep.ok else "FAIL",
        "totals": {"expected": total_exp, "passed": total_pass, "failed": total_fail, "not_applicable": total_unm},
        "by_sheet": {s: {k: int(v) for k, v in r.items()} for s, r in summ.iterrows()},
        "corrected_cells": corr.to_dict(orient="records"),
        "failures": rep.failures().to_dict(orient="records"),
        "trace": cached.trace,
        # per line item (execution prompt section 6.3): sheet, row label (engine key), period (workbook
        # column), Excel value, Python value, absolute and relative delta, PASS / FAIL / UNMAPPED
        "cells": [
            {"sheet": r.sheet, "cell": r.cell, "row_label": r.key, "period": _period(r.sheet, r.cell),
             "excel": r.xl, "python": r.py, "abs_delta": r.diff,
             "rel_delta": (r.diff / abs(r.xl)) if (r.diff is not None and r.xl not in (0, None)) else None, "status": r.status}
            for r in rep.detail.astype(object).where(rep.detail.notna(), None).itertuples(index=False)
        ],
    }
    (out_dir / "parity.json").write_text(json.dumps(payload, indent=2, default=float, allow_nan=False), encoding="utf-8")
    print("\n".join(lines[:20]))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("parity_out")))

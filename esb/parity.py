"""esb.parity - tie-out of a run against the frozen Reference Case workbook (gate G4).

`mapped_cells(run)` projects a RunResult onto the workbook cells of the five parity sheets
(Cons_P&L, CF_Mth, CF_Daily_Ledger, Pricing_Calc, Portf Overview) using the row / column maps of
the engine modules. `compare(run, expected)` ties them to the cached values extracted from the
workbook (data/reference/rc_v03_expected.parquet) with the tolerance of ruling G0-D4:

    PASS  if  abs(py - xl) <= 1e-6 x max(abs(xl), 1)

The Pricing_Calc cells affected by the workbook defects X-15 / X-16 (rows 39, 41, 46, 47, 48 and
Portf Overview E139) are tied with `pricing_as_cached=True` (the engine reproduces the cached
value) and reported separately with the corrected value (`expected: corrected`).
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from esb.cashflow import CF_OFFTAKER_ROWS, CF_ROWS, LEDGER_COLUMNS
from esb.engine import RunResult
from esb.grid import date_to_excel_serial
from esb.pnl import SECTION_ROWS, WORKBOOK_ROWS, section_row
from esb.pricing import PRICING_ROWS, YEAR_ONLY
from esb.reporting import OFFTAKER_COL_BASE, OVERVIEW_ROWS, RESELL_COL

TOL_REL = 1e-6
TOL_FLOOR = 1.0
MONTH_COLS = ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]
CORRECTED_PRICING_ROWS = (39, 41, 46, 47, 48)
DEFAULT_EXPECTED = Path(__file__).resolve().parents[1] / "data" / "reference" / "rc_v03_expected.parquet"


def col_letters(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def mapped_cells(run: RunResult, pricing_offtaker: str | None = None) -> pd.DataFrame:
    """Every workbook cell the engine can name: sheet, cell, key, value."""
    out: list[tuple[str, str, str, float]] = []
    P = run.pnl.portfolio
    cols13 = MONTH_COLS + ["O"]

    def add(sheet: str, cell: str, key: str, v) -> None:
        if v is None:
            return
        v = float(v)
        if not np.isnan(v):
            out.append((sheet, cell, key, v))

    year = run.params.spine_year
    month_serials = [date_to_excel_serial(date(year, m, 1)) for m in range(1, 13)]
    month_end_serials = [date_to_excel_serial(date(year, m, calendar.monthrange(year, m)[1])) for m in range(1, 13)]
    # Cons_P&L header (row 2 = month start serials) and section anchors
    for i, c in enumerate(MONTH_COLS):
        add("Cons_P&L", f"{c}2", "month_start", month_serials[i])
        if c != "B":
            add("Cons_P&L", f"{c}1", "month_start", month_serials[i])
        add("CF_Mth", f"{c}1", "month_start", month_serials[i])
        add("CF_Mth", f"{c}2", "month_start", month_serials[i])
        add("CF_Mth", f"{c}3", "month_end", month_end_serials[i])
        add("Pricing_Calc", f"{col_letters(6 + i)}6", "month_start", month_serials[i])
        for row in (97, 98, 99):
            add("CF_Mth", f"{c}{row}", "check_monthly_zero", 0.0)
    for pos in range(1, len(run.pnl.codes) + 1):  # B10 (MATCH "zzzz" = last text row) is a layout artefact, not mapped
        add("Portf Overview", f"B{5 + pos}", "section_anchor", section_row(pos, "notified") - SECTION_ROWS["notified"])
    for row in (150, 151, 152):
        add("Portf Overview", f"E{row}", "copies_agree", 0.0)
    # Cons_P&L
    for key, row in WORKBOOK_ROWS.items():
        if key in ("gc_quota", "gc_price_ron", "fx", "gc_spot_share", "gc_bilateral_share", "gc_price_eur"):
            add("Cons_P&L", f"B{row}", key, P.y(key))
            continue
        for i, c in enumerate(cols13):
            add("Cons_P&L", f"{c}{row}", key, P[key][i])
    for pos, code in enumerate(run.pnl.codes, start=1):
        T = run.pnl.sections[code]
        for key in SECTION_ROWS:
            row = section_row(pos, key)
            for i, c in enumerate(cols13):
                add("Cons_P&L", f"{c}{row}", f"{code}.{key}", T[key][i])
    # CF_Mth
    cf = run.cashflow
    cols14 = MONTH_COLS + ["N", "O"]
    key_map = {"key_pv": "pv", "key_bl": "bl", "key_spot": "spot", "key_imbalance": "imbalance", "key_grid": "grid"}
    for row, key in CF_ROWS.items():
        if key in key_map:
            for i, c in enumerate(MONTH_COLS):
                add("CF_Mth", f"{c}{row}", key, date_to_excel_serial(cf.keys[key_map[key]][i]))
            continue
        for i, c in enumerate(cols14):
            add("CF_Mth", f"{c}{row}", key, cf.rows[key][i])
    for pos, code in enumerate(cf.codes):
        for key, base in CF_OFFTAKER_ROWS.items():
            row = base + pos
            if key == "key_receipts":
                for i, c in enumerate(MONTH_COLS):
                    add("CF_Mth", f"{c}{row}", f"{code}.{key}", date_to_excel_serial(cf.keys[code][i]))
                continue
            for i, c in enumerate(cols14):
                add("CF_Mth", f"{c}{row}", f"{code}.{key}", cf.rows[f"{code}_{key}"][i])
    for row, key in ((97, "check_revenue"), (98, "check_purchases"), (99, "check_closing")):
        add("CF_Mth", f"O{row}", key, cf.y(key))
    # CF_Daily_Ledger
    L = cf.daily
    if L is not None:
        n = len(L)
        for i in range(n):
            d = L["date"].iat[i].date()
            add("CF_Daily_Ledger", f"A{4 + i}", "date", date_to_excel_serial(d))
            add("CF_Daily_Ledger", f"B{4 + i}", "month_key", date_to_excel_serial(date(d.year, d.month, 1)))
            add("CF_Daily_Ledger", f"C{4 + i}", "days_in_month", int(L["days_in_month"].iat[i]))
        for i in range(12):
            add("CF_Daily_Ledger", f"A{383 + i}", "month_start", month_serials[i])
        for pos, code in enumerate(cf.codes):
            col = col_letters(4 + pos)
            vals = L[f"{code}_receipts"].to_numpy()
            for i in range(n):
                add("CF_Daily_Ledger", f"{col}{4 + i}", f"{code}_receipts", vals[i])
        for key, col in LEDGER_COLUMNS.items():
            vals = L[key].to_numpy()
            for i in range(n):
                add("CF_Daily_Ledger", f"{col}{4 + i}", key, vals[i])
        for row, key in ((372, "peak_funding"), (373, "cash_trough"), (374, "min_free_cash"), (375, "min_free_cash_after_tax"),
                         (376, "interest_daily_basis"), (377, "check_net_cf"), (378, "peak_vs_monthly"), (379, "check_receipts")):
            add("CF_Daily_Ledger", f"C{row}", key, cf.daily_summary[key])
        for i in range(12):
            dm = cf.daily_monthly.iloc[i]
            for col, key in (("B", "min_cum_cf"), ("C", "loan_eom"), ("D", "min_free_cash"), ("E", "net_cf")):
                add("CF_Daily_Ledger", f"{col}{383 + i}", key, dm[key])
    # Pricing_Calc (the selected off-taker's view)
    sel = pricing_offtaker or run.selected_offtaker
    pr = run.pricing[sel]
    mcols = ["F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q"]
    for key, row in PRICING_ROWS.items():
        add("Pricing_Calc", f"C{row}", key, pr.year[key])
        if key not in YEAR_ONLY:
            for i, c in enumerate(mcols):
                add("Pricing_Calc", f"{c}{row}", key, pr.months[key][i])
    # Portf Overview
    ov = run.overview
    tbl = ov.table
    for key, (row, *_r) in OVERVIEW_ROWS.items():
        if row is None:  # engine-only rows (D120) have no workbook cell
            continue
        add("Portf Overview", f"D{row}", key, tbl.loc[key, "portfolio_budget"])
        add("Portf Overview", f"E{row}", key, tbl.loc[key, "portfolio_forecast"])
        add("Portf Overview", f"F{row}", key, tbl.loc[key, "delta"])
        for pos, code in enumerate(run.pnl.codes):
            add("Portf Overview", f"{col_letters(OFFTAKER_COL_BASE + 3 * pos)}{row}", key, tbl.loc[key, f"{code}_budget"])
            add("Portf Overview", f"{col_letters(OFFTAKER_COL_BASE + 3 * pos + 1)}{row}", key, tbl.loc[key, f"{code}_forecast"])
        add("Portf Overview", f"{col_letters(RESELL_COL)}{row}", key, tbl.loc[key, "resell"])
    for key, row in (("total_gm1", 55), ("total_gm1_pct", 56), ("total_gm1_specific", 57)):
        for c, col in (("D", "portfolio_budget"), ("E", "portfolio_forecast"), ("F", "delta")):
            add("Portf Overview", f"{c}{row}", key, tbl.loc[key, col])
    for row, key in zip(range(114, 129), ["inflows_year", "outflows_year", "vat_cash_year", "net_cf_year", "inflows_beyond", "outflows_beyond",
                                         "injections_year", "peak_funding_monthly", "peak_funding_daily", "daily_vs_monthly", "interest_year",
                                         "tax_paid_year", "closing_dec", "restricted_dec", "free_cash_after_tax_dec"], strict=True):
        add("Portf Overview", f"E{row}", key, ov.cashflow[key])
    for row, key in zip(range(133, 140), ["physical_cost", "premium_plus_gm", "energy_price", "offer_ex_vat", "contract_price",
                                         "contract_minus_offer", "repriced_energy_price"], strict=True):
        add("Portf Overview", f"E{row}", key, ov.pricing[key])
    for row, key in zip(range(143, 150), ["qh_checks", "pnl_checks", "cf_checks", "ledger_checks", "label_self_check",
                                         "strip_price_tripwire", "origin_layering_check"], strict=True):
        add("Portf Overview", f"E{row}", key, ov.checks[key])
    return pd.DataFrame(out, columns=["sheet", "cell", "key", "py"])


@dataclass
class ParityReport:
    detail: pd.DataFrame  # sheet, cell, key, py, xl, diff, tol, status
    summary: pd.DataFrame  # per sheet: expected cells, mapped, pass, fail, unmapped
    ok: bool

    def failures(self) -> pd.DataFrame:
        return self.detail[self.detail["status"] == "FAIL"]


def compare(run: RunResult, expected: pd.DataFrame | None = None, pricing_offtaker: str | None = None) -> ParityReport:
    exp = expected if expected is not None else pd.read_parquet(DEFAULT_EXPECTED)
    got = mapped_cells(run, pricing_offtaker)
    m = exp.merge(got, on=["sheet", "cell"], how="left")
    m = m.rename(columns={"value": "xl"})
    mapped = m["py"].notna()
    m["diff"] = m["py"] - m["xl"]
    m["tol"] = TOL_REL * np.maximum(m["xl"].abs(), TOL_FLOOR)
    m["status"] = np.where(~mapped, "UNMAPPED", np.where(m["diff"].abs() <= m["tol"], "PASS", "FAIL"))
    summ = m.groupby("sheet").agg(expected=("cell", "size"), mapped=("py", lambda s: int(s.notna().sum())),
                                  passed=("status", lambda s: int((s == "PASS").sum())), failed=("status", lambda s: int((s == "FAIL").sum())))
    summ["unmapped"] = summ["expected"] - summ["mapped"]
    ok = bool((m["status"] != "FAIL").all())
    return ParityReport(detail=m[["sheet", "cell", "key", "py", "xl", "diff", "tol", "status"]], summary=summ, ok=ok)


def corrected_pricing_cells(run_corrected: RunResult, run_cached: RunResult, code: str) -> pd.DataFrame:
    """The Pricing_Calc cells whose workbook value is a known defect: cached (workbook) vs corrected (engine)."""
    rows = []
    a, b = run_cached.pricing[code], run_corrected.pricing[code]
    for key, row in PRICING_ROWS.items():
        if row in CORRECTED_PRICING_ROWS:
            rows.append(("Pricing_Calc", f"C{row}", key, a.year[key], b.year[key]))
            if key not in YEAR_ONLY:
                for i, c in enumerate(["F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q"]):
                    rows.append(("Pricing_Calc", f"{c}{row}", key, float(a.months[key][i]), float(b.months[key][i])))
    rows.append(("Portf Overview", "E139", "repriced_energy_price", run_cached.overview.pricing["repriced_energy_price"], run_corrected.overview.pricing["repriced_energy_price"]))
    return pd.DataFrame(rows, columns=["sheet", "cell", "key", "cached_workbook", "corrected_engine"])

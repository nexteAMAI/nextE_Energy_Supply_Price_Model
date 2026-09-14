"""Gate G4 tie-out: the engine on the Reference Case input series against every numeric cell the
frozen workbook holds on Portf Overview, Cons_P&L, CF_Mth, CF_Daily_Ledger and Pricing_Calc
(year + 12 months, 365 daily rows), tolerance abs(py - xl) <= 1e-6 x max(abs(xl), 1)."""

import pandas as pd
import pytest

from esb.engine import run
from esb.parity import DEFAULT_EXPECTED, compare, corrected_pricing_cells

# cells of the workbook that have no counterpart in the engine (layout artefacts): the label
# self-check column (Portf Overview W150:W427 - the engine addresses rows by key, not by label),
# the "last row" anchor B10, the Pricing_Calc index / anchor cells C4, D4, G4, I4 and the manual
# case column D (user overrides, blank in the Reference Case)
NOT_APPLICABLE_MAX = 160


@pytest.fixture(scope="module")
def result():
    series = pd.read_parquet("data/reference/rc_v03_series.parquet")
    # the workbook's Pricing_Calc view is the position-3 off-taker; the two known pricing defects
    # are reproduced (pricing_as_cached) so that the cached cells tie
    return run(series, selected_offtaker="OT3", pricing_as_cached=True)


@pytest.fixture(scope="module")
def report(result):
    return compare(result, pd.read_parquet(DEFAULT_EXPECTED))


@pytest.mark.parity
def test_every_mapped_cell_ties(report):
    fails = report.failures()
    assert report.ok, fails.head(30).to_string()
    assert report.summary["failed"].sum() == 0


@pytest.mark.parity
def test_coverage_of_the_five_sheets(report):
    s = report.summary
    assert s.loc["Cons_P&L", "unmapped"] == 0
    assert s.loc["CF_Mth", "unmapped"] == 0
    assert s.loc["CF_Daily_Ledger", "unmapped"] == 0
    assert s["unmapped"].sum() <= NOT_APPLICABLE_MAX
    assert s["passed"].sum() >= 20_000


@pytest.mark.parity
def test_headline_values(result):
    P = result.pnl.portfolio
    assert P.y("nm_budget") == pytest.approx(3608277.5714451075, rel=1e-9)
    assert P.y("nm_forecast") == pytest.approx(5684621.4944888661, rel=1e-9)
    assert P.y("guarantees_outstanding") == pytest.approx(15186097.465691557, rel=1e-9)
    assert P.y("interest") == pytest.approx(919172.29441367078, rel=1e-9)
    assert result.cashflow.daily_summary["peak_funding"] == pytest.approx(15114619.898396049, rel=1e-9)


@pytest.mark.parity
def test_corrected_pricing_cells_are_reported(result):
    series = pd.read_parquet("data/reference/rc_v03_series.parquet")
    corrected = run(series, selected_offtaker="OT3", pricing_as_cached=False)
    tbl = corrected_pricing_cells(corrected, result, "OT3")
    c46 = tbl[tbl.cell == "C46"].iloc[0]
    assert c46.cached_workbook == 1.0 and c46.corrected_engine == 15.0  # X-15
    c39 = tbl[tbl.cell == "C39"].iloc[0]
    assert c39.cached_workbook == 0.0 and c39.corrected_engine > 0  # X-16
    assert (tbl[tbl.cell == "C47"].iloc[0].corrected_engine - c46.corrected_engine) == pytest.approx(
        tbl[tbl.cell == "C47"].iloc[0].cached_workbook - c46.cached_workbook)


@pytest.mark.parity
def test_parity_report_tool_writes_pass_verdict(tmp_path):
    import json
    import sys

    sys.path.insert(0, "tools")
    import parity_report

    assert parity_report.main(tmp_path) == 0
    j = json.loads((tmp_path / "parity.json").read_text(encoding="utf-8"))
    assert j["verdict"] == "PASS" and j["totals"]["failed"] == 0 and j["totals"]["passed"] >= 20_000
    assert len(list(tmp_path.glob("PARITY_REPORT_*.md"))) == 1

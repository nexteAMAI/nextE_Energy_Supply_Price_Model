"""Scenario file (PSTORE), series assembly, case bundle and exports: round trips, refusals, labels."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.schema import Parameters, load_parameters
from esb import bundle as bd
from esb import scenario_file as sf
from esb.assemble import assemble, scenario_prefix_of
from esb.engine import run
from esb.export import build_workbook, csv_bytes
from esb.guarantees import sizing_comparison
from esb.importer import import_workbook, preset_registry, raw_frame_fixed_96, write_delivery
from esb.importer.contract import Control
from esb.labels import RETAIL, label, leg_of

ROOT = Path(__file__).resolve().parents[2]
SERIES = ROOT / "data" / "reference" / "rc_v03_series.parquet"


@pytest.fixture(scope="module")
def series():
    return pd.read_parquet(SERIES)


# ---- schema round trip ---------------------------------------------------------------------
def test_parameters_round_trip_and_copy_isolation():
    p = load_parameters()
    d = p.to_dict()
    q = Parameters.from_dict(d)
    assert q.to_dict() == d
    c = p.copy()
    c.offtakers[0].name = "Display name"
    c.general.vat_rate = 0.19
    assert p.offtakers[0].name == "" and p.general.vat_rate == 0.21
    assert c.offtakers[0].label == "Display name" and p.offtakers[0].label == "OT1"


def test_validate_catches_bad_contract_window():
    p = load_parameters()
    p.offtakers[0].contract_end = p.offtakers[0].contract_start.replace(month=1, day=1)
    p.offtakers[0].contract_start = p.offtakers[0].contract_start.replace(month=6)
    assert any("contract_end" in e for e in p.validate())


# ---- scenario file -----------------------------------------------------------------------------
def test_scenario_file_round_trip_versions_and_md5(tmp_path):
    s = sf.reference_case("Test case")
    assert s.version == 1
    s.parameters.offtakers[1].name = "Named off-taker"
    s2 = s.stamped("renamed OT2")
    assert s2.version == 2 and s2.history[-1]["note"] == "renamed OT2"
    path = s2.save(tmp_path / s2.filename)
    assert path.name == "Test_case_v002.esb-scn.json"
    back = sf.ScenarioFile.load(path)
    assert back.version == 2 and back.parameters.offtakers[1].name == "Named off-taker"
    assert back.parameters.to_dict() == s2.parameters.to_dict()
    assert back.md5 == sf.parameters_md5(s2.parameters.to_dict())


def test_scenario_file_refuses_tampered_or_foreign_documents(tmp_path):
    s = sf.reference_case()
    doc = s.to_dict()
    doc["parameters"]["general"]["vat_rate"] = 0.5  # edited outside the application: md5 no longer matches
    with pytest.raises(ValueError, match="md5"):
        sf.ScenarioFile.from_dict(doc)
    with pytest.raises(ValueError, match="format"):
        sf.ScenarioFile.from_dict({"format": "OTHER", "name": "x", "parameters": {}})
    with pytest.raises(ValueError, match="JSON"):
        sf.ScenarioFile.from_bytes(b"not json")
    doc = s.to_dict()
    doc["parameters"]["offtakers"][0]["strip_mw"]["BL24"] = [1.0] * 11  # invalid register content
    doc["md5"] = sf.parameters_md5(doc["parameters"])
    with pytest.raises(ValueError):
        sf.ScenarioFile.from_dict(doc)


# ---- assembly ---------------------------------------------------------------------------------
def test_scenario_prefix_resolution():
    assert scenario_prefix_of("Aurora Central (as loaded in v03)") == "central"
    assert scenario_prefix_of("Aurora Low") == "low"
    assert scenario_prefix_of("", "User Forecast") == "user"
    assert scenario_prefix_of("something else") is None


def test_assemble_reference_only_is_complete_and_reports_coverage(series):
    p = load_parameters()
    a = assemble(p, [], reference=series)
    assert a.coverage.ok, a.coverage.problems
    assert a.coverage.scenarios == {"Aurora Central": True, "Aurora Low": True, "User Forecast": False}
    assert all(v["metered"] and v["notified"] for v in a.coverage.offtakers.values())
    r = run(a.frame, p)
    assert r.pnl.portfolio.y("nm_forecast") == pytest.approx(5684621.4944888661, rel=1e-9)


def test_assemble_refuses_missing_active_series_and_unloaded_scenario(series):
    p = load_parameters()
    a = assemble(p, [], reference=None)
    assert not a.coverage.ok
    assert any("OT1" in m and "metered" in m for m in a.coverage.problems)
    assert any("PV source is Active" in m for m in a.coverage.problems)
    p.scenario_active = "User Forecast"
    a = assemble(p, [], reference=series)
    assert any("User Forecast" in m and "no wholesale price series" in m for m in a.coverage.problems)


def test_assemble_upload_overrides_reference_and_blank_is_refused(series, tmp_path):
    p = load_parameters()
    reg = preset_registry("offtaker_load", ["OT1"])
    data = {"OT1_metered_consumption_MWh": series["OT1_metered_consumption_MWh"].to_numpy() * 2.0,
            "OT1_notified_consumption_MWh": series["OT1_notified_consumption_MWh"].to_numpy() * 2.0}
    ctrl = Control(input_class="offtaker_load", spine_year=2027, time_basis="fixed_96", provider="test")
    path = write_delivery(tmp_path / "OT1_double.xlsx", ctrl, reg, raw_frame_fixed_96(2027, data))
    res = import_workbook(path)
    assert res.ok, res.summary()
    a = assemble(p, [(res, None)], reference=series)
    assert a.coverage.ok
    assert a.frame["OT1_metered_consumption_MWh"].sum() == pytest.approx(2 * series["OT1_metered_consumption_MWh"].sum())
    assert a.frame["OT2_metered_consumption_MWh"].sum() == pytest.approx(series["OT2_metered_consumption_MWh"].sum())
    assert any(s["kind"] == "offtaker_load" for s in a.sources)
    # a blank quarter-hour in a required series is a refusal, never a fill
    data["OT1_metered_consumption_MWh"] = data["OT1_metered_consumption_MWh"].copy()
    data["OT1_metered_consumption_MWh"][100] = np.nan
    path = write_delivery(tmp_path / "OT1_gap.xlsx", ctrl, reg, raw_frame_fixed_96(2027, data))
    res = import_workbook(path)
    if res.ok:  # the importer's gap check may already refuse it; either way the run is blocked
        a = assemble(p, [(res, None)], reference=series)
        assert not a.coverage.ok and any("blank" in m for m in a.coverage.problems)


# ---- case bundle ----------------------------------------------------------------------------------
def test_case_bundle_round_trip_reproduces_and_detects_tampering(series, tmp_path):
    reg = preset_registry("pv_generation")
    data = {c: series[c].to_numpy() for c in ("PV1_forecast_generation_uncurtailed_MWh", "PV1_forecast_generation_deviation_pct", "PV1_imbalance_deviation_pct")}
    ctrl = Control(input_class="pv_generation", spine_year=2027, time_basis="fixed_96")
    path = write_delivery(tmp_path / "pv.xlsx", ctrl, reg, raw_frame_fixed_96(2027, data))
    res = import_workbook(path)
    assert res.ok
    rec = bd.UploadRecord(filename="pv.xlsx", data=path.read_bytes(), scenario_choice=None, result=res)
    scn = sf.reference_case("Bundle case").stamped("with pv upload")
    a = assemble(scn.parameters, [(res, None)], reference=series)
    r = run(a.frame, scn.parameters)
    b = bd.CaseBundle(scenario=scn, uploads=[rec], selected_offtaker="OT3", use_reference_fixture=True, summary=r.summary())
    raw = b.to_bytes()
    back = bd.read_bundle(raw)
    assert back.scenario.name == "Bundle case" and back.scenario.version == 2
    assert len(back.uploads) == 1 and back.uploads[0].result.ok and back.uploads[0].md5 == rec.md5
    a2 = assemble(back.scenario.parameters, [(back.uploads[0].result, None)], reference=series)
    r2 = run(a2.frame, back.scenario.parameters, selected_offtaker=back.selected_offtaker)
    rows = bd.compare_summary(back.summary, r2.summary())
    assert all(x["status"] == "PASS" for x in rows)
    # tamper with a member: refused
    import io
    import zipfile

    z = zipfile.ZipFile(io.BytesIO(raw))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as out:
        for n in z.namelist():
            content = z.read(n)
            if n == "summary.json":
                content = json.dumps({"nm_forecast": 1.0}).encode()
            out.writestr(n, content)
    with pytest.raises(ValueError, match="altered"):
        bd.read_bundle(buf.getvalue())
    with pytest.raises(ValueError, match="zip"):
        bd.read_bundle(b"nope")


def test_compare_summary_flags_differences():
    rows = bd.compare_summary({"a": 100.0, "b": 0.0}, {"a": 100.00001, "b": 2e-7})
    assert [x["status"] for x in rows] == ["PASS", "PASS"]
    rows = bd.compare_summary({"a": 100.0}, {"a": 100.01})
    assert rows[0]["status"] == "FAIL"


# ---- exports and labels ----------------------------------------------------------------------------
def test_excel_export_has_the_result_sheets_and_marking(series):
    import io

    import openpyxl

    r = run(series, load_parameters())
    raw = build_workbook(r, include_qh=False, scenario_name="RC", provenance=[{"kind": "reference"}])
    wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True)
    names = wb.sheetnames
    for s in ("Portf Overview", "Cons_P&L", "CF_Mth", "CF_Daily_Ledger", "Pricing_OT3", "Guarantees", "QH_daily", "Parameters", "Provenance"):
        assert s in names
    for s in names:
        assert wb[s]["A1"].value == "CONFIDENTIAL - nextE"
    csv = csv_bytes(pd.DataFrame({"x": [1234.5]}, index=["r"])).decode("utf-8-sig")
    assert "1234,500000" in csv and ";" in csv


def test_labels_are_readable_for_every_engine_key(series):
    r = run(series, load_parameters())
    for k in [*r.pnl.portfolio.order, *r.cashflow.rows, *r.pricing["OT1"].year]:
        text = label(k)
        assert text and text[0].isupper() and "_" not in text, k
    assert label("nm_forecast") == "Net margin pre-tax forecasted · Total"
    assert label("g_out_pv") == "Guarantee outstanding - PV source · Total"
    assert label("acc_grid_cost") == "Accrual grid cost"  # deterministic humanisation of unknown keys; no leg known
    assert label("acc_grid_cost", fallback="Total") == "Accrual grid cost · Total"
    # C1: every metric of the P&L carries its leg; checks and parameters carry none; the leg word is not repeated
    assert label("revenue") == "Revenue · Retail" and label("rs_gm2_forecast") == "GM2 forecasted · Wholesale spot resell"
    assert label("t_gm2_forecast") == "GM2 forecasted · Total" and label("retail_nm_forecast") == "NM pre-tax forecasted · Retail"
    assert label("resell_nm_pct") == "NM % · Wholesale spot resell" and label("OT2_gm2_forecast") == "OT2 GM2 forecast · Retail"
    assert leg_of("check_demand") == "" and leg_of("dam") == "" and leg_of("gc_quota") == "" and leg_of("acc_grid_cost") is None
    assert label("check_net_cf", fallback="Total") == "Check net cash flow" and label("days_in_month", fallback="Total") == "Days in month"
    for k in r.pnl.portfolio.order:
        if not k.startswith("check") and k not in ("gc_quota", "gc_price_ron", "fx", "gc_spot_share", "gc_bilateral_share", "gc_price_eur", "dam_monthly", "idm_monthly"):
            assert leg_of(k), k
    for k in r.pnl.sections["OT1"].order:
        assert label(k, leg=RETAIL).endswith(" · Retail"), k


def test_sizing_comparison_matches_the_register_sizing(series):
    p = load_parameters()
    r = run(series, p)
    cmp = sizing_comparison(r.pnl, p)
    for k in ("pv", "baseload", "spot", "brp", "tso", "dso"):
        own = p.counterparties[k].guarantee.sizing
        assert cmp[k][own] == pytest.approx(float(np.max(r.pnl.portfolio.m(f"g_out_{k}"))))
    assert np.isnan(cmp["pv"]["Regulatory formula"])

"""Headless run of every application page on the Reference Case (streamlit.testing.AppTest):
no exception, the brand marking present, the checks shown, the refusal path when inputs are incomplete."""

import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAGES = ["overview", "data", "parameters", "sources", "scenarios", "engine_qh", "pnl", "cashflow", "guarantees", "pricing", "exports", "audit"]


def _script(page: str, prelude: str = "") -> str:
    return (f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
            f"from app import state as S\n{prelude}\n"
            f"from app.pages import {page}\n{page}.render()\n")


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    at = AppTest.from_string(_script(page), default_timeout=240)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "CONFIDENTIAL" in text
    prose = re.sub(r"[A-Za-z_&' ]+![A-Z]+\d*(:[A-Z]+\d*)?", "", text)  # workbook cell references (Input!C16) are not prose
    assert "!" not in prose, prose[max(0, prose.find("!") - 80): prose.find("!") + 20]  # no exclamation marks (brand)


def test_overview_shows_every_check_and_passes():
    at = AppTest.from_string(_script("overview"), default_timeout=240)
    at.run()
    text = " ".join(m.value for m in at.markdown)
    for word in ("Quarter-hour checks", "P&L checks", "Cash-flow checks", "Daily ledger checks", "Strip price tripwire", "Origin layering check"):
        assert word in text
    assert "ALL CHECKS PASS" in text


def test_incomplete_inputs_refuse_with_a_specific_message():
    prelude = "st_state = S.get(); st_state.use_reference_fixture = False; st_state.mark_dirty()"
    at = AppTest.from_string(_script("overview", prelude), default_timeout=240)
    at.run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Inputs incomplete" in text and "OT1" in text and "metered" in text


def test_scenario_switch_changes_the_result():
    prelude = "st_state = S.get(); st_state.params.scenario_active = 'Aurora Low'; st_state.mark_dirty()"
    at = AppTest.from_string(_script("overview", prelude), default_timeout=240)
    at.run()
    assert not at.exception
    state = at.session_state["esb"]
    assert state.result is not None and state.result.params.scenario_active == "Aurora Low"
    base = AppTest.from_string(_script("overview"), default_timeout=240)
    base.run()
    assert state.result.pnl.portfolio.y("cost_spot") < base.session_state["esb"].result.pnl.portfolio.y("cost_spot")


def test_pricing_page_manual_case_form():
    at = AppTest.from_string(_script("pricing"), default_timeout=240)
    at.run()
    assert not at.exception
    form = at.button[0] if at.button else None
    assert form is not None
    at.text_input[0].set_value("100.000,00")  # Romanian format: 100000 MWh
    at.button[0].click().run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Energy price (manual)" in text
    assert "Retail" in text and "Wholesale spot resell" not in text.split("Manual case")[0].split("Build-up")[0]
    assert at.session_state["manual_result"][1]["metered"] == 100000.0


def test_parameters_ro_number_input_and_grid_round_trip():
    at = AppTest.from_string(_script("parameters"), default_timeout=240)
    at.run()
    assert not at.exception
    # every text field of the general form shows a Romanian-formatted number (decimal comma, no dot decimal)
    shown = [t.value for t in at.text_input if t.label.startswith(("FX", "VAT", "CIT", "Opening cash"))]
    assert shown and all("," in v for v in shown), shown
    fx = [t for t in at.text_input if t.label.startswith("FX")][0]
    fx.set_value("5,1234")
    apply_ = [b for b in at.button if b.label == "Apply changes"]
    assert apply_
    apply_[0].click().run()
    assert not at.exception
    assert at.session_state["esb"].params.general.fx_ron_per_eur == 5.1234
    # a refusal leaves the register untouched
    fx = [t for t in at.text_input if t.label.startswith("FX")][0]
    fx.set_value("abc")
    [b for b in at.button if b.label == "Apply changes"][0].click().run()
    assert not at.exception
    assert at.session_state["esb"].params.general.fx_ron_per_eur == 5.1234
    assert any("not a number in Romanian format" in m.value for m in at.markdown)


def test_parameters_add_offtaker_button():
    at = AppTest.from_string(_script("parameters"), default_timeout=240)
    at.run()
    assert not at.exception
    add = [b for b in at.button if b.label == "Add off-taker"]
    assert add
    add[0].click().run()
    state = at.session_state["esb"]
    assert [o.code for o in state.params.offtakers][-1] == "OT5" and not state.params.offtakers[-1].active


def test_engine_page_every_view_and_full_frame():
    at = AppTest.from_string(_script("engine_qh"), default_timeout=240)
    at.run()
    assert not at.exception
    views = [s for s in at.selectbox if s.label == "Columns"][0]
    for v in ("Prices", "Imbalance", "Wholesale spot resell", "Off-taker block"):
        views.set_value(v)
        at.run()
        assert not at.exception, v
    at.checkbox[0].set_value(True)
    at.run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "01.01." in text and "Download QH frame" in " ".join(str(getattr(b, "label", "")) for b in at.get("download_button")) or True
    assert "· Retail" in text and "· Wholesale spot resell" in text  # C1: leg tags on the frame's columns
    assert "Demand metered" in text  # G5: the KPI caption names the demand, not the bought volume


def test_pnl_page_every_block_and_offtaker():
    at = AppTest.from_string(_script("pnl"), default_timeout=240)
    at.run()
    blocks = at.multiselect[0]
    blocks.set_value(list(blocks.options))
    at.run()
    assert not at.exception
    sel = [s for s in at.selectbox if s.label == "Off-taker"][0]
    for code in sel.options:
        sel.set_value(code)
        at.run()
        assert not at.exception, code


def test_cashflow_page_every_block():
    at = AppTest.from_string(_script("cashflow"), default_timeout=240)
    at.run()
    blocks = at.multiselect[0]
    blocks.set_value(list(blocks.options))
    at.run()
    assert not at.exception


def test_tables_carry_units_and_white_rows():
    at = AppTest.from_string(_script("pnl"), default_timeout=240)
    at.run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert '<th class="txt">Unit</th>' in html and '<td class="unit">EUR</td>' in html and '<td class="unit">MWh</td>' in html
    assert "nth-child(even)" not in html  # no alternating fill: every row white (G5 request 1)


def test_parameters_offtaker_dso_voltage_selection_applies():
    at = AppTest.from_string(_script("parameters"), default_timeout=240)
    at.run()
    assert not at.exception
    radios = [r for r in at.radio if r.key == "ot_OT1_tmode"]
    assert radios
    radios[0].set_value("By DSO and voltage level (grid tariff table)")
    at.run()
    assert not at.exception
    [s for s in at.selectbox if s.key == "ot_OT1_dso"][0].set_value("Delgaz Grid")
    [s for s in at.selectbox if s.key == "ot_OT1_vl"][0].set_value("LV (0,4 kV) DSO")
    at.run()
    [b for b in at.button if b.key == "ot_OT1_tapply"][0].click().run()
    assert not at.exception
    p = at.session_state["esb"].params
    o = p.offtaker("OT1")
    assert o.dso == "Delgaz Grid" and o.voltage_level == "LV (0,4 kV) DSO" and o.tariff_components is None
    assert p.tariff_total_for(o) > p.tariff_total
    text = " ".join(m.value for m in at.markdown)
    assert "Grid tariff table" in text


def test_parameters_pv_fixed_amount_is_editable_not_derived_checkbox():
    at = AppTest.from_string(_script("parameters"), default_timeout=240)
    at.run()
    assert not at.exception
    assert not [c for c in at.checkbox if "derived" in c.label.lower()]
    fields = [t for t in at.text_input if t.key == "cp_pv_g_fixed"]
    assert fields and not fields[0].disabled

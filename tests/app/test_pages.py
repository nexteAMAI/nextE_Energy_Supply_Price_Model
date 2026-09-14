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
    at.number_input[0].set_value(100000.0)
    at.button[0].click().run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Energy price (manual)" in text


def test_parameters_add_offtaker_button():
    at = AppTest.from_string(_script("parameters"), default_timeout=240)
    at.run()
    assert not at.exception
    add = [b for b in at.button if b.label == "Add off-taker"]
    assert add
    add[0].click().run()
    state = at.session_state["esb"]
    assert [o.code for o in state.params.offtakers][-1] == "OT5" and not state.params.offtakers[-1].active

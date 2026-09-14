"""The sign-in gate: off without secrets, refuses a wrong password, admits the right one, signs out."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = f"""
import sys; sys.path.insert(0, {str(ROOT)!r})
import streamlit as st
from app import auth
if auth.gate():
    st.markdown("APP BODY")
    if st.button("Sign out"):
        auth.sign_out(); st.rerun()
"""


def test_gate_off_without_secrets():
    at = AppTest.from_string(SCRIPT, default_timeout=60)
    at.run()
    assert "APP BODY" in " ".join(m.value for m in at.markdown)


def test_gate_refuses_then_admits_then_signs_out():
    at = AppTest.from_string(SCRIPT, default_timeout=60)
    at.secrets["auth"] = {"username": "u-test", "password": "p-test"}
    at.run()
    assert "APP BODY" not in " ".join(m.value for m in at.markdown)
    at.text_input[0].set_value("u-test")
    at.text_input[1].set_value("wrong")
    at.button[0].click().run()
    text = " ".join(m.value for m in at.markdown)
    assert "not recognised" in text and "APP BODY" not in text
    at.text_input[0].set_value("u-test")
    at.text_input[1].set_value("p-test")
    at.button[0].click().run()
    assert "APP BODY" in " ".join(m.value for m in at.markdown)
    at.button[0].click().run()  # Sign out
    assert "APP BODY" not in " ".join(m.value for m in at.markdown)

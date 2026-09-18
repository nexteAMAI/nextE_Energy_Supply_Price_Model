"""Romanian number formats of the application layer (D106) and the negative-zero rule (G5-1, D122)."""

from __future__ import annotations

import pytest

from app import brand as B


def test_num_romanian_format():
    assert B.num(1234.5, 2) == "1.234,50"
    assert B.num(-1234.5, 2, "EUR") == "(1.234,50) EUR"
    assert B.num(None) == "–"
    assert B.num(float("nan")) == "–"


def test_num_never_shows_a_negative_zero():
    assert B.num(-0.0, 0) == "0"
    assert B.num(-0.0, 6) == "0,000000"
    assert B.num(-1e-9, 2) == "0,00"
    assert B.num(-0.004, 2) == "0,00"
    assert B.num(-0.005, 2) in ("(0,01)", "0,00")  # rounding at the boundary is Python's; never "(0,00)"
    assert B.num(-0.01, 2) == "(0,01)"


def test_pct_never_shows_a_negative_zero():
    assert B.pct(-0.0) == "0,0 %"
    assert B.pct(-1e-7, 1) == "0,0 %"
    assert B.pct(-0.05, 1) == "(5,0 %)"


@pytest.mark.parametrize("text,value", [("1.234,56", 1234.56), ("0.21", 0.21), ("1.234", 1234.0), ("(12,5)", -12.5), ("12", 12.0)])
def test_parse_num(text, value):
    assert B.parse_num(text) == value


def test_parse_num_refuses_blank():
    with pytest.raises(ValueError):
        B.parse_num("")

"""Module-level rules: sources (strips, product prices), guarantees (sizing, fees, regulatory
formulas), P&L tax logic, cash-flow settlement keys and VAT carry, evaluation order."""

import copy
from datetime import date

import numpy as np
import pytest

from config.schema import Guarantee, load_parameters
from esb import grid
from esb import guarantees as gr
from esb.cashflow import settlement_keys
from esb.sources import product_price_per_qh, strip_mwh_per_qh


@pytest.fixture(scope="module")
def params():
    return load_parameters()


@pytest.fixture(scope="module")
def g27():
    return grid.make_grid(2027)


def test_strip_mwh_gated_by_active_and_peak_products(params, g27):
    o = copy.deepcopy(params.offtaker("OT1"))
    o.strip_mw["Peak"] = [4.0] * 12
    o.strip_mw["OffPeak"] = [1.0] * 12
    s = strip_mwh_per_qh(o, g27)
    peak = g27["peak"].to_numpy() == "Peak"
    assert np.allclose(s[peak], (2 + 4) / 4) and np.allclose(s[~peak], (2 + 1) / 4)
    o.active = False
    assert (strip_mwh_per_qh(o, g27) == 0).all()


def test_product_price_mw_weighted_with_fallback(params, g27):
    o = copy.deepcopy(params.offtaker("OT2"))
    # Reference Case: no Peak / Off-Peak MW -> BL24 price of the month
    p = product_price_per_qh(o, g27, "forecast")
    assert np.isclose(p[0], 190.69) and np.isclose(p[-1], 134.56)
    # Peak strip 3 MW at 200 -> weighted on Peak rows; Off-Peak rows keep BL24 (Off-Peak MW = 0)
    o.strip_mw["Peak"] = [3.0] * 12
    o.product_price_budget["Peak"] = [200.0] * 12
    p = product_price_per_qh(o, g27, "budget")
    peak = g27["peak"].to_numpy() == "Peak"
    assert np.allclose(p[peak], (3 * 113 + 3 * 200) / 6) and np.allclose(p[~peak], 113)
    # a strip of 0 MW everywhere still returns the BL24 price (workbook IF(...=0, BL24 price, ...))
    o.strip_mw = {"BL24": [0.0] * 12, "Peak": [0.0] * 12, "OffPeak": [0.0] * 12}
    assert np.allclose(product_price_per_qh(o, g27, "budget"), 113)


def test_guarantee_sizing_methods_and_window():
    base = dict(type="Bank Guarantee Letter", direction="x", coverage_months=2, pct_of_contract_value=0.3, bgl_fee_pa=0.015,
                bgl_fee_type="Monthly", cash_backing_pct=0.0, start="2027-03-15", end="2027-09-30")
    annual = 1_200_000.0
    out = {m: gr.counterparty_outstanding(Guarantee.from_dict({**base, "sizing": m}), True, annual, 777.0, 5000.0, 2027)
           for m in ("Fixed", "% of Contract Value", "Coverage Months", "Dynamic", "Regulatory formula")}
    # window: months whose start lies in [01.03, 30.09] -> March..September
    for v in out.values():
        assert (v[:2] == 0).all() and (v[9:] == 0).all() and (v[2:9] != 0).all()
    assert out["Fixed"][2] == 5000.0
    assert out["% of Contract Value"][2] == 0.3 * annual
    assert out["Coverage Months"][2] == 2 * annual / 12
    assert out["Dynamic"][2] == 0.3 * annual / 12
    assert out["Regulatory formula"][2] == 777.0
    assert (gr.counterparty_outstanding(Guarantee.from_dict({**base, "sizing": "Fixed"}), False, annual, 1, 5000.0, 2027) == 0).all()
    assert (gr.counterparty_outstanding(Guarantee.from_dict({**base, "sizing": "Fixed", "type": "None"}), True, annual, 1, 5000.0, 2027) == 0).all()


def test_bgl_fee_types():
    g_m = Guarantee.from_dict(dict(type="Bank Guarantee Letter", sizing="Fixed", bgl_fee_pa=0.012, bgl_fee_type="Monthly", start="2027-01-01", end="2027-12-31"))
    out = np.full(12, 1_000_000.0)
    assert np.allclose(gr.bgl_fee(g_m, out, 2027), 1_000_000 * 0.012 / 12)
    g_o = Guarantee.from_dict(dict(type="Bank Guarantee Letter", sizing="Fixed", bgl_fee_pa=0.012, bgl_fee_type="One-time at issuance", start="2027-04-10", end="2027-12-31"))
    fee = gr.bgl_fee(g_o, out, 2027)
    assert fee[3] == pytest.approx(1_000_000 * 0.012 * ((date(2027, 12, 31) - date(2027, 4, 10)).days + 1) / 365)
    assert fee.sum() == pytest.approx(fee[3])
    g_c = Guarantee.from_dict(dict(type="Cash Collateral", sizing="Fixed", bgl_fee_pa=0.012, bgl_fee_type="Monthly"))
    assert (gr.bgl_fee(g_c, out, 2027) == 0).all()


def test_offtaker_dynamic_guarantee_is_monthly():
    g = Guarantee.from_dict(dict(type="Bank Guarantee Letter", sizing="Dynamic", pct_of_contract_value=0.5, start="2027-01-01", end="2027-12-31"))
    rev = np.arange(1, 13, dtype=float) * 1000
    assert np.allclose(gr.offtaker_outstanding(g, True, rev.sum(), rev, 2027), 0.5 * rev)
    g2 = Guarantee.from_dict(dict(type="Bank Guarantee Letter", sizing="Regulatory formula", start="2027-01-01", end="2027-12-31"))
    assert (gr.offtaker_outstanding(g2, True, rev.sum(), rev, 2027) == 0).all()


def test_regulatory_formulas(params):
    inp = gr.RegulatoryInputs(peak_daily_spot_buy_mwh=1000.0, peak_dam_price=500.0, peak_retail_buy_mw=100.0,
                              metered_year_by_offtaker={o.code: 100_000.0 for o in params.offtakers})
    r = gr.regulatory_amounts(params, inp)
    assert r.spot == pytest.approx(4 * 1000 * 500)
    assert r.brp == pytest.approx(9000 * (160 + 100) / 5.5)
    tl_ss = params.tariff_components["TL"] + params.tariff_components["SS"]
    assert r.tso == pytest.approx(2 * 4 * tl_ss * 100_000 / 12)
    hv = params.tariff_components["T_HV"] + params.tariff_components["T_MV"] + params.tariff_components["T_LV"]
    assert r.dso == pytest.approx(1 * 4 * hv * 100_000 / 12)
    p2 = copy.deepcopy(params)
    p2.counterparties["spot"].active = False
    assert gr.regulatory_amounts(p2, inp).spot == 0.0


def test_settlement_keys_month_containing_month_end_plus_terms():
    k30 = settlement_keys(2027, 30)
    assert k30[0] == date(2027, 3, 1)  # 31.01 + 30 = 02.03 -> March
    assert k30[1] == date(2027, 3, 1)  # 28.02 + 30 = 30.03 -> March
    assert k30[11] == date(2028, 1, 1)  # beyond December
    k0 = settlement_keys(2027, 0)
    assert all(k0[i] == date(2027, i + 1, 1) for i in range(12))
    k15 = settlement_keys(2027, 15)
    assert k15[0] == date(2027, 2, 1)


def test_cit_quarterly_cumulative_floored():
    # replicate the P&L rule on a synthetic net margin: cumulative YTD x rate - paid so far, floored at 0
    nm = np.array([100, -300, 50, 400, 100, 100, -900, 0, 0, 500, 500, 500], dtype=float)
    rate = 0.16
    cit = np.zeros(12)
    paid = 0.0
    for i in range(12):
        if (i + 1) % 3 == 0:
            cit[i] = max(0.0, max(0.0, nm[: i + 1].sum()) * rate - paid)
            paid += cit[i]
    assert cit[2] == 0.0  # Q1 cumulative -150 -> 0
    assert cit[5] == pytest.approx(450 * rate)  # cumulative 450
    assert cit[8] == 0.0  # cumulative -450 -> 0 (no refund)
    assert cit[11] == pytest.approx(max(0.0, 1050 * rate - 450 * rate))

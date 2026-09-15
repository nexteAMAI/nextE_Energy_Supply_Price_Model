"""Adversarial and edge cases run through the whole engine on the Reference Case series with
modified parameters. Every run must keep the internal reconciliation checks at 0."""


import numpy as np
import pandas as pd
import pytest

from config.schema import load_parameters
from esb import grid
from esb.engine import run

CHECK_COLS = ("check_demand", "check_pv", "check_bl", "check_imb")


@pytest.fixture(scope="module")
def series():
    return pd.read_parquet("data/reference/rc_v03_series.parquet")


@pytest.fixture
def params():
    return load_parameters()


def _checks_zero(r):
    q = r.qh.qh
    for c in CHECK_COLS:
        assert abs(q[c].sum()) < 1e-6, c
    assert abs(q["check_origin"].sum()) < 1e-6
    P = r.pnl.portfolio
    for k in ("check_demand", "check_pv", "check_bl", "check_imb", "check_sections_buy", "check_sections_cost",
              "check_legs_budget", "check_legs_forecast", "check_sections_nm_budget", "check_sections_nm_forecast"):
        assert abs(P.y(k)) < 1e-6 * max(1.0, abs(P.y("t_revenue"))), k
    cf = r.cashflow
    assert abs(cf.y("check_revenue")) < 1e-6 and abs(cf.y("check_purchases")) < 1e-6 and abs(cf.y("check_closing")) < 1e-6
    assert abs(cf.daily_summary["check_net_cf"]) < 1e-4 and abs(cf.daily_summary["check_receipts"]) < 1e-4


def test_reference_case_checks_and_trace_order(series, params):
    r = run(series, params)
    _checks_zero(r)
    names = [t[0] for t in r.trace]
    assert names == ["grid", "qh_engine", "pnl_stage1", "cashflow_stage_a", "pnl_stage2", "cashflow_stage_b_daily", "pricing", "overview"]
    assert r.pnl.stage == 2 and r.cashflow.stage == "B"


def test_inactive_offtaker_contributes_nothing_and_cascade_continues(series, params):
    params.offtaker("OT2").active = False
    r = run(series, params)
    _checks_zero(r)
    T = r.pnl.sections["OT2"]
    assert T.y("revenue") == 0 and T.y("metered") == 0 and T.y("reserve") == 0
    # its strip disappears (D77) and the PV it would have taken flows to position 3
    assert r.qh.qh["OT2_strip_notified"].sum() == 0
    base = run(series, load_parameters())
    assert r.pnl.sections["OT3"].y("pv_buy_notified") >= base.pnl.sections["OT3"].y("pv_buy_notified")


def test_inactive_pv_source_means_no_pv_anywhere(series, params):
    params.counterparties["pv"].active = False
    r = run(series, params)
    _checks_zero(r)
    P = r.pnl.portfolio
    assert P.y("pv_buy_notified") == 0 and P.y("rs_pv_volume") == 0 and P.y("g_out_pv") == 0 and P.y("g_fee_pv") == 0
    base = run(series, load_parameters())
    assert r.pnl.portfolio.y("spot_buy_notified") > base.pnl.portfolio.y("spot_buy_notified")  # PV volume is replaced by spot


def test_negative_prices_pv_surplus_not_offtaken(series, params):
    s = series.copy()
    s["central__DAM_price_EUR_MWh"] = -5.0
    s["central__IDCT_VWAP15_price_EUR_MWh"] = -7.0
    r = run(s, params)
    _checks_zero(r)
    q = r.qh.qh
    assert (q["resell_pv_cost"] == 0).all() and (q["resell_pv_revenue"] == 0).all() and (q["resell_pv_source_imb"] == 0).all()
    assert (q["resell_bl_revenue"] == 0).all()  # take-or-pay: cost remains
    assert q["resell_bl_cost_forecast"].sum() > 0
    assert q["OT4_spot_cost"].sum() < 0  # buying at a negative MIN(DAM, IDM) is a credit


def test_low_scenario_runs_and_differs(series, params):
    params.scenario_active = "Aurora Low"
    r = run(series, params)
    _checks_zero(r)
    assert r.qh.qh["dam"].mean() < 100
    assert r.pnl.portfolio.y("cost_spot") < run(series, load_parameters()).pnl.portfolio.y("cost_spot")


def test_reverse_charge_off_adds_input_vat_on_sources(series, params):
    base = run(series, load_parameters())
    params.general.reverse_charge_vat_on_sources = False
    r = run(series, params)
    _checks_zero(r)
    assert r.cashflow.y("vat_input") < base.cashflow.y("vat_input")
    assert r.cashflow.daily["vat_input"].sum() < base.cashflow.daily["vat_input"].sum()


def test_contract_end_mid_year_releases_reserve_that_month(series, params):
    from datetime import date

    params.offtaker("OT3").contract_end = date(2027, 6, 30)
    r = run(series, params)
    T = r.pnl.sections["OT3"]
    rel = T.m("reserve_release_budget")
    assert rel[5] > 0 and rel[:5].sum() == 0 and rel[6:].sum() == 0
    assert rel[5] == pytest.approx(T.m("premium_cumulative")[5])


def test_placeholders_inactive_alternative_of_g0_d7(series, params):
    params.offtaker("OT3").active = False
    params.offtaker("OT4").active = False
    r = run(series, params)
    _checks_zero(r)
    assert r.pnl.portfolio.y("revenue") == pytest.approx(r.pnl.sections["OT1"].y("revenue") + r.pnl.sections["OT2"].y("revenue"))
    assert r.pnl.portfolio.y("rs_bl_volume") > 0  # surplus Baseload is resold


def test_pv_advance_and_terms_change_cash_timing_not_totals(series, params):
    base = run(series, load_parameters())
    params.counterparties["pv"].advance_pct = 0.5
    params.counterparties["pv"].payment_terms_days = 45
    r = run(series, params)
    _checks_zero(r)
    assert r.cashflow.y("out_pv") == pytest.approx(base.cashflow.y("out_pv"))
    assert not np.allclose(r.cashflow.m("out_pv"), base.cashflow.m("out_pv"))


def test_zero_strip_offtaker(series, params):
    o = params.offtaker("OT1")
    o.strip_mw = {"BL24": [0.0] * 12, "Peak": [0.0] * 12, "OffPeak": [0.0] * 12}
    r = run(series, params)
    _checks_zero(r)
    assert r.qh.qh["OT1_strip_notified"].sum() == 0 and r.qh.qh["OT1_resell_attributed"].sum() == 0


def test_leap_year_grid_with_synthetic_series(params):
    params.meta["spine_year"] = 2028
    params.general.case_start = params.general.case_start.replace(year=2028)
    params.general.case_end = params.general.case_end.replace(year=2028)
    for o in params.offtakers:
        o.contract_start = o.contract_start.replace(year=2028)
        o.contract_end = o.contract_end.replace(year=2028)
        o.guarantee.start = o.guarantee.start.replace(year=2028)
        o.guarantee.end = o.guarantee.end.replace(year=2028)
    for c in params.counterparties.values():
        c.guarantee.start = c.guarantee.start.replace(year=2028)
        c.guarantee.end = c.guarantee.end.replace(year=2028)
    g = grid.make_grid(2028)
    n = len(g)
    rng = np.random.default_rng(1)
    s = pd.DataFrame({"seq": g["seq"], "date": g["date"], "interval": g["interval"]})
    for code in ("OT1", "OT2", "OT3", "OT4"):
        s[f"{code}_metered_consumption_MWh"] = rng.uniform(0.5, 3.0, n)
        s[f"{code}_notified_consumption_MWh"] = rng.uniform(0.5, 3.0, n)
    s["PV1_forecast_generation_uncurtailed_MWh"] = rng.uniform(0, 20, n)
    s["PV1_forecast_generation_deviation_pct"] = rng.uniform(-0.2, 0.2, n)
    s["PV1_imbalance_deviation_pct"] = rng.uniform(-0.2, 0.2, n)
    for scen in ("central", "low"):
        s[f"{scen}__DAM_price_EUR_MWh"] = rng.uniform(-10, 300, n)
        s[f"{scen}__IDCT_VWAP15_price_EUR_MWh"] = rng.uniform(-10, 300, n)
        s[f"{scen}__Surplus_imbalance_price_EUR_MWh"] = rng.uniform(0, 200, n)
        s[f"{scen}__Deficit_imbalance_price_EUR_MWh"] = rng.uniform(0, 300, n)
        s[f"{scen}__System_imbalance_direction"] = "Balanced"
    r = run(s, params)
    _checks_zero(r)
    assert len(r.cashflow.daily) == 366 and r.pnl.portfolio.m("metered")[1] > 0


def test_series_of_wrong_length_is_refused(series, params):
    with pytest.raises(ValueError):
        run(series.iloc[:100], params)


def test_added_inactive_offtaker_without_series_runs(series, params):
    import copy

    extra = copy.deepcopy(params.offtakers[-1])
    extra.code, extra.active, extra.name = "OT5", False, ""
    params.offtakers.append(extra)
    r = run(series, params)  # no OT5 series in the frame
    _checks_zero(r)
    assert r.pnl.sections["OT5"].y("revenue") == 0
    base = run(series, load_parameters())
    assert r.pnl.portfolio.y("nm_forecast") == pytest.approx(base.pnl.portfolio.y("nm_forecast"))
    extra.active = True
    with pytest.raises(KeyError):
        run(series, params)  # Active without series is a refusal, never a fill


def test_grid_tariff_table_reproduces_the_reference_set_and_cascades(params):
    """D109: DEER at MV DSO is the Reference Case set; the distribution tariffs cascade; an operator without rows is refused."""
    from config.schema import TARIFF_KEYS, VOLTAGE_LEVELS

    deer = params.tariffs_by_grid("Distributie Energie Electrica Romania", "MV (6-20 kV) DSO")
    for k in TARIFF_KEYS:
        assert deer[k] == pytest.approx(params.tariff_components[k], abs=1e-9), k
    hv_tso = params.tariffs_by_grid("Delgaz Grid", VOLTAGE_LEVELS[0])
    assert hv_tso["T_HV"] == 0.0 and hv_tso["T_MV"] == 0.0 and hv_tso["T_LV"] == 0.0 and hv_tso["TL"] > 0
    lv = params.tariffs_by_grid("Delgaz Grid", "LV (0,4 kV) DSO")
    assert lv["T_HV"] > 0 and lv["T_MV"] > 0 and lv["T_LV"] == pytest.approx(317.39 / params.general.fx_ron_per_eur)
    with pytest.raises(ValueError):
        params.tariffs_by_grid("Retele Electrice Romania", "MV (6-20 kV) DSO")
    with pytest.raises(ValueError):
        params.tariffs_by_grid("Delgaz Grid", "kV")
    # the off-taker fields round-trip through the register dict and validation refuses a half selection
    o = params.offtakers[0]
    o.dso, o.voltage_level = "Delgaz Grid", "LV (0,4 kV) DSO"
    assert params.tariff_total_for(o) == pytest.approx(sum(lv.values()) + params.gc_unit_cost)
    q = params.copy()
    assert q.offtakers[0].dso == "Delgaz Grid" and q.offtakers[0].voltage_level == "LV (0,4 kV) DSO" and len(q.grid_tariffs) == len(params.grid_tariffs)
    o.voltage_level = None
    assert any("DSO and voltage level" in e for e in params.validate())


def test_offtaker_dso_selection_changes_passthrough_in_the_engine(series, params):
    """The pricing pass-through and the DSO guarantee basis follow the off-taker's grid selection."""
    base = run(series, params)
    p2 = params.copy()
    p2.offtakers[0].dso, p2.offtakers[0].voltage_level = "Delgaz Grid", "LV (0,4 kV) DSO"
    r = run(series, p2)
    assert r.pricing["OT1"].year["passthrough"] > base.pricing["OT1"].year["passthrough"]
    assert r.pricing["OT2"].year["passthrough"] == pytest.approx(base.pricing["OT2"].year["passthrough"])
    _checks_zero(r)


def test_pv_fixed_guarantee_is_a_user_input_with_the_workbook_derivation_as_default(series, params):
    """D110: fixed_amount None -> Input!C85 derivation (parity); a value set by the user drives the PV guarantee."""
    base = run(series, params)
    derived = base.pnl.pv_fixed_guarantee
    assert derived == pytest.approx(float(base.pnl.portfolio.m("cost_pv_budget").mean() + base.pnl.portfolio.m("rs_pv_cost").mean()))
    p2 = params.copy()
    p2.counterparties["pv"].guarantee.fixed_amount = 1_000_000.0
    r = run(series, p2)
    assert r.pnl.pv_fixed_guarantee == 1_000_000.0
    assert float(np.max(r.pnl.portfolio.m("g_out_pv"))) == pytest.approx(1_000_000.0)
    assert float(np.max(base.pnl.portfolio.m("g_out_pv"))) == pytest.approx(derived)

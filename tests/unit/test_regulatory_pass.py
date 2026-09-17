"""Regulatory verification pass of 17.09.2026 (T12.7): catalogue statuses, the verified 2026 tariff table and the
TAR-2026 finding (applied tariffs summed by the cascade in the v1.1 table)."""

from __future__ import annotations

import copy

import pytest

from config.schema import load_grid_tariffs, load_parameters
from esb.catalogue import STATUSES, _load, entry_for

APPLIED_2026 = {  # lei/MWh IT / MT / JT as the suppliers' sheets publish them (sum of the specific tariffs)
    "Distributie Energie Electrica Romania": (31.96, 115.32, 355.34),
    "Distributie Energie Oltenia": (39.37, 122.80, 333.25),
    "Delgaz Grid": (40.46, 125.71, 387.91),
    "Retele Electrice Romania": (23.83, 92.19, 317.39),
}


def test_catalogue_statuses_and_pass_fields():
    _, entries = _load()
    for _, e in entries:
        assert e.get("source_status", "") in ("",) + STATUSES
        if e.get("source_status") in ("verified", "verified_secondary", "contradicted"):
            assert e.get("checked") == "17.09.2026", e["path"]
        if e.get("source_status") == "contradicted":
            assert e.get("note"), e["path"]  # a contradiction names the open item and the correction
    cit = entry_for("general.cit_rate")
    assert cit.source_status == "verified" and "art. 17" in cit.source
    rc = entry_for("general.reverse_charge_vat_on_sources")
    assert rc.validity == "until 31.12.2026 inclusive" and "RC-2027" in rc.note
    brp = entry_for("market_guarantees.brp.rate_ron_per_mw")
    assert brp.source_status == "contradicted" and "TEL 00.45" in brp.source
    assert entry_for("market_guarantees.tso.vtm_multiplier").source_status == "verified"
    assert entry_for("market_guarantees.dso.vdm_multiplier").source_status == "verified"
    gc = entry_for("green_certificates.quota_gc_per_mwh")
    assert gc.source_status == "verified" and "81/16.12.2025" in gc.source and gc.validity.startswith("2026")
    assert entry_for("calendar.public_holidays").source_status == "verified_secondary"
    text = entry_for("tariff_components_eur_per_mwh.TL").source_text
    assert "[contradicted; validity 2026; checked 17.09.2026]" in text and "36,54" in text


def test_verified_2026_table_sums_to_the_applied_tariffs():
    cfg = load_grid_tariffs()  # the shipped table is the ANRE 2026 table since 0.7.3 (TAR-2026 (a), D119)
    assert cfg["meta"]["source_status"] == "verified" and cfg["meta"]["basis"] == "specific"
    rows = {(r["owner"], r["component"]): r["ron_per_mwh"] for r in cfg["tariffs"]}
    for dso, (it, mt, jt) in APPLIED_2026.items():
        hv, mv, lv = rows[(dso, "T_HV")], rows[(dso, "T_MV")], rows[(dso, "T_LV")]
        assert hv == pytest.approx(it) and hv + mv == pytest.approx(mt) and hv + mv + lv == pytest.approx(jt), dso
    assert rows[("TSO", "TL")] == 36.45 and rows[("TSO", "TG")] == 3.63 and rows[("TSO", "SS")] == 14.70


def test_tar_2026_finding_deer_mv():
    """The superseded v1.1 table yielded the Reference Case set for DEER at MV DSO: 39,37 + 122,80 lei/MWh - Oltenia's
    applied tariffs under DEER's name, HV counted twice. The shipped ANRE table yields DEER's applied MV tariff 115,32.
    The frozen Reference Case register keeps its explicit components (parity)."""
    p = load_parameters()
    o = copy.deepcopy(p.offtaker("OT1"))
    fx = p.general.fx_ron_per_eur
    o.dso, o.voltage_level = "Distributie Energie Electrica Romania", "MV (6-20 kV) DSO"
    q = copy.deepcopy(p)
    q.grid_tariffs = load_grid_tariffs("config/tariffs_ro_v11_template_superseded.yaml")["tariffs"]
    v11 = q.tariffs_by_grid(o.dso, o.voltage_level)
    assert (v11["T_HV"] + v11["T_MV"]) * fx == pytest.approx(39.37 + 122.80, abs=1e-6)
    assert v11["T_HV"] * fx == pytest.approx(p.tariff_components["T_HV"] * fx, abs=1e-6)  # the Reference Case set (D109)
    anre = p.tariffs_by_grid(o.dso, o.voltage_level)
    assert (anre["T_HV"] + anre["T_MV"]) * fx == pytest.approx(115.32, abs=1e-6)
    assert anre["T_LV"] == 0.0 and anre["TL"] * fx == pytest.approx(36.45, abs=1e-6)
    # the finding in EUR/MWh at the register FX: what the Reference Case adds on top of DEER's applied MV tariff
    assert ((39.37 + 122.80) - 115.32) / fx == pytest.approx(8.5182, abs=1e-3)
    assert p.tariff_components["TL"] * fx == pytest.approx(36.54, abs=1e-6)  # the frozen register is untouched


def test_d119_bid_defaults_and_delegated_pre_guarantee():
    """RC-2027, BRP-GF, TAR-2026 (a): the bid defaults change a copy, never the Reference Case; the delegated-PRE
    guarantee is max(initial, months x average monthly imbalance value x (1 + VAT))."""
    import numpy as np

    from esb import guarantees as gr

    p = load_parameters()
    assert p.general.reverse_charge_vat_on_sources and p.market_guarantees["brp"]["method"] == "rate_per_mw"
    assert any(w.startswith("RC-2027") for w in p.regulatory_warnings()) and any(w.startswith("BRP-GF") for w in p.regulatory_warnings())
    q = copy.deepcopy(p)
    done = q.apply_bid_defaults()
    assert len(done) == 2 and not q.general.reverse_charge_vat_on_sources and q.market_guarantees["brp"]["method"] == "pre_delegated"
    assert q.apply_bid_defaults() == [] and q.regulatory_warnings() == []
    assert p.general.reverse_charge_vat_on_sources  # the source register is untouched
    inp = gr.RegulatoryInputs(peak_daily_spot_buy_mwh=0.0, peak_dam_price=0.0, peak_retail_buy_mw=40.0, metered_year_by_offtaker={},
                              imbalance_value_monthly=np.array([-30000.0, 20000.0, -10000.0] + [0.0] * 9))
    fx, vat = p.general.fx_ron_per_eur, p.general.vat_rate
    assert gr.brp_required(p, inp) == pytest.approx(9000.0 * (160.0 + 40.0) / fx)
    avg = 60000.0 / 12
    assert gr.brp_required(q, inp) == pytest.approx(max(100000.0 / fx, 2.0 * avg * (1 + vat)))
    q.market_guarantees["brp"]["pre_months_of_imbalance"] = 1.0
    q.market_guarantees["brp"]["pre_vat_inclusive"] = False
    assert gr.brp_required(q, inp) == pytest.approx(100000.0 / fx)  # the initial amount floors it


def test_d120_pre_service_fee_and_gain():
    """D120: under the workbook rule the PRE service rows are zero (parity); under the delegated rule the fixed fee of the
    contract (2.500 RON/month) and, when a gain is assumed, the 5 % variable fee flow into NM and the cash flow."""
    import pandas as pd

    from esb.engine import run

    series = pd.read_parquet("data/reference/rc_v03_series.parquet")
    p = load_parameters()
    base = run(series, p)
    P = base.pnl.portfolio
    assert P.y("pre_service_fee") == 0.0 and P.y("pre_aggregation_gain") == 0.0 and base.cashflow.y("out_pre_service") == 0.0
    q = copy.deepcopy(p)
    q.market_guarantees["brp"]["method"] = "pre_delegated"  # the method alone (the bid defaults also move the reverse charge, hence interest)
    r = run(series, q)
    fx = q.general.fx_ron_per_eur
    R = r.pnl.portfolio
    assert R.y("pre_service_fee") == pytest.approx(12 * 2500.0 / fx)
    delta_other = (P.y("g_fee_brp") - R.y("g_fee_brp")) + (P.y("interest") - R.y("interest"))
    assert R.y("nm_forecast") == pytest.approx(P.y("nm_forecast") - 12 * 2500.0 / fx + delta_other, rel=1e-9)
    cf = r.cashflow
    assert cf.m13("out_pre_service")[0] == 0.0 and cf.m13("out_pre_service")[1] == pytest.approx(-2500.0 / fx)
    assert cf.m13("out_pre_service")[12] == pytest.approx(-2500.0 / fx)  # December's fee is paid beyond the year
    q.market_guarantees["brp"]["pre_aggregation_gain_pct_of_imbalance"] = 0.2
    r2 = run(series, q)
    gain = 0.2 * float(abs(r2.pnl.portfolio.m("t_imb")).sum())
    assert r2.pnl.portfolio.y("pre_aggregation_gain") == pytest.approx(gain)
    assert r2.pnl.portfolio.y("pre_service_fee") == pytest.approx(12 * 2500.0 / fx + 0.05 * gain)
    d_int = r.pnl.portfolio.y("interest") - r2.pnl.portfolio.y("interest")
    d_g = r.pnl.portfolio.y("g_fee_brp") - r2.pnl.portfolio.y("g_fee_brp")
    assert r2.pnl.portfolio.y("nm_forecast") == pytest.approx(r.pnl.portfolio.y("nm_forecast") + 0.95 * gain + d_int + d_g, rel=1e-9)


"""Parameter catalogue (D-I) and the CSV presentation (compliance check 17.09.2026, C-1 / C-2)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from config.schema import load_parameters
from esb.catalogue import entry_for, unit_by_rule
from esb.engine import run
from esb.export import _flatten, csv_columns, csv_rows
from esb.layout import qh_csv_frame


def test_catalogue_units_and_entries():
    assert unit_by_rule("offtakers[OT1].contract_price_eur_per_mwh") == "EUR/MWh"
    assert unit_by_rule("offtakers[OT1].guarantee.bgl_fee_pa") == "% p.a."
    assert unit_by_rule("general.debt_facility.capacity_pct_of_peak_funding") == "%"
    assert unit_by_rule("offtakers[OT1].contract_start") == "date"
    assert unit_by_rule("meta.currency") == ""  # no rule, no guess
    e = entry_for("general.vat_rate")
    assert e.unit == "%" and e.source_status == "unverified" and "227/2015" in e.source
    e = entry_for("offtakers[OT3].premium_budget.volume")
    assert e.unit == "EUR/MWh" and e.source_status == "assumption"
    flat = _flatten(load_parameters().to_dict())
    numeric = [k for k, v in flat.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    missing = [k for k in numeric if not entry_for(k).unit]
    assert not missing, missing  # every numeric register path carries a unit; text and flags legitimately none


def test_csv_rows_and_columns_presentation():
    df = pd.DataFrame({"Jan": [1234.5678, -0.0000001], "Year": [10.0, 0.5]}, index=["revenue", "gm1_pct"])
    out = csv_rows(df, {"revenue": "EUR", "gm1_pct": "%"}).decode("utf-8-sig").splitlines()
    assert out[0] == "key;label;unit;Jan;Year"
    assert out[1] == "revenue;Revenue · Retail;EUR;1235;10"
    assert out[2].startswith("gm1_pct;") and out[2].endswith(";%;0,0;0,5")  # negative zero normalised, 1 dp for shares
    d = pd.DataFrame({"date": pd.to_datetime(["2027-01-01", "2027-01-02"]), "pay_pv": [-0.0, 1418733.6], "spot_buy_mwh": [304.8986349, 0.0]})
    out = csv_columns(d, {"pay_pv": "EUR", "spot_buy_mwh": "MWh"}, "daily").decode("utf-8-sig").splitlines()
    assert out[0] == "date;pay_pv;spot_buy_mwh"
    assert out[1] == "01.01.2027;0;304,899"
    assert out[2] == "02.01.2027;1418734;0,000"


def test_qh_csv_order_is_the_workbook_order():
    r = run(pd.read_parquet("data/reference/rc_v03_series.parquet"), load_parameters())
    f = qh_csv_frame(r, full=False)
    cols = list(f.columns)
    assert cols[:2] == ["Year_EET", "Year_CET"] and "seq" in cols and cols.index("seq") == 31
    assert cols[cols.index("seq") + 1] == "dam"  # first engine block after the calendar and the sequence
    assert len(f) == 365 and f["Date_EET"].iloc[0] == date(2027, 1, 1)

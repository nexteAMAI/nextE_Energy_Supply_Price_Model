"""Reference Case input fixtures (data/reference/rc_v03_series.parquet) tie to the workbook's own
header totals (row 2 sums and row 3 non-zero means of the source sheets, values as cached in the
frozen workbook) and to the engine grid."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from esb import grid

ROOT = Path(__file__).resolve().parents[2]
PQ = ROOT / "data" / "reference" / "rc_v03_series.parquet"
MANIFEST = ROOT / "data" / "reference" / "rc_v03_series_manifest.json"

# workbook cached totals: FW_Retail_Volume row 2, FW_Source_Purch_Volume row 2, Whol_Sport_Imb_Fcst row 3
WORKBOOK_SUMS = {
    "OT1_metered_consumption_MWh": 54979.817999999825,
    "OT1_notified_consumption_MWh": 55017.820900839513,
    "OT2_metered_consumption_MWh": 88285.089153613691,
    "OT2_notified_consumption_MWh": 87834.057878677238,
    "OT3_metered_consumption_MWh": 199999.99999999965,
    "OT3_notified_consumption_MWh": 199372.26857283467,
    "OT4_metered_consumption_MWh": 220000.00000000064,
    "OT4_notified_consumption_MWh": 219309.49543011526,
    "PV1_forecast_generation_uncurtailed_MWh": 411705.00000000664,
}
WORKBOOK_NONZERO_MEANS = {
    "central__DAM_price_EUR_MWh": 119.6106962606093,
    "central__IDCT_VWAP15_price_EUR_MWh": 122.3536381278532,
}


@pytest.fixture(scope="module")
def series() -> pd.DataFrame:
    assert PQ.exists(), "fixture parquet missing"
    return pd.read_parquet(PQ)


@pytest.mark.parity
def test_shape_and_grid(series):
    g = grid.make_grid(2027)
    assert len(series) == 35040
    assert (series["seq"].values == g["seq"].values).all()
    assert (pd.to_datetime(series["date"]).values == g["date"].values).all()
    assert (series["interval"].values == g["interval"].values).all()
    assert (series["peak_v03"].values == g["peak"].values).all()


@pytest.mark.parity
@pytest.mark.parametrize("name,total", list(WORKBOOK_SUMS.items()))
def test_sums_tie_to_workbook(series, name, total):
    s = float(series[name].sum())
    assert abs(s - total) <= 1e-6 * max(abs(total), 1.0), (name, s, total)


@pytest.mark.parity
@pytest.mark.parametrize("name,mean", list(WORKBOOK_NONZERO_MEANS.items()))
def test_nonzero_means_tie_to_workbook(series, name, mean):
    v = series[name].values
    m = float(v[v != 0].mean())
    assert abs(m - mean) <= 1e-6 * max(abs(mean), 1.0)


@pytest.mark.parity
def test_no_blank_and_coded_entities(series):
    numeric = [c for c in series.columns if series[c].dtype.kind == "f"]
    assert not series[numeric].isna().any().any()
    assert set(series["central__System_imbalance_direction"].unique()) <= {"Positive (Long)", "Negative (Short)", "Balanced"}
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["source_md5"] == "1c873718bcd08957229f1f46d5449b2a"
    # entities appear only as codes
    entity_cols = [c for c in series.columns if c.startswith("OT") or c.startswith("PV")]
    assert {c.split("_")[0] for c in entity_cols} == {"OT1", "OT2", "OT3", "OT4", "PV1"}


@pytest.mark.parity
def test_surplus_equals_deficit_in_loaded_scenarios(series):
    # X-19: the loaded scenarios carry identical surplus and deficit prices; the rule-book
    # tests therefore use a synthetic dual-price case
    for scen in ("central", "low"):
        assert np.array_equal(series[f"{scen}__Surplus_imbalance_price_EUR_MWh"].values, series[f"{scen}__Deficit_imbalance_price_EUR_MWh"].values)

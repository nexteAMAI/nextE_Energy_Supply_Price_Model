"""Contract ESB-STD-QH 1.1 (D112): scenario blocks in one wholesale file, entity names for reference, an entirely
blank declared column = not delivered; 1.0 files keep their behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import load_parameters
from esb import grid
from esb.assemble import assemble
from esb.importer import import_workbook
from esb.importer.contract import Control, offtaker_load_registry, wholesale_prices_registry
from esb.importer.template import write_delivery

YEAR = 2027


def _fixed96(year: int, series: dict[str, np.ndarray]) -> pd.DataFrame:
    g = grid.make_grid(year)
    out = pd.DataFrame({"Date_EET": [d.date() for d in g["date"]], "Start_EET": [t.time() for t in g["start_eet"]],
                        "End_EET": [t.time() for t in g["end_eet"]]})
    for k, v in series.items():
        out[k] = v
    return out


def test_wholesale_file_with_two_scenario_blocks_is_keyed_by_scenario(tmp_path):
    reg = wholesale_prices_registry(["Aurora Central", "Aurora Low"])
    assert reg.validate() == []  # same series names in two blocks: unique per block
    g = grid.make_grid(YEAR)
    n = len(g)
    data = _fixed96(YEAR, {
        "Aurora Central__DAM_price_EUR_MWh": 100.0 + g["interval"].values, "Aurora Central__IDCT_VWAP15_price_EUR_MWh": 101.0 + g["interval"].values,
        "Aurora Central__Surplus_imbalance_price_EUR_MWh": np.full(n, 90.0), "Aurora Central__Deficit_imbalance_price_EUR_MWh": np.full(n, 110.0),
        "Aurora Central__System_imbalance_direction": np.where(g["interval"].values % 2 == 0, "Positive (Long)", "Negative (Short)"),
        "Aurora Low__DAM_price_EUR_MWh": 50.0 + g["interval"].values, "Aurora Low__IDCT_VWAP15_price_EUR_MWh": 51.0 + g["interval"].values,
        "Aurora Low__Surplus_imbalance_price_EUR_MWh": np.full(n, 40.0), "Aurora Low__Deficit_imbalance_price_EUR_MWh": np.full(n, 60.0),
        "Aurora Low__System_imbalance_direction": np.full(n, "Balanced"),
    })
    ctl = Control(input_class="wholesale_prices", spine_year=YEAR, time_basis="fixed_96")
    path = tmp_path / "wh.xlsx"
    write_delivery(path, ctl, reg, data)
    res = import_workbook(path)
    assert res.ok, res.summary()
    assert res.provenance.scenario_blocks == ["Aurora Central", "Aurora Low"]
    assert "Aurora Low__DAM_price_EUR_MWh" in res.frame.columns and "Aurora Central__DAM_price_EUR_MWh" in res.frame.columns
    p = load_parameters()
    a = assemble(p, [(res, None)], reference=None)
    assert "central__DAM_price_EUR_MWh" in a.frame.columns and "low__DAM_price_EUR_MWh" in a.frame.columns
    assert a.coverage.scenarios["Aurora Central"] and a.coverage.scenarios["Aurora Low"] and not a.coverage.scenarios["User Forecast"]
    assert float(a.frame["low__DAM_price_EUR_MWh"].iloc[0]) == 51.0


def test_entirely_blank_declared_columns_are_not_delivered_under_11(tmp_path):
    reg = offtaker_load_registry(["OT1", "OT2", "OT3"])  # six required slots declared
    g = grid.make_grid(YEAR)
    data = _fixed96(YEAR, {"OT1_metered_consumption_MWh": np.full(len(g), 1.0), "OT1_notified_consumption_MWh": np.full(len(g), 1.0)})
    for s in reg.slots[2:]:
        data[s.name] = np.nan  # OT2 and OT3 columns present, entirely blank
    ctl = Control(input_class="offtaker_load", spine_year=YEAR, time_basis="fixed_96")
    path = tmp_path / "ot.xlsx"
    write_delivery(path, ctl, reg, data)
    res = import_workbook(path)
    assert res.ok, res.summary()
    assert sorted(res.provenance.not_delivered) == ["E03", "E04", "E05", "E06"]
    assert "OT2_metered_consumption_MWh" not in res.frame.columns
    # a partly blank required series is still a gap
    data2 = data.copy()
    data2.loc[100, "OT1_metered_consumption_MWh"] = np.nan
    path2 = tmp_path / "ot2.xlsx"
    write_delivery(path2, ctl, reg, data2)
    res2 = import_workbook(path2)
    assert not res2.ok and any(c.code == "6" and not c.passed for c in res2.checks)


def test_10_file_keeps_the_old_behaviour(tmp_path):
    reg = offtaker_load_registry(["OT1", "OT2"])
    g = grid.make_grid(YEAR)
    data = _fixed96(YEAR, {"OT1_metered_consumption_MWh": np.full(len(g), 1.0), "OT1_notified_consumption_MWh": np.full(len(g), 1.0),
                           "OT2_metered_consumption_MWh": np.full(len(g), np.nan), "OT2_notified_consumption_MWh": np.full(len(g), np.nan)})
    ctl = Control(input_class="offtaker_load", spine_year=YEAR, time_basis="fixed_96", template_version="ESB-STD-QH 1.0")
    path = tmp_path / "ot10.xlsx"
    write_delivery(path, ctl, reg, data)
    res = import_workbook(path)
    assert not res.ok and res.provenance.not_delivered == []
    assert Control(input_class="offtaker_load", spine_year=YEAR, template_version="ESB-STD-QH 2.0").validate()  # unknown version refused

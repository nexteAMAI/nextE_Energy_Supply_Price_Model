"""Daylight-saving rules and layout logic of the importer, on synthetic local-clock deliveries."""

from datetime import date

import numpy as np
import pytest

from esb import grid
from esb.importer.standardise import label_rows, standardise
from tests.conftest import YEAR, local_clock_frame


def _std(registry, control, raw=None):
    raw = local_clock_frame() if raw is None else raw
    labelled, layout = label_rows(raw, control)
    return labelled, layout, standardise(labelled, control, registry)


def test_layout_of_a_correct_local_clock_delivery(registry, control):
    _, layout, std = _std(registry, control)
    assert layout.ok
    assert layout.delivered_rows == 365 * 96  # -4 + 4
    spring, autumn = grid.dst_dates(YEAR)
    assert layout.per_day_delivered[spring] == 92 and layout.per_day_delivered[autumn] == 100
    assert len(std) == 365 * 96


def test_extensive_injected_zero_and_merged_sum(registry, control):
    _, _, std = _std(registry, control)
    spring, autumn = grid.dst_dates(YEAR)
    inj = std[(std["date"].dt.date == spring) & std["interval"].between(13, 16)]
    assert (inj["vol_MWh"] == 0).all()
    mer = std[(std["date"].dt.date == autumn) & std["interval"].between(13, 16)]
    # first occurrence = seq, second = seq + 1000 -> sum
    assert np.allclose(mer["vol_MWh"].values, mer["seq"].values * 2 + 1000)
    # energy conserved: sum of std = sum of raw
    raw = local_clock_frame()
    assert abs(std["vol_MWh"].sum() - raw["vol_MWh"].sum()) < 1e-6


def test_intensive_interpolate_and_mean(registry, control):
    _, _, std = _std(registry, control)
    spring, autumn = grid.dst_dates(YEAR)
    inj = std[(std["date"].dt.date == spring) & std["interval"].between(13, 16)]
    # prev (interval 12) = 112, next (interval 17) = 117 -> 113, 114, 115, 116
    assert np.allclose(inj["price_EUR_MWh"].values, [113, 114, 115, 116])
    mer = std[(std["date"].dt.date == autumn) & std["interval"].between(13, 16)]
    # mean of (100 + iv) and (110 + iv)
    assert np.allclose(mer["price_EUR_MWh"].values, 105 + mer["interval"].values)


def test_intensive_locf_and_volume_weighted(registry, control):
    _, _, std = _std(registry, control)
    spring, autumn = grid.dst_dates(YEAR)
    inj = std[(std["date"].dt.date == spring) & std["interval"].between(13, 16)]
    assert np.allclose(inj["ratio"].values, 0.12)  # last value before the gap (interval 12)
    mer = std[(std["date"].dt.date == autumn) & std["interval"].between(13, 16)]
    seq = mer["seq"].values.astype(float)
    r1 = 0.01 * mer["interval"].values
    r2 = r1 + 1
    expected = (r1 * seq + r2 * (seq + 1000)) / (seq + seq + 1000)
    assert np.allclose(mer["ratio"].values, expected)


def test_categorical_carry_forward_and_first(registry, control):
    _, _, std = _std(registry, control)
    spring, autumn = grid.dst_dates(YEAR)
    inj = std[(std["date"].dt.date == spring) & std["interval"].between(13, 16)]
    assert (inj["direction"] == "Positive (Long)").all()  # interval 12 is even -> Positive
    mer = std[(std["date"].dt.date == autumn) & std["interval"].between(13, 16)]
    assert "Balanced" not in set(mer["direction"])  # first occurrence wins


def test_normal_rows_are_copied_unchanged(registry, control):
    _, _, std = _std(registry, control)
    normal = std[std["dst_status"] == "Normal"]
    assert np.array_equal(normal["vol_MWh"].values, normal["seq"].values.astype(float))
    assert np.array_equal(normal["price_EUR_MWh"].values, 100.0 + normal["interval"].values)


def test_fixed_96_basis_is_positional(registry, control):
    control.time_basis = "fixed_96"
    g = grid.make_grid(YEAR)
    raw = local_clock_frame()
    # a fixed_96 file has 96 rows on every day: rebuild from the grid
    import pandas as pd

    raw = pd.DataFrame(
        {
            "Date_EET": g["date"].dt.date.values,
            "Start_EET": [t.time() for t in g["start_eet"]],
            "End_EET": [t.time() for t in g["end_eet"]],
            "vol_MWh": g["seq"].astype(float).values,
            "price_EUR_MWh": 100.0 + g["interval"].values,
            "ratio": 0.01 * g["interval"].values,
            "direction": "Balanced",
        }
    )
    _, layout, std = _std(registry, control, raw)
    assert layout.ok and layout.delivered_rows == 365 * 96
    assert np.array_equal(std["vol_MWh"].values, g["seq"].values.astype(float))


def test_blank_cell_survives_as_nan(registry, control):
    raw = local_clock_frame()
    d = date(YEAR, 6, 15)
    mask = (raw["Date_EET"] == d) & (raw["Start_EET"] == grid.interval_start(40))
    raw.loc[mask, "vol_MWh"] = np.nan
    _, _, std = _std(registry, control, raw)
    cell = std[(std["date"].dt.date == d) & (std["interval"] == 40)]["vol_MWh"]
    assert cell.isna().all()


@pytest.mark.parametrize("year", [2026, 2028])
def test_other_years(year, registry):
    from esb.importer.contract import Control

    control = Control(input_class="wholesale_prices", spine_year=year, time_basis="local_clock")
    raw = local_clock_frame(year)
    _, layout, std = _std(registry, control, raw)
    assert layout.ok and len(std) == grid.days_in_year(year) * 96

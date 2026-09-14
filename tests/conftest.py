"""Shared helpers for the importer tests: synthetic deliveries on the real local clock."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from esb import grid
from esb.importer.contract import Control, Registry, SlotSpec

YEAR = 2027


def synthetic_registry() -> Registry:
    return Registry(
        [
            SlotSpec("E01", "vol_MWh", "MWh", "Extensive", "inject_zero", "sum", k=-1, basis="metered"),
            SlotSpec("E02", "price_EUR_MWh", "EUR/MWh", "Intensive", "interpolate", "mean", basis="price"),
            SlotSpec("E03", "ratio", "ratio", "Intensive", "locf", "volume_weighted", paired_volume_slot="E01", basis="ratio"),
            SlotSpec("C01", "direction", "flag", "Categorical", "carry_forward", "first", basis="flag", required=False),
        ]
    )


def local_clock_frame(year: int = YEAR) -> pd.DataFrame:
    """A delivery on the real local clock: 92 rows on the spring day, 100 on the autumn day.

    Values are deterministic functions of the grid position so that every rule can be checked:
    vol = seq (as float), price = 100 + interval, ratio = 0,01 x interval, direction alternates.
    The second autumn occurrence carries vol + 1000, price + 10, ratio + 1.
    """
    g = grid.make_grid(year)
    df = pd.DataFrame(
        {
            "Date_EET": g["date"].dt.date.values,
            "Start_EET": [t.time() for t in g["start_eet"]],
            "End_EET": [t.time() for t in g["end_eet"]],
            "vol_MWh": g["seq"].astype(float).values,
            "price_EUR_MWh": 100.0 + g["interval"].values,
            "ratio": 0.01 * g["interval"].values,
            "direction": np.where(g["interval"].values % 2 == 0, "Positive (Long)", "Negative (Short)"),
            "_status": g["dst_status"].values,
        }
    )
    df = df[df["_status"] != "Injected"]
    second = df[df["_status"] == "Merged"].copy()
    second["vol_MWh"] += 1000
    second["price_EUR_MWh"] += 10
    second["ratio"] += 1
    second["direction"] = "Balanced"
    out = pd.concat([df, second]).sort_values(["Date_EET", "Start_EET"], kind="stable").reset_index(drop=True)
    return out.drop(columns="_status")


@pytest.fixture
def registry() -> Registry:
    return synthetic_registry()


@pytest.fixture
def control() -> Control:
    return Control(input_class="wholesale_prices", spine_year=YEAR, time_basis="local_clock")


@pytest.fixture
def spring_autumn() -> tuple[date, date]:
    return grid.dst_dates(YEAR)

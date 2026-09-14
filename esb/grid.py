"""esb.grid - the quarter-hourly calendar grid of the engine.

Conventions (Reference Case, workbook `Energy_Supply_Portfolio_Tracking_v03`, SPEC section 4):

* One row per quarter-hour of the spine year, 96 rows per calendar day, labelled in EET
  (Romanian local clock) with a fixed one-hour offset to CET. There are no 92- or 100-row
  days: daylight-saving transitions are absorbed at import (see esb.importer), so the engine
  grid is purely positional. A common year has 365 x 96 = 35.040 rows, a leap year 35.136.
* The workbook's fixed 35.136-row container (96 placeholder rows at 29.02 in common years,
  or at the end of the STD template) is an Excel artefact and is not reproduced; every
  aggregation in the workbook is keyed by date, so dropping the placeholder rows changes nothing.
* Peak / Off-Peak follows the CET clock, 08:00 <= start < 20:00 CET, evaluated on the row's
  fixed-offset CET label. In EET interval numbers (1..96) that is intervals 37..84 on every day,
  which is exactly the workbook's `Peak_Off_Peak_Interval_08_20_CET` column (verified on all
  35.040 dated rows of the Reference Case).
* Daylight-saving dates follow the EU rule (last Sunday of March and of October, Directive
  2000/84/EC); they are needed by the importer, not by the engine.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

INTERVALS_PER_DAY = 96
INTERVAL_MINUTES = 15
OFFSET_EET_CET_H = 1
PEAK_START_CET = time(8, 0)
PEAK_END_CET = time(20, 0)
PEAK_LABEL = "Peak"
OFFPEAK_LABEL = "Off-Peak"
# EET interval numbers (1-based) whose CET start lies in [08:00, 20:00)
PEAK_INTERVALS = range(37, 85)
# Local-clock intervals affected by the DST transitions (03:00-03:45 EET), 1-based
DST_INTERVALS = (13, 14, 15, 16)
EXCEL_EPOCH = date(1899, 12, 30)


def last_sunday(year: int, month: int) -> date:
    """Last Sunday of a month (EU daylight-saving rule)."""
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    return d - timedelta(days=(d.weekday() + 1) % 7)


def dst_dates(year: int) -> tuple[date, date]:
    """(spring transition date, autumn transition date) for the spine year."""
    return last_sunday(year, 3), last_sunday(year, 10)


def days_in_year(year: int) -> int:
    return 366 if calendar.isleap(year) else 365


def interval_start(interval: int) -> time:
    """Start time (local clock) of a 1-based interval number."""
    minutes = (interval - 1) * INTERVAL_MINUTES
    return time(minutes // 60, minutes % 60)


def interval_of(t: time) -> int:
    """1-based interval number of a local-clock start time; raises if not on the 15-minute raster."""
    minutes = t.hour * 60 + t.minute
    if minutes % INTERVAL_MINUTES or t.second:
        raise ValueError(f"start time {t} is not on the 15-minute raster")
    return minutes // INTERVAL_MINUTES + 1


def excel_serial_to_date(serial: float | int) -> date:
    """Excel 1900-system serial number -> date (46388 -> 01.01.2027)."""
    return EXCEL_EPOCH + timedelta(days=int(serial))


def date_to_excel_serial(d: date) -> int:
    return (d - EXCEL_EPOCH).days


def peak_flag(interval: int) -> str:
    return PEAK_LABEL if interval in PEAK_INTERVALS else OFFPEAK_LABEL


@dataclass(frozen=True)
class GridInfo:
    year: int
    days: int
    rows: int
    spring: date
    autumn: date


def grid_info(year: int) -> GridInfo:
    d = days_in_year(year)
    s, a = dst_dates(year)
    return GridInfo(year=year, days=d, rows=d * INTERVALS_PER_DAY, spring=s, autumn=a)


def make_grid(year: int) -> pd.DataFrame:
    """The engine grid for one spine year.

    Columns: seq (1..rows), date (datetime64[ns], midnight), day_of_year (1-based), interval
    (1..96), start_eet, end_eet, start_cet (datetime64[ns]; end of day 96 rolls to the next day),
    month (1..12), weekday (0 = Monday), peak ('Peak' / 'Off-Peak'), dst_status
    ('Normal' / 'Injected' / 'Merged' - informational: which rows were affected by a
    daylight-saving rule at import).
    """
    info = grid_info(year)
    day0 = datetime(year, 1, 1)
    day_idx = np.repeat(np.arange(info.days), INTERVALS_PER_DAY)
    interval = np.tile(np.arange(1, INTERVALS_PER_DAY + 1), info.days)
    dates = pd.to_datetime(day0) + pd.to_timedelta(day_idx, unit="D")
    start_eet = dates + pd.to_timedelta((interval - 1) * INTERVAL_MINUTES, unit="m")
    end_eet = start_eet + pd.Timedelta(minutes=INTERVAL_MINUTES)
    start_cet = start_eet - pd.Timedelta(hours=OFFSET_EET_CET_H)
    peak = np.where(np.isin(interval, list(PEAK_INTERVALS)), PEAK_LABEL, OFFPEAK_LABEL)
    status = np.full(info.rows, "Normal", dtype=object)
    for d, label in ((info.spring, "Injected"), (info.autumn, "Merged")):
        doy = (d - date(year, 1, 1)).days
        for iv in DST_INTERVALS:
            status[doy * INTERVALS_PER_DAY + iv - 1] = label
    return pd.DataFrame(
        {
            "seq": np.arange(1, info.rows + 1),
            "date": dates,
            "day_of_year": day_idx + 1,
            "interval": interval,
            "start_eet": start_eet,
            "end_eet": end_eet,
            "start_cet": start_cet,
            "month": dates.month,
            "weekday": dates.weekday,
            "peak": peak,
            "dst_status": status,
        }
    )

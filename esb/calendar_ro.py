"""esb.calendar_ro - Romanian public holidays and the calendar block of the QH exports (D113, O-19/O-20).

The calendar block reproduces the descriptive columns of the CEO's output template (EET and CET twins of
year, month, week, date, weekday, season, day-of-year, interval and the weekend / public-holiday flag).
None of it feeds the engine; it is written next to the engine keys so a reader can filter the frame.
Holidays come from config/calendar_ro.yaml (rules, not dated lists); the Orthodox Easter is computed.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from esb.grid import INTERVAL_MINUTES, OFFSET_EET_CET_H

CALENDAR_YAML = Path(__file__).resolve().parents[1] / "config" / "calendar_ro.yaml"
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
           9: "Autumn", 10: "Autumn", 11: "Autumn"}  # meteorological seasons (interpretation - O-20)


def orthodox_easter(year: int) -> date:
    """Orthodox Easter Sunday in the Gregorian calendar (Meeus Julian algorithm + 13 days, valid 1900-2099)."""
    if not 1900 <= year <= 2099:
        raise ValueError(f"Orthodox Easter conversion offset valid for 1900-2099, got {year}")
    a, b, c = year % 4, year % 7, year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    month, day = divmod(d + e + 114, 31)
    return date(year, month, day + 1) + timedelta(days=13)


@cache
def _rules() -> dict:
    with open(CALENDAR_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f).get("public_holidays", {}) or {}


def public_holidays(year: int) -> dict[date, str]:
    """{date: name} of the Romanian legal public holidays of one year, from the yaml rules."""
    rules = _rules()
    out: dict[date, str] = {}
    for h in rules.get("fixed", []):
        m, d = (int(x) for x in str(h["date"]).split("-"))
        out[date(year, m, d)] = h["name"]
    easter = orthodox_easter(year)
    for h in rules.get("movable", []):
        out[easter + timedelta(days=int(h["offset"]))] = h["name"]
    return dict(sorted(out.items()))


def holiday_source() -> tuple[str, str]:
    r = _rules()
    return str(r.get("source", "")), str(r.get("source_status", ""))


# ---- calendar block ----------------------------------------------------------------------------------
CALENDAR_COLUMNS = [  # (name, dtype hint for the number format) in the template's order, without the duplicate flag (O-19)
    ("Year_EET", "int"), ("Year_CET", "int"), ("Month_year_EET", "month"), ("Month_year_CET", "month"),
    ("Week_year_EET", "int"), ("Week_year_CET", "int"), ("Month_name_EET", "text"), ("Month_name_CET", "text"),
    ("Date_EET", "date"), ("Date_CET", "date"), ("Day_year_EET", "int"), ("Day_year_CET", "int"),
    ("Weekday_EET", "int"), ("Weekday_CET", "int"), ("Season_EET", "text"), ("Season_CET", "text"),
    ("Weekday_name_EET", "text"), ("Weekday_name_CET", "text"),
    ("Is_Weekend_or_RO_public_holiday_flag_EET", "int"), ("Is_Weekend_or_RO_public_holiday_flag_CET", "int"),
    ("Day_of_year_EET", "int"), ("Day_of_year_CET", "int"), ("Day_hour_interval_EET", "int"), ("Day_hour_interval_CET", "int"),
    ("Day_interval_EET", "int"), ("Day_interval_CET", "int"), ("Peak_Off_Peak_Interval_08_20_CET_DAM", "text"),
    ("Start_time_interval_CET", "time"), ("End_time_interval_CET", "time"), ("Start_time_interval_EET", "time"), ("End_time_interval_EET", "time"),
]
CALENDAR_INTERPRETATION = {  # what each descriptive column means (O-20: to be confirmed by the CEO)
    "Week_year": "ISO week number of the interval's date",
    "Day_year": "day of the month (1-31); Day_of_year is the ordinal day (1-366)",
    "Weekday": "ISO weekday number, Monday = 1",
    "Season": "meteorological season: Winter Dec-Feb, Spring Mar-May, Summer Jun-Aug, Autumn Sep-Nov",
    "Day_hour_interval": "hour of the day (0-23) in which the interval starts",
    "Day_interval": "interval index within the day (1-96); the CET twin counts within the CET day",
    "Is_Weekend_or_RO_public_holiday_flag": "1 when the date is a Saturday, a Sunday or a Romanian legal public holiday, else 0",
}


def calendar_block(grid: pd.DataFrame) -> pd.DataFrame:
    """The calendar columns for the engine grid (one row per quarter-hour of the spine year)."""
    start_eet = pd.to_datetime(grid["start_eet"])
    start_cet = pd.to_datetime(grid["start_cet"]) if "start_cet" in grid else start_eet - pd.Timedelta(hours=OFFSET_EET_CET_H)
    end_eet = start_eet + pd.Timedelta(minutes=INTERVAL_MINUTES)
    end_cet = start_cet + pd.Timedelta(minutes=INTERVAL_MINUTES)
    years = sorted({int(y) for y in start_eet.dt.year.unique()} | {int(y) for y in start_cet.dt.year.unique()})
    holidays = {d for y in years for d in public_holidays(y)}

    def flag(ts: pd.Series) -> np.ndarray:
        d = ts.dt.date
        return np.array([1 if (t.weekday() >= 5 or t in holidays) else 0 for t in d], dtype=int)

    def block(ts: pd.Series, tag: str) -> dict[str, object]:
        return {
            f"Year_{tag}": ts.dt.year.to_numpy(),
            f"Month_year_{tag}": ts.dt.to_period("M").dt.to_timestamp().dt.date.to_numpy(),
            f"Week_year_{tag}": ts.dt.isocalendar().week.astype(int).to_numpy(),
            f"Month_name_{tag}": np.array([MONTH_NAMES[m - 1] for m in ts.dt.month], dtype=object),
            f"Date_{tag}": ts.dt.date.to_numpy(),
            f"Day_year_{tag}": ts.dt.day.to_numpy(),
            f"Weekday_{tag}": (ts.dt.weekday + 1).to_numpy(),
            f"Season_{tag}": np.array([SEASONS[m] for m in ts.dt.month], dtype=object),
            f"Weekday_name_{tag}": np.array([WEEKDAY_NAMES[w] for w in ts.dt.weekday], dtype=object),
            f"Is_Weekend_or_RO_public_holiday_flag_{tag}": flag(ts),
            f"Day_of_year_{tag}": ts.dt.dayofyear.to_numpy(),
            f"Day_hour_interval_{tag}": ts.dt.hour.to_numpy(),
            f"Day_interval_{tag}": ((ts.dt.hour * 60 + ts.dt.minute) // INTERVAL_MINUTES + 1).to_numpy(),
        }

    cols = {**block(start_eet, "EET"), **block(start_cet, "CET")}
    cols["Peak_Off_Peak_Interval_08_20_CET_DAM"] = grid["peak"].to_numpy()
    cols["Start_time_interval_CET"] = start_cet.dt.time.to_numpy()
    cols["End_time_interval_CET"] = end_cet.dt.time.to_numpy()
    cols["Start_time_interval_EET"] = start_eet.dt.time.to_numpy()
    cols["End_time_interval_EET"] = end_eet.dt.time.to_numpy()
    return pd.DataFrame({name: cols[name] for name, _ in CALENDAR_COLUMNS})

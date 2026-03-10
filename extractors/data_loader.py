"""
Core Data Loader — Unified loading for all pre-loaded CSV datasets.

Handles the diverse file formats across the project:
  - Standard ENTSO-E CSVs (comma-separated, UTC offset timestamps)
  - Balancing Services CSVs (ISO timestamps with +0300 offset)
  - NEXTE IDM data (semicolon-delimited, comma decimal, cp1252)
  - Aurora forecast (semicolon-delimited, comma decimal, cp1252)
  - Montel backcasts (multi-row headers, date index)
  - FX rate data (semicolon-delimited, comma decimal, DD.MM.YYYY dates)
  - Sensitivity scenarios (multi-level header)

All loaded DataFrames are timezone-aligned to Europe/Bucharest (EET/EEST).
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data directory resolution (pre-loaded files may be in project or data/static)
# ---------------------------------------------------------------------------

_DATA_SEARCH_PATHS = [
    settings.static_dir,
    settings.raw_dir,
    settings.project_root / "data",
    Path("/mnt/project"),  # Claude project files mount
]


def _resolve_file(filename: str) -> Path:
    """Locate a pre-loaded data file across known directories."""
    for base in _DATA_SEARCH_PATHS:
        candidate = base / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Dataset file '{filename}' not found in any search path: "
        f"{[str(p) for p in _DATA_SEARCH_PATHS]}"
    )


def _to_bucharest_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeIndex is in Europe/Bucharest timezone."""
    if not isinstance(df.index, pd.DatetimeIndex):
        # Attempt to convert to DatetimeIndex
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Europe/Bucharest")
    else:
        df.index = df.index.tz_convert("Europe/Bucharest")
    df.index.name = "timestamp_eet"
    return df


# ===========================================================================
# GENERIC ENTSO-E / STANDARD CSV LOADER
# ===========================================================================

def load_entsoe_csv(
    filename: str,
    *,
    usecols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load a standard ENTSO-E-format CSV.
    Format: comma-separated, index_col='Interval_Start_EET', datetime w/ tz offset.
    """
    path = _resolve_file(filename)
    df = pd.read_csv(
        path,
        index_col="Interval_Start_EET",
        parse_dates=True,
        usecols=usecols,
    )
    df = _to_bucharest_tz(df)
    df = df.sort_index()
    logger.info("Loaded %s: %d rows, %s — %s", filename, len(df),
                df.index.min(), df.index.max())
    return df


# ===========================================================================
# BALANCING SERVICES CSV LOADER
# ===========================================================================

def load_balancing_services_csv(
    filename: str,
    *,
    index_col: str = "period_startAt",
) -> pd.DataFrame:
    """
    Load Balancing Services API CSV.
    Format: comma-separated, ISO timestamps with +0300 offset.
    """
    path = _resolve_file(filename)
    df = pd.read_csv(path)
    df[index_col] = pd.to_datetime(df[index_col], utc=True)
    df = df.set_index(index_col)
    df = _to_bucharest_tz(df)
    df = df.sort_index()
    logger.info("Loaded %s: %d rows, %s — %s", filename, len(df),
                df.index.min(), df.index.max())
    return df


# ===========================================================================
# NEXTE IDM LOADER
# ===========================================================================

def load_idm_nexte(filename: str = "RO_EMWSH_QH_DATA_SET_2_IDM_IDCT_Price_Vol_V1_NEXTE.csv") -> pd.DataFrame:
    """
    Load NEXTE IDM Continuous Trading dataset.
    Format: semicolon-delimited, comma decimal, cp1252 encoding.
    Date = DD.MM.YYYY string, time intervals as Start_time_interval / End_time_interval.

    Returns DataFrame with proper datetime index at 15-min resolution.
    """
    path = _resolve_file(filename)
    df = pd.read_csv(path, sep=";", decimal=",", encoding="cp1252")

    # Build datetime from Date + Start_time_interval
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Start_time_interval"],
        format="%d.%m.%Y %H:%M:%S",
    )
    df = df.set_index("timestamp")
    df.index = df.index.tz_localize("Europe/Bucharest", ambiguous="NaT", nonexistent="shift_forward")
    # Drop rows where DST ambiguity resulted in NaT
    df = df[df.index.notna()]
    df.index.name = "timestamp_eet"
    df = df.sort_index()

    # Standardize column names (remove inconsistent prefix "I")
    rename_map = {}
    for col in df.columns:
        if col.startswith("IIDCT_"):
            rename_map[col] = col.replace("IIDCT_", "IDCT_", 1)
    if rename_map:
        df = df.rename(columns=rename_map)

    logger.info("Loaded IDM NEXTE: %d rows, %s — %s", len(df),
                df.index.min(), df.index.max())
    return df


# ===========================================================================
# AURORA FORECAST LOADER
# ===========================================================================

def load_aurora_forecast(
    filename: str = "RO_EMWSF_Monthly_Aurora_Oct25_ROU_Nominal_AURORA.csv",
) -> pd.DataFrame:
    """
    Load Aurora Oct 2025 monthly power & renewables forecast.
    Format: semicolon-delimited, comma decimal, cp1252, no datetime index.

    Returns DataFrame indexed by (Calendar_Year, Month) with all scenario columns.
    """
    path = _resolve_file(filename)
    df = pd.read_csv(path, sep=";", decimal=",", encoding="cp1252")

    # Build a proper date index for time-series analysis
    # Month column may be float (1.0) due to comma-decimal parsing
    # Drop rows with NaN in key columns (trailing empty rows in CSV)
    df = df.dropna(subset=["Calendar_Year", "Month"])
    df["Calendar_Year"] = df["Calendar_Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df["date"] = pd.to_datetime(
        df["Calendar_Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-01"
    )
    df = df.set_index("date")
    df.index.name = "month_start"
    df = df.sort_index()

    logger.info("Loaded Aurora forecast: %d months, %s — %s", len(df),
                df.index.min(), df.index.max())
    return df


# ===========================================================================
# FX RATE LOADER
# ===========================================================================

def load_fx_eur_ron(
    filename: str = "EURO_RON_Conversion_rate_01_01_2009_26_02_2026.csv",
) -> pd.DataFrame:
    """
    Load EUR/RON BNR daily reference rate.
    Format: semicolon-delimited, comma decimal, DD.MM.YYYY dates.
    """
    path = _resolve_file(filename)
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8")

    # First column is Date, second is rate; remaining columns may be empty
    date_col = df.columns[0]
    rate_col = df.columns[1]
    df = df[[date_col, rate_col]].copy()
    df.columns = ["date", "eur_ron"]
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
    df = df.set_index("date")
    df = df.sort_index()
    df["eur_ron"] = pd.to_numeric(df["eur_ron"], errors="coerce")

    # Forward-fill weekends/holidays
    df = df.asfreq("D", method="ffill")

    logger.info("Loaded FX EUR/RON: %d days, %s — %s, latest rate: %.4f",
                len(df), df.index.min(), df.index.max(), df["eur_ron"].iloc[-1])
    return df


# ===========================================================================
# MONTEL BACKCAST LOADER (Residual Load / Imbalance Volume)
# ===========================================================================

def load_montel_backcast(
    filename: str,
    *,
    value_col_name: str = "value",
) -> pd.DataFrame:
    """
    Load Montel/EQ backcast CSVs with multi-row headers.
    Format: first 3 rows are metadata; row 4 is 'date' header; data starts row 5.
    Index is datetime with timezone offset.
    """
    path = _resolve_file(filename)
    df = pd.read_csv(path, skiprows=3, index_col=0, parse_dates=True)
    df.columns = [value_col_name]
    df.index.name = "timestamp"
    df.index = pd.to_datetime(df.index, utc=True)
    df = _to_bucharest_tz(df)
    df = df.sort_index()
    df[value_col_name] = pd.to_numeric(df[value_col_name], errors="coerce")

    logger.info("Loaded Montel backcast %s: %d rows, %s — %s",
                filename, len(df), df.index.min(), df.index.max())
    return df


# ===========================================================================
# SENSITIVITY SCENARIO LOADER
# ===========================================================================

def load_sensitivity_scenarios(
    filename: str = "timeseries_RO_Sensitivity_Spot_EUR_MWh_H_Scenario.csv",
) -> pd.DataFrame:
    """
    Load EQ price sensitivity scenarios.
    Format: multi-level header (3 rows), date index.
    Returns DataFrame with columns named by MW shift (e.g., '-1400MW', '+100MW').
    """
    path = _resolve_file(filename)
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0, parse_dates=True)

    # Flatten multi-level columns to just the MW shift level (row 3)
    df.columns = [str(c[2]).strip() if isinstance(c, tuple) else str(c) for c in df.columns]
    df.index.name = "timestamp"
    df.index = pd.to_datetime(df.index, utc=True)
    df = _to_bucharest_tz(df)
    df = df.sort_index()

    logger.info("Loaded sensitivity scenarios: %d rows × %d scenarios, %s — %s",
                len(df), len(df.columns), df.index.min(), df.index.max())
    return df


# ===========================================================================
# CONVENIENCE: Load any named dataset from the registry
# ===========================================================================

_LOADER_MAP = {
    "dam_prices_entsoe": lambda: load_entsoe_csv("RO_day_ahead_prices_ENTSOE.csv"),
    "spot_backcast_montel": lambda: load_entsoe_csv(
        "RO_Price_Spot_EUR_MWh_Power2Sim_H_Backcast_MONTEL.csv"),
    "imbalance_prices_entsoe": lambda: load_entsoe_csv("RO_imbalance_prices_ENTSOE.csv"),
    "imbalance_volumes_entsoe": lambda: load_entsoe_csv("RO_imbalance_volumes_ENTSOE.csv"),
    "generation_entsoe": lambda: load_entsoe_csv("RO_generation_ENTSOE.csv"),
    "generation_forecast_entsoe": lambda: load_entsoe_csv("RO_generation_forecast_ENTSOE.csv"),
    "load_entsoe": lambda: load_entsoe_csv("RO_load_ENTSOE.csv"),
    "load_forecast_entsoe": lambda: load_entsoe_csv("RO_load_forecast_ENTSOE.csv"),
    "load_and_forecast_entsoe": lambda: load_entsoe_csv("RO_load_and_forecast_ENTSOE.csv"),
    "wind_solar_forecast_entsoe": lambda: load_entsoe_csv("RO_wind_and_solar_forecast_ENTSOE.csv"),
    "intraday_wind_solar_forecast_entsoe": lambda: load_entsoe_csv(
        "RO_intraday_wind_and_solar_forecast_ENTSOE.csv"),
    "imports_entsoe": lambda: load_entsoe_csv("RO_import_ENTSOE.csv"),
    "generation_import_entsoe": lambda: load_entsoe_csv("RO_generation_import_ENTSOE.csv"),
    "imbalance_prices_bal_serv": lambda: load_balancing_services_csv(
        "RO_imbalance_prices_NA_BAL_SERV.csv"),
    "imbalance_totalvolumes_bal_serv": lambda: load_balancing_services_csv(
        "RO_imbalance_totalvolumes_NA_BAL_SERV.csv"),
    "idm_nexte": load_idm_nexte,
    "aurora_forecast": load_aurora_forecast,
    "fx_eur_ron": load_fx_eur_ron,
    "residual_load_montel": lambda: load_montel_backcast(
        "RO_Residual_Load_MWh_h_15min_Backcast_MONTEL.csv",
        value_col_name="residual_load_mwh"),
    "imbalance_volume_montel": lambda: load_montel_backcast(
        "RO_Volume_Imbalance_Net_MWh_15min_Actual_MONTEL.csv",
        value_col_name="net_imbalance_mwh"),
    "sensitivity_scenarios": load_sensitivity_scenarios,
}


def load_dataset(name: str) -> pd.DataFrame:
    """Load a named dataset from the pre-loaded file registry."""
    if name not in _LOADER_MAP:
        raise ValueError(
            f"Unknown dataset: '{name}'. Available: {sorted(_LOADER_MAP.keys())}"
        )
    return _LOADER_MAP[name]()


def list_datasets() -> list:
    """Return list of available dataset names."""
    return sorted(_LOADER_MAP.keys())

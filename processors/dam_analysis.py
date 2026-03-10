"""
DAM Price Analysis Processor (Module 1).

Processes day-ahead market prices from ENTSO-E / EQ data:
  - Base / Peak / Off-Peak decomposition
  - Monthly weighted average prices (per ANRE Order 15/2022)
  - Percentile distributions (P10, P25, P50, P75, P90)
  - Daily and weekly aggregations
  - Trailing averages (6M, 12M)
  - Volume-weighted statistics (where volume data available)

Input:  RO_day_ahead_prices_ENTSOE.csv (108K rows, 2015–2026)
Output: dam_monthly_summary.csv, dam_hourly_latest.csv
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


def classify_hour_block(hour: int) -> str:
    """Classify an hour into Base/Peak/Off-Peak."""
    if hour in settings.peak_hours:
        return "Peak"
    else:
        return "Off-Peak"


def add_time_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived time dimension columns to a time-indexed DataFrame."""
    df = df.copy()
    df["hour"] = df.index.hour
    df["date"] = df.index.date
    df["month"] = df.index.to_period("M")
    df["quarter"] = df.index.to_period("Q")
    df["year"] = df.index.year
    df["weekday"] = df.index.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6])
    df["block"] = df["hour"].apply(classify_hour_block)
    return df


def compute_daily_averages(
    dam: pd.DataFrame,
    price_col: str = "Value [EUR/MWh]",
) -> pd.DataFrame:
    """
    Compute daily Base, Peak, and Off-Peak average prices.

    Returns
    -------
    DataFrame indexed by date with columns: base, peak, offpeak.
    """
    df = add_time_dimensions(dam)

    daily_base = df.groupby("date")[price_col].mean().rename("base_eur_mwh")
    daily_peak = (
        df[df["block"] == "Peak"]
        .groupby("date")[price_col]
        .mean()
        .rename("peak_eur_mwh")
    )
    daily_offpeak = (
        df[df["block"] == "Off-Peak"]
        .groupby("date")[price_col]
        .mean()
        .rename("offpeak_eur_mwh")
    )

    result = pd.concat([daily_base, daily_peak, daily_offpeak], axis=1)
    result.index = pd.DatetimeIndex(result.index)
    result.index.name = "date"
    result["peak_offpeak_spread"] = result["peak_eur_mwh"] - result["offpeak_eur_mwh"]

    logger.info("Daily DAM averages: %d days computed", len(result))
    return result


def compute_monthly_summary(
    dam: pd.DataFrame,
    price_col: str = "Value [EUR/MWh]",
) -> pd.DataFrame:
    """
    Compute comprehensive monthly DAM statistics.

    Returns
    -------
    DataFrame indexed by month with:
      - base_avg, peak_avg, offpeak_avg
      - p10, p25, p50, p75, p90
      - min, max, std
      - count (number of intervals)
      - peak_offpeak_spread
    """
    df = add_time_dimensions(dam)

    # Monthly base average
    monthly_base = df.groupby("month")[price_col].agg(
        base_avg="mean",
        p10=lambda x: np.nanpercentile(x, 10),
        p25=lambda x: np.nanpercentile(x, 25),
        p50="median",
        p75=lambda x: np.nanpercentile(x, 75),
        p90=lambda x: np.nanpercentile(x, 90),
        min_price="min",
        max_price="max",
        std_dev="std",
        count="count",
    )

    # Monthly peak average
    monthly_peak = (
        df[df["block"] == "Peak"]
        .groupby("month")[price_col]
        .mean()
        .rename("peak_avg")
    )

    # Monthly off-peak average
    monthly_offpeak = (
        df[df["block"] == "Off-Peak"]
        .groupby("month")[price_col]
        .mean()
        .rename("offpeak_avg")
    )

    result = pd.concat([monthly_base, monthly_peak, monthly_offpeak], axis=1)
    result["peak_offpeak_spread"] = result["peak_avg"] - result["offpeak_avg"]
    result.index = result.index.to_timestamp()
    result.index.name = "month_start"

    logger.info("Monthly DAM summary: %d months computed", len(result))
    return result


def compute_trailing_averages(
    monthly: pd.DataFrame,
    windows: Tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    """
    Compute trailing N-month rolling averages for base price.

    Parameters
    ----------
    monthly : DataFrame with 'base_avg' column, monthly index.
    windows : tuple of ints, rolling window sizes in months.

    Returns
    -------
    DataFrame with additional columns: trailing_3m, trailing_6m, trailing_12m.
    """
    result = monthly.copy()
    for w in windows:
        result[f"trailing_{w}m"] = result["base_avg"].rolling(w, min_periods=1).mean()
    return result


def compute_percentile_bands(
    dam: pd.DataFrame,
    price_col: str = "Value [EUR/MWh]",
    lookback_months: int = 12,
) -> pd.DataFrame:
    """
    Compute P10/P50/P90 bands on a rolling 12-month basis.
    Useful for scenario calibration (Section 10.1).
    """
    df = add_time_dimensions(dam)

    monthly_stats = df.groupby("month")[price_col].agg(
        p10=lambda x: np.nanpercentile(x, 10),
        p50="median",
        p90=lambda x: np.nanpercentile(x, 90),
    )
    monthly_stats.index = monthly_stats.index.to_timestamp()

    # Rolling lookback
    for col in ["p10", "p50", "p90"]:
        monthly_stats[f"rolling_{col}"] = (
            monthly_stats[col].rolling(lookback_months, min_periods=3).mean()
        )

    return monthly_stats


def compute_hourly_profile(
    dam: pd.DataFrame,
    price_col: str = "Value [EUR/MWh]",
    period: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute average price by hour of day.

    Parameters
    ----------
    period : str, optional
        Filter to specific period (e.g., '2025-01'). None = full dataset.

    Returns
    -------
    DataFrame with hourly average price profile.
    """
    df = add_time_dimensions(dam)
    if period:
        df = df[df["month"].astype(str) == period]

    hourly = df.groupby("hour")[price_col].agg(
        avg="mean", median="median", std="std",
        p10=lambda x: np.nanpercentile(x, 10),
        p90=lambda x: np.nanpercentile(x, 90),
    )
    hourly.index.name = "hour"
    return hourly


def run_dam_analysis(
    dam: pd.DataFrame,
    price_col: str = "Value [EUR/MWh]",
) -> dict:
    """
    Execute full DAM price analysis pipeline.

    Returns
    -------
    Dict of DataFrames:
      - 'daily': daily Base/Peak/Off-Peak
      - 'monthly': monthly summary with percentiles
      - 'trailing': monthly with rolling averages
      - 'percentile_bands': rolling P10/P50/P90
      - 'hourly_profile': average price by hour
    """
    daily = compute_daily_averages(dam, price_col)
    monthly = compute_monthly_summary(dam, price_col)
    trailing = compute_trailing_averages(monthly)
    bands = compute_percentile_bands(dam, price_col)
    profile = compute_hourly_profile(dam, price_col)

    logger.info("DAM analysis pipeline complete: %d months, %d days",
                len(monthly), len(daily))

    return {
        "daily": daily,
        "monthly": monthly,
        "trailing": trailing,
        "percentile_bands": bands,
        "hourly_profile": profile,
    }

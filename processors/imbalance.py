"""
Imbalance Cost Processor (Module 5).

Processes three overlapping imbalance data sources:
  1. ENTSO-E: imbalance prices (Long/Short) + volumes (2015–2026)
  2. Balancing Services API: prices (positive/negative, RON) + volumes (Jul 2024+)
  3. Montel: net imbalance volume (15-min, to Dec 2025)

Computes:
  - Imbalance cost adder (EUR/MWh) for the price build-up
  - P50 and P90 imbalance cost for well-managed vs. poorly-managed portfolios
  - Monthly Long/Short price spread to DAM
  - Seasonal and hourly imbalance cost profiles
  - Rolling 30-day imbalance cost tracker

Reference: Prompt Section 6 (Module 5).
Typical well-managed portfolio adder: 1.5–4.0 EUR/MWh.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


def compute_imbalance_spread_to_dam(
    imb_prices: pd.DataFrame,
    dam_prices: pd.DataFrame,
    dam_col: str = "Value [EUR/MWh]",
) -> pd.DataFrame:
    """
    Calculate imbalance price spread relative to DAM.

    Long spread = DAM - Long_Price (cost of being long/surplus)
    Short spread = Short_Price - DAM (cost of being short/deficit)

    For a supplier, being short is more expensive (buys at Short price).
    Being long means selling surplus at Long price (usually below DAM).
    """
    long_col = [c for c in imb_prices.columns if "Long" in c or "long" in c.lower()]
    short_col = [c for c in imb_prices.columns if "Short" in c or "short" in c.lower()]

    if not long_col or not short_col:
        logger.error("Long/Short columns not found in imbalance prices")
        return pd.DataFrame()

    long_col = long_col[0]
    short_col = short_col[0]

    # Align with DAM (resample DAM to match imbalance resolution if needed)
    dam_aligned = dam_prices[dam_col].reindex(imb_prices.index, method="ffill")

    result = pd.DataFrame(index=imb_prices.index)
    result["dam_price"] = dam_aligned
    result["long_price"] = imb_prices[long_col]
    result["short_price"] = imb_prices[short_col]
    result["long_spread"] = result["dam_price"] - result["long_price"]
    result["short_spread"] = result["short_price"] - result["dam_price"]
    result["long_short_spread"] = result["short_price"] - result["long_price"]

    result = result.dropna(subset=["dam_price", "long_price", "short_price"])

    logger.info("Imbalance-DAM spread: %d intervals, mean L/S spread: %.2f EUR/MWh",
                len(result), result["long_short_spread"].mean())
    return result


def compute_imbalance_cost_adder(
    imb_spread: pd.DataFrame,
    imbalance_rate: float = 0.05,
    direction_bias: str = "neutral",
) -> pd.DataFrame:
    """
    Estimate imbalance cost adder (EUR/MWh) for inclusion in final price.

    Formula (Section 6.2):
      Imbalance_Cost = Expected_Imbalance_Rate × Average_Imbalance_Price_Spread

    Parameters
    ----------
    imb_spread : DataFrame
        Output of compute_imbalance_spread_to_dam()
    imbalance_rate : float
        Expected imbalance as fraction of delivered volume (default 5%)
    direction_bias : str
        'neutral' (equal probability), 'short_bias' (70/30 short), 'long_bias' (70/30 long)
    """
    if imb_spread.empty:
        return pd.DataFrame()

    imb_spread["month"] = imb_spread.index.to_period("M")

    if direction_bias == "neutral":
        short_weight = 0.5
    elif direction_bias == "short_bias":
        short_weight = 0.7
    else:
        short_weight = 0.3

    monthly = imb_spread.groupby("month").agg(
        avg_long_spread=("long_spread", "mean"),
        avg_short_spread=("short_spread", "mean"),
        p50_long_spread=("long_spread", "median"),
        p50_short_spread=("short_spread", "median"),
        p90_short_spread=("short_spread", lambda x: np.nanpercentile(x, 90)),
        avg_ls_spread=("long_short_spread", "mean"),
    )

    # Weighted imbalance cost
    monthly["imbalance_cost_p50_eur_mwh"] = imbalance_rate * (
        short_weight * monthly["p50_short_spread"]
        + (1 - short_weight) * monthly["p50_long_spread"]
    )

    monthly["imbalance_cost_p90_eur_mwh"] = imbalance_rate * (
        short_weight * monthly["p90_short_spread"]
        + (1 - short_weight) * monthly["avg_long_spread"]
    )

    monthly["imbalance_cost_avg_eur_mwh"] = imbalance_rate * (
        short_weight * monthly["avg_short_spread"]
        + (1 - short_weight) * monthly["avg_long_spread"]
    )

    monthly.index = monthly.index.to_timestamp()
    monthly.index.name = "month_start"

    logger.info("Monthly imbalance cost adder: %d months, P50 range: %.2f–%.2f EUR/MWh",
                len(monthly),
                monthly["imbalance_cost_p50_eur_mwh"].min(),
                monthly["imbalance_cost_p50_eur_mwh"].max())
    return monthly


def compute_hourly_imbalance_profile(
    imb_spread: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute average imbalance cost by hour of day.
    Useful for identifying peak imbalance risk hours.
    """
    if imb_spread.empty:
        return pd.DataFrame()

    df = imb_spread.copy()
    df["hour"] = df.index.hour

    profile = df.groupby("hour").agg(
        avg_short_spread=("short_spread", "mean"),
        avg_long_spread=("long_spread", "mean"),
        avg_ls_spread=("long_short_spread", "mean"),
        std_ls_spread=("long_short_spread", "std"),
    )
    profile.index.name = "hour"
    return profile


def compute_rolling_imbalance_cost(
    imb_spread: pd.DataFrame,
    window_days: int = 30,
    imbalance_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Rolling N-day imbalance cost tracker for Streamlit Layer 3.
    """
    if imb_spread.empty:
        return pd.DataFrame()

    daily_spread = imb_spread.resample("D").agg(
        avg_short_spread=("short_spread", "mean"),
        avg_long_spread=("long_spread", "mean"),
        avg_ls_spread=("long_short_spread", "mean"),
    )

    rolling = daily_spread.rolling(window_days, min_periods=7).mean()
    rolling["rolling_imbalance_cost_eur_mwh"] = (
        imbalance_rate * 0.5 * (rolling["avg_short_spread"] + rolling["avg_long_spread"])
    )

    logger.info("Rolling %d-day imbalance cost: %d days", window_days, len(rolling))
    return rolling


def run_imbalance_analysis(
    imb_prices: pd.DataFrame,
    dam_prices: pd.DataFrame,
    imb_volumes: Optional[pd.DataFrame] = None,
    dam_col: str = "Value [EUR/MWh]",
    imbalance_rate: float = 0.05,
) -> dict:
    """
    Execute full imbalance analysis pipeline.

    Returns dict:
      - 'spread': per-interval imbalance-DAM spread
      - 'monthly_cost': monthly imbalance cost adder (P50/P90)
      - 'hourly_profile': average cost by hour
      - 'rolling_30d': rolling 30-day tracker
    """
    spread = compute_imbalance_spread_to_dam(imb_prices, dam_prices, dam_col)
    monthly_cost = compute_imbalance_cost_adder(spread, imbalance_rate)
    hourly_profile = compute_hourly_imbalance_profile(spread)
    rolling = compute_rolling_imbalance_cost(spread)

    logger.info("Imbalance analysis pipeline complete")
    return {
        "spread": spread,
        "monthly_cost": monthly_cost,
        "hourly_profile": hourly_profile,
        "rolling_30d": rolling,
    }

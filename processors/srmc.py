"""
Short-Run Marginal Cost (SRMC) Processor (Module 2 → Module 3).

Calculates marginal generation costs for price formation analysis:
  - Gas-SRMC (CCGT / OCGT) = (Gas_Price / Efficiency) + (CO2_Price × CO2_Intensity)
  - Coal-SRMC (Hard coal / Lignite) = (Coal_Price / Efficiency) + (CO2_Price × CO2_Intensity)
  - Clean Spark Spread = Power_Price - Gas_SRMC
  - Clean Dark Spread = Power_Price - Coal_SRMC

Reference: Prompt Section 3 (Module 2), Addendum E.6.
All efficiency and emission factor assumptions from config/assumptions.yaml.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

# API-2 coal energy content: 6,000 kcal/kg = 25.12 GJ/t = 6.978 MWh_th/t
COAL_API2_MWH_TH_PER_TONNE = 6.978


def compute_gas_srmc(
    gas_price_eur_mwh_th: pd.Series,
    co2_price_eur_t: pd.Series,
    efficiency: Optional[float] = None,
    co2_intensity: Optional[float] = None,
) -> pd.DataFrame:
    """
    Calculate Gas-SRMC (EUR/MWh_e).

    Formula: SRMC = (Gas / η) + (CO2_price × CO2_intensity_gas)

    Parameters
    ----------
    gas_price_eur_mwh_th : Series
        TTF gas price in EUR/MWh thermal
    co2_price_eur_t : Series
        EUA carbon price in EUR/tCO2
    efficiency : float, optional
        Net thermal efficiency (default: CCGT 55%)
    co2_intensity : float, optional
        tCO2/MWh_th (default: 0.37 for natural gas)

    Returns
    -------
    DataFrame with srmc_ccgt, srmc_ocgt columns.
    """
    ccgt_eff = efficiency or settings.ccgt_efficiency
    ocgt_eff = settings.ocgt_efficiency
    co2_int = co2_intensity or settings.gas_co2_intensity

    # Align series on common dates
    aligned = pd.concat([gas_price_eur_mwh_th, co2_price_eur_t], axis=1, join="inner")
    aligned.columns = ["gas_price", "co2_price"]

    result = pd.DataFrame(index=aligned.index)
    result["srmc_gas_ccgt_eur_mwh"] = (
        aligned["gas_price"] / ccgt_eff
        + aligned["co2_price"] * co2_int
    )
    result["srmc_gas_ocgt_eur_mwh"] = (
        aligned["gas_price"] / ocgt_eff
        + aligned["co2_price"] * co2_int
    )
    result["gas_price_eur_mwh_th"] = aligned["gas_price"]
    result["co2_price_eur_t"] = aligned["co2_price"]

    logger.info("Gas-SRMC computed: %d data points, CCGT latest: %.2f EUR/MWh",
                len(result),
                result["srmc_gas_ccgt_eur_mwh"].iloc[-1] if len(result) > 0 else 0)
    return result


def compute_coal_srmc(
    coal_price_usd_t: pd.Series,
    co2_price_eur_t: pd.Series,
    usd_eur_rate: pd.Series,
    coal_type: str = "hard_coal",
) -> pd.DataFrame:
    """
    Calculate Coal-SRMC (EUR/MWh_e).

    Formula: SRMC = (Coal_EUR_per_MWh_th / η) + (CO2_price × CO2_intensity_coal)

    Parameters
    ----------
    coal_price_usd_t : Series
        API-2 coal price in USD/tonne
    co2_price_eur_t : Series
        EUA carbon price in EUR/tCO2
    usd_eur_rate : Series
        USD per EUR exchange rate
    coal_type : str
        'hard_coal' or 'lignite' (different efficiency and CO2 intensity)
    """
    if coal_type == "hard_coal":
        eff = settings.hard_coal_efficiency
        co2_int = settings.hard_coal_co2_intensity
    else:
        eff = settings.lignite_efficiency
        co2_int = settings.lignite_co2_intensity

    # Align all three series
    aligned = pd.concat([coal_price_usd_t, co2_price_eur_t, usd_eur_rate],
                         axis=1, join="inner")
    aligned.columns = ["coal_usd_t", "co2_price", "usd_eur"]

    # Convert coal from USD/t to EUR/MWh_th
    aligned["coal_eur_t"] = aligned["coal_usd_t"] * aligned["usd_eur"]
    aligned["coal_eur_mwh_th"] = aligned["coal_eur_t"] / COAL_API2_MWH_TH_PER_TONNE

    result = pd.DataFrame(index=aligned.index)
    result[f"srmc_{coal_type}_eur_mwh"] = (
        aligned["coal_eur_mwh_th"] / eff
        + aligned["co2_price"] * co2_int
    )
    result["coal_eur_mwh_th"] = aligned["coal_eur_mwh_th"]
    result["co2_price_eur_t"] = aligned["co2_price"]

    logger.info("Coal-SRMC (%s) computed: %d data points, latest: %.2f EUR/MWh",
                coal_type, len(result),
                result[f"srmc_{coal_type}_eur_mwh"].iloc[-1] if len(result) > 0 else 0)
    return result


def compute_clean_spreads(
    power_price_eur_mwh: pd.Series,
    gas_srmc: pd.DataFrame,
    coal_srmc: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Calculate clean spark spread and clean dark spread.

    Clean Spark Spread = Power − Gas_SRMC_CCGT
    Clean Dark Spread  = Power − Coal_SRMC

    Returns
    -------
    DataFrame with spread columns aligned to the shortest common date range.
    """
    result = pd.DataFrame(index=power_price_eur_mwh.index)
    result["power_price_eur_mwh"] = power_price_eur_mwh

    # Join gas SRMC
    if "srmc_gas_ccgt_eur_mwh" in gas_srmc.columns:
        joined = result.join(gas_srmc["srmc_gas_ccgt_eur_mwh"], how="inner")
        result = result.reindex(joined.index)
        result["srmc_gas_ccgt_eur_mwh"] = joined["srmc_gas_ccgt_eur_mwh"]
        result["clean_spark_spread"] = result["power_price_eur_mwh"] - result["srmc_gas_ccgt_eur_mwh"]

    # Join coal SRMC
    if coal_srmc is not None:
        coal_col = [c for c in coal_srmc.columns if c.startswith("srmc_")]
        if coal_col:
            joined = result.join(coal_srmc[coal_col[0]], how="inner")
            result = result.reindex(joined.index)
            result[coal_col[0]] = joined[coal_col[0]]
            result["clean_dark_spread"] = result["power_price_eur_mwh"] - result[coal_col[0]]

    logger.info("Clean spreads computed: %d data points", len(result))
    return result


def compute_srmc_daily(
    gas_srmc: pd.DataFrame,
    coal_srmc: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Produce the daily SRMC summary for Layer 2 export.
    Columns: date, srmc_gas_ccgt, srmc_gas_ocgt, srmc_hard_coal, srmc_lignite.
    """
    result = gas_srmc[["srmc_gas_ccgt_eur_mwh", "srmc_gas_ocgt_eur_mwh",
                        "gas_price_eur_mwh_th", "co2_price_eur_t"]].copy()

    if coal_srmc is not None:
        coal_cols = [c for c in coal_srmc.columns if c.startswith("srmc_")]
        for c in coal_cols:
            result = result.join(coal_srmc[c], how="outer")

    result.index.name = "date"
    logger.info("SRMC daily summary: %d days", len(result))
    return result

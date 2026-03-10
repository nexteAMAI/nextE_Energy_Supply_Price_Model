"""
Sensitivity Analysis Processor (Module 6).

Processes EQ price sensitivity scenario data (20 scenarios × hourly × Jul 2021–Dec 2025):
  - Marginal price elasticity to load shifts (±100MW to ±1400MW)
  - Tornado chart inputs for final price sensitivity
  - Model parameter sensitivity (±TTF, ±EUA, ±RES, ±FX, ±hedge ratio)

Input:  timeseries_RO_Sensitivity_Spot_EUR_MWh_H_Scenario.csv
Output: streamlit_sensitivity.parquet, sensitivity tables for Layer 2

Reference: Prompt Sections 7.3, 14.2.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


def compute_price_elasticity(
    scenarios: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate marginal price elasticity (EUR/MWh per MW) from scenario data.

    The scenario file columns represent price changes (EUR/MWh) from DAM baseline
    for each MW shift level (-1400MW to +1400MW).

    Elasticity = ΔPrice / ΔLoad (EUR/MWh per MW)
    """
    if scenarios.empty:
        return pd.DataFrame()

    # Parse MW shift values from column names
    mw_shifts = []
    for col in scenarios.columns:
        try:
            mw = float(col.replace("MW", "").replace("+", ""))
            mw_shifts.append((col, mw))
        except (ValueError, AttributeError):
            continue

    if not mw_shifts:
        logger.error("Could not parse MW shift values from scenario columns: %s",
                     list(scenarios.columns))
        return pd.DataFrame()

    mw_shifts.sort(key=lambda x: x[1])

    # Average price change per MW shift
    avg_changes = []
    for col, mw in mw_shifts:
        avg_change = scenarios[col].mean()
        median_change = scenarios[col].median()
        p10_change = scenarios[col].quantile(0.10)
        p90_change = scenarios[col].quantile(0.90)
        avg_changes.append({
            "mw_shift": mw,
            "avg_price_change_eur_mwh": avg_change,
            "median_price_change": median_change,
            "p10_price_change": p10_change,
            "p90_price_change": p90_change,
        })

    elasticity = pd.DataFrame(avg_changes)
    elasticity = elasticity.set_index("mw_shift")

    # Marginal elasticity (finite difference)
    elasticity["marginal_elasticity_eur_per_mw"] = (
        elasticity["avg_price_change_eur_mwh"].diff()
        / elasticity.index.to_series().diff()
    )

    logger.info("Price elasticity computed: %d shift levels, "
                "avg elasticity at +500MW: %.4f EUR/MWh per MW",
                len(elasticity),
                elasticity.loc[500, "marginal_elasticity_eur_per_mw"]
                if 500 in elasticity.index else 0)
    return elasticity


def compute_seasonal_elasticity(
    scenarios: pd.DataFrame,
    shift_mw: float = 500,
) -> pd.DataFrame:
    """
    Compute seasonal variation in price sensitivity for a given load shift.

    Returns monthly average price impact for the selected MW shift.
    """
    shift_col = [c for c in scenarios.columns if str(int(shift_mw)) in c.replace("-", "")]
    if not shift_col:
        logger.warning("Shift column for %d MW not found", shift_mw)
        return pd.DataFrame()

    col = shift_col[0]
    df = scenarios[[col]].copy()
    df.columns = ["price_impact"]
    df["month"] = df.index.month
    df["hour"] = df.index.hour

    seasonal = df.groupby("month")["price_impact"].agg(
        avg="mean", std="std",
        p10=lambda x: np.nanpercentile(x, 10),
        p90=lambda x: np.nanpercentile(x, 90),
    )
    seasonal.index.name = "month"

    return seasonal


def build_tornado_inputs(
    base_price_eur_mwh: float,
    assumptions: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Build tornado chart sensitivity inputs per Section 14.2.

    Sensitivities:
      - ±10 EUR/MWh TTF gas price
      - ±5 EUR/t EUA price
      - ±10% RES production
      - ±5% procurement weight shifts
      - ±0.10 EUR/RON exchange rate
      - ±10% bilateral PPA volume

    Returns DataFrame with variable, low_change, high_change, low_impact, high_impact.
    """
    if assumptions is None:
        assumptions = {}

    # Default sensitivities from Section 14.2
    sensitivities = [
        {
            "variable": "TTF Gas Price",
            "unit": "EUR/MWh",
            "low_shock": -10,
            "high_shock": +10,
            "transmission_rate": assumptions.get("gas_weight", 0.20) / settings.ccgt_efficiency,
        },
        {
            "variable": "EUA Carbon Price",
            "unit": "EUR/t",
            "low_shock": -5,
            "high_shock": +5,
            "transmission_rate": assumptions.get("gas_weight", 0.20) * settings.gas_co2_intensity,
        },
        {
            "variable": "RES Production",
            "unit": "%",
            "low_shock": -10,
            "high_shock": +10,
            "transmission_rate": assumptions.get("res_price_impact_per_pct", 0.5),
        },
        {
            "variable": "Procurement Weight Shift",
            "unit": "%",
            "low_shock": -5,
            "high_shock": +5,
            "transmission_rate": assumptions.get("weight_sensitivity", 0.3),
        },
        {
            "variable": "EUR/RON FX Rate",
            "unit": "EUR/RON",
            "low_shock": -0.10,
            "high_shock": +0.10,
            "transmission_rate": assumptions.get("fx_sensitivity", 1.0),
        },
        {
            "variable": "Bilateral PPA Volume",
            "unit": "%",
            "low_shock": -10,
            "high_shock": +10,
            "transmission_rate": assumptions.get("ppa_hedge_sensitivity", 0.2),
        },
    ]

    results = []
    for s in sensitivities:
        low_impact = s["low_shock"] * s["transmission_rate"]
        high_impact = s["high_shock"] * s["transmission_rate"]
        results.append({
            "variable": s["variable"],
            "unit": s["unit"],
            "low_shock": s["low_shock"],
            "high_shock": s["high_shock"],
            "low_impact_eur_mwh": low_impact,
            "high_impact_eur_mwh": high_impact,
            "low_total_price": base_price_eur_mwh + low_impact,
            "high_total_price": base_price_eur_mwh + high_impact,
            "impact_range": abs(high_impact - low_impact),
        })

    tornado = pd.DataFrame(results)
    tornado = tornado.sort_values("impact_range", ascending=True)  # For tornado chart

    logger.info("Tornado chart inputs: %d variables, base price: %.2f EUR/MWh",
                len(tornado), base_price_eur_mwh)
    return tornado


def run_sensitivity_analysis(
    scenarios: pd.DataFrame,
    base_price_eur_mwh: float = 120.0,
) -> dict:
    """
    Execute full sensitivity analysis pipeline.

    Returns dict:
      - 'elasticity': price elasticity by MW shift
      - 'seasonal_500mw': seasonal variation at ±500MW
      - 'tornado': tornado chart inputs
    """
    elasticity = compute_price_elasticity(scenarios)
    seasonal = compute_seasonal_elasticity(scenarios, shift_mw=500)
    tornado = build_tornado_inputs(base_price_eur_mwh)

    logger.info("Sensitivity analysis complete")
    return {
        "elasticity": elasticity,
        "seasonal_500mw": seasonal,
        "tornado": tornado,
    }

"""
Merit Order & Generation Stack Processor (Module 3).

Computes:
  - Residual demand = Total_Load - Wind - Solar - Nuclear - Must_Run_Hydro
  - Marginal price regime identification (hydro/nuclear, gas CCGT, gas OCGT/coal, extreme)
  - Monthly generation mix summary by fuel type
  - Capacity factor analysis by technology

Input:  RO_generation_ENTSOE.csv, RO_load_ENTSOE.csv, RO_wind_and_solar_forecast_ENTSOE.csv
Output: generation_monthly.csv (Layer 2), streamlit_generation_stack.parquet (Layer 3)

Reference: Prompt Section 4 (Module 3), generation mix shares from ANRE 2024.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

# Fuel type column mapping for RO_generation_ENTSOE.csv
FUEL_TYPES = [
    "Biomass [MW]",
    "Fossil Brown coal/Lignite [MW]",
    "Fossil Gas [MW]",
    "Fossil Hard coal [MW]",
    "Hydro Run-of-river and poundage [MW]",
    "Hydro Water Reservoir [MW]",
    "Nuclear [MW]",
    "Solar [MW]",
    "Wind Onshore [MW]",
    "Energy storage [MW]",
]

# Short names for outputs
FUEL_SHORT = {
    "Biomass [MW]": "biomass",
    "Fossil Brown coal/Lignite [MW]": "lignite",
    "Fossil Gas [MW]": "gas",
    "Fossil Hard coal [MW]": "hard_coal",
    "Hydro Run-of-river and poundage [MW]": "hydro_ror",
    "Hydro Water Reservoir [MW]": "hydro_reservoir",
    "Nuclear [MW]": "nuclear",
    "Solar [MW]": "solar",
    "Wind Onshore [MW]": "wind",
    "Energy storage [MW]": "storage",
}

# Approximate installed capacities (MW) for capacity factor calculation (2024 reference)
INSTALLED_CAPACITY_MW = {
    "nuclear": 1400,       # Cernavoda U1+U2
    "wind": 3200,          # ~3.2 GW installed
    "solar": 4500,         # ~4.5 GW after 2024 expansion
    "hydro_ror": 3000,     # Run-of-river
    "hydro_reservoir": 3400,  # Reservoir (Hidroelectrica)
    "gas": 2800,           # CCGT + OCGT fleet
    "lignite": 2000,       # CE Oltenia (declining)
    "hard_coal": 600,      # Residual hard coal
    "biomass": 120,
    "storage": 50,         # Early-stage BESS
}


def compute_residual_demand(
    load: pd.DataFrame,
    generation: pd.DataFrame,
    load_col: str = "Actual Load [MW]",
) -> pd.DataFrame:
    """
    Calculate residual demand after subtracting must-run/zero-marginal-cost generation.

    Residual_Demand = Total_Load - Wind - Solar - Nuclear - Must_Run_Hydro(RoR)

    Returns DataFrame with residual demand and its components.
    """
    # Align timestamps
    gen = generation.copy()
    gen_cols = {
        "wind": [c for c in gen.columns if "Wind" in c],
        "solar": [c for c in gen.columns if "Solar" in c],
        "nuclear": [c for c in gen.columns if "Nuclear" in c],
        "hydro_ror": [c for c in gen.columns if "Run-of-river" in c],
    }

    result = pd.DataFrame(index=gen.index)
    result["total_load_mw"] = load[load_col].reindex(gen.index)

    for key, cols in gen_cols.items():
        if cols:
            result[f"{key}_mw"] = gen[cols[0]]
        else:
            result[f"{key}_mw"] = 0

    result["must_run_total_mw"] = (
        result["wind_mw"].fillna(0)
        + result["solar_mw"].fillna(0)
        + result["nuclear_mw"].fillna(0)
        + result["hydro_ror_mw"].fillna(0)
    )

    result["residual_demand_mw"] = result["total_load_mw"] - result["must_run_total_mw"]
    result["residual_demand_mw"] = result["residual_demand_mw"].clip(lower=0)

    logger.info("Residual demand: %d intervals, mean: %.0f MW, max: %.0f MW",
                len(result),
                result["residual_demand_mw"].mean(),
                result["residual_demand_mw"].max())
    return result


def classify_price_regime(
    residual_demand_mw: pd.Series,
    dam_price_eur_mwh: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Classify each interval into marginal price regime.

    Regimes (from Section 4.2):
      - 'hydro_nuclear': Low residual demand → 40–60 EUR/MWh
      - 'gas_ccgt': Moderate residual demand → 80–120 EUR/MWh
      - 'gas_ocgt_coal': High residual demand → 120–250 EUR/MWh
      - 'extreme': Scarcity/constrained → >250 EUR/MWh

    Uses residual demand thresholds calibrated to historical data.
    """
    regimes = settings.get_assumption("generation_stack", "price_regimes")

    # Dynamic thresholds based on residual demand percentiles
    p25 = residual_demand_mw.quantile(0.25)
    p60 = residual_demand_mw.quantile(0.60)
    p85 = residual_demand_mw.quantile(0.85)

    conditions = [
        residual_demand_mw <= p25,
        (residual_demand_mw > p25) & (residual_demand_mw <= p60),
        (residual_demand_mw > p60) & (residual_demand_mw <= p85),
        residual_demand_mw > p85,
    ]
    labels = ["hydro_nuclear", "gas_ccgt", "gas_ocgt_coal", "extreme"]

    return pd.Series(
        np.select(conditions, labels, default="gas_ccgt"),
        index=residual_demand_mw.index,
        name="price_regime",
    )


def compute_generation_mix_monthly(
    generation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Monthly generation summary by fuel type (MW average and MWh total).

    Returns DataFrame indexed by month with columns per fuel type.
    """
    gen = generation.copy()

    # Rename to short names
    rename_cols = {}
    for orig, short in FUEL_SHORT.items():
        if orig in gen.columns:
            rename_cols[orig] = short
    gen = gen.rename(columns=rename_cols)

    fuel_cols = [c for c in gen.columns if c in FUEL_SHORT.values()]

    gen["month"] = gen.index.to_period("M")

    # Monthly averages (MW)
    monthly_avg = gen.groupby("month")[fuel_cols].mean()
    monthly_avg.columns = [f"{c}_avg_mw" for c in monthly_avg.columns]

    # Monthly totals (MWh — approximate from avg MW × hours)
    # Determine hours per interval
    if len(gen) > 1:
        typical_delta = (gen.index[1] - gen.index[0]).total_seconds() / 3600
    else:
        typical_delta = 1.0

    monthly_sum = gen.groupby("month")[fuel_cols].sum() * typical_delta
    monthly_sum.columns = [f"{c}_total_mwh" for c in monthly_sum.columns]

    # Total generation per month
    total_cols = [c for c in monthly_sum.columns]
    monthly_sum["total_generation_mwh"] = monthly_sum[total_cols].sum(axis=1)

    # Share calculation
    for c in fuel_cols:
        monthly_sum[f"{c}_share_pct"] = (
            monthly_sum[f"{c}_total_mwh"] / monthly_sum["total_generation_mwh"] * 100
        )

    result = pd.concat([monthly_avg, monthly_sum], axis=1)
    result.index = result.index.to_timestamp()
    result.index.name = "month_start"

    logger.info("Monthly generation mix: %d months, %d fuel types", len(result), len(fuel_cols))
    return result


def compute_capacity_factors(
    generation: pd.DataFrame,
    period: str = "M",
) -> pd.DataFrame:
    """
    Calculate capacity factors by technology for a given aggregation period.

    CF = Actual_Generation_MWh / (Installed_Capacity_MW × Hours_in_Period)
    """
    gen = generation.copy()
    rename_cols = {}
    for orig, short in FUEL_SHORT.items():
        if orig in gen.columns:
            rename_cols[orig] = short
    gen = gen.rename(columns=rename_cols)

    gen["period"] = gen.index.to_period(period)
    fuel_cols = [c for c in gen.columns if c in FUEL_SHORT.values()]

    # Hours per interval
    if len(gen) > 1:
        typical_delta = (gen.index[1] - gen.index[0]).total_seconds() / 3600
    else:
        typical_delta = 1.0

    period_gen = gen.groupby("period")[fuel_cols].sum() * typical_delta
    hours_per_period = gen.groupby("period").size() * typical_delta

    result = pd.DataFrame(index=period_gen.index)
    for col in fuel_cols:
        installed = INSTALLED_CAPACITY_MW.get(col, None)
        if installed and installed > 0:
            result[f"{col}_cf_pct"] = (
                period_gen[col] / (installed * hours_per_period) * 100
            ).clip(0, 100)

    result.index = result.index.to_timestamp()
    result.index.name = "period_start"

    logger.info("Capacity factors computed for %d periods", len(result))
    return result


def run_merit_order_analysis(
    generation: pd.DataFrame,
    load: pd.DataFrame,
    dam: Optional[pd.DataFrame] = None,
    load_col: str = "Actual Load [MW]",
) -> dict:
    """
    Execute full merit order / generation stack analysis.

    Returns dict:
      - 'residual_demand': per-interval residual demand
      - 'price_regimes': per-interval price regime classification
      - 'monthly_mix': monthly generation summary
      - 'capacity_factors': monthly capacity factors
    """
    residual = compute_residual_demand(load, generation, load_col)
    regimes = classify_price_regime(residual["residual_demand_mw"])

    dam_prices = None
    if dam is not None:
        dam_col = [c for c in dam.columns if "EUR/MWh" in c or "price" in c.lower()]
        if dam_col:
            dam_prices = dam[dam_col[0]].reindex(residual.index)

    residual["price_regime"] = regimes

    monthly_mix = compute_generation_mix_monthly(generation)
    cap_factors = compute_capacity_factors(generation)

    logger.info("Merit order analysis complete")
    return {
        "residual_demand": residual,
        "price_regimes": regimes,
        "monthly_mix": monthly_mix,
        "capacity_factors": cap_factors,
    }

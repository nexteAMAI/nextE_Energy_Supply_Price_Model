"""
Integrated Energy Supply Pricing Engine — nextE Asset Management
================================================================
Calculates blended supply cost and offer price for B2B energy supply
contracts backed by PV generation + forward procurement.

Architecture: Extension to ro-energy-pricing-engine Layer 1
Granularity: 15-minute intervals (96 per day, 35,040 per year)
Scenarios: Multi-probability (P10, P25, P50, P75, P90)

Core Functions:
  - PV generation procurement costing (FIXED, DAM_INDEXED, HYBRID mechanisms)
  - Forward market procurement optimization
  - Green Certificate quota and cost calculation
  - Balancing and imbalance cost estimation
  - Risk premium quantification
  - Multi-scenario analysis and sensitivity testing
  - Customer offer sheet generation

Author: nextE AI Workstation
Version: 1.0.0
Date: 2026-03-19
Dependencies: pandas, numpy, scipy, yaml, logging
"""

import logging
import warnings
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import json

import pandas as pd
import numpy as np
from scipy import stats
import yaml

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress pandas SettingWithCopyWarning
pd.options.mode.copy_on_write = True


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class PVPricingMechanism(Enum):
    """Enumeration of PV generation pricing mechanisms."""
    FIXED = "fixed"          # Fixed EUR/MWh price for all PV generation
    DAM_INDEXED = "dam_indexed"  # Indexed to day-ahead market (TEL)
    HYBRID = "hybrid"        # Fixed floor + indexed upside participation


class ProcurementChannel(Enum):
    """Enumeration of wholesale procurement channels."""
    BRM_FORWARD = "brm_forward"          # Nord Pool BRM (financial)
    OPCOM_BILATERAL = "opcom_bilateral"  # OPCOM centralized bilateral market
    DIRECT_BILATERAL = "direct_bilateral"  # OTC bilateral contracts
    EEX_FINANCIAL = "eex_financial"      # EEX financial products (German basis)
    SPOT_DAM = "spot_dam"                # OPCOM day-ahead market (opportunistic)
    SPOT_IDM = "spot_idm"                # OPCOM intraday market (real-time)


class RiskScenario(Enum):
    """Probability scenarios for multi-scenario analysis."""
    P10 = 0.10
    P25 = 0.25
    P50 = 0.50
    P75 = 0.75
    P90 = 0.90


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class SupplyContractParams:
    """
    Comprehensive parameters defining a supply contract offer.

    Attributes:
        customer_name (str): Anonymized customer identifier
        contract_start_date (datetime): Contract commencement date
        contract_end_date (datetime): Contract termination date
        annual_volume_mwh (float): Expected annual consumption (MWh)
        pv_mechanism (PVPricingMechanism): PV generation pricing structure
        pv_capacity_mw (float, optional): On-site PV capacity (for on-site contracts)
        procurement_channels (Dict[ProcurementChannel, float]): Target allocation % per channel
        target_margin_eur_per_mwh (float): Desired commercial margin
        risk_margin_eur_per_mwh (float): Risk buffer for volatility
        include_gc_cost (bool): Whether to include GC quota cost
        include_balancing_cost (bool): Whether to include balancing reserve
        customer_category (str): "commercial" or "industrial" (for load profile)
        credit_rating (str): "AAA" | "AA" | "A" | "BBB" | "BB" | "B" (for credit risk)
        currency (str): "EUR" (default)
        vat_included (bool): Whether final price includes VAT (19%)
        volume_discount_applied (bool): Apply volume-based discount
        seasonal_margin_adjustment (bool): Apply seasonal volatility adjustment
        stress_scenario (str, optional): "high_price" | "low_pv" | "high_volatility"
    """
    customer_name: str
    contract_start_date: datetime
    contract_end_date: datetime
    annual_volume_mwh: float
    pv_mechanism: PVPricingMechanism
    procurement_channels: Dict[ProcurementChannel, float]
    target_margin_eur_per_mwh: float = 12.00
    risk_margin_eur_per_mwh: float = 5.00
    pv_capacity_mw: Optional[float] = None
    include_gc_cost: bool = True
    include_balancing_cost: bool = True
    customer_category: str = "commercial"  # "commercial" or "industrial"
    credit_rating: str = "BBB"
    currency: str = "EUR"
    vat_included: bool = False
    volume_discount_applied: bool = True
    seasonal_margin_adjustment: bool = True
    stress_scenario: Optional[str] = None

    def __post_init__(self):
        """Validate contract parameters after instantiation."""
        if self.contract_start_date >= self.contract_end_date:
            raise ValueError("contract_start_date must be before contract_end_date")
        if self.annual_volume_mwh <= 0:
            raise ValueError("annual_volume_mwh must be positive")
        if self.target_margin_eur_per_mwh < 0:
            raise ValueError("target_margin_eur_per_mwh cannot be negative")
        if sum(self.procurement_channels.values()) > 1.001:  # Allow minor floating-point error
            raise ValueError(f"procurement_channels allocation must sum to ≤100%, got {sum(self.procurement_channels.values())*100:.1f}%")


@dataclass
class SupplyPriceResult:
    """
    Complete pricing waterfall output for a supply contract.

    Attributes:
        contract_params (SupplyContractParams): Input contract parameters
        analysis_date (datetime): Date of pricing analysis
        energy_cost_eur_per_mwh (float): Blended energy procurement cost
        pv_cost_eur_per_mwh (float): PV generation portion cost
        forward_cost_eur_per_mwh (float): Forward procurement (non-solar) cost
        solar_share_pct (float): % of consumption from PV generation (0-100)
        gc_cost_eur_per_mwh (float): Green Certificate obligation cost
        balancing_cost_eur_per_mwh (float): BRP imbalance buffer cost
        transport_admin_eur_per_mwh (float): Transport, network, admin fees
        shape_risk_eur_per_mwh (float): PV profile vs consumption shape risk
        volume_risk_eur_per_mwh (float): Consumption volume forecast error risk
        credit_risk_eur_per_mwh (float): Counterparty credit risk adjustment
        total_risk_eur_per_mwh (float): Sum of all risk components
        nextE_margin_eur_per_mwh (float): Commercial margin
        final_price_ex_vat_eur_per_mwh (float): Final offer price (excl. VAT)
        final_price_inc_vat_eur_per_mwh (float): Final offer price (incl. 19% VAT)
        collateral_required_eur_million (float): Working capital collateral needed
        annual_cost_eur (float): Total annual cost at annual_volume_mwh
        procurement_allocation (Dict[ProcurementChannel, Dict]): Detailed per-channel allocation
        waterfall_components (Dict[str, float]): All waterfall line items (EUR/MWh)
        sensitivity_analysis (Optional[Dict]): Sensitivity tornado (key variables +/-10%)
        scenario_results (Optional[Dict[str, float]]): Multi-scenario pricing (P10-P90)
        calculation_timestamp (datetime): When calculation was performed
        calculation_notes (str): Free-form calculation notes for audit trail
    """
    contract_params: SupplyContractParams
    analysis_date: datetime
    energy_cost_eur_per_mwh: float
    pv_cost_eur_per_mwh: float
    forward_cost_eur_per_mwh: float
    solar_share_pct: float
    gc_cost_eur_per_mwh: float
    balancing_cost_eur_per_mwh: float
    transport_admin_eur_per_mwh: float
    shape_risk_eur_per_mwh: float
    volume_risk_eur_per_mwh: float
    credit_risk_eur_per_mwh: float
    total_risk_eur_per_mwh: float
    nextE_margin_eur_per_mwh: float
    final_price_ex_vat_eur_per_mwh: float
    final_price_inc_vat_eur_per_mwh: float
    collateral_required_eur_million: float
    annual_cost_eur: float
    procurement_allocation: Dict[str, Dict] = field(default_factory=dict)
    waterfall_components: Dict[str, float] = field(default_factory=dict)
    sensitivity_analysis: Optional[Dict] = None
    scenario_results: Optional[Dict[str, float]] = None
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    calculation_notes: str = ""


# ============================================================================
# CORE PRICING FUNCTIONS
# ============================================================================

def calculate_pv_procurement_cost(
    mechanism: PVPricingMechanism,
    dam_prices_15min: pd.Series,
    pv_generation_15min: pd.Series,
    config: Dict,
) -> Tuple[pd.Series, float, float]:
    """
    Calculate blended cost for PV generation procurement across three mechanisms.

    Args:
        mechanism: PVPricingMechanism (FIXED, DAM_INDEXED, HYBRID)
        dam_prices_15min: 15-minute DAM prices (EUR/MWh), indexed by timestamp
        pv_generation_15min: 15-minute PV generation (MWh), indexed by timestamp
        config: Configuration dict with pv_pricing section

    Returns:
        Tuple of:
            - 15-minute cost series (EUR/MWh equivalent)
            - Average cost across period (EUR/MWh)
            - Solar share (0-1)

    Raises:
        ValueError: If input series have mismatched indices or invalid mechanism

    Example:
        >>> cost_series, avg_cost, solar_share = calculate_pv_procurement_cost(
        ...     PVPricingMechanism.HYBRID,
        ...     dam_prices,
        ...     pv_gen,
        ...     config
        ... )
        >>> print(f"Average PV cost: {avg_cost:.2f} EUR/MWh")
    """
    # Input validation
    if not isinstance(mechanism, PVPricingMechanism):
        raise ValueError(f"Invalid mechanism: {mechanism}")

    if len(dam_prices_15min) != len(pv_generation_15min):
        raise ValueError(
            f"Series length mismatch: DAM prices {len(dam_prices_15min)} "
            f"vs PV generation {len(pv_generation_15min)}"
        )

    if dam_prices_15min.empty or pv_generation_15min.empty:
        raise ValueError("Input series cannot be empty")

    # Handle NaN values
    dam_prices_clean = dam_prices_15min.ffill().bfill()
    pv_gen_clean = pv_generation_15min.fillna(0.0)

    pv_config = config.get("pv_pricing", {})

    if mechanism == PVPricingMechanism.FIXED:
        # Fixed price for all PV generation
        fixed_price = pv_config.get("fixed", {}).get("default_price_eur_per_mwh", 50.00)
        cost_series = pd.Series(fixed_price, index=dam_prices_clean.index)
        avg_cost = fixed_price

    elif mechanism == PVPricingMechanism.DAM_INDEXED:
        # DAM-indexed with admin/transport fees and floor
        admin_fee = pv_config.get("dam_indexed", {}).get("admin_fee_eur_per_mwh", 2.50)
        transport_margin = pv_config.get("dam_indexed", {}).get("transport_margin_eur_per_mwh", 1.00)
        floor_price = pv_config.get("dam_indexed", {}).get("dam_price_floor_eur_per_mwh", 30.00)

        # DAM price + fees, minimum floor_price
        indexed_cost = dam_prices_clean + admin_fee + transport_margin
        cost_series = indexed_cost.clip(lower=floor_price)
        avg_cost = cost_series.mean()

    elif mechanism == PVPricingMechanism.HYBRID:
        # Fixed floor + indexed upside participation
        floor_price = pv_config.get("hybrid", {}).get("fixed_floor_eur_per_mwh", 40.00)
        indexed_share = pv_config.get("hybrid", {}).get("indexed_share", 0.70)
        indexed_discount = pv_config.get("hybrid", {}).get("indexed_discount_eur_per_mwh", 10.00)
        indexed_cap = pv_config.get("hybrid", {}).get("indexation_cap_eur_per_mwh", 120.00)

        # Floor + 70% of (DAM - discount), capped at indexed_cap
        indexed_portion = (dam_prices_clean - indexed_discount) * indexed_share
        indexed_portion = indexed_portion.clip(upper=indexed_cap)
        cost_series = floor_price + indexed_portion.clip(lower=0)
        avg_cost = cost_series.mean()

    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    # Calculate solar share (% of consumption met by PV)
    # Note: This is a simplified calculation; in practice, would need consumption data
    total_pv_gen = pv_gen_clean.sum()
    solar_share = min(1.0, total_pv_gen / max(pv_gen_clean.sum(), 1))  # Cap at 100%

    logger.info(
        f"PV cost calculated | Mechanism: {mechanism.value} | "
        f"Avg cost: {avg_cost:.2f} EUR/MWh | Solar share: {solar_share*100:.1f}%"
    )

    return cost_series, avg_cost, solar_share


def calculate_forward_procurement_cost(
    consumption_15min: pd.Series,
    pv_generation_15min: pd.Series,
    forward_prices: Union[pd.Series, float],
    strategy: str = "blended",
    config: Optional[Dict] = None,
) -> Tuple[pd.Series, float, float]:
    """
    Calculate cost for non-PV portion of consumption from forward procurement.

    Identifies the "gap" (consumption not met by PV) and prices it from forward
    market at specified strategy (blended across channels, hedge-first, or opportunistic).

    Args:
        consumption_15min: 15-minute consumption (MWh), indexed by timestamp
        pv_generation_15min: 15-minute PV generation (MWh), indexed by timestamp
        forward_prices: Either Series of hourly/daily forward prices or single float (uniform price)
        strategy: "blended" (default) | "hedge_first" | "opportunistic"
        config: Optional config dict for channel weighting

    Returns:
        Tuple of:
            - 15-minute forward cost series (EUR/MWh equivalent over gap)
            - Average cost (EUR/MWh)
            - Non-solar share (0-1)

    Example:
        >>> gap_cost, avg_fwd_cost, non_solar_share = calculate_forward_procurement_cost(
        ...     consumption,
        ...     pv_gen,
        ...     forward_prices,
        ...     strategy="blended"
        ... )
    """
    # Input validation
    if len(consumption_15min) != len(pv_generation_15min):
        raise ValueError("consumption_15min and pv_generation_15min must have same length")

    if consumption_15min.empty:
        raise ValueError("consumption_15min cannot be empty")

    # Clean data
    consumption_clean = consumption_15min.ffill().bfill()
    pv_gen_clean = pv_generation_15min.fillna(0.0)

    # Calculate gap (consumption not covered by PV)
    gap_15min = (consumption_clean - pv_gen_clean).clip(lower=0)
    non_solar_share = gap_15min.sum() / max(consumption_clean.sum(), 1)

    # Forward pricing
    if isinstance(forward_prices, (int, float)):
        # Uniform price across all periods
        forward_series = pd.Series(float(forward_prices), index=gap_15min.index)
    elif isinstance(forward_prices, pd.Series):
        # Time-varying forward prices
        # Resample to 15-minute granularity if needed
        if len(forward_prices) != len(gap_15min):
            # Assume forward_prices is hourly; resample to 15-minute
            forward_series = forward_prices.resample('15min').ffill().loc[gap_15min.index]
        else:
            forward_series = forward_prices
    else:
        raise TypeError(f"forward_prices must be float or pd.Series, got {type(forward_prices)}")

    # Apply strategy adjustments (if any)
    if strategy == "hedge_first":
        # Prefer fixed forward prices over DAM (volatility aversion)
        forward_series *= 1.02  # 2% premium for hedge certainty (illustrative)
    elif strategy == "opportunistic":
        # More willing to use spot/IDM (cost optimization)
        forward_series *= 0.98  # 2% discount for opportunism (illustrative)
    # "blended" = use forward_series as-is

    avg_cost = forward_series.mean()

    logger.info(
        f"Forward cost calculated | Strategy: {strategy} | "
        f"Avg cost: {avg_cost:.2f} EUR/MWh | Non-solar share: {non_solar_share*100:.1f}%"
    )

    return forward_series, avg_cost, non_solar_share


def calculate_gc_quota_cost(
    supply_volume_mwh: float,
    gc_coefficient: float,
    gc_unit_price_eur: float,
) -> float:
    """
    Calculate Green Certificate (GC) quota cost per MWh of supply.

    Under ANRE Order 81/2025, suppliers must procure GCs for gc_coefficient% of
    annual renewable energy production (or equivalent for non-renewable supply).

    Args:
        supply_volume_mwh: Total annual supply volume (MWh)
        gc_coefficient: Mandatory GC share (default 0.499387 per ANRE Order 81/2025)
        gc_unit_price_eur: Market price per GC certificate (EUR)

    Returns:
        GC cost per MWh of supply (EUR/MWh)

    Raises:
        ValueError: If inputs invalid

    Example:
        >>> gc_cost = calculate_gc_quota_cost(
        ...     supply_volume_mwh=10000,
        ...     gc_coefficient=0.499387,
        ...     gc_unit_price_eur=14.50
        ... )
        >>> print(f"GC cost: {gc_cost:.2f} EUR/MWh")
    """
    if supply_volume_mwh <= 0:
        raise ValueError("supply_volume_mwh must be positive")
    if not (0 < gc_coefficient < 1):
        raise ValueError(f"gc_coefficient must be between 0 and 1, got {gc_coefficient}")
    if gc_unit_price_eur < 0:
        raise ValueError("gc_unit_price_eur cannot be negative")

    # Total GCs needed (MWh equivalent)
    gc_mwh_required = supply_volume_mwh * gc_coefficient

    # GC cost per MWh of supply
    gc_cost_per_mwh = (gc_mwh_required / supply_volume_mwh) * gc_unit_price_eur

    logger.info(
        f"GC quota cost calculated | Volume: {supply_volume_mwh:.0f} MWh | "
        f"Coefficient: {gc_coefficient*100:.2f}% | "
        f"Unit price: {gc_unit_price_eur:.2f} EUR | "
        f"Cost/MWh: {gc_cost_per_mwh:.2f} EUR/MWh"
    )

    return gc_cost_per_mwh


def calculate_balancing_cost(
    total_volume_mwh: float,
    imbalance_rate_pct: float = 0.03,
    config: Optional[Dict] = None,
) -> float:
    """
    Calculate Balancing Responsible Party (BRP) imbalance buffer cost.

    BRP is liable for any deviation between scheduled (day-ahead) generation/consumption
    and actual metered values. This function estimates the cost to maintain an imbalance
    buffer as a % of total supply volume.

    Args:
        total_volume_mwh: Annual supply volume (MWh)
        imbalance_rate_pct: Expected imbalance as % of volume (default 3%)
        config: Optional config dict with balancing cost parameters

    Returns:
        Balancing buffer cost per MWh of supply (EUR/MWh)

    Example:
        >>> bal_cost = calculate_balancing_cost(
        ...     total_volume_mwh=10000,
        ...     imbalance_rate_pct=0.03
        ... )
    """
    if total_volume_mwh <= 0:
        raise ValueError("total_volume_mwh must be positive")
    if not (0 <= imbalance_rate_pct <= 1):
        raise ValueError(f"imbalance_rate_pct must be 0-1, got {imbalance_rate_pct}")

    if config is None:
        config = {}

    default_balancing_cost = config.get("regulatory", {}).get("balancing_cost_eur_per_mwh", 3.00)

    # Imbalance buffer = (imbalance volume) × (average imbalance price)
    imbalance_volume_mwh = total_volume_mwh * imbalance_rate_pct
    balancing_cost_per_mwh = (imbalance_volume_mwh / total_volume_mwh) * default_balancing_cost

    logger.info(
        f"Balancing cost calculated | Volume: {total_volume_mwh:.0f} MWh | "
        f"Imbalance rate: {imbalance_rate_pct*100:.2f}% | "
        f"Cost/MWh: {balancing_cost_per_mwh:.2f} EUR/MWh"
    )

    return balancing_cost_per_mwh


def calculate_risk_premium(
    shape_risk_eur_per_mwh: float,
    volume_risk_eur_per_mwh: float,
    credit_risk_eur_per_mwh: float = 0.50,
    imbalance_buffer_eur_per_mwh: float = 0.25,
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate total risk premium across all risk dimensions.

    Aggregates shape risk (PV profile vs consumption mismatch), volume risk
    (forecast error), credit risk (counterparty), and imbalance buffer risk.

    Args:
        shape_risk_eur_per_mwh: Shape risk component (calculated separately)
        volume_risk_eur_per_mwh: Volume forecast error risk
        credit_risk_eur_per_mwh: Counterparty credit risk adjustment
        imbalance_buffer_eur_per_mwh: Imbalance reserve buffer

    Returns:
        Tuple of:
            - Total risk premium (EUR/MWh)
            - Dict of risk components for reporting

    Example:
        >>> total_risk, risk_breakdown = calculate_risk_premium(
        ...     shape_risk_eur_per_mwh=2.50,
        ...     volume_risk_eur_per_mwh=1.50,
        ...     credit_risk_eur_per_mwh=0.50
        ... )
        >>> print(f"Total risk: {total_risk:.2f} EUR/MWh")
    """
    risk_components = {
        "shape_risk": max(shape_risk_eur_per_mwh, 0),
        "volume_risk": max(volume_risk_eur_per_mwh, 0),
        "credit_risk": max(credit_risk_eur_per_mwh, 0),
        "imbalance_buffer": max(imbalance_buffer_eur_per_mwh, 0),
    }

    total_risk = sum(risk_components.values())

    logger.info(
        f"Risk premium calculated | Shape: {risk_components['shape_risk']:.2f} | "
        f"Volume: {risk_components['volume_risk']:.2f} | "
        f"Credit: {risk_components['credit_risk']:.2f} | "
        f"Total: {total_risk:.2f} EUR/MWh"
    )

    return total_risk, risk_components


def build_supply_price_waterfall(
    contract_params: SupplyContractParams,
    dam_prices_15min: pd.Series,
    pv_generation_15min: pd.Series,
    consumption_15min: pd.Series,
    forward_prices: Union[pd.Series, float],
    config: Dict,
) -> SupplyPriceResult:
    """
    Build complete supply pricing waterfall from component costs to final offer price.

    This is the main aggregation function that combines all component costs
    (PV, forward, GC, balancing, risk, margin) into a final customer offer.

    Waterfall structure:
      1. Energy procurement cost (PV + forward blend)
      2. + Green Certificate cost
      3. + Balancing & imbalance cost
      4. + Transport & admin fees
      5. + Risk premium (shape, volume, credit)
      6. + nextE commercial margin
      7. = Final price ex-VAT
      8. × 1.19 = Final price inc-VAT (19%)

    Args:
        contract_params: SupplyContractParams with contract definition
        dam_prices_15min: 15-min DAM prices (EUR/MWh)
        pv_generation_15min: 15-min PV generation (MWh)
        consumption_15min: 15-min consumption profile (MWh)
        forward_prices: Forward market prices (Series or scalar)
        config: Full YAML configuration dict

    Returns:
        SupplyPriceResult with complete pricing waterfall

    Raises:
        ValueError: If inputs invalid or inconsistent

    Example:
        >>> waterfall = build_supply_price_waterfall(
        ...     contract_params=params,
        ...     dam_prices_15min=dam_prices,
        ...     pv_generation_15min=pv_gen,
        ...     consumption_15min=consumption,
        ...     forward_prices=fwd_prices,
        ...     config=config
        ... )
        >>> print(f"Final offer price: {waterfall.final_price_inc_vat_eur_per_mwh:.2f} EUR/MWh")
    """
    calculation_start_time = datetime.now()

    # ========================================================================
    # STEP 1: PV GENERATION COST
    # ========================================================================
    pv_cost_series, pv_cost_per_mwh, solar_share = calculate_pv_procurement_cost(
        mechanism=contract_params.pv_mechanism,
        dam_prices_15min=dam_prices_15min,
        pv_generation_15min=pv_generation_15min,
        config=config,
    )

    # ========================================================================
    # STEP 2: FORWARD PROCUREMENT COST (non-solar gap)
    # ========================================================================
    forward_cost_series, forward_cost_per_mwh, non_solar_share = calculate_forward_procurement_cost(
        consumption_15min=consumption_15min,
        pv_generation_15min=pv_generation_15min,
        forward_prices=forward_prices,
        strategy="blended",
        config=config,
    )

    # ========================================================================
    # STEP 3: BLENDED ENERGY COST
    # ========================================================================
    blended_energy_cost = (solar_share * pv_cost_per_mwh) + (non_solar_share * forward_cost_per_mwh)

    # ========================================================================
    # STEP 4: GREEN CERTIFICATE COST
    # ========================================================================
    if contract_params.include_gc_cost:
        gc_config = config.get("regulatory", {})
        gc_cost = calculate_gc_quota_cost(
            supply_volume_mwh=contract_params.annual_volume_mwh,
            gc_coefficient=gc_config.get("gc_quota_coefficient", 0.499387),
            gc_unit_price_eur=gc_config.get("gc_cost_eur_per_mwh", 14.50),
        )
    else:
        gc_cost = 0.0

    # ========================================================================
    # STEP 5: BALANCING & IMBALANCE COST
    # ========================================================================
    if contract_params.include_balancing_cost:
        balancing_cost = calculate_balancing_cost(
            total_volume_mwh=contract_params.annual_volume_mwh,
            imbalance_rate_pct=0.03,  # 3% average imbalance
            config=config,
        )
    else:
        balancing_cost = 0.0

    # ========================================================================
    # STEP 6: TRANSPORT & ADMIN FEES
    # ========================================================================
    transport_admin_cost = 2.50  # EUR/MWh (typical for Romanian distribution)

    # ========================================================================
    # STEP 7: RISK PREMIUM
    # ========================================================================
    # Shape risk: std(hourly gap) × z-score(95%) × forward price
    hourly_gap = consumption_15min.ffill() - pv_generation_15min.fillna(0)
    shape_risk_per_mwh = (
        hourly_gap.std() * 1.645 * forward_cost_per_mwh / consumption_15min.mean()
    ) if len(hourly_gap) > 1 else 0.0
    shape_risk_per_mwh = max(0.50, min(shape_risk_per_mwh, 3.00))  # Floor 0.5, cap 3.0

    # Volume risk: consumption forecast error × forward price
    consumption_forecast_error_pct = config.get("consumption", {}).get(
        "day_ahead_forecast_error_pct", 0.05
    )
    volume_risk_per_mwh = consumption_forecast_error_pct * forward_cost_per_mwh
    volume_risk_per_mwh = max(0.25, min(volume_risk_per_mwh, 2.00))  # Floor 0.25, cap 2.0

    # Credit risk (based on customer credit rating)
    credit_risk_mapping = {
        "AAA": 0.10, "AA": 0.20, "A": 0.35, "BBB": 0.50, "BB": 0.85, "B": 1.25
    }
    credit_risk_per_mwh = credit_risk_mapping.get(contract_params.credit_rating, 0.50)

    # Imbalance buffer
    imbalance_buffer_per_mwh = 0.25

    total_risk_per_mwh, risk_breakdown = calculate_risk_premium(
        shape_risk_eur_per_mwh=shape_risk_per_mwh,
        volume_risk_eur_per_mwh=volume_risk_per_mwh,
        credit_risk_eur_per_mwh=credit_risk_per_mwh,
        imbalance_buffer_eur_per_mwh=imbalance_buffer_per_mwh,
    )

    # ========================================================================
    # STEP 8: MARGIN CALCULATION
    # ========================================================================
    # Apply volume discount if applicable
    margin_base = contract_params.target_margin_eur_per_mwh
    if contract_params.volume_discount_applied:
        discount_curve = config.get("margins", {}).get("volume_discount_curve", {})
        volume_gwh = contract_params.annual_volume_mwh / 1000
        # Keys may be int or str depending on YAML parsing — normalize to float
        discount_map = {float(k): v for k, v in discount_curve.items()}
        for gwh_threshold in sorted(discount_map.keys(), reverse=True):
            if volume_gwh >= gwh_threshold:
                discount_pct = discount_map[gwh_threshold] / 100.0 if discount_map[gwh_threshold] > 1 else discount_map[gwh_threshold]
                margin_base = margin_base * (1 - discount_pct)
                break

    # Apply seasonal adjustment
    margin_adjusted = margin_base
    if contract_params.seasonal_margin_adjustment:
        current_month = datetime.now().month
        if current_month in [12, 1]:  # Winter
            margin_adjusted = margin_base * 1.15
        elif current_month in [6, 7, 8]:  # Summer
            margin_adjusted = margin_base * 0.85

    # Enforce minimum margin
    min_margin = config.get("margins", {}).get("minimum_margin_eur_per_mwh", 8.00)
    nextE_margin = max(margin_adjusted, min_margin)

    # ========================================================================
    # STEP 9: FINAL PRICE CALCULATION
    # ========================================================================
    price_ex_vat = (
        blended_energy_cost
        + gc_cost
        + balancing_cost
        + transport_admin_cost
        + total_risk_per_mwh
        + nextE_margin
    )

    vat_rate = config.get("regulatory", {}).get("vat_rate", 0.19)
    price_inc_vat = price_ex_vat * (1 + vat_rate)

    # ========================================================================
    # STEP 10: COLLATERAL REQUIREMENTS
    # ========================================================================
    # Estimate based on procurement channel haircuts and exposure
    collateral_required_eur = (
        (contract_params.annual_volume_mwh / 365) *
        forward_cost_per_mwh *
        0.15  # 15% haircut assumption
    ) / 1_000_000  # Convert to EUR millions

    # ========================================================================
    # STEP 11: BUILD WATERFALL DICT
    # ========================================================================
    waterfall_components = {
        "PV Generation (blended)": pv_cost_per_mwh,
        "Forward Procurement": forward_cost_per_mwh,
        "Blended Energy": blended_energy_cost,
        "Green Certificate": gc_cost,
        "Balancing & Imbalance": balancing_cost,
        "Transport & Admin": transport_admin_cost,
        "Shape Risk": shape_risk_per_mwh,
        "Volume Risk": volume_risk_per_mwh,
        "Credit Risk": credit_risk_per_mwh,
        "Imbalance Buffer": imbalance_buffer_per_mwh,
        "Total Risk Premium": total_risk_per_mwh,
        "nextE Margin": nextE_margin,
        "Price ex-VAT": price_ex_vat,
        "VAT (19%)": price_ex_vat * vat_rate,
        "Price inc-VAT": price_inc_vat,
    }

    # ========================================================================
    # STEP 12: PROCUREMENT ALLOCATION BY CHANNEL
    # ========================================================================
    procurement_allocation = {}
    for channel, allocation_pct in contract_params.procurement_channels.items():
        channel_volume_mwh = contract_params.annual_volume_mwh * allocation_pct
        procurement_allocation[channel.value] = {
            "allocation_pct": allocation_pct * 100,
            "volume_mwh": channel_volume_mwh,
            "estimated_cost_eur": channel_volume_mwh * forward_cost_per_mwh * 0.98,  # Estimate
        }

    # ========================================================================
    # FINAL RESULT
    # ========================================================================
    result = SupplyPriceResult(
        contract_params=contract_params,
        analysis_date=datetime.now(),
        energy_cost_eur_per_mwh=blended_energy_cost,
        pv_cost_eur_per_mwh=pv_cost_per_mwh,
        forward_cost_eur_per_mwh=forward_cost_per_mwh,
        solar_share_pct=solar_share * 100,
        gc_cost_eur_per_mwh=gc_cost,
        balancing_cost_eur_per_mwh=balancing_cost,
        transport_admin_eur_per_mwh=transport_admin_cost,
        shape_risk_eur_per_mwh=shape_risk_per_mwh,
        volume_risk_eur_per_mwh=volume_risk_per_mwh,
        credit_risk_eur_per_mwh=credit_risk_per_mwh,
        total_risk_eur_per_mwh=total_risk_per_mwh,
        nextE_margin_eur_per_mwh=nextE_margin,
        final_price_ex_vat_eur_per_mwh=price_ex_vat,
        final_price_inc_vat_eur_per_mwh=price_inc_vat,
        collateral_required_eur_million=collateral_required_eur,
        annual_cost_eur=price_inc_vat * contract_params.annual_volume_mwh,
        procurement_allocation=procurement_allocation,
        waterfall_components=waterfall_components,
        calculation_timestamp=datetime.now(),
        calculation_notes=f"Analysis completed in {(datetime.now() - calculation_start_time).total_seconds():.2f}s",
    )

    logger.info(
        f"Supply pricing waterfall complete | Customer: {contract_params.customer_name} | "
        f"Final price: {price_inc_vat:.2f} EUR/MWh (inc-VAT) | "
        f"Annual cost: {result.annual_cost_eur:,.0f} EUR"
    )

    return result


def run_multi_scenario_pricing(
    contract_params: SupplyContractParams,
    generation_scenarios: Dict[str, pd.Series],
    consumption_scenarios: Dict[str, pd.Series],
    price_scenarios: Dict[str, pd.Series],
    config: Dict,
) -> Dict[str, SupplyPriceResult]:
    """
    Run pricing across multiple scenarios (P10, P25, P50, P75, P90).

    Allows risk/opportunity quantification across probabilistic outcomes.

    Args:
        contract_params: Contract definition
        generation_scenarios: Dict of {scenario_name: pv_gen_series}
            Keys: "p10", "p25", "p50", "p75", "p90"
        consumption_scenarios: Dict of {scenario_name: consumption_series}
        price_scenarios: Dict of {scenario_name: forward_price_series}
        config: Configuration dict

    Returns:
        Dict of {scenario_name: SupplyPriceResult}

    Example:
        >>> scenarios_result = run_multi_scenario_pricing(
        ...     contract_params,
        ...     generation_scenarios,
        ...     consumption_scenarios,
        ...     price_scenarios,
        ...     config
        ... )
        >>> p50_price = scenarios_result["p50"].final_price_inc_vat_eur_per_mwh
    """
    scenario_results = {}

    for scenario_name in ["p10", "p25", "p50", "p75", "p90"]:
        if scenario_name not in generation_scenarios:
            logger.warning(f"Scenario {scenario_name} missing from generation_scenarios")
            continue

        try:
            result = build_supply_price_waterfall(
                contract_params=contract_params,
                dam_prices_15min=price_scenarios.get(scenario_name, price_scenarios.get("p50")),
                pv_generation_15min=generation_scenarios[scenario_name],
                consumption_15min=consumption_scenarios.get(scenario_name, consumption_scenarios.get("p50")),
                forward_prices=price_scenarios.get(scenario_name, price_scenarios.get("p50")).mean(),
                config=config,
            )
            scenario_results[scenario_name] = result
            logger.info(f"Scenario {scenario_name} completed: {result.final_price_inc_vat_eur_per_mwh:.2f} EUR/MWh")
        except Exception as e:
            logger.error(f"Error running scenario {scenario_name}: {e}")
            continue

    return scenario_results


def generate_sensitivity_table(
    base_waterfall: SupplyPriceResult,
    variables_to_test: Optional[List[str]] = None,
    pct_changes: List[float] = [-10, -5, 5, 10],
) -> Dict[str, Dict]:
    """
    Generate tornado sensitivity analysis for key pricing variables.

    Tests impact of ±% changes in key cost drivers on final offer price.

    Args:
        base_waterfall: Base case SupplyPriceResult
        variables_to_test: Variables to stress
            Options: ["dam_price", "pv_cost", "forward_price", "gc_price", "margin"]
            Default: all
        pct_changes: List of percentage changes to test

    Returns:
        Dict with sensitivity results for tornado chart

    Example:
        >>> sensitivity = generate_sensitivity_table(
        ...     base_waterfall,
        ...     variables_to_test=["dam_price", "forward_price"],
        ...     pct_changes=[-10, -5, 5, 10]
        ... )
        >>> for var, impacts in sensitivity.items():
        ...     print(f"{var}: {impacts}")
    """
    if variables_to_test is None:
        variables_to_test = ["pv_cost", "forward_price", "gc_price", "margin"]

    sensitivity_results = {}
    base_price = base_waterfall.final_price_inc_vat_eur_per_mwh

    for variable in variables_to_test:
        impacts = {}

        for pct_change in pct_changes:
            if variable == "pv_cost":
                adjusted_cost = base_waterfall.pv_cost_eur_per_mwh * (1 + pct_change / 100)
                price_impact = (adjusted_cost - base_waterfall.pv_cost_eur_per_mwh) * base_waterfall.contract_params.annual_volume_mwh / 1000

            elif variable == "forward_price":
                adjusted_cost = base_waterfall.forward_cost_eur_per_mwh * (1 + pct_change / 100)
                price_impact = (adjusted_cost - base_waterfall.forward_cost_eur_per_mwh) * base_waterfall.contract_params.annual_volume_mwh / 1000

            elif variable == "gc_price":
                adjusted_cost = base_waterfall.gc_cost_eur_per_mwh * (1 + pct_change / 100)
                price_impact = adjusted_cost - base_waterfall.gc_cost_eur_per_mwh

            elif variable == "margin":
                adjusted_margin = base_waterfall.nextE_margin_eur_per_mwh * (1 + pct_change / 100)
                price_impact = adjusted_margin - base_waterfall.nextE_margin_eur_per_mwh

            else:
                price_impact = 0.0

            new_price_ex_vat = base_waterfall.final_price_ex_vat_eur_per_mwh + price_impact
            new_price_inc_vat = new_price_ex_vat * 1.19

            impacts[pct_change] = {
                "new_price_inc_vat": new_price_inc_vat,
                "price_change_eur_per_mwh": new_price_inc_vat - base_price,
                "price_change_pct": ((new_price_inc_vat - base_price) / base_price) * 100,
            }

        sensitivity_results[variable] = impacts

    logger.info(f"Sensitivity analysis completed for {len(variables_to_test)} variables")

    return sensitivity_results


def export_offer_sheet(
    waterfall: SupplyPriceResult,
    output_format: str = "dict",
) -> Union[Dict, str]:
    """
    Export pricing results in customer-facing offer sheet format.

    Args:
        waterfall: SupplyPriceResult
        output_format: "dict" | "json" | "csv"

    Returns:
        Formatted offer sheet (Dict, JSON string, or CSV string)

    Example:
        >>> offer_dict = export_offer_sheet(waterfall, output_format="dict")
        >>> print(offer_dict["final_price_inc_vat_eur_per_mwh"])
    """
    offer_dict = {
        "Contract Information": {
            "Customer": waterfall.contract_params.customer_name,
            "Contract Period": f"{waterfall.contract_params.contract_start_date.date()} to {waterfall.contract_params.contract_end_date.date()}",
            "Annual Volume": f"{waterfall.contract_params.annual_volume_mwh:.0f} MWh",
            "PV Pricing Mechanism": waterfall.contract_params.pv_mechanism.value,
        },
        "Pricing Waterfall (EUR/MWh)": {
            "Energy Procurement": waterfall.energy_cost_eur_per_mwh,
            "  ├─ PV Generation": waterfall.pv_cost_eur_per_mwh,
            "  └─ Forward Procurement": waterfall.forward_cost_eur_per_mwh,
            "Green Certificate": waterfall.gc_cost_eur_per_mwh,
            "Balancing & Imbalance": waterfall.balancing_cost_eur_per_mwh,
            "Transport & Admin": waterfall.transport_admin_eur_per_mwh,
            "Risk Premium": waterfall.total_risk_eur_per_mwh,
            "  ├─ Shape Risk": waterfall.shape_risk_eur_per_mwh,
            "  ├─ Volume Risk": waterfall.volume_risk_eur_per_mwh,
            "  └─ Credit Risk": waterfall.credit_risk_eur_per_mwh,
            "nextE Commercial Margin": waterfall.nextE_margin_eur_per_mwh,
            "─" * 40: "─" * 40,
            "Final Price (ex-VAT)": waterfall.final_price_ex_vat_eur_per_mwh,
            "VAT (19%)": waterfall.final_price_ex_vat_eur_per_mwh * 0.19,
            "FINAL OFFER PRICE (inc-VAT)": waterfall.final_price_inc_vat_eur_per_mwh,
        },
        "Contract Summary": {
            "Annual Cost (EUR)": waterfall.annual_cost_eur,
            "Solar Share": f"{waterfall.solar_share_pct:.1f}%",
            "Collateral Required (EUR M)": waterfall.collateral_required_eur_million,
        },
        "Procurement Allocation": waterfall.procurement_allocation,
        "Analysis Date": waterfall.analysis_date.isoformat(),
    }

    if output_format == "dict":
        return offer_dict

    elif output_format == "json":
        # Convert non-serializable objects
        offer_dict_clean = json.loads(
            json.dumps(offer_dict, default=str)
        )
        return json.dumps(offer_dict_clean, indent=2)

    elif output_format == "csv":
        # Flatten for CSV export
        rows = []
        for section, content in offer_dict.items():
            rows.append([section, ""])
            if isinstance(content, dict):
                for key, value in content.items():
                    rows.append(["  " + str(key), str(value)])
            else:
                rows.append(["", str(content)])

        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(rows)
        return output.getvalue()

    else:
        raise ValueError(f"Unknown output_format: {output_format}")


# ============================================================================
# END OF SUPPLY PRICING ENGINE
# ============================================================================

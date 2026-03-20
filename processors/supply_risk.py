"""
Supply Risk Quantification Engine — nextE Energy Supply
========================================================
Comprehensive risk modeling for B2B energy supply contracts:
  - Shape risk (PV profile vs consumption mismatch)
  - Volume risk (consumption forecast error)
  - Price risk (forward curve movement)
  - Credit risk (counterparty default)
  - Portfolio VaR (Monte Carlo simulation)
  - Position limit monitoring

Methodology: Parametric VaR, historical simulation, Monte Carlo

Author: nextE AI Workstation
Version: 1.0.0
Date: 2026-03-19
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ShapeRiskAnalysis:
    """Shape risk quantification results."""
    hourly_gap_profile: pd.Series  # Hourly (consumption - PV)
    gap_mean_mwh: float
    gap_std_mwh: float
    gap_coeff_variation: float  # std/mean
    gap_skewness: float  # Asymmetry
    gap_kurtosis: float  # Tail fatness
    price_weighted_shape_risk_eur_per_mwh: float
    var_95_shape_risk_eur_per_mwh: float  # 95% confidence
    analysis_timestamp: str = ""


@dataclass
class VolumeRiskAnalysis:
    """Volume/consumption forecast error risk."""
    expected_volume_mwh: float
    forecast_error_pct: float  # Standard deviation (%)
    volume_uncertainty_mwh_95: float  # 95% confidence interval
    volume_risk_eur_per_mwh_95: float  # Translated to price impact
    seasonal_adjustment: Dict[str, float] = field(default_factory=dict)
    daily_volatility_pct: float = 0.0


@dataclass
class PriceRiskAnalysis:
    """Forward curve and spot price volatility."""
    forward_price_mean_eur_per_mwh: float
    forward_price_std_eur_per_mwh: float
    forward_price_vol_pct: float
    dam_price_mean_eur_per_mwh: float
    dam_price_std_eur_per_mwh: float
    dam_price_vol_pct: float
    basis_spread_mean_eur_per_mwh: float  # Forward - DAM
    basis_spread_std_eur_per_mwh: float
    var_95_price_risk_eur_per_mwh: float
    price_correlation_dam_forward: float  # -1 to 1


@dataclass
class CreditRiskAnalysis:
    """Counterparty credit risk assessment."""
    counterparty_name: str
    credit_rating: str  # AAA, AA, A, BBB, BB, B, etc.
    probability_of_default_bps: int  # Basis points (100 bps = 1%)
    loss_given_default_pct: float  # LGD (0-1)
    exposure_eur: float  # Current MTM exposure
    exposure_at_default_eur: float  # EAD (potential future exposure)
    expected_loss_eur: float  # PD × LGD × EAD
    expected_loss_eur_per_mwh: float
    credit_limit_eur: float
    utilization_pct: float


@dataclass
class PortfolioVaRAnalysis:
    """Multi-factor Value-at-Risk for supply book."""
    confidence_level_pct: float
    holding_period_days: int
    var_eur: float
    var_eur_per_mwh: float
    cvar_eur: float  # Conditional VaR (expected shortfall)
    cvar_eur_per_mwh: float
    monte_carlo_iterations: int
    risk_factors_included: List[str]
    portfolio_pnl_mean_eur: float
    portfolio_pnl_std_eur: float


@dataclass
class PositionLimitStatus:
    """Real-time position limit monitoring."""
    unhedged_exposure_mwh: float
    max_unhedged_mwh: float
    unhedged_utilization_pct: float
    unhedged_status: str  # "OK", "WARNING", "BREACH"

    customer_concentration_gwh: float
    max_customer_concentration_gwh: float
    concentration_utilization_pct: float
    concentration_status: str

    var_95_30d_eur: float
    max_var_95_30d_eur: float
    var_utilization_pct: float
    var_status: str

    collateral_requirement_eur: float
    max_collateral_eur: float
    collateral_utilization_pct: float
    collateral_status: str

    overall_limit_status: str  # "GREEN", "YELLOW", "RED"


# ============================================================================
# SHAPE RISK FUNCTIONS
# ============================================================================

def calculate_shape_risk(
    hourly_consumption: pd.Series,
    hourly_pv_generation: pd.Series,
    forward_prices: pd.Series,
    confidence_level_pct: float = 95,
) -> ShapeRiskAnalysis:
    """
    Quantify shape risk: mismatch between PV generation profile and consumption.

    PV generates heavily during daytime; consumption may be shifted (morning peak,
    evening peak, or flat industrial). This mismatch creates a gap that must be
    filled from forward markets at potentially unfavorable prices.

    Args:
        hourly_consumption: Hourly consumption profile (MWh/h)
        hourly_pv_generation: Hourly PV generation profile (MWh/h)
        forward_prices: Hourly forward prices (EUR/MWh)
        confidence_level_pct: Confidence level for VaR calculation

    Returns:
        ShapeRiskAnalysis with full risk metrics

    Example:
        >>> shape_risk = calculate_shape_risk(
        ...     hourly_consumption,
        ...     hourly_pv,
        ...     forward_prices,
        ...     confidence_level_pct=95
        ... )
        >>> print(f"Shape risk (95%): {shape_risk.var_95_shape_risk_eur_per_mwh:.2f} EUR/MWh")
    """
    # Calculate hourly gap (consumption not covered by PV)
    gap = (hourly_consumption - hourly_pv_generation).clip(lower=0)

    # Gap statistics
    gap_mean = gap.mean()
    gap_std = gap.std()
    gap_coeff_var = gap_std / max(gap_mean, 0.01)
    gap_skewness = stats.skew(gap)
    gap_kurtosis = stats.kurtosis(gap)

    # Price-weighted shape risk
    # Shape risk = volatility of (gap × forward price)
    gap_weighted_by_price = gap * forward_prices
    price_weighted_std = gap_weighted_by_price.std()
    price_weighted_mean = gap_weighted_by_price.mean()

    # VaR 95% shape risk
    z_score = stats.norm.ppf(confidence_level_pct / 100)
    var_95_shape_risk = max(0, price_weighted_mean + z_score * price_weighted_std) - price_weighted_mean

    return ShapeRiskAnalysis(
        hourly_gap_profile=gap,
        gap_mean_mwh=gap_mean,
        gap_std_mwh=gap_std,
        gap_coeff_variation=gap_coeff_var,
        gap_skewness=gap_skewness,
        gap_kurtosis=gap_kurtosis,
        price_weighted_shape_risk_eur_per_mwh=price_weighted_std,
        var_95_shape_risk_eur_per_mwh=var_95_shape_risk,
        analysis_timestamp=pd.Timestamp.now().isoformat(),
    )


# ============================================================================
# VOLUME RISK FUNCTIONS
# ============================================================================

def calculate_volume_risk(
    expected_annual_volume_mwh: float,
    forecast_error_pct: float,
    forward_price_eur_per_mwh: float,
    seasonal_variations: Optional[Dict[str, float]] = None,
) -> VolumeRiskAnalysis:
    """
    Quantify volume risk: uncertainty in customer consumption forecast.

    Customers' actual consumption may deviate from contracted forecast,
    creating additional procurement costs (if consumption > forecast) or
    surplus selling opportunity (if consumption < forecast).

    Args:
        expected_annual_volume_mwh: Forecasted annual consumption
        forecast_error_pct: Forecast error as % of volume (std dev, e.g., 0.05 for 5%)
        forward_price_eur_per_mwh: Forward procurement cost baseline
        seasonal_variations: Dict of seasonal adjustments (e.g., {"winter": 1.5, "summer": 0.8})

    Returns:
        VolumeRiskAnalysis

    Example:
        >>> vol_risk = calculate_volume_risk(
        ...     expected_annual_volume_mwh=10000,
        ...     forecast_error_pct=0.05,
        ...     forward_price_eur_per_mwh=65.0
        ... )
        >>> print(f"Volume uncertainty (95%): {vol_risk.volume_uncertainty_mwh_95:.0f} MWh")
    """
    # Volume uncertainty (95% confidence)
    z_score = stats.norm.ppf(0.95)
    volume_uncertainty_mwh_95 = expected_annual_volume_mwh * forecast_error_pct * z_score

    # Volume risk translated to price impact
    volume_risk_eur_per_mwh_95 = (volume_uncertainty_mwh_95 / expected_annual_volume_mwh) * forward_price_eur_per_mwh

    # Daily volatility (assume linear scaling from annual)
    daily_volatility_pct = forecast_error_pct * np.sqrt(365)

    if seasonal_variations is None:
        seasonal_variations = {
            "winter": 1.2,
            "summer": 0.8,
            "shoulder": 1.0,
        }

    return VolumeRiskAnalysis(
        expected_volume_mwh=expected_annual_volume_mwh,
        forecast_error_pct=forecast_error_pct,
        volume_uncertainty_mwh_95=volume_uncertainty_mwh_95,
        volume_risk_eur_per_mwh_95=volume_risk_eur_per_mwh_95,
        seasonal_adjustment=seasonal_variations,
        daily_volatility_pct=daily_volatility_pct,
    )


# ============================================================================
# PRICE RISK FUNCTIONS
# ============================================================================

def calculate_price_risk(
    forward_prices: pd.Series,
    dam_prices: pd.Series,
    annual_volume_mwh: float,
) -> PriceRiskAnalysis:
    """
    Quantify price risk: forward curve and DAM volatility exposure.

    Args:
        forward_prices: Time series of forward prices (EUR/MWh)
        dam_prices: Time series of DAM prices (EUR/MWh)
        annual_volume_mwh: Annual volume exposure

    Returns:
        PriceRiskAnalysis with full metrics

    Example:
        >>> price_risk = calculate_price_risk(
        ...     forward_prices,
        ...     dam_prices,
        ...     annual_volume_mwh=10000
        ... )
        >>> print(f"DAM volatility: {price_risk.dam_price_vol_pct:.1f}%")
    """
    # Forward price statistics
    fwd_mean = forward_prices.mean()
    fwd_std = forward_prices.std()
    fwd_vol_pct = (fwd_std / fwd_mean) * 100 if fwd_mean > 0 else 0

    # DAM price statistics
    dam_mean = dam_prices.mean()
    dam_std = dam_prices.std()
    dam_vol_pct = (dam_std / dam_mean) * 100 if dam_mean > 0 else 0

    # Basis spread (forward - DAM)
    basis_spread = (forward_prices - dam_prices).dropna()
    basis_mean = basis_spread.mean()
    basis_std = basis_spread.std()

    # Correlation between DAM and forward
    correlation = forward_prices.corr(dam_prices)

    # VaR 95% price risk (annual)
    z_score = stats.norm.ppf(0.95)
    var_95_price_risk_abs = z_score * fwd_std
    var_95_price_risk_per_mwh = var_95_price_risk_abs

    return PriceRiskAnalysis(
        forward_price_mean_eur_per_mwh=fwd_mean,
        forward_price_std_eur_per_mwh=fwd_std,
        forward_price_vol_pct=fwd_vol_pct,
        dam_price_mean_eur_per_mwh=dam_mean,
        dam_price_std_eur_per_mwh=dam_std,
        dam_price_vol_pct=dam_vol_pct,
        basis_spread_mean_eur_per_mwh=basis_mean,
        basis_spread_std_eur_per_mwh=basis_std,
        var_95_price_risk_eur_per_mwh=var_95_price_risk_per_mwh,
        price_correlation_dam_forward=correlation,
    )


# ============================================================================
# CREDIT RISK FUNCTIONS
# ============================================================================

def calculate_credit_risk(
    counterparty_name: str,
    credit_rating: str,
    exposure_eur: float,
    annual_volume_mwh: float,
    forward_price_eur_per_mwh: float,
    credit_limit_eur: float,
) -> CreditRiskAnalysis:
    """
    Quantify credit risk: counterparty default probability and loss.

    Uses industry-standard mapping of credit rating to probability of default.

    Args:
        counterparty_name: Counterparty identifier
        credit_rating: Standard rating (AAA, AA, A, BBB, BB, B, CCC, etc.)
        exposure_eur: Current mark-to-market exposure
        annual_volume_mwh: Annual contract volume
        forward_price_eur_per_mwh: Price basis for EAD calculation
        credit_limit_eur: Risk limit for this counterparty

    Returns:
        CreditRiskAnalysis

    Example:
        >>> credit_risk = calculate_credit_risk(
        ...     counterparty_name="Customer ABC",
        ...     credit_rating="BBB",
        ...     exposure_eur=100000,
        ...     annual_volume_mwh=10000,
        ...     forward_price_eur_per_mwh=65.0,
        ...     credit_limit_eur=500000
        ... )
        >>> print(f"Expected loss: {credit_risk.expected_loss_eur:,.0f} EUR")
    """
    # PD mapping (basis points)
    pd_mapping = {
        "AAA": 10,
        "AA": 25,
        "A": 50,
        "BBB": 100,
        "BB": 250,
        "B": 500,
        "CCC": 1000,
        "D": 10000,
    }

    pd_bps = pd_mapping.get(credit_rating, 100)
    pd_pct = pd_bps / 10000

    # LGD (Loss Given Default)
    lgd_mapping = {
        "AAA": 0.20,
        "AA": 0.25,
        "A": 0.35,
        "BBB": 0.45,
        "BB": 0.60,
        "B": 0.75,
        "CCC": 0.85,
        "D": 1.00,
    }
    lgd = lgd_mapping.get(credit_rating, 0.45)

    # EAD (Exposure at Default)
    # Current exposure + 30% of annual procurement value (potential future exposure)
    annual_exposure_eur = annual_volume_mwh * forward_price_eur_per_mwh
    ead = exposure_eur + (0.30 * annual_exposure_eur)

    # Expected Loss
    expected_loss = pd_pct * lgd * ead
    expected_loss_per_mwh = expected_loss / max(annual_volume_mwh, 1)

    # Utilization
    utilization_pct = (exposure_eur / credit_limit_eur * 100) if credit_limit_eur > 0 else 0

    return CreditRiskAnalysis(
        counterparty_name=counterparty_name,
        credit_rating=credit_rating,
        probability_of_default_bps=pd_bps,
        loss_given_default_pct=lgd,
        exposure_eur=exposure_eur,
        exposure_at_default_eur=ead,
        expected_loss_eur=expected_loss,
        expected_loss_eur_per_mwh=expected_loss_per_mwh,
        credit_limit_eur=credit_limit_eur,
        utilization_pct=utilization_pct,
    )


# ============================================================================
# PORTFOLIO VAR FUNCTIONS
# ============================================================================

def calculate_portfolio_var(
    supply_book_customers: List[Dict],
    confidence_level_pct: float = 95,
    holding_period_days: int = 30,
    monte_carlo_iterations: int = 10000,
    random_seed: Optional[int] = None,
) -> PortfolioVaRAnalysis:
    """
    Monte Carlo VaR calculation for supply book (multi-customer portfolio).

    Simulates joint P&L distribution across multiple risk factors:
      - Price (forward curve movement)
      - Volume (consumption deviation)
      - Credit (default events)

    Args:
        supply_book_customers: List of customer dicts with:
            {"customer_id", "annual_volume_mwh", "base_price_eur_per_mwh", "credit_rating"}
        confidence_level_pct: Confidence level (default 95%)
        holding_period_days: Holding period for VaR (default 30 days)
        monte_carlo_iterations: Number of MC simulations
        random_seed: Random seed for reproducibility

    Returns:
        PortfolioVaRAnalysis

    Example:
        >>> var_analysis = calculate_portfolio_var(
        ...     supply_book_customers=[
        ...         {"customer_id": "C001", "annual_volume_mwh": 5000, "base_price_eur_per_mwh": 65, "credit_rating": "BBB"},
        ...         {"customer_id": "C002", "annual_volume_mwh": 3000, "base_price_eur_per_mwh": 62, "credit_rating": "A"},
        ...     ],
        ...     confidence_level_pct=95,
        ...     holding_period_days=30,
        ...     monte_carlo_iterations=10000
        ... )
        >>> print(f"Portfolio VaR (95%, 30d): {var_analysis.var_eur:,.0f} EUR")
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    total_volume_mwh = sum(c.get("annual_volume_mwh", 0) for c in supply_book_customers)

    if total_volume_mwh == 0:
        raise ValueError("Total supply book volume is zero")

    # Volatility assumptions (annualized)
    price_volatility_annual = 0.25  # 25% annual price volatility
    volume_volatility_annual = 0.05  # 5% annual volume volatility
    credit_default_rate = 0.005  # 0.5% annual default rate (portfolio average)

    # Scale to holding period
    holding_period_fraction = holding_period_days / 365
    price_vol_holding = price_volatility_annual * np.sqrt(holding_period_fraction)
    volume_vol_holding = volume_volatility_annual * np.sqrt(holding_period_fraction)

    # MC simulation
    pnl_simulations = []

    for _ in range(monte_carlo_iterations):
        # Sample price shock (% change in forward prices)
        price_shock = np.random.normal(0, price_vol_holding)

        # Sample volume shock (% change in consumption)
        volume_shock = np.random.normal(0, volume_vol_holding)

        # Sample credit events (binary)
        credit_events = np.random.uniform(0, 1, len(supply_book_customers)) < (credit_default_rate * holding_period_fraction)

        # Calculate portfolio P&L
        portfolio_pnl = 0

        for i, customer in enumerate(supply_book_customers):
            customer_volume = customer.get("annual_volume_mwh", 0) * (holding_period_days / 365)
            base_price = customer.get("base_price_eur_per_mwh", 65)

            # Price impact (positive price shock = profit for supplier)
            price_pnl = customer_volume * base_price * price_shock

            # Volume impact (positive volume shock = more cost for supplier)
            volume_pnl = -customer_volume * base_price * volume_shock

            # Credit impact (default = loss of margin on full contract)
            margin_eur_per_mwh = 12  # Assumed margin
            credit_pnl = -customer_volume * margin_eur_per_mwh if credit_events[i] else 0

            portfolio_pnl += price_pnl + volume_pnl + credit_pnl

        pnl_simulations.append(portfolio_pnl)

    pnl_array = np.array(pnl_simulations)

    # VaR and CVaR
    confidence_index = int((1 - confidence_level_pct / 100) * len(pnl_array))
    var = -np.percentile(pnl_array, confidence_level_pct)
    cvar = -np.mean(pnl_array[pnl_array <= -var])

    return PortfolioVaRAnalysis(
        confidence_level_pct=confidence_level_pct,
        holding_period_days=holding_period_days,
        var_eur=var,
        var_eur_per_mwh=var / max(total_volume_mwh * holding_period_days / 365, 1),
        cvar_eur=cvar,
        cvar_eur_per_mwh=cvar / max(total_volume_mwh * holding_period_days / 365, 1),
        monte_carlo_iterations=monte_carlo_iterations,
        risk_factors_included=["price", "volume", "credit"],
        portfolio_pnl_mean_eur=np.mean(pnl_array),
        portfolio_pnl_std_eur=np.std(pnl_array),
    )


# ============================================================================
# POSITION LIMIT MONITORING
# ============================================================================

def monitor_position_limits(
    unhedged_exposure_mwh: float,
    max_unhedged_mwh: float,
    customer_concentration_gwh: float,
    max_customer_concentration_gwh: float,
    var_95_30d_eur: float,
    max_var_95_30d_eur: float,
    collateral_requirement_eur: float,
    max_collateral_eur: float,
) -> PositionLimitStatus:
    """
    Monitor position against all risk limits.

    Args:
        unhedged_exposure_mwh: Current unhedged MWh
        max_unhedged_mwh: Limit
        customer_concentration_gwh: Single-customer volume (GWh)
        max_customer_concentration_gwh: Limit
        var_95_30d_eur: Current 95% VaR (30-day)
        max_var_95_30d_eur: Limit
        collateral_requirement_eur: Current requirement
        max_collateral_eur: Limit

    Returns:
        PositionLimitStatus with overall status

    Example:
        >>> status = monitor_position_limits(
        ...     unhedged_exposure_mwh=500,
        ...     max_unhedged_mwh=1000,
        ...     customer_concentration_gwh=45,
        ...     max_customer_concentration_gwh=50,
        ...     var_95_30d_eur=2000000,
        ...     max_var_95_30d_eur=2500000,
        ...     collateral_requirement_eur=8000000,
        ...     max_collateral_eur=10000000
        ... )
        >>> print(f"Overall status: {status.overall_limit_status}")
    """
    def _status(utilization_pct: float) -> str:
        if utilization_pct >= 100:
            return "BREACH"
        elif utilization_pct >= 85:
            return "WARNING"
        else:
            return "OK"

    unhedged_util = (unhedged_exposure_mwh / max(max_unhedged_mwh, 1)) * 100
    concentration_util = (customer_concentration_gwh / max(max_customer_concentration_gwh, 1)) * 100
    var_util = (var_95_30d_eur / max(max_var_95_30d_eur, 1)) * 100
    collateral_util = (collateral_requirement_eur / max(max_collateral_eur, 1)) * 100

    result = PositionLimitStatus(
        unhedged_exposure_mwh=unhedged_exposure_mwh,
        max_unhedged_mwh=max_unhedged_mwh,
        unhedged_utilization_pct=unhedged_util,
        unhedged_status=_status(unhedged_util),
        customer_concentration_gwh=customer_concentration_gwh,
        max_customer_concentration_gwh=max_customer_concentration_gwh,
        concentration_utilization_pct=concentration_util,
        concentration_status=_status(concentration_util),
        var_95_30d_eur=var_95_30d_eur,
        max_var_95_30d_eur=max_var_95_30d_eur,
        var_utilization_pct=var_util,
        var_status=_status(var_util),
        collateral_requirement_eur=collateral_requirement_eur,
        max_collateral_eur=max_collateral_eur,
        collateral_utilization_pct=collateral_util,
        collateral_status=_status(collateral_util),
        overall_limit_status="RED" if any(s == "BREACH" for s in [_status(x) for x in [unhedged_util, concentration_util, var_util, collateral_util]])
                              else "YELLOW" if any(s == "WARNING" for s in [_status(x) for x in [unhedged_util, concentration_util, var_util, collateral_util]])
                              else "GREEN"
    )

    logger.info(
        f"Position limits monitored | "
        f"Unhedged: {unhedged_util:.1f}% | "
        f"Concentration: {concentration_util:.1f}% | "
        f"VaR: {var_util:.1f}% | "
        f"Collateral: {collateral_util:.1f}% | "
        f"Overall: {result.overall_limit_status}"
    )

    return result


# ============================================================================
# END OF SUPPLY RISK ENGINE
# ============================================================================

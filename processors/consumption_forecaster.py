"""
Consumption Profile Forecaster — Multi-Probability 15-min Load Curves
=====================================================================
Produces probabilistic consumption forecasts for B2B supply customers
at 15-minute granularity. Used by the supply pipeline to quantify
demand volume uncertainty and compute hedging/procurement requirements.

Methodology:
  1. Base load profile: customer-specific or standard (commercial/industrial)
  2. Calendar adjustment: day-of-week, public holidays, seasonal patterns
  3. Temperature sensitivity: HDD/CDD regression model
  4. Growth/trend adjustment: contractual volume trajectory
  5. Statistical uncertainty: forecast error distribution → P10-P90 bands

Customer Segments:
  - Commercial: office/retail, strong diurnal pattern, weekend dip
  - Industrial: flat baseload (24/7), minimal diurnal variation
  - Mixed: weighted combination

Output:
  - 15-min consumption profiles for D+1 to D+365
  - Multi-probability: P10, P25, P50, P75, P90
  - Per-customer and portfolio-level aggregation

Author: nextE AI Workstation
Version: 1.0.0
Date: 2026-03-20
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# CONSTANTS
# ============================================================================

INTERVALS_PER_HOUR = 4
INTERVALS_PER_DAY = 96
HOURS_PER_DAY = 24
PERCENTILES = [10, 25, 50, 75, 90]
PERCENTILE_LABELS = ["P10", "P25", "P50", "P75", "P90"]

# Romanian public holidays (fixed dates, 2026)
RO_PUBLIC_HOLIDAYS_2026 = [
    date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 24),
    date(2026, 4, 17), date(2026, 4, 20),  # Orthodox Easter (approx)
    date(2026, 5, 1), date(2026, 6, 1), date(2026, 6, 8),  # Whit Monday (approx)
    date(2026, 8, 15), date(2026, 11, 30), date(2026, 12, 1),
    date(2026, 12, 25), date(2026, 12, 26),
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class CustomerSegment(Enum):
    """B2B customer load profile type."""
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    MIXED = "mixed"


class ForecastHorizon(Enum):
    """Forecast time horizon."""
    DAY_AHEAD = "day_ahead"       # D+1
    WEEK_AHEAD = "week_ahead"     # D+7
    MONTH_AHEAD = "month_ahead"   # D+30
    QUARTER_AHEAD = "quarter"     # D+90
    YEAR_AHEAD = "annual"         # D+365


@dataclass
class CustomerProfile:
    """Customer consumption specification."""
    customer_id: str
    name: str
    segment: CustomerSegment
    annual_consumption_mwh: float
    peak_demand_mw: float

    # Load shape parameters
    load_factor: float = 0.65  # Annual load factor (avg/peak)
    weekend_factor: float = 0.75  # Weekend consumption as fraction of weekday
    holiday_factor: float = 0.60  # Holiday consumption fraction

    # Temperature sensitivity
    hdd_sensitivity_mwh_per_degree: float = 0.0  # Heating degree day response
    cdd_sensitivity_mwh_per_degree: float = 0.0  # Cooling degree day response
    base_temperature_c: float = 18.0  # Reference temperature

    # Forecast uncertainty
    day_ahead_error_pct: float = 0.05  # Day-ahead forecast RMSE
    week_ahead_error_pct: float = 0.08  # Week-ahead RMSE
    month_ahead_error_pct: float = 0.12  # Month-ahead RMSE

    # Growth / contractual parameters
    annual_growth_pct: float = 0.02  # 2% annual demand growth


@dataclass
class ConsumptionForecast:
    """Multi-probability consumption forecast result."""
    customer_id: str
    segment: CustomerSegment
    annual_consumption_mwh: float
    peak_demand_mw: float
    forecast_date: date
    horizon_days: int
    granularity_minutes: int  # 15

    # Core: DataFrame with 15-min timestamps and P10-P90 columns (MW)
    profiles: pd.DataFrame

    # Summary
    daily_consumption_mwh: Dict[str, float] = field(default_factory=dict)
    load_factor_pct: Dict[str, float] = field(default_factory=dict)
    peak_demand_forecast_mw: Dict[str, float] = field(default_factory=dict)

    # Metadata
    generation_timestamp: datetime = field(default_factory=datetime.now)
    methodology: str = ""


@dataclass
class PortfolioConsumptionForecast:
    """Aggregated portfolio consumption forecast."""
    forecast_date: date
    horizon_days: int
    total_annual_mwh: float
    total_peak_mw: float
    customer_forecasts: List[ConsumptionForecast]
    portfolio_profiles: pd.DataFrame  # Aggregated P10-P90
    segment_breakdown: Dict[str, pd.DataFrame] = field(default_factory=dict)


# ============================================================================
# LOAD PROFILE TEMPLATES
# ============================================================================

# Hourly load shape factors (normalized, 24 values per season)
# From supply_config.yaml + enhanced 15-min interpolation

COMMERCIAL_WINTER = np.array([
    0.85, 0.80, 0.78, 0.80, 0.88, 1.05, 1.15, 1.10, 1.05, 0.98,
    0.95, 0.92, 0.90, 0.92, 0.95, 0.98, 1.02, 1.08, 1.05, 1.00,
    0.95, 0.90, 0.85, 0.80,
])

COMMERCIAL_SUMMER = np.array([
    0.70, 0.65, 0.60, 0.62, 0.75, 0.95, 1.10, 1.15, 1.10, 1.00,
    0.95, 0.90, 0.85, 0.88, 0.92, 0.95, 0.98, 1.05, 1.00, 0.95,
    0.85, 0.78, 0.70, 0.65,
])

COMMERCIAL_SHOULDER = (COMMERCIAL_WINTER + COMMERCIAL_SUMMER) / 2

INDUSTRIAL_FLAT = np.ones(24)  # 24/7 baseload

PROFILE_LIBRARY = {
    CustomerSegment.COMMERCIAL: {
        "winter": COMMERCIAL_WINTER,
        "summer": COMMERCIAL_SUMMER,
        "shoulder": COMMERCIAL_SHOULDER,
    },
    CustomerSegment.INDUSTRIAL: {
        "winter": INDUSTRIAL_FLAT,
        "summer": INDUSTRIAL_FLAT,
        "shoulder": INDUSTRIAL_FLAT,
    },
    CustomerSegment.MIXED: {
        "winter": 0.6 * COMMERCIAL_WINTER + 0.4 * INDUSTRIAL_FLAT,
        "summer": 0.6 * COMMERCIAL_SUMMER + 0.4 * INDUSTRIAL_FLAT,
        "shoulder": 0.6 * COMMERCIAL_SHOULDER + 0.4 * INDUSTRIAL_FLAT,
    },
}


# ============================================================================
# FORECASTER IMPLEMENTATION
# ============================================================================

class ConsumptionForecaster:
    """
    Probabilistic consumption forecaster for B2B customers.

    Produces 15-min multi-probability load curves based on:
    - Standard load profiles (commercial / industrial)
    - Calendar patterns (weekday, weekend, holiday)
    - Seasonal variation
    - Statistical uncertainty bands
    """

    def __init__(self, customer: CustomerProfile):
        self.customer = customer
        self.profiles = PROFILE_LIBRARY.get(
            customer.segment,
            PROFILE_LIBRARY[CustomerSegment.MIXED]
        )

    def forecast(
        self,
        target_date: date,
        horizon_days: int = 7,
        n_scenarios: int = 100,
        seed: Optional[int] = None,
    ) -> ConsumptionForecast:
        """
        Generate multi-probability consumption forecast.

        Args:
            target_date: First forecast day.
            horizon_days: Number of days.
            n_scenarios: Ensemble size.
            seed: Random seed.

        Returns:
            ConsumptionForecast with 15-min P10-P90 profiles.
        """
        rng = np.random.default_rng(seed)

        timestamps = pd.date_range(
            start=target_date,
            periods=INTERVALS_PER_DAY * horizon_days,
            freq="15min",
            tz="Europe/Bucharest",
        )
        n_intervals = len(timestamps)

        # Step 1: Deterministic base profile
        base_profile = self._build_base_profile(timestamps)

        # Step 2: Scale to annual consumption
        scaled_profile = self._scale_to_annual_volume(base_profile, timestamps)

        # Step 3: Apply calendar adjustments
        calendar_profile = self._apply_calendar_adjustments(scaled_profile, timestamps)

        # Step 4: Temperature sensitivity (simplified, using seasonal proxy)
        temp_adjusted = self._apply_temperature_effect(calendar_profile, timestamps, rng)

        # Step 5: Generate uncertainty scenarios
        scenarios = self._generate_scenarios(
            temp_adjusted, timestamps, n_scenarios, rng
        )

        # Step 6: Extract percentiles
        percentile_profiles = np.percentile(scenarios, PERCENTILES, axis=0)

        profiles_df = pd.DataFrame({"timestamp": timestamps})
        for i, label in enumerate(PERCENTILE_LABELS):
            profiles_df[label] = np.round(
                np.clip(percentile_profiles[i], 0, self.customer.peak_demand_mw * 1.1),
                3,
            )

        # Summary
        daily_consumption = {}
        load_factor = {}
        peak_demand = {}

        for label in PERCENTILE_LABELS:
            total_mwh = profiles_df[label].sum() * 0.25
            daily_consumption[label] = round(total_mwh / max(horizon_days, 1), 2)
            peak_mw = profiles_df[label].max()
            peak_demand[label] = round(peak_mw, 3)
            if peak_mw > 0:
                load_factor[label] = round(
                    (total_mwh / (peak_mw * 24 * horizon_days)) * 100, 2
                )
            else:
                load_factor[label] = 0.0

        return ConsumptionForecast(
            customer_id=self.customer.customer_id,
            segment=self.customer.segment,
            annual_consumption_mwh=self.customer.annual_consumption_mwh,
            peak_demand_mw=self.customer.peak_demand_mw,
            forecast_date=target_date,
            horizon_days=horizon_days,
            granularity_minutes=15,
            profiles=profiles_df,
            daily_consumption_mwh=daily_consumption,
            load_factor_pct=load_factor,
            peak_demand_forecast_mw=peak_demand,
            methodology="standard profile + calendar + temperature + MC uncertainty",
        )

    def _build_base_profile(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Build hourly base profile, interpolated to 15-min."""
        n_intervals = len(timestamps)
        profile_15min = np.zeros(n_intervals)

        for i, ts in enumerate(timestamps):
            month = ts.month
            hour = ts.hour
            minute = ts.minute

            # Season selection
            if month in [12, 1, 2]:
                season = "winter"
            elif month in [6, 7, 8]:
                season = "summer"
            else:
                season = "shoulder"

            hourly_profile = self.profiles[season]

            # Linear interpolation between hours for 15-min resolution
            h0 = hour
            h1 = (hour + 1) % 24
            frac = minute / 60.0
            profile_15min[i] = hourly_profile[h0] * (1 - frac) + hourly_profile[h1] * frac

        return profile_15min

    def _scale_to_annual_volume(
        self,
        base_profile: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        """Scale base profile to match annual consumption volume."""
        # Target average MW = annual_mwh / 8760
        target_avg_mw = self.customer.annual_consumption_mwh / 8760

        # Current average of normalized profile
        current_avg = base_profile.mean()

        if current_avg > 0:
            scale = target_avg_mw / current_avg
        else:
            scale = target_avg_mw

        return base_profile * scale

    def _apply_calendar_adjustments(
        self,
        profile: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        """Apply weekday/weekend/holiday multipliers."""
        adjusted = profile.copy()

        for i, ts in enumerate(timestamps):
            d = ts.date()
            dow = ts.dayofweek  # 0=Monday, 6=Sunday

            if d in RO_PUBLIC_HOLIDAYS_2026:
                adjusted[i] *= self.customer.holiday_factor
            elif dow >= 5:  # Saturday, Sunday
                adjusted[i] *= self.customer.weekend_factor

        return adjusted

    def _apply_temperature_effect(
        self,
        profile: np.ndarray,
        timestamps: pd.DatetimeIndex,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply temperature sensitivity using seasonal proxy temperatures."""
        if (self.customer.hdd_sensitivity_mwh_per_degree == 0 and
                self.customer.cdd_sensitivity_mwh_per_degree == 0):
            return profile

        adjusted = profile.copy()
        base_temp = self.customer.base_temperature_c

        for i, ts in enumerate(timestamps):
            month = ts.month
            hour = ts.hour

            # Synthetic seasonal temperature (Romania average)
            seasonal_temp = {
                1: -2, 2: 0, 3: 7, 4: 13, 5: 18, 6: 23,
                7: 26, 8: 25, 9: 19, 10: 12, 11: 5, 12: 0,
            }.get(month, 12)

            # Diurnal variation (±5°C)
            diurnal = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
            temp = seasonal_temp + diurnal + rng.normal(0, 2)

            # HDD/CDD adjustment
            hdd = max(0, base_temp - temp)
            cdd = max(0, temp - base_temp)

            temp_adjustment_mw = (
                self.customer.hdd_sensitivity_mwh_per_degree * hdd
                + self.customer.cdd_sensitivity_mwh_per_degree * cdd
            ) / HOURS_PER_DAY  # Convert daily MWh to hourly MW

            adjusted[i] += temp_adjustment_mw / INTERVALS_PER_HOUR

        return adjusted

    def _generate_scenarios(
        self,
        base: np.ndarray,
        timestamps: pd.DatetimeIndex,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate stochastic consumption scenarios.

        Uses forecast error model with temporal correlation.
        """
        n_intervals = len(base)
        scenarios = np.zeros((n_scenarios, n_intervals))

        # Forecast error grows with horizon
        # Day-ahead: ~5%, month-ahead: ~12%
        base_error = self.customer.day_ahead_error_pct

        for s in range(n_scenarios):
            # Correlated noise (AR(1) process)
            noise = rng.normal(0, 1, n_intervals)
            ar_coeff = 0.90  # High temporal correlation in demand

            for j in range(1, n_intervals):
                noise[j] = ar_coeff * noise[j - 1] + np.sqrt(1 - ar_coeff**2) * noise[j]

            # Scale by forecast error (grows with forecast day)
            days_ahead = np.arange(n_intervals) // INTERVALS_PER_DAY
            error_pct = base_error * (1 + 0.02 * days_ahead)  # Gradually increases
            error_pct = np.clip(error_pct, base_error, 0.20)

            perturbation = noise * error_pct * base
            scenarios[s] = base + perturbation

        return np.clip(scenarios, 0, None)


# ============================================================================
# PORTFOLIO AGGREGATION
# ============================================================================

def forecast_consumption_portfolio(
    customers: List[CustomerProfile],
    target_date: date,
    horizon_days: int = 7,
    n_scenarios: int = 100,
    seed: Optional[int] = None,
) -> PortfolioConsumptionForecast:
    """
    Generate aggregated portfolio consumption forecast.

    Args:
        customers: List of customer profiles.
        target_date: First forecast day.
        horizon_days: Forecast horizon.
        n_scenarios: Ensemble size.
        seed: Random seed.

    Returns:
        PortfolioConsumptionForecast.
    """
    logger.info(
        f"Forecasting consumption: {len(customers)} customers, "
        f"{sum(c.annual_consumption_mwh for c in customers):.0f} MWh/yr total"
    )

    rng = np.random.default_rng(seed)
    customer_forecasts = []

    for customer in customers:
        fc = ConsumptionForecaster(customer).forecast(
            target_date=target_date,
            horizon_days=horizon_days,
            n_scenarios=n_scenarios,
            seed=rng.integers(0, 2**31),
        )
        customer_forecasts.append(fc)
        logger.info(
            f"  {customer.customer_id}: {customer.annual_consumption_mwh:.0f} MWh/yr "
            f"{customer.segment.value} | "
            f"P50 daily={fc.daily_consumption_mwh.get('P50', 0):.1f} MWh"
        )

    if not customer_forecasts:
        raise ValueError("No customers to forecast")

    ref_timestamps = customer_forecasts[0].profiles["timestamp"]
    portfolio_profiles = pd.DataFrame({"timestamp": ref_timestamps})

    for label in PERCENTILE_LABELS:
        portfolio_profiles[label] = sum(
            fc.profiles[label].values for fc in customer_forecasts
        )
        portfolio_profiles[label] = np.round(portfolio_profiles[label], 3)

    # Segment breakdown
    segment_breakdown = {}
    for seg in CustomerSegment:
        seg_fcs = [fc for fc in customer_forecasts if fc.segment == seg]
        if seg_fcs:
            seg_df = pd.DataFrame({"timestamp": ref_timestamps})
            for label in PERCENTILE_LABELS:
                seg_df[label] = sum(fc.profiles[label].values for fc in seg_fcs)
            segment_breakdown[seg.value] = seg_df

    total_annual = sum(c.annual_consumption_mwh for c in customers)
    total_peak = sum(c.peak_demand_mw for c in customers)

    return PortfolioConsumptionForecast(
        forecast_date=target_date,
        horizon_days=horizon_days,
        total_annual_mwh=total_annual,
        total_peak_mw=total_peak,
        customer_forecasts=customer_forecasts,
        portfolio_profiles=portfolio_profiles,
        segment_breakdown=segment_breakdown,
    )


# ============================================================================
# DEMO DATA
# ============================================================================

def generate_demo_customers() -> List[CustomerProfile]:
    """
    Generate demo B2B customer portfolio for testing.

    Matches the 5-contract demo in 09_supply_portfolio.py.
    """
    return [
        CustomerProfile(
            customer_id="CUST-METRO-01",
            name="Metro Cash & Carry",
            segment=CustomerSegment.COMMERCIAL,
            annual_consumption_mwh=12000,
            peak_demand_mw=4.5,
            load_factor=0.60,
            weekend_factor=0.80,
            hdd_sensitivity_mwh_per_degree=1.5,
            cdd_sensitivity_mwh_per_degree=2.0,
        ),
        CustomerProfile(
            customer_id="CUST-ARCELORMITTAL-01",
            name="ArcelorMittal Galati",
            segment=CustomerSegment.INDUSTRIAL,
            annual_consumption_mwh=85000,
            peak_demand_mw=12.0,
            load_factor=0.85,
            weekend_factor=0.95,
            annual_growth_pct=0.01,
        ),
        CustomerProfile(
            customer_id="CUST-KAUFLAND-01",
            name="Kaufland Romania",
            segment=CustomerSegment.COMMERCIAL,
            annual_consumption_mwh=25000,
            peak_demand_mw=8.0,
            load_factor=0.55,
            weekend_factor=0.85,
            hdd_sensitivity_mwh_per_degree=2.0,
            cdd_sensitivity_mwh_per_degree=3.5,
        ),
        CustomerProfile(
            customer_id="CUST-CONTINENTAL-01",
            name="Continental Automotive",
            segment=CustomerSegment.INDUSTRIAL,
            annual_consumption_mwh=42000,
            peak_demand_mw=6.5,
            load_factor=0.78,
            weekend_factor=0.70,
        ),
        CustomerProfile(
            customer_id="CUST-DEDEMAN-01",
            name="Dedeman DIY",
            segment=CustomerSegment.MIXED,
            annual_consumption_mwh=18000,
            peak_demand_mw=5.5,
            load_factor=0.58,
            weekend_factor=0.90,
            hdd_sensitivity_mwh_per_degree=1.0,
            cdd_sensitivity_mwh_per_degree=1.5,
        ),
    ]


def generate_demo_consumption_forecast(
    horizon_days: int = 7,
    seed: int = 42,
) -> PortfolioConsumptionForecast:
    """Generate demo portfolio consumption forecast."""
    customers = generate_demo_customers()
    target = date.today() + timedelta(days=1)

    return forecast_consumption_portfolio(
        customers=customers,
        target_date=target,
        horizon_days=horizon_days,
        n_scenarios=50,
        seed=seed,
    )


# ============================================================================
# END OF CONSUMPTION FORECASTER
# ============================================================================

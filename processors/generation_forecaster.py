"""
Generation Profile Forecaster — Multi-Probability 15-min Curves
================================================================
Produces probabilistic generation forecasts for PV and Wind assets
at 15-minute granularity. Used by the supply pipeline to quantify
generation volume uncertainty and compute imbalance exposure.

Methodology:
  PV Forecasting:
    1. Clear-sky irradiance model (simplified Ineichen / PVLib-lite)
    2. Cloud cover ensemble → irradiance scenarios
    3. PV system model (DC/AC, temperature, soiling, degradation)
    4. Statistical post-processing → P10/P25/P50/P75/P90 quantiles

  Wind Forecasting:
    1. Wind speed distribution (Weibull) at hub height
    2. Power curve mapping (turbine-specific)
    3. Wake and availability loss
    4. Statistical uncertainty bands → P10-P90

Output:
  - 15-min generation profiles for D+1 to D+7
  - Multi-probability bands: P10, P25, P50, P75, P90
  - Technology-specific: PV and Wind treated separately
  - Aggregation: portfolio-level curves from asset-level forecasts

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

INTERVALS_PER_HOUR = 4   # 15-minute intervals
INTERVALS_PER_DAY = 96   # 24 * 4
HOURS_PER_DAY = 24

# Romania geographic center (for default solar calculations)
DEFAULT_LATITUDE = 44.4   # Bucharest approximate
DEFAULT_LONGITUDE = 26.1
DEFAULT_ALTITUDE = 90     # meters

# Standard percentile levels
PERCENTILES = [10, 25, 50, 75, 90]
PERCENTILE_LABELS = ["P10", "P25", "P50", "P75", "P90"]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class Technology(Enum):
    """Generation technology type."""
    PV = "pv"
    WIND = "wind"
    BESS = "bess"         # For hybrid dispatch
    PV_BESS = "pv_bess"   # Hybrid PV+BESS


@dataclass
class AssetSpec:
    """Asset specification for forecast generation."""
    asset_id: str
    name: str
    technology: Technology
    capacity_mw: float
    latitude: float = DEFAULT_LATITUDE
    longitude: float = DEFAULT_LONGITUDE

    # PV-specific
    tilt_deg: float = 25.0
    azimuth_deg: float = 180.0  # South-facing
    dc_ac_ratio: float = 1.25
    tracking: bool = True       # Single-axis tracking
    degradation_year: int = 1   # Years since commissioning
    annual_degradation_pct: float = 0.007
    first_year_lid_pct: float = 0.025
    soiling_loss_pct: float = 0.02
    inverter_efficiency: float = 0.975
    temp_coefficient: float = -0.005  # %/°C

    # Wind-specific
    hub_height_m: float = 100.0
    rotor_diameter_m: float = 136.0
    rated_wind_speed_ms: float = 12.0
    cut_in_speed_ms: float = 3.0
    cut_out_speed_ms: float = 25.0
    weibull_k: float = 2.1         # Shape parameter (Romania typical)
    weibull_a: float = 7.5         # Scale parameter (m/s, Romania typical)
    availability_pct: float = 0.97
    wake_loss_pct: float = 0.08
    electrical_loss_pct: float = 0.02


@dataclass
class GenerationForecast:
    """Multi-probability generation forecast result."""
    asset_id: str
    technology: Technology
    capacity_mw: float
    forecast_date: date
    horizon_days: int
    granularity_minutes: int  # 15

    # Core forecast: DataFrame with 15-min timestamps and P10-P90 columns
    profiles: pd.DataFrame  # columns: timestamp, P10, P25, P50, P75, P90 (MW)

    # Summary statistics
    daily_energy_mwh: Dict[str, float] = field(default_factory=dict)  # P10-P90
    capacity_factor_pct: Dict[str, float] = field(default_factory=dict)
    peak_generation_mw: Dict[str, float] = field(default_factory=dict)
    zero_generation_hours: int = 0

    # Metadata
    generation_timestamp: datetime = field(default_factory=datetime.now)
    methodology: str = ""


@dataclass
class PortfolioForecast:
    """Aggregated portfolio-level generation forecast."""
    forecast_date: date
    horizon_days: int
    total_capacity_mw: float
    asset_forecasts: List[GenerationForecast]
    portfolio_profiles: pd.DataFrame  # Aggregated P10-P90
    technology_breakdown: Dict[str, pd.DataFrame] = field(default_factory=dict)
    correlation_matrix: Optional[pd.DataFrame] = None


# ============================================================================
# PV GENERATION FORECASTER
# ============================================================================

class PVForecaster:
    """
    Probabilistic PV generation forecaster.

    Produces 15-min P10-P90 generation curves using a simplified
    clear-sky + cloud ensemble approach.
    """

    def __init__(self, asset: AssetSpec):
        """
        Initialize PV forecaster for a specific asset.

        Args:
            asset: Asset specification.
        """
        if asset.technology not in (Technology.PV, Technology.PV_BESS):
            raise ValueError(f"PVForecaster requires PV asset, got {asset.technology}")
        self.asset = asset

    def forecast(
        self,
        target_date: date,
        horizon_days: int = 7,
        n_scenarios: int = 100,
        seed: Optional[int] = None,
    ) -> GenerationForecast:
        """
        Generate multi-probability PV forecast.

        Args:
            target_date: First forecast day (D+1).
            horizon_days: Number of days to forecast (default 7).
            n_scenarios: Monte Carlo ensemble size (default 100).
            seed: Random seed for reproducibility.

        Returns:
            GenerationForecast with 15-min P10-P90 profiles.
        """
        rng = np.random.default_rng(seed)

        # Generate 15-min timestamps
        timestamps = pd.date_range(
            start=target_date,
            periods=INTERVALS_PER_DAY * horizon_days,
            freq="15min",
            tz="Europe/Bucharest",
        )

        n_intervals = len(timestamps)

        # Step 1: Clear-sky irradiance model (GHI)
        clear_sky_ghi = self._compute_clear_sky_ghi(timestamps)

        # Step 2: Cloud cover ensemble → GHI scenarios
        ghi_scenarios = self._apply_cloud_ensemble(
            clear_sky_ghi, timestamps, n_scenarios, rng
        )

        # Step 3: GHI → POA irradiance (simplified transposition)
        poa_scenarios = self._transpose_to_poa(ghi_scenarios, timestamps)

        # Step 4: POA → DC power → AC power
        ac_scenarios = self._pv_system_model(poa_scenarios, timestamps, rng)

        # Step 5: Extract percentiles
        percentile_profiles = np.percentile(
            ac_scenarios, PERCENTILES, axis=0
        )  # shape: (5, n_intervals)

        # Build result DataFrame
        profiles_df = pd.DataFrame(
            {"timestamp": timestamps},
        )
        for i, label in enumerate(PERCENTILE_LABELS):
            profiles_df[label] = np.round(percentile_profiles[i], 3)

        # Clip to capacity
        for label in PERCENTILE_LABELS:
            profiles_df[label] = profiles_df[label].clip(0, self.asset.capacity_mw)

        # Summary statistics
        daily_energy = {}
        capacity_factor = {}
        peak_gen = {}

        for label in PERCENTILE_LABELS:
            total_mwh = profiles_df[label].sum() * 0.25  # 15-min → MWh
            daily_energy[label] = round(total_mwh / horizon_days, 2)
            capacity_factor[label] = round(
                total_mwh / (self.asset.capacity_mw * 24 * horizon_days) * 100, 2
            )
            peak_gen[label] = round(profiles_df[label].max(), 3)

        zero_hours = int((profiles_df["P50"] < 0.001).sum() / INTERVALS_PER_HOUR)

        return GenerationForecast(
            asset_id=self.asset.asset_id,
            technology=self.asset.technology,
            capacity_mw=self.asset.capacity_mw,
            forecast_date=target_date,
            horizon_days=horizon_days,
            granularity_minutes=15,
            profiles=profiles_df,
            daily_energy_mwh=daily_energy,
            capacity_factor_pct=capacity_factor,
            peak_generation_mw=peak_gen,
            zero_generation_hours=zero_hours,
            methodology="clear-sky + cloud ensemble + PV system model",
        )

    def _compute_clear_sky_ghi(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Simplified clear-sky GHI model.

        Uses the Ineichen approximation: GHI_cs ≈ I0 * sin(α) * τ
        where α is solar elevation, τ is atmospheric transmittance.
        """
        lat_rad = np.radians(self.asset.latitude)
        n_intervals = len(timestamps)
        ghi = np.zeros(n_intervals)

        for i, ts in enumerate(timestamps):
            # Day of year
            doy = ts.timetuple().tm_yday

            # Solar declination (Spencer, 1971)
            B = 2 * np.pi * (doy - 1) / 365
            declination = (
                0.006918 - 0.399912 * np.cos(B) + 0.070257 * np.sin(B)
                - 0.006758 * np.cos(2 * B) + 0.000907 * np.sin(2 * B)
            )

            # Hour angle
            solar_time = ts.hour + ts.minute / 60.0  # Local solar time approx
            hour_angle = np.radians(15 * (solar_time - 12))

            # Solar elevation
            sin_elevation = (
                np.sin(lat_rad) * np.sin(declination)
                + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)
            )

            if sin_elevation > 0.01:
                # Extraterrestrial radiation
                i0 = 1361 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365))
                # Atmospheric transmittance (Ineichen simplified)
                air_mass = 1.0 / (sin_elevation + 0.50572 * (np.degrees(np.arcsin(sin_elevation)) + 6.07995) ** (-1.6364))
                air_mass = min(air_mass, 40)
                transmittance = 0.75  # Clear-sky atmospheric transmittance
                ghi[i] = i0 * sin_elevation * transmittance ** air_mass
            else:
                ghi[i] = 0.0

        return np.clip(ghi, 0, 1200)  # W/m²

    def _apply_cloud_ensemble(
        self,
        clear_sky_ghi: np.ndarray,
        timestamps: pd.DatetimeIndex,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Apply stochastic cloud cover ensemble to clear-sky GHI.

        Uses a beta-distributed cloud factor with temporal correlation.
        """
        n_intervals = len(clear_sky_ghi)
        scenarios = np.zeros((n_scenarios, n_intervals))

        for s in range(n_scenarios):
            # Daily cloud factor: Beta distribution (clear-sky bias)
            # alpha > beta → skewed toward clear skies
            n_days = int(np.ceil(n_intervals / INTERVALS_PER_DAY))
            daily_cloud = rng.beta(3.5, 2.0, n_days)  # Mean ~0.64

            # Expand to 15-min with intra-day variability
            cloud_factor = np.repeat(daily_cloud, INTERVALS_PER_DAY)[:n_intervals]

            # Add intra-day noise (correlated)
            intra_day_noise = rng.normal(0, 0.08, n_intervals)
            # Apply exponential smoothing for temporal correlation
            for j in range(1, n_intervals):
                intra_day_noise[j] = 0.85 * intra_day_noise[j - 1] + 0.15 * intra_day_noise[j]

            cloud_factor = np.clip(cloud_factor + intra_day_noise, 0.05, 1.0)
            scenarios[s] = clear_sky_ghi * cloud_factor

        return scenarios  # shape: (n_scenarios, n_intervals)

    def _transpose_to_poa(
        self,
        ghi_scenarios: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        Simplified GHI → POA transposition.

        For tracked systems: POA ≈ GHI * 1.15 (single-axis tracking gain)
        For fixed-tilt: POA ≈ GHI * cos(tilt) adjustment
        """
        if self.asset.tracking:
            # Single-axis tracking: ~15% gain over horizontal
            tracking_gain = 1.15
        else:
            # Fixed tilt: simplified Perez-like correction
            tracking_gain = 1.0 + 0.1 * np.cos(np.radians(self.asset.tilt_deg - self.asset.latitude))

        return ghi_scenarios * tracking_gain

    def _pv_system_model(
        self,
        poa_scenarios: np.ndarray,
        timestamps: pd.DatetimeIndex,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        POA irradiance → AC power output.

        Applies:
          - Module efficiency and area (implicit via capacity)
          - Temperature derating
          - Soiling
          - Degradation (LID + annual)
          - Inverter efficiency and DC/AC clipping
        """
        n_scenarios, n_intervals = poa_scenarios.shape

        # Reference irradiance at STC
        ghi_stc = 1000.0  # W/m²

        # Nominal DC power per unit irradiance
        # capacity_mw is AC nameplate; DC = AC * DC/AC ratio
        dc_capacity_mw = self.asset.capacity_mw * self.asset.dc_ac_ratio

        # Temperature effect (simplified: assume ambient = 10-35°C seasonal)
        months = np.array(timestamps.month)
        ambient_temp = np.where(
            np.isin(months, [12, 1, 2]), 2,
            np.where(np.isin(months, [6, 7, 8]), 28,
            np.where(np.isin(months, [3, 4, 5, 9, 10, 11]), 15, 15))
        ).astype(float)
        # Cell temperature ≈ ambient + 0.03 * POA (simplified NOCT)
        # We'll apply per-scenario
        results = np.zeros_like(poa_scenarios)

        for s in range(n_scenarios):
            poa = poa_scenarios[s]

            # DC power (linear model)
            dc_power = dc_capacity_mw * (poa / ghi_stc)

            # Temperature derating
            cell_temp = ambient_temp + 0.03 * poa
            temp_loss = 1.0 + self.asset.temp_coefficient * (cell_temp - 25.0)
            dc_power *= np.clip(temp_loss, 0.8, 1.05)

            # Soiling
            dc_power *= (1.0 - self.asset.soiling_loss_pct)

            # Degradation
            if self.asset.degradation_year <= 1:
                deg_factor = 1.0 - self.asset.first_year_lid_pct
            else:
                deg_factor = (
                    (1.0 - self.asset.first_year_lid_pct)
                    * (1.0 - self.asset.annual_degradation_pct) ** (self.asset.degradation_year - 1)
                )
            dc_power *= deg_factor

            # Inverter: DC/AC clipping + efficiency
            ac_power = np.minimum(dc_power, self.asset.capacity_mw) * self.asset.inverter_efficiency

            # Small random perturbation (equipment variability)
            ac_power *= (1.0 + rng.normal(0, 0.01, n_intervals))

            results[s] = np.clip(ac_power, 0, self.asset.capacity_mw)

        return results


# ============================================================================
# WIND GENERATION FORECASTER
# ============================================================================

class WindForecaster:
    """
    Probabilistic wind generation forecaster.

    Produces 15-min P10-P90 generation curves using Weibull wind speed
    distribution and turbine power curve mapping.
    """

    def __init__(self, asset: AssetSpec):
        if asset.technology != Technology.WIND:
            raise ValueError(f"WindForecaster requires WIND asset, got {asset.technology}")
        self.asset = asset

    def forecast(
        self,
        target_date: date,
        horizon_days: int = 7,
        n_scenarios: int = 100,
        seed: Optional[int] = None,
    ) -> GenerationForecast:
        """
        Generate multi-probability wind forecast.

        Args:
            target_date: First forecast day.
            horizon_days: Days to forecast.
            n_scenarios: Ensemble size.
            seed: Random seed.

        Returns:
            GenerationForecast with P10-P90 wind profiles.
        """
        rng = np.random.default_rng(seed)

        timestamps = pd.date_range(
            start=target_date,
            periods=INTERVALS_PER_DAY * horizon_days,
            freq="15min",
            tz="Europe/Bucharest",
        )
        n_intervals = len(timestamps)

        # Step 1: Wind speed scenarios (Weibull + temporal correlation)
        wind_scenarios = self._generate_wind_scenarios(
            timestamps, n_scenarios, rng
        )

        # Step 2: Power curve mapping
        power_scenarios = self._apply_power_curve(wind_scenarios)

        # Step 3: Apply losses (wake, availability, electrical)
        power_scenarios = self._apply_losses(power_scenarios, rng)

        # Step 4: Extract percentiles
        percentile_profiles = np.percentile(
            power_scenarios, PERCENTILES, axis=0
        )

        profiles_df = pd.DataFrame({"timestamp": timestamps})
        for i, label in enumerate(PERCENTILE_LABELS):
            profiles_df[label] = np.round(percentile_profiles[i], 3)
            profiles_df[label] = profiles_df[label].clip(0, self.asset.capacity_mw)

        # Summary
        daily_energy = {}
        capacity_factor = {}
        peak_gen = {}

        for label in PERCENTILE_LABELS:
            total_mwh = profiles_df[label].sum() * 0.25
            daily_energy[label] = round(total_mwh / horizon_days, 2)
            capacity_factor[label] = round(
                total_mwh / (self.asset.capacity_mw * 24 * horizon_days) * 100, 2
            )
            peak_gen[label] = round(profiles_df[label].max(), 3)

        zero_hours = int((profiles_df["P50"] < 0.001).sum() / INTERVALS_PER_HOUR)

        return GenerationForecast(
            asset_id=self.asset.asset_id,
            technology=Technology.WIND,
            capacity_mw=self.asset.capacity_mw,
            forecast_date=target_date,
            horizon_days=horizon_days,
            granularity_minutes=15,
            profiles=profiles_df,
            daily_energy_mwh=daily_energy,
            capacity_factor_pct=capacity_factor,
            peak_generation_mw=peak_gen,
            zero_generation_hours=zero_hours,
            methodology="Weibull ensemble + power curve + loss model",
        )

    def _generate_wind_scenarios(
        self,
        timestamps: pd.DatetimeIndex,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate wind speed scenarios using Weibull distribution with
        temporal correlation (AR(1) process).
        """
        n_intervals = len(timestamps)
        hours = np.array(timestamps.hour)
        months = np.array(timestamps.month)

        # Seasonal Weibull scale adjustment
        seasonal_factor = np.where(
            np.isin(months, [11, 12, 1, 2, 3]), 1.15,  # Winter: higher wind
            np.where(np.isin(months, [6, 7, 8]), 0.85,  # Summer: lower wind
                     1.0)
        )

        # Diurnal pattern: slightly higher afternoon wind
        diurnal_factor = 1.0 + 0.08 * np.sin(2 * np.pi * (hours - 6) / 24)

        effective_scale = self.asset.weibull_a * seasonal_factor * diurnal_factor

        scenarios = np.zeros((n_scenarios, n_intervals))

        for s in range(n_scenarios):
            # Weibull samples
            ws = rng.weibull(self.asset.weibull_k, n_intervals) * effective_scale

            # Apply temporal correlation (AR(1), ρ ≈ 0.92 at 15-min)
            ar_coeff = 0.92
            for j in range(1, n_intervals):
                ws[j] = ar_coeff * ws[j - 1] + (1 - ar_coeff) * ws[j]

            scenarios[s] = np.clip(ws, 0, 35)  # m/s

        return scenarios

    def _apply_power_curve(self, wind_scenarios: np.ndarray) -> np.ndarray:
        """
        Map wind speed to power output using simplified IEC power curve.

        Cubic relationship between cut-in and rated, flat at rated,
        zero above cut-out.
        """
        v_ci = self.asset.cut_in_speed_ms
        v_r = self.asset.rated_wind_speed_ms
        v_co = self.asset.cut_out_speed_ms
        p_rated = self.asset.capacity_mw

        power = np.zeros_like(wind_scenarios)

        # Cubic region: cut-in to rated
        cubic_mask = (wind_scenarios >= v_ci) & (wind_scenarios < v_r)
        power[cubic_mask] = p_rated * (
            (wind_scenarios[cubic_mask] - v_ci) / (v_r - v_ci)
        ) ** 3

        # Rated region: rated to cut-out
        rated_mask = (wind_scenarios >= v_r) & (wind_scenarios <= v_co)
        power[rated_mask] = p_rated

        # Cut-out: zero above cut-out
        # (default: power stays 0)

        return np.clip(power, 0, p_rated)

    def _apply_losses(
        self,
        power_scenarios: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply wake, availability, and electrical losses."""
        wake_factor = 1.0 - self.asset.wake_loss_pct
        elec_factor = 1.0 - self.asset.electrical_loss_pct

        # Availability: random outage events
        n_scenarios, n_intervals = power_scenarios.shape
        availability = np.ones((n_scenarios, n_intervals))

        for s in range(n_scenarios):
            # Random forced outages (Poisson process, ~3% downtime)
            n_outages = rng.poisson(n_intervals * 0.0005)
            if n_outages > 0:
                outage_starts = rng.integers(0, n_intervals, n_outages)
                outage_durations = rng.integers(4, 48, n_outages)  # 1-12 hours
                for start, dur in zip(outage_starts, outage_durations):
                    end = min(start + dur, n_intervals)
                    availability[s, start:end] = 0.0

        return power_scenarios * wake_factor * elec_factor * availability


# ============================================================================
# PORTFOLIO AGGREGATION
# ============================================================================

def forecast_portfolio(
    assets: List[AssetSpec],
    target_date: date,
    horizon_days: int = 7,
    n_scenarios: int = 100,
    seed: Optional[int] = None,
    correlation: float = 0.6,
) -> PortfolioForecast:
    """
    Generate aggregated portfolio-level generation forecast.

    Produces individual asset forecasts and aggregates them with
    inter-asset correlation handling.

    Args:
        assets: List of asset specifications.
        target_date: First forecast day.
        horizon_days: Forecast horizon.
        n_scenarios: Ensemble size.
        seed: Random seed.
        correlation: Inter-asset correlation (same technology).

    Returns:
        PortfolioForecast with asset and portfolio-level profiles.
    """
    logger.info(
        f"Forecasting portfolio: {len(assets)} assets, "
        f"{sum(a.capacity_mw for a in assets):.1f} MW total, "
        f"{horizon_days} days"
    )

    asset_forecasts = []
    rng = np.random.default_rng(seed)

    for asset in assets:
        asset_seed = rng.integers(0, 2**31)

        if asset.technology in (Technology.PV, Technology.PV_BESS):
            forecaster = PVForecaster(asset)
        elif asset.technology == Technology.WIND:
            forecaster = WindForecaster(asset)
        else:
            logger.warning(f"Unsupported technology {asset.technology} for {asset.asset_id}, skipping")
            continue

        fc = forecaster.forecast(
            target_date=target_date,
            horizon_days=horizon_days,
            n_scenarios=n_scenarios,
            seed=asset_seed,
        )
        asset_forecasts.append(fc)
        logger.info(
            f"  {asset.asset_id}: {asset.capacity_mw:.1f} MW {asset.technology.value} | "
            f"CF P50={fc.capacity_factor_pct.get('P50', 0):.1f}%"
        )

    if not asset_forecasts:
        raise ValueError("No valid assets to forecast")

    # Aggregate: sum asset profiles
    # Note: simple summation assumes imperfect correlation already captured
    # in the individual scenarios. For strict correlation control, use
    # copula-based aggregation (future enhancement).
    ref_timestamps = asset_forecasts[0].profiles["timestamp"]
    portfolio_profiles = pd.DataFrame({"timestamp": ref_timestamps})

    for label in PERCENTILE_LABELS:
        portfolio_profiles[label] = sum(
            fc.profiles[label].values for fc in asset_forecasts
        )
        portfolio_profiles[label] = np.round(portfolio_profiles[label], 3)

    total_capacity = sum(a.capacity_mw for a in assets)

    # Technology breakdown
    tech_breakdown = {}
    for tech in [Technology.PV, Technology.WIND]:
        tech_fcs = [fc for fc in asset_forecasts if fc.technology == tech]
        if tech_fcs:
            tech_df = pd.DataFrame({"timestamp": ref_timestamps})
            for label in PERCENTILE_LABELS:
                tech_df[label] = sum(fc.profiles[label].values for fc in tech_fcs)
            tech_breakdown[tech.value] = tech_df

    return PortfolioForecast(
        forecast_date=target_date,
        horizon_days=horizon_days,
        total_capacity_mw=total_capacity,
        asset_forecasts=asset_forecasts,
        portfolio_profiles=portfolio_profiles,
        technology_breakdown=tech_breakdown,
    )


# ============================================================================
# DEMO DATA GENERATION
# ============================================================================

def generate_demo_portfolio() -> List[AssetSpec]:
    """
    Generate a demo portfolio matching nextE's ~130 MW RtM PV fleet
    plus illustrative wind assets.

    Returns:
        List of AssetSpec for demo/testing.
    """
    return [
        AssetSpec(
            asset_id="PV-GIURGIU-01",
            name="Giurgiu Solar Park",
            technology=Technology.PV,
            capacity_mw=50.0,
            latitude=43.9,
            longitude=25.9,
            tracking=True,
            degradation_year=2,
        ),
        AssetSpec(
            asset_id="PV-CONSTANTA-01",
            name="Constanta Solar Park",
            technology=Technology.PV,
            capacity_mw=35.0,
            latitude=44.2,
            longitude=28.6,
            tracking=True,
            degradation_year=1,
        ),
        AssetSpec(
            asset_id="PV-DOLJ-01",
            name="Dolj Solar Park",
            technology=Technology.PV,
            capacity_mw=25.0,
            latitude=44.3,
            longitude=23.8,
            tracking=False,
            tilt_deg=30.0,
            degradation_year=3,
        ),
        AssetSpec(
            asset_id="PV-IALOMITA-01",
            name="Ialomita Solar Park",
            technology=Technology.PV,
            capacity_mw=20.0,
            latitude=44.6,
            longitude=27.3,
            tracking=True,
            degradation_year=1,
        ),
        AssetSpec(
            asset_id="WIND-DOBROGEA-01",
            name="Dobrogea Wind Farm",
            technology=Technology.WIND,
            capacity_mw=48.0,
            latitude=44.5,
            longitude=28.3,
            weibull_k=2.2,
            weibull_a=8.2,  # Dobrogea has good wind resource
            hub_height_m=110.0,
        ),
    ]


def generate_demo_forecast(
    horizon_days: int = 7,
    seed: int = 42,
) -> PortfolioForecast:
    """
    Generate a complete demo portfolio forecast for testing.

    Args:
        horizon_days: Forecast horizon.
        seed: Random seed.

    Returns:
        PortfolioForecast with demo data.
    """
    assets = generate_demo_portfolio()
    target = date.today() + timedelta(days=1)

    return forecast_portfolio(
        assets=assets,
        target_date=target,
        horizon_days=horizon_days,
        n_scenarios=50,  # Fewer for demo speed
        seed=seed,
    )


# ============================================================================
# END OF GENERATION FORECASTER
# ============================================================================

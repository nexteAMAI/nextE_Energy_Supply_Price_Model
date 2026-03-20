"""
Supply Pipeline Orchestrator — End-to-End Supply Workflow
=========================================================
Orchestrates the complete supply pricing and risk management workflow,
coordinating data extraction, forecasting, pricing, procurement
optimization, and risk quantification into a single execution pipeline.

Pipeline Stages:
  Stage 1 — Data Extraction:   DAMAS system data + market data
  Stage 2 — Generation Forecast: PV + Wind multi-probability profiles
  Stage 3 — Consumption Forecast: Customer load profiles
  Stage 4 — Gap Analysis:       Generation vs. consumption mismatch
  Stage 5 — Pricing Waterfall:  Per-contract supply price calculation
  Stage 6 — Procurement:        Channel allocation optimization
  Stage 7 — Risk Assessment:    Portfolio VaR, shape/volume/price risk
  Stage 8 — P&L Projection:     Budget vs. actual margin projection
  Stage 9 — Reporting:          Summary output + alerts

Execution Modes:
  - full:      All stages end-to-end (daily operational run)
  - pricing:   Stages 2-5 only (new offer preparation)
  - risk:      Stages 2-3 + 7 only (risk monitoring refresh)
  - rebalance: Stages 2-4 + 6 only (intraday procurement adjustment)

Author: nextE AI Workstation
Version: 1.0.0
Date: 2026-03-20
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# CONSTANTS
# ============================================================================

PIPELINE_VERSION = "1.0.0"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class PipelineMode(Enum):
    """Pipeline execution mode."""
    FULL = "full"
    PRICING = "pricing"
    RISK = "risk"
    REBALANCE = "rebalance"


class StageStatus(Enum):
    """Individual stage execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result of a single pipeline stage execution."""
    stage_name: str
    status: StageStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    records_processed: int = 0
    output_data: Any = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Pipeline execution configuration."""
    mode: PipelineMode = PipelineMode.FULL
    target_date: Optional[date] = None  # Default: tomorrow
    forecast_horizon_days: int = 7
    n_scenarios: int = 50
    seed: int = 42

    # Data extraction
    entsoe_api_key: Optional[str] = None
    lookback_days: int = 30  # Historical data for context

    # Pricing
    use_live_forward_curve: bool = False
    margin_override_eur_per_mwh: Optional[float] = None

    # Output
    export_excel: bool = True
    export_json: bool = True
    output_dir: Optional[str] = None

    # Alerting
    alert_on_margin_breach: bool = True
    alert_on_risk_breach: bool = True
    min_margin_floor_eur_per_mwh: float = 8.0


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    pipeline_id: str
    mode: PipelineMode
    config: PipelineConfig
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    stages: Dict[str, StageResult] = field(default_factory=dict)
    alerts: List[Dict] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    success: bool = False


# ============================================================================
# PIPELINE STAGES
# ============================================================================

def _run_stage(
    stage_name: str,
    func,
    *args,
    **kwargs,
) -> StageResult:
    """Execute a pipeline stage with timing and error handling."""
    start = datetime.now()
    logger.info(f"  [{stage_name}] Starting...")

    try:
        result_data = func(*args, **kwargs)
        end = datetime.now()
        duration = (end - start).total_seconds()

        records = 0
        if isinstance(result_data, pd.DataFrame):
            records = len(result_data)
        elif isinstance(result_data, dict):
            records = sum(
                len(v) for v in result_data.values() if isinstance(v, (pd.DataFrame, list))
            )

        logger.info(f"  [{stage_name}] Completed in {duration:.1f}s ({records} records)")

        return StageResult(
            stage_name=stage_name,
            status=StageStatus.COMPLETED,
            start_time=start,
            end_time=end,
            duration_seconds=duration,
            records_processed=records,
            output_data=result_data,
        )

    except Exception as e:
        end = datetime.now()
        duration = (end - start).total_seconds()
        logger.error(f"  [{stage_name}] FAILED after {duration:.1f}s: {e}")

        return StageResult(
            stage_name=stage_name,
            status=StageStatus.FAILED,
            start_time=start,
            end_time=end,
            duration_seconds=duration,
            error_message=str(e),
        )


# ============================================================================
# STAGE 1: DATA EXTRACTION
# ============================================================================

def stage_data_extraction(config: PipelineConfig) -> Dict:
    """
    Stage 1: Extract system data from DAMAS / ENTSO-E.

    Returns:
        Dict with keys: imbalance_prices, generation_mix, cross_border, system_load
    """
    from extractors.damas_client import DAMASClient

    target = config.target_date or (date.today() + timedelta(days=1))
    lookback_start = target - timedelta(days=config.lookback_days)

    client = DAMASClient(entsoe_api_key=config.entsoe_api_key)
    try:
        bundle = client.get_full_data_bundle(lookback_start, target)
        return {
            "imbalance_prices": bundle.imbalance_prices,
            "generation_mix": bundle.generation_mix,
            "cross_border_flows": bundle.cross_border_flows,
            "system_load": bundle.system_load,
            "metadata": bundle.metadata,
        }
    finally:
        client.close()


# ============================================================================
# STAGE 2: GENERATION FORECAST
# ============================================================================

def stage_generation_forecast(config: PipelineConfig) -> Dict:
    """
    Stage 2: Generate multi-probability PV + Wind forecasts.

    Returns:
        Dict with portfolio_forecast and per-asset forecasts.
    """
    from processors.generation_forecaster import (
        forecast_portfolio,
        generate_demo_portfolio,
    )

    target = config.target_date or (date.today() + timedelta(days=1))

    # Use demo portfolio (in production, load from Supabase asset register)
    assets = generate_demo_portfolio()

    portfolio_fc = forecast_portfolio(
        assets=assets,
        target_date=target,
        horizon_days=config.forecast_horizon_days,
        n_scenarios=config.n_scenarios,
        seed=config.seed,
    )

    return {
        "portfolio_forecast": portfolio_fc,
        "portfolio_profiles": portfolio_fc.portfolio_profiles,
        "technology_breakdown": portfolio_fc.technology_breakdown,
        "total_capacity_mw": portfolio_fc.total_capacity_mw,
        "asset_count": len(portfolio_fc.asset_forecasts),
    }


# ============================================================================
# STAGE 3: CONSUMPTION FORECAST
# ============================================================================

def stage_consumption_forecast(config: PipelineConfig) -> Dict:
    """
    Stage 3: Generate multi-probability customer consumption forecasts.

    Returns:
        Dict with portfolio consumption forecast and per-customer forecasts.
    """
    from processors.consumption_forecaster import (
        forecast_consumption_portfolio,
        generate_demo_customers,
    )

    target = config.target_date or (date.today() + timedelta(days=1))

    # Use demo customers (in production, load from Supabase CRM)
    customers = generate_demo_customers()

    consumption_fc = forecast_consumption_portfolio(
        customers=customers,
        target_date=target,
        horizon_days=config.forecast_horizon_days,
        n_scenarios=config.n_scenarios,
        seed=config.seed + 100,  # Different seed from generation
    )

    return {
        "portfolio_forecast": consumption_fc,
        "portfolio_profiles": consumption_fc.portfolio_profiles,
        "segment_breakdown": consumption_fc.segment_breakdown,
        "total_annual_mwh": consumption_fc.total_annual_mwh,
        "total_peak_mw": consumption_fc.total_peak_mw,
        "customer_count": len(consumption_fc.customer_forecasts),
    }


# ============================================================================
# STAGE 4: GAP ANALYSIS
# ============================================================================

def stage_gap_analysis(
    generation_data: Dict,
    consumption_data: Dict,
) -> Dict:
    """
    Stage 4: Compute generation-consumption gap (procurement requirement).

    The gap = consumption - generation represents the volume that must be
    procured from wholesale markets. Negative gap = excess generation to sell.

    Returns:
        Dict with gap profiles (P10-P90), gap statistics, and hourly summary.
    """
    gen_profiles = generation_data["portfolio_profiles"]
    con_profiles = consumption_data["portfolio_profiles"]

    # Align timestamps
    merged = pd.merge(
        gen_profiles.rename(columns={c: f"gen_{c}" for c in gen_profiles.columns if c != "timestamp"}),
        con_profiles.rename(columns={c: f"con_{c}" for c in con_profiles.columns if c != "timestamp"}),
        on="timestamp",
        how="inner",
    )

    gap_profiles = pd.DataFrame({"timestamp": merged["timestamp"]})

    percentile_labels = ["P10", "P25", "P50", "P75", "P90"]

    for label in percentile_labels:
        gen_col = f"gen_{label}"
        con_col = f"con_{label}"
        # Gap: positive = need to buy, negative = excess generation
        gap_profiles[label] = merged[con_col] - merged[gen_col]

    # Statistics
    gap_stats = {}
    for label in percentile_labels:
        gap_series = gap_profiles[label]
        gap_stats[label] = {
            "mean_mw": round(gap_series.mean(), 2),
            "max_mw": round(gap_series.max(), 2),
            "min_mw": round(gap_series.min(), 2),
            "total_mwh": round(gap_series.sum() * 0.25, 2),  # 15-min → MWh
            "hours_positive": int((gap_series > 0).sum() / 4),
            "hours_negative": int((gap_series < 0).sum() / 4),
        }

    # Hourly aggregation for overview
    gap_profiles_hourly = gap_profiles.copy()
    gap_profiles_hourly["hour"] = gap_profiles_hourly["timestamp"].dt.hour
    hourly_avg = gap_profiles_hourly.groupby("hour")[percentile_labels].mean().reset_index()

    # Self-supply ratio (P50)
    gen_p50_total = gen_profiles["P50"].sum() * 0.25 if "P50" in gen_profiles.columns else 0
    con_p50_total = con_profiles["P50"].sum() * 0.25 if "P50" in con_profiles.columns else 0
    self_supply_ratio = gen_p50_total / con_p50_total if con_p50_total > 0 else 0

    return {
        "gap_profiles": gap_profiles,
        "gap_statistics": gap_stats,
        "hourly_average_gap": hourly_avg,
        "self_supply_ratio": round(self_supply_ratio, 4),
        "total_procurement_mwh_p50": gap_stats.get("P50", {}).get("total_mwh", 0),
        "total_excess_mwh_p50": abs(min(0, gap_stats.get("P50", {}).get("min_mw", 0))) * 0.25,
    }


# ============================================================================
# STAGE 5: PRICING WATERFALL
# ============================================================================

def stage_pricing(
    generation_data: Dict,
    consumption_data: Dict,
    gap_data: Dict,
    config: PipelineConfig,
) -> Dict:
    """
    Stage 5: Compute supply pricing waterfall for portfolio.

    Uses the processors.supply_pricing engine if available,
    otherwise computes a simplified waterfall.

    Returns:
        Dict with waterfall components, final price, and scenario analysis.
    """
    try:
        from config.settings import settings
        supply_cfg = settings._load_yaml("supply_config")
    except Exception:
        supply_cfg = {}

    reg = supply_cfg.get("regulatory", {})
    margins = supply_cfg.get("margins", {})

    # Extract key metrics
    gen_p50_mwh = generation_data["portfolio_profiles"]["P50"].sum() * 0.25
    con_p50_mwh = consumption_data["portfolio_profiles"]["P50"].sum() * 0.25
    gap_p50_mwh = gap_data.get("total_procurement_mwh_p50", 0)

    total_gen_cap = generation_data.get("total_capacity_mw", 130)

    # Solar share
    solar_share = gen_p50_mwh / con_p50_mwh if con_p50_mwh > 0 else 0

    # Simplified pricing waterfall
    pv_cost = 50.0  # EUR/MWh (PV procurement average)
    forward_price = 105.0  # EUR/MWh (forward curve Q2-Q4 average)
    blended_energy = solar_share * pv_cost + (1 - solar_share) * forward_price

    gc_cost = reg.get("gc_quota_coefficient", 0.499387) * reg.get("gc_cost_eur_per_mwh", 14.50) / reg.get("gc_quota_coefficient", 0.499387)
    gc_cost = reg.get("gc_cost_eur_per_mwh", 14.50)

    balancing_cost = reg.get("balancing_cost_eur_per_mwh", 3.0)
    transport_admin = 3.50  # EUR/MWh
    risk_premium = margins.get("default_risk_margin_eur_per_mwh", 5.0)

    margin = config.margin_override_eur_per_mwh or margins.get(
        "default_nexte_margin_eur_per_mwh", 12.0
    )

    final_price_ex_vat = (
        blended_energy + gc_cost + balancing_cost
        + transport_admin + risk_premium + margin
    )

    vat_rate = reg.get("vat_rate", 0.19)
    final_price_inc_vat = final_price_ex_vat * (1 + vat_rate)

    # Margin floor check
    min_floor = config.min_margin_floor_eur_per_mwh
    margin_breach = margin < min_floor

    waterfall = {
        "pv_procurement_cost": round(pv_cost, 2),
        "forward_procurement_cost": round(forward_price, 2),
        "blended_energy_cost": round(blended_energy, 2),
        "gc_quota_cost": round(gc_cost, 2),
        "balancing_cost": round(balancing_cost, 2),
        "transport_admin": round(transport_admin, 2),
        "risk_premium": round(risk_premium, 2),
        "nexte_margin": round(margin, 2),
        "final_price_ex_vat": round(final_price_ex_vat, 2),
        "vat_amount": round(final_price_ex_vat * vat_rate, 2),
        "final_price_inc_vat": round(final_price_inc_vat, 2),
    }

    # Multi-scenario (simplified P10-P90)
    scenario_prices = {}
    price_multipliers = {"P10": 0.85, "P25": 0.93, "P50": 1.0, "P75": 1.08, "P90": 1.18}
    for label, mult in price_multipliers.items():
        scenario_prices[label] = round(final_price_ex_vat * mult, 2)

    return {
        "waterfall": waterfall,
        "solar_share_pct": round(solar_share * 100, 2),
        "scenario_prices": scenario_prices,
        "margin_breach": margin_breach,
        "total_volume_mwh_p50": round(con_p50_mwh, 2),
    }


# ============================================================================
# STAGE 6: PROCUREMENT OPTIMIZATION
# ============================================================================

def stage_procurement(
    gap_data: Dict,
    config: PipelineConfig,
) -> Dict:
    """
    Stage 6: Optimize procurement channel allocation.

    Returns:
        Dict with channel allocation, total cost, and collateral requirement.
    """
    gap_mwh = gap_data.get("total_procurement_mwh_p50", 0)
    if gap_mwh <= 0:
        return {
            "allocation": {},
            "total_cost_eur": 0,
            "collateral_eur": 0,
            "status": "no_procurement_needed",
        }

    # Simplified channel allocation (in production, use MILP optimizer)
    channels = {
        "BRM Forward": {"share": 0.35, "price": 106.0, "collateral_pct": 0.15},
        "OPCOM Bilateral": {"share": 0.25, "price": 104.0, "collateral_pct": 0.20},
        "Direct Bilateral": {"share": 0.10, "price": 108.0, "collateral_pct": 0.25},
        "Spot DAM": {"share": 0.20, "price": 100.0, "collateral_pct": 0.05},
        "Spot IDM": {"share": 0.10, "price": 102.0, "collateral_pct": 0.08},
    }

    allocation = {}
    total_cost = 0
    total_collateral = 0

    for channel, params in channels.items():
        vol_mwh = gap_mwh * params["share"]
        cost = vol_mwh * params["price"]
        collateral = cost * params["collateral_pct"]

        allocation[channel] = {
            "volume_mwh": round(vol_mwh, 2),
            "price_eur_per_mwh": params["price"],
            "cost_eur": round(cost, 2),
            "collateral_eur": round(collateral, 2),
            "share_pct": round(params["share"] * 100, 1),
        }
        total_cost += cost
        total_collateral += collateral

    weighted_avg_price = total_cost / gap_mwh if gap_mwh > 0 else 0

    return {
        "allocation": allocation,
        "total_procurement_mwh": round(gap_mwh, 2),
        "total_cost_eur": round(total_cost, 2),
        "weighted_avg_price_eur_per_mwh": round(weighted_avg_price, 2),
        "total_collateral_eur": round(total_collateral, 2),
        "status": "optimized",
    }


# ============================================================================
# STAGE 7: RISK ASSESSMENT
# ============================================================================

def stage_risk_assessment(
    generation_data: Dict,
    consumption_data: Dict,
    gap_data: Dict,
    pricing_data: Dict,
    config: PipelineConfig,
) -> Dict:
    """
    Stage 7: Portfolio risk quantification.

    Returns:
        Dict with VaR, risk decomposition, position limits, and alerts.
    """
    gen_profiles = generation_data["portfolio_profiles"]
    con_profiles = consumption_data["portfolio_profiles"]

    # Volume risk: P10-P90 spread on generation
    gen_p10_mwh = gen_profiles["P10"].sum() * 0.25
    gen_p90_mwh = gen_profiles["P90"].sum() * 0.25
    gen_spread_mwh = gen_p90_mwh - gen_p10_mwh

    # Shape risk: hourly gap volatility
    gap_p50 = gap_data["gap_profiles"]["P50"]
    shape_risk_mw = gap_p50.std()

    # Price risk: simplified (forward price uncertainty)
    forward_price = pricing_data["waterfall"]["forward_procurement_cost"]
    price_vol_pct = 0.15  # 15% annualized volatility assumption
    price_risk_eur = (
        gap_data.get("total_procurement_mwh_p50", 0)
        * forward_price * price_vol_pct
        * np.sqrt(config.forecast_horizon_days / 365)
    )

    # Portfolio VaR (95%, horizon)
    # Simplified: sqrt(shape² + volume² + price²)
    volume_risk_eur = gen_spread_mwh * forward_price * 0.5
    shape_risk_eur = shape_risk_mw * forward_price * np.sqrt(config.forecast_horizon_days * 24)
    total_var_95 = np.sqrt(volume_risk_eur**2 + shape_risk_eur**2 + price_risk_eur**2)

    # Imbalance exposure
    gen_forecast_error_pct = 0.12
    imbalance_exposure_mw = generation_data.get("total_capacity_mw", 130) * gen_forecast_error_pct
    imbalance_cost_daily = imbalance_exposure_mw * 3.0 * 24  # 3 EUR/MWh * 24h

    # Risk limits check
    var_limit = 2_500_000  # 2.5M EUR
    imbalance_limit = 50_000  # 50k EUR/day
    var_status = "GREEN" if total_var_95 < var_limit * 0.8 else ("YELLOW" if total_var_95 < var_limit else "RED")
    imb_status = "GREEN" if imbalance_cost_daily < imbalance_limit * 0.8 else ("YELLOW" if imbalance_cost_daily < imbalance_limit else "RED")

    return {
        "var_95_eur": round(total_var_95, 2),
        "var_95_status": var_status,
        "volume_risk_eur": round(volume_risk_eur, 2),
        "shape_risk_eur": round(shape_risk_eur, 2),
        "price_risk_eur": round(price_risk_eur, 2),
        "shape_risk_mw": round(shape_risk_mw, 3),
        "gen_volume_spread_mwh": round(gen_spread_mwh, 2),
        "imbalance_exposure_mw": round(imbalance_exposure_mw, 2),
        "imbalance_cost_daily_eur": round(imbalance_cost_daily, 2),
        "imbalance_status": imb_status,
        "risk_decomposition": {
            "volume": round(volume_risk_eur / max(total_var_95, 1) * 100, 1),
            "shape": round(shape_risk_eur / max(total_var_95, 1) * 100, 1),
            "price": round(price_risk_eur / max(total_var_95, 1) * 100, 1),
        },
    }


# ============================================================================
# STAGE 8: P&L PROJECTION
# ============================================================================

def stage_pnl_projection(
    pricing_data: Dict,
    procurement_data: Dict,
    risk_data: Dict,
    config: PipelineConfig,
) -> Dict:
    """
    Stage 8: Project P&L for the forecast horizon.

    Returns:
        Dict with revenue, cost, margin projections.
    """
    total_volume = pricing_data.get("total_volume_mwh_p50", 0)
    final_price = pricing_data["waterfall"]["final_price_ex_vat"]
    procurement_cost = procurement_data.get("weighted_avg_price_eur_per_mwh", 0)
    solar_share = pricing_data.get("solar_share_pct", 0) / 100

    # Revenue
    revenue_eur = total_volume * final_price

    # Costs
    pv_cost = total_volume * solar_share * pricing_data["waterfall"]["pv_procurement_cost"]
    forward_cost = total_volume * (1 - solar_share) * procurement_cost
    gc_cost = total_volume * pricing_data["waterfall"]["gc_quota_cost"]
    balancing_cost = total_volume * pricing_data["waterfall"]["balancing_cost"]
    transport_cost = total_volume * pricing_data["waterfall"]["transport_admin"]
    risk_cost = total_volume * pricing_data["waterfall"]["risk_premium"]

    total_cost = pv_cost + forward_cost + gc_cost + balancing_cost + transport_cost + risk_cost

    # Margin
    gross_margin = revenue_eur - total_cost
    margin_per_mwh = gross_margin / max(total_volume, 1)
    margin_pct = gross_margin / max(revenue_eur, 1) * 100

    # Annualized
    days = config.forecast_horizon_days
    annual_factor = 365 / max(days, 1)
    annual_revenue = revenue_eur * annual_factor
    annual_margin = gross_margin * annual_factor

    return {
        "revenue_eur": round(revenue_eur, 2),
        "total_cost_eur": round(total_cost, 2),
        "gross_margin_eur": round(gross_margin, 2),
        "margin_per_mwh": round(margin_per_mwh, 2),
        "margin_pct": round(margin_pct, 2),
        "annualized_revenue_eur": round(annual_revenue, 2),
        "annualized_margin_eur": round(annual_margin, 2),
        "cost_breakdown": {
            "pv_procurement": round(pv_cost, 2),
            "forward_procurement": round(forward_cost, 2),
            "gc_quota": round(gc_cost, 2),
            "balancing": round(balancing_cost, 2),
            "transport_admin": round(transport_cost, 2),
            "risk_buffer": round(risk_cost, 2),
        },
        "volume_mwh": round(total_volume, 2),
    }


# ============================================================================
# ALERT ENGINE
# ============================================================================

def generate_alerts(
    pricing_data: Dict,
    risk_data: Dict,
    pnl_data: Dict,
    config: PipelineConfig,
) -> List[Dict]:
    """Generate actionable alerts from pipeline results."""
    alerts = []

    # Margin floor breach
    if pricing_data.get("margin_breach"):
        alerts.append({
            "severity": "HIGH",
            "category": "margin",
            "message": f"Margin {pnl_data['margin_per_mwh']:.2f} EUR/MWh below floor {config.min_margin_floor_eur_per_mwh:.2f}",
            "action": "Review pricing or decline offer",
        })

    # VaR breach
    if risk_data.get("var_95_status") == "RED":
        alerts.append({
            "severity": "HIGH",
            "category": "risk",
            "message": f"Portfolio VaR {risk_data['var_95_eur']:,.0f} EUR exceeds limit",
            "action": "Reduce unhedged exposure or increase hedging",
        })
    elif risk_data.get("var_95_status") == "YELLOW":
        alerts.append({
            "severity": "MEDIUM",
            "category": "risk",
            "message": f"Portfolio VaR {risk_data['var_95_eur']:,.0f} EUR approaching limit",
            "action": "Monitor and prepare hedging actions",
        })

    # Imbalance cost warning
    if risk_data.get("imbalance_status") in ("YELLOW", "RED"):
        alerts.append({
            "severity": risk_data["imbalance_status"],
            "category": "imbalance",
            "message": f"Daily imbalance cost {risk_data['imbalance_cost_daily_eur']:,.0f} EUR",
            "action": "Improve generation forecast or increase IDM trading",
        })

    # Negative margin
    if pnl_data.get("margin_per_mwh", 0) < 0:
        alerts.append({
            "severity": "CRITICAL",
            "category": "pnl",
            "message": f"Negative margin: {pnl_data['margin_per_mwh']:.2f} EUR/MWh",
            "action": "STOP — do not proceed with this offer",
        })

    return alerts


# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================

def run_supply_pipeline(
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """
    Execute the supply pipeline end-to-end.

    This is the main entry point. Runs all stages according to the
    configured mode and returns a comprehensive result object.

    Args:
        config: Pipeline configuration (default: full mode, demo data).

    Returns:
        PipelineResult with all stage outputs and alerts.

    Usage:
        >>> from processors.supply_pipeline import run_supply_pipeline, PipelineConfig, PipelineMode
        >>> result = run_supply_pipeline(PipelineConfig(mode=PipelineMode.FULL))
        >>> print(f"Pipeline {'SUCCESS' if result.success else 'FAILED'}")
        >>> print(f"Final price: {result.stages['pricing'].output_data['waterfall']['final_price_ex_vat']} EUR/MWh")
    """
    if config is None:
        config = PipelineConfig()

    pipeline_id = f"SP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    start_time = datetime.now()

    logger.info("=" * 70)
    logger.info(f"SUPPLY PIPELINE [{pipeline_id}] — Mode: {config.mode.value}")
    logger.info("=" * 70)

    result = PipelineResult(
        pipeline_id=pipeline_id,
        mode=config.mode,
        config=config,
        start_time=start_time,
    )

    stages_to_run = _get_stages_for_mode(config.mode)

    # Stage 1: Data Extraction
    if "extraction" in stages_to_run:
        sr = _run_stage("extraction", stage_data_extraction, config)
        result.stages["extraction"] = sr
    else:
        result.stages["extraction"] = StageResult(
            stage_name="extraction",
            status=StageStatus.SKIPPED,
            start_time=datetime.now(),
        )

    # Stage 2: Generation Forecast
    if "generation" in stages_to_run:
        sr = _run_stage("generation", stage_generation_forecast, config)
        result.stages["generation"] = sr
    else:
        result.stages["generation"] = StageResult(
            stage_name="generation",
            status=StageStatus.SKIPPED,
            start_time=datetime.now(),
        )

    # Stage 3: Consumption Forecast
    if "consumption" in stages_to_run:
        sr = _run_stage("consumption", stage_consumption_forecast, config)
        result.stages["consumption"] = sr
    else:
        result.stages["consumption"] = StageResult(
            stage_name="consumption",
            status=StageStatus.SKIPPED,
            start_time=datetime.now(),
        )

    # Stage 4: Gap Analysis
    gen_data = result.stages.get("generation", StageResult("generation", StageStatus.SKIPPED, datetime.now()))
    con_data = result.stages.get("consumption", StageResult("consumption", StageStatus.SKIPPED, datetime.now()))

    if ("gap" in stages_to_run
            and gen_data.status == StageStatus.COMPLETED
            and con_data.status == StageStatus.COMPLETED):
        sr = _run_stage("gap", stage_gap_analysis, gen_data.output_data, con_data.output_data)
        result.stages["gap"] = sr
    else:
        result.stages["gap"] = StageResult(
            stage_name="gap",
            status=StageStatus.SKIPPED,
            start_time=datetime.now(),
        )

    # Stage 5: Pricing
    gap_result = result.stages.get("gap")
    if ("pricing" in stages_to_run
            and gen_data.status == StageStatus.COMPLETED
            and con_data.status == StageStatus.COMPLETED
            and gap_result and gap_result.status == StageStatus.COMPLETED):
        sr = _run_stage(
            "pricing", stage_pricing,
            gen_data.output_data, con_data.output_data,
            gap_result.output_data, config,
        )
        result.stages["pricing"] = sr
    else:
        result.stages["pricing"] = StageResult(
            stage_name="pricing",
            status=StageStatus.SKIPPED,
            start_time=datetime.now(),
        )

    # Stage 6: Procurement
    if ("procurement" in stages_to_run
            and gap_result and gap_result.status == StageStatus.COMPLETED):
        sr = _run_stage("procurement", stage_procurement, gap_result.output_data, config)
        result.stages["procurement"] = sr
    else:
        result.stages["procurement"] = StageResult(
            stage_name="procurement",
            status=StageStatus.SKIPPED,
            start_time=datetime.now(),
        )

    # Stage 7: Risk
    pricing_result = result.stages.get("pricing")
    if ("risk" in stages_to_run
            and gen_data.status == StageStatus.COMPLETED
            and con_data.status == StageStatus.COMPLETED
            and gap_result and gap_result.status == StageStatus.COMPLETED
            and pricing_result and pricing_result.status == StageStatus.COMPLETED):
        sr = _run_stage(
            "risk", stage_risk_assessment,
            gen_data.output_data, con_data.output_data,
            gap_result.output_data, pricing_result.output_data, config,
        )
        result.stages["risk"] = sr
    else:
        result.stages["risk"] = StageResult(
            stage_name="risk",
            status=StageStatus.SKIPPED,
            start_time=datetime.now(),
        )

    # Stage 8: P&L
    procurement_result = result.stages.get("procurement")
    risk_result = result.stages.get("risk")
    if ("pnl" in stages_to_run
            and pricing_result and pricing_result.status == StageStatus.COMPLETED
            and procurement_result and procurement_result.status == StageStatus.COMPLETED
            and risk_result and risk_result.status == StageStatus.COMPLETED):
        sr = _run_stage(
            "pnl", stage_pnl_projection,
            pricing_result.output_data, procurement_result.output_data,
            risk_result.output_data, config,
        )
        result.stages["pnl"] = sr
    else:
        result.stages["pnl"] = StageResult(
            stage_name="pnl",
            status=StageStatus.SKIPPED,
            start_time=datetime.now(),
        )

    # Stage 9: Alerts
    pnl_result = result.stages.get("pnl")
    if (pricing_result and pricing_result.status == StageStatus.COMPLETED
            and risk_result and risk_result.status == StageStatus.COMPLETED
            and pnl_result and pnl_result.status == StageStatus.COMPLETED):
        result.alerts = generate_alerts(
            pricing_result.output_data,
            risk_result.output_data,
            pnl_result.output_data,
            config,
        )

    # Finalize
    end_time = datetime.now()
    result.end_time = end_time
    result.total_duration_seconds = (end_time - start_time).total_seconds()

    completed = sum(1 for s in result.stages.values() if s.status == StageStatus.COMPLETED)
    failed = sum(1 for s in result.stages.values() if s.status == StageStatus.FAILED)
    skipped = sum(1 for s in result.stages.values() if s.status == StageStatus.SKIPPED)

    result.success = failed == 0 and completed > 0

    result.summary = {
        "pipeline_id": pipeline_id,
        "mode": config.mode.value,
        "duration_seconds": round(result.total_duration_seconds, 1),
        "stages_completed": completed,
        "stages_failed": failed,
        "stages_skipped": skipped,
        "alerts_count": len(result.alerts),
        "high_alerts": sum(1 for a in result.alerts if a["severity"] in ("HIGH", "CRITICAL")),
    }

    logger.info("=" * 70)
    logger.info(
        f"PIPELINE {'SUCCESS' if result.success else 'FAILED'} | "
        f"{completed} completed, {failed} failed, {skipped} skipped | "
        f"{result.total_duration_seconds:.1f}s | "
        f"{len(result.alerts)} alerts"
    )
    logger.info("=" * 70)

    return result


def _get_stages_for_mode(mode: PipelineMode) -> set:
    """Map pipeline mode to required stages."""
    stages = {
        PipelineMode.FULL: {
            "extraction", "generation", "consumption", "gap",
            "pricing", "procurement", "risk", "pnl",
        },
        PipelineMode.PRICING: {
            "generation", "consumption", "gap", "pricing",
        },
        PipelineMode.RISK: {
            "generation", "consumption", "gap", "pricing", "risk",
        },
        PipelineMode.REBALANCE: {
            "generation", "consumption", "gap", "procurement",
        },
    }
    return stages.get(mode, stages[PipelineMode.FULL])


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """CLI entry point for supply pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="nextE Supply Pipeline Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["full", "pricing", "risk", "rebalance"],
        default="full",
        help="Pipeline execution mode",
    )
    parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon (days)")
    parser.add_argument("--scenarios", type=int, default=50, help="Monte Carlo scenarios")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--entsoe-key", type=str, default=None, help="ENTSO-E API key")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = PipelineConfig(
        mode=PipelineMode(args.mode),
        forecast_horizon_days=args.horizon,
        n_scenarios=args.scenarios,
        seed=args.seed,
        entsoe_api_key=args.entsoe_key,
    )

    result = run_supply_pipeline(config)

    # Print summary
    print(f"\nPipeline {result.pipeline_id}: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Duration: {result.total_duration_seconds:.1f}s")

    if result.alerts:
        print(f"\nAlerts ({len(result.alerts)}):")
        for alert in result.alerts:
            print(f"  [{alert['severity']}] {alert['message']}")

    if "pricing" in result.stages and result.stages["pricing"].output_data:
        wf = result.stages["pricing"].output_data["waterfall"]
        print(f"\nFinal price: {wf['final_price_ex_vat']:.2f} EUR/MWh (ex-VAT)")


if __name__ == "__main__":
    main()


# ============================================================================
# END OF SUPPLY PIPELINE
# ============================================================================

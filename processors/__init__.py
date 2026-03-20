"""
Layer 1 Processors — Analytical computation modules.

  dam_analysis:  DAM price statistics (Base/Peak/Off-Peak, percentiles, monthly avg)
  idm_analysis:  IDM VWAP, spread to DAM, buy/sell decomposition
  srmc:          Gas-SRMC, Coal-SRMC, clean spark/dark spread
  merit_order:   Residual demand, marginal price regime identification
  imbalance:     Imbalance cost P50/P90, seasonal profiles, spread to DAM
  forward_curve: Forward curve construction, contango/backwardation, Aurora comparison
  sensitivity:   Price elasticity from scenario data, tornado chart inputs
  statistics:    Volatility, correlations, distribution fitting

  --- Supply Extension (v1.0.0, 2026-03-19) ---
  supply_pricing:          B2B supply cost waterfall and offer generation
  procurement_optimizer:   Procurement channel allocation optimization (MILP)
  supply_risk:             Shape/volume/price/credit risk + portfolio VaR

  --- Supply P&L Extension (v1.0.0, 2026-03-20) ---
  supply_pnl:              Per-contract P&L, budget vs actuals, variance decomposition

  --- Forecasters & Pipeline (v1.0.0, 2026-03-20) ---
  generation_forecaster:   Multi-probability 15-min PV/Wind generation curves
  consumption_forecaster:  Multi-probability 15-min B2B load curves
  supply_pipeline:         End-to-end supply workflow orchestrator
"""

# --- Supply Extension Exports ---
# Lazy imports to avoid breaking existing functionality if dependencies are missing
try:
    from .supply_pricing import (
        PVPricingMechanism,
        ProcurementChannel,
        RiskScenario,
        SupplyContractParams,
        SupplyPriceResult,
        calculate_pv_procurement_cost,
        calculate_forward_procurement_cost,
        calculate_gc_quota_cost,
        calculate_balancing_cost,
        calculate_risk_premium,
        build_supply_price_waterfall,
        run_multi_scenario_pricing,
        generate_sensitivity_table,
        export_offer_sheet,
    )

    from .procurement_optimizer import (
        ChannelConstraints,
        ProcurementOptimization,
        optimize_procurement_allocation,
        analyze_procurement_scenarios,
        compare_procurement_strategies,
        estimate_channel_cost_sensitivity,
    )

    from .supply_risk import (
        ShapeRiskAnalysis,
        VolumeRiskAnalysis,
        PriceRiskAnalysis,
        CreditRiskAnalysis,
        PortfolioVaRAnalysis,
        PositionLimitStatus,
        calculate_shape_risk,
        calculate_volume_risk,
        calculate_price_risk,
        calculate_credit_risk,
        calculate_portfolio_var,
        monitor_position_limits,
    )

    from .supply_pnl import (
        PnLPeriod,
        VarianceType,
        ContractBudget,
        ContractActuals,
        ContractPnL,
        PortfolioPnL,
        compute_contract_pnl,
        compute_portfolio_pnl,
        decompose_margin_variance,
        build_variance_bridge,
        generate_monthly_pnl_series,
        generate_demo_pnl_data,
        pnl_to_dataframe,
        portfolio_summary_table,
    )

    from .generation_forecaster import (
        Technology,
        AssetSpec,
        GenerationForecast,
        PortfolioForecast,
        PVForecaster,
        WindForecaster,
        forecast_portfolio,
        generate_demo_portfolio,
        generate_demo_forecast,
    )

    from .consumption_forecaster import (
        CustomerSegment,
        CustomerProfile,
        ConsumptionForecast,
        PortfolioConsumptionForecast,
        ConsumptionForecaster,
        forecast_consumption_portfolio,
        generate_demo_customers,
        generate_demo_consumption_forecast,
    )

    from .supply_pipeline import (
        PipelineMode,
        PipelineConfig,
        PipelineResult,
        StageResult,
        StageStatus,
        run_supply_pipeline,
    )

    _SUPPLY_EXTENSION_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(
        f"Supply extension modules not fully loaded: {e}. "
        "Existing Layer 1 processors are unaffected."
    )
    _SUPPLY_EXTENSION_AVAILABLE = False

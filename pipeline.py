"""
Main Pipeline Orchestrator — RO Energy Pricing Engine Layer 1.

Runs the full processing chain:
  1. Load pre-loaded datasets (or extract from APIs)
  2. Process through all analytical modules (DAM, IDM, SRMC, Merit Order, Imbalance)
  3. Export results for Layer 2 (Excel) and Layer 3 (Streamlit)
  4. Run validation checks

Usage:
  python pipeline.py --mode backtest         # Process all historical data
  python pipeline.py --mode daily            # Daily refresh (D-1 data)
  python pipeline.py --mode weekly           # Weekly refresh (forward curves + commodities)
  python pipeline.py --mode export-only      # Re-export from processed data
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


def run_backtest_pipeline(data_dir: str = "/mnt/project"):
    """
    Full historical backtest: load all pre-loaded CSVs, run all processors, export.
    This is the most comprehensive pipeline mode.
    """
    from extractors.data_loader import (
        load_entsoe_csv, load_idm_nexte, load_aurora_forecast,
        load_fx_eur_ron, load_montel_backcast, load_sensitivity_scenarios,
        load_balancing_services_csv,
    )
    from processors.dam_analysis import run_dam_analysis
    from processors.idm_analysis import run_idm_analysis
    from processors.merit_order import run_merit_order_analysis
    from processors.imbalance import run_imbalance_analysis
    from processors.sensitivity import run_sensitivity_analysis
    from processors.statistics import compute_volatility, compute_percentile_table
    from outputs.excel_export import export_for_excel, export_dam_hourly_latest
    from outputs.streamlit_data import export_timeseries_parquet, export_kpis
    from outputs.validation import run_full_validation

    logger.info("=" * 70)
    logger.info("STARTING BACKTEST PIPELINE")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # STEP 1: Load all pre-loaded datasets
    # ------------------------------------------------------------------
    logger.info("STEP 1: Loading pre-loaded datasets...")

    dam = load_entsoe_csv("RO_day_ahead_prices_ENTSOE.csv")
    generation = load_entsoe_csv("RO_generation_ENTSOE.csv")
    load_actual = load_entsoe_csv("RO_load_ENTSOE.csv")
    load_forecast = load_entsoe_csv("RO_load_forecast_ENTSOE.csv")
    wind_solar_fcst = load_entsoe_csv("RO_wind_and_solar_forecast_ENTSOE.csv")
    imports = load_entsoe_csv("RO_import_ENTSOE.csv")
    imb_prices_entsoe = load_entsoe_csv("RO_imbalance_prices_ENTSOE.csv")
    imb_volumes_entsoe = load_entsoe_csv("RO_imbalance_volumes_ENTSOE.csv")

    # Balancing Services (shorter history — Jul 2024+)
    imb_prices_bs = load_balancing_services_csv("RO_imbalance_prices_NA_BAL_SERV.csv")
    imb_volumes_bs = load_balancing_services_csv("RO_imbalance_totalvolumes_NA_BAL_SERV.csv")

    # IDM (NEXTE)
    idm = load_idm_nexte()

    # Aurora long-term forecast
    aurora = load_aurora_forecast()

    # FX rates
    fx = load_fx_eur_ron()

    # Sensitivity scenarios
    sensitivity = load_sensitivity_scenarios()

    logger.info("All datasets loaded successfully.")

    # ------------------------------------------------------------------
    # STEP 2: Run analytical processors
    # ------------------------------------------------------------------
    logger.info("STEP 2: Running analytical processors...")

    # 2a. DAM analysis
    logger.info("  Processing DAM prices...")
    dam_results = run_dam_analysis(dam, price_col="Value [EUR/MWh]")

    # 2b. IDM analysis
    logger.info("  Processing IDM data...")
    idm_results = run_idm_analysis(idm, dam=dam, dam_price_col="Value [EUR/MWh]")

    # 2c. Merit order / generation stack
    logger.info("  Processing generation stack & merit order...")
    merit_results = run_merit_order_analysis(
        generation, load_actual, dam=dam, load_col="Actual Load [MW]"
    )

    # 2d. Imbalance analysis
    logger.info("  Processing imbalance data...")
    imbalance_results = run_imbalance_analysis(
        imb_prices_entsoe, dam, dam_col="Value [EUR/MWh]"
    )

    # 2e. Sensitivity analysis
    logger.info("  Processing sensitivity scenarios...")
    sensitivity_results = run_sensitivity_analysis(sensitivity)

    # 2f. Volatility
    logger.info("  Computing price volatility...")
    dam_daily = dam_results["daily"]
    vol = compute_volatility(dam_daily["base_eur_mwh"].dropna())

    logger.info("All processors completed.")

    # ------------------------------------------------------------------
    # STEP 3: Export for Layer 2 (Excel)
    # ------------------------------------------------------------------
    logger.info("STEP 3: Exporting for Layer 2 (Excel)...")

    excel_data = {
        "dam_monthly": dam_results["monthly"],
        "dam_hourly": dam,
        "idm_monthly": idm_results.get("monthly", pd.DataFrame()),
        "imbalance_monthly": imbalance_results.get("monthly_cost", pd.DataFrame()),
        "fx_daily": fx,
        "aurora_forecast": aurora,
        "generation_monthly": merit_results.get("monthly_mix", pd.DataFrame()),
        "cross_border_monthly": imports.resample("MS").mean(),
    }
    export_for_excel(excel_data)
    export_dam_hourly_latest(dam, days=90)

    # ------------------------------------------------------------------
    # STEP 4: Export for Layer 3 (Streamlit)
    # ------------------------------------------------------------------
    logger.info("STEP 4: Exporting for Layer 3 (Streamlit)...")

    streamlit_data = {
        "dam_timeseries": dam,
        "generation_stack": generation,
        "imbalance": imb_prices_entsoe,
        "cross_border": imports,
        "sensitivity": sensitivity,
    }
    export_timeseries_parquet(streamlit_data)

    # KPIs
    latest_month = dam_results["monthly"].iloc[-1] if len(dam_results["monthly"]) > 0 else {}
    kpis = {
        "dam_base_avg_latest_month": float(latest_month.get("base_avg", 0)),
        "dam_peak_avg_latest_month": float(latest_month.get("peak_avg", 0)),
        "trailing_6m_avg": float(
            dam_results["trailing"]["trailing_6m"].iloc[-1]
        ) if "trailing_6m" in dam_results["trailing"].columns else None,
        "imbalance_cost_p50": float(
            imbalance_results["monthly_cost"]["imbalance_cost_p50_eur_mwh"].iloc[-1]
        ) if not imbalance_results["monthly_cost"].empty else None,
        "last_updated": pd.Timestamp.now().isoformat(),
    }
    export_kpis(kpis)

    # ------------------------------------------------------------------
    # STEP 5: Validation
    # ------------------------------------------------------------------
    logger.info("STEP 5: Running validation checks...")

    validation_datasets = {
        "dam": dam,
        "generation": generation,
        "load": load_actual,
        "imbalance_prices": imb_prices_entsoe,
        "imports": imports,
    }
    validation_report = run_full_validation(
        validation_datasets,
        dam_monthly=dam_results["monthly"],
    )
    validation_path = settings_processed() / "validation_report.csv"
    validation_report.to_csv(validation_path, index=False)

    # Summary
    pass_n = (validation_report["status"] == "PASS").sum()
    warn_n = (validation_report["status"] == "WARN").sum()
    fail_n = (validation_report["status"] == "FAIL").sum()

    logger.info("=" * 70)
    logger.info("BACKTEST PIPELINE COMPLETE")
    logger.info("Validation: %d PASS | %d WARN | %d FAIL", pass_n, warn_n, fail_n)
    logger.info("=" * 70)

    return {
        "dam_results": dam_results,
        "idm_results": idm_results,
        "merit_results": merit_results,
        "imbalance_results": imbalance_results,
        "sensitivity_results": sensitivity_results,
        "validation": validation_report,
    }


def settings_processed():
    """Get the processed data directory from settings."""
    from config.settings import settings
    return settings.processed_dir


def main():
    parser = argparse.ArgumentParser(
        description="RO Energy Pricing Engine — Layer 1 Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "daily", "weekly", "export-only"],
        default="backtest",
        help="Pipeline execution mode",
    )
    parser.add_argument(
        "--data-dir",
        default="/mnt/project",
        help="Directory containing pre-loaded data files",
    )
    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest_pipeline(data_dir=args.data_dir)
    elif args.mode == "daily":
        logger.info("Daily refresh mode — requires live API access")
        logger.info("Use: extractors.entsoe_client.ENTSOEClient().daily_refresh()")
        logger.info("     extractors.balancing_client.BalancingServicesClient().daily_refresh()")
    elif args.mode == "weekly":
        logger.info("Weekly refresh mode — requires EQ API access")
        logger.info("Use: extractors.eq_client.EQClient().get_commodity_settlements()")
    else:
        logger.info("Export-only mode — re-export existing processed data")


if __name__ == "__main__":
    main()

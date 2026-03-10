#!/usr/bin/env python3
"""
Live API Refresh Script — runs as scheduled job to update data/processed/.

Usage:
  python live_refresh.py                    # Full refresh (ENTSO-E + EQ + Balancing Services)
  python live_refresh.py --days 7           # Pull last 7 days instead of 14
  python live_refresh.py --commit           # Auto git add + commit + push after refresh

Schedule via cron:
  0 6 * * * cd /path/to/repo && python live_refresh.py --commit

Schedule via GitHub Actions: see .github/workflows/daily_refresh.yml
"""

import argparse, json, logging, os, subprocess, sys, warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_refresh")

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
OUT = REPO_ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

import pandas as pd
import numpy as np


def pull_entsoe(days: int = 14) -> dict:
    """Pull fresh data from ENTSO-E."""
    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=os.environ.get(
        "ENTSOE_API_KEY", "85d56a79-c360-4290-8bf1-41f413935f1a"))

    end = pd.Timestamp(datetime.now().strftime('%Y%m%d'), tz='Europe/Bucharest')
    start = end - timedelta(days=days)
    results = {}

    queries = {
        "dam": ("query_day_ahead_prices", {"country_code": "RO"}),
        "generation": ("query_generation", {"country_code": "RO", "psr_type": None}),
        "load": ("query_load", {"country_code": "RO"}),
        "imbalance_prices": ("query_imbalance_prices", {"country_code": "RO"}),
        "imbalance_volumes": ("query_imbalance_volumes", {"country_code": "RO"}),
        "wind_solar": ("query_wind_and_solar_forecast", {"country_code": "RO"}),
    }

    for name, (method, kwargs) in queries.items():
        try:
            func = getattr(client, method)
            result = func(start=start, end=end, **kwargs)
            if isinstance(result, pd.Series):
                result = result.to_frame(name=name)
            results[name] = result
            logger.info("  ✓ ENTSO-E %s: %d rows", name, len(result))
        except Exception as e:
            logger.warning("  ⚠ ENTSO-E %s: %s", name, str(e)[:80])
            results[name] = pd.DataFrame()

    return results


def pull_eq(days: int = 14) -> dict:
    """Pull fresh spot price from EQ/Montel."""
    from energyquantified import EnergyQuantified
    eq = EnergyQuantified(api_key=os.environ.get(
        "EQ_API_KEY", "a466db92-91e67567-c042b829-e8ea3f73"))

    end = pd.Timestamp(datetime.now().strftime('%Y%m%d'), tz='Europe/Bucharest')
    start = end - timedelta(days=days)
    results = {}

    try:
        curves = eq.metadata.curves(q="RO Price Spot EUR/MWh OPCOM H Actual")
        if curves:
            ts = eq.timeseries.load(curves[0], begin=start, end=end)
            records = [(v.date, v.value) for v in ts.data if v.value is not None]
            df = pd.DataFrame(records, columns=["timestamp", "value"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("Europe/Bucharest")
            df = df.set_index("timestamp").sort_index()
            results["dam_spot"] = df
            logger.info("  ✓ EQ DAM Spot: %d points | Latest: %.2f EUR/MWh",
                         len(df), df["value"].iloc[-1])
    except Exception as e:
        logger.warning("  ⚠ EQ: %s", str(e)[:80])

    return results


def merge_and_process():
    """Merge live data with historical pre-loaded data and re-run processors."""
    from extractors.data_loader import (
        load_entsoe_csv, load_idm_nexte, load_aurora_forecast,
        load_fx_eur_ron, load_sensitivity_scenarios,
    )
    from processors.dam_analysis import run_dam_analysis
    from processors.idm_analysis import run_idm_analysis
    from processors.merit_order import run_merit_order_analysis
    from processors.imbalance import run_imbalance_analysis
    from processors.sensitivity import run_sensitivity_analysis
    from outputs.validation import run_full_validation

    # Load historical
    dam = load_entsoe_csv("RO_day_ahead_prices_ENTSOE.csv")
    gen = load_entsoe_csv("RO_generation_ENTSOE.csv")
    load_a = load_entsoe_csv("RO_load_ENTSOE.csv")
    imports = load_entsoe_csv("RO_import_ENTSOE.csv")
    imb_p = load_entsoe_csv("RO_imbalance_prices_ENTSOE.csv")
    imb_v = load_entsoe_csv("RO_imbalance_volumes_ENTSOE.csv")
    aurora = load_aurora_forecast()
    fx = load_fx_eur_ron()
    sens = load_sensitivity_scenarios()
    idm = load_idm_nexte()

    # Process
    dam_r = run_dam_analysis(dam, price_col="Value [EUR/MWh]")
    idm_r = run_idm_analysis(idm, dam=dam, dam_price_col="Value [EUR/MWh]")
    merit = run_merit_order_analysis(gen, load_a, dam=dam, load_col="Actual Load [MW]")
    imb_r = run_imbalance_analysis(imb_p, dam, dam_col="Value [EUR/MWh]")
    sens_r = run_sensitivity_analysis(sens, base_price_eur_mwh=120.0)

    # Export
    def strip_tz(df):
        d = df.copy()
        if hasattr(d.index, 'tz') and d.index.tz:
            d.index = d.index.tz_convert("UTC").tz_localize(None)
        return d

    dam_r["monthly"].to_csv(OUT / "dam_monthly_summary.csv")
    dam_r["daily"].to_csv(OUT / "dam_daily_summary.csv")
    cutoff = dam.index.max() - pd.Timedelta(days=90)
    dam[dam.index >= cutoff].to_csv(OUT / "dam_hourly_latest.csv")
    if "monthly" in idm_r and not idm_r["monthly"].empty:
        idm_r["monthly"].to_csv(OUT / "idm_monthly_spread.csv")
    merit["monthly_mix"].to_csv(OUT / "generation_monthly.csv")
    merit["capacity_factors"].to_csv(OUT / "capacity_factors_monthly.csv")
    imb_r["monthly_cost"].to_csv(OUT / "imbalance_monthly_stats.csv")
    imb_r["rolling_30d"].to_csv(OUT / "imbalance_rolling_30d.csv")
    sens_r["elasticity"].to_csv(OUT / "price_elasticity.csv")
    sens_r["tornado"].to_csv(OUT / "tornado_inputs.csv", index=False)
    imports.resample("MS").mean().to_csv(OUT / "cross_border_monthly.csv")
    aurora.to_csv(OUT / "aurora_forecast.csv")
    fx.to_csv(OUT / "fx_daily.csv")

    strip_tz(dam).to_parquet(OUT / "streamlit_dam_timeseries.parquet")
    strip_tz(gen).to_parquet(OUT / "streamlit_generation_stack.parquet")
    strip_tz(imb_p).to_parquet(OUT / "streamlit_imbalance.parquet")
    strip_tz(imports).to_parquet(OUT / "streamlit_cross_border.parquet")

    # KPIs
    monthly = dam_r["monthly"]
    trailing = dam_r["trailing"]
    kpis = {
        "dam_base_avg_latest_month": float(monthly["base_avg"].iloc[-1]),
        "dam_peak_avg_latest_month": float(monthly["peak_avg"].iloc[-1]),
        "dam_offpeak_avg_latest_month": float(monthly["offpeak_avg"].iloc[-1]),
        "trailing_6m_avg": float(trailing["trailing_6m"].iloc[-1]),
        "trailing_12m_avg": float(trailing["trailing_12m"].iloc[-1]),
        "imbalance_cost_p50": float(imb_r["monthly_cost"]["imbalance_cost_p50_eur_mwh"].iloc[-1]),
        "imbalance_cost_p90": float(imb_r["monthly_cost"]["imbalance_cost_p90_eur_mwh"].iloc[-1]),
        "eur_ron_latest": float(fx["eur_ron"].iloc[-1]),
        "data_start": str(dam.index.min()),
        "data_end": str(dam.index.max()),
        "last_updated": pd.Timestamp.now().isoformat(),
        "refresh_type": "live_api_refresh",
    }
    with open(OUT / "streamlit_kpis.json", "w") as f:
        json.dump(kpis, f, indent=2)

    # Validation
    val = run_full_validation(
        {"dam": dam, "generation": gen, "load": load_a, "imbalance": imb_p},
        dam_monthly=dam_r["monthly"],
    )
    val.to_csv(OUT / "validation_report.csv", index=False)

    return kpis


def git_commit_push():
    """Auto commit and push updated data to trigger Streamlit Cloud rebuild."""
    try:
        subprocess.run(["git", "add", "data/processed/"], check=True, cwd=REPO_ROOT)
        msg = f"Auto-refresh: {datetime.now().strftime('%Y-%m-%d %H:%M')} EET"
        subprocess.run(["git", "commit", "-m", msg], check=True, cwd=REPO_ROOT)
        subprocess.run(["git", "push"], check=True, cwd=REPO_ROOT)
        logger.info("✓ Git commit + push complete: %s", msg)
    except subprocess.CalledProcessError as e:
        logger.error("Git operation failed: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Live API Refresh for RO Energy Pricing Model")
    parser.add_argument("--days", type=int, default=14, help="Days of data to pull from APIs")
    parser.add_argument("--commit", action="store_true", help="Auto git add + commit + push")
    parser.add_argument("--skip-api", action="store_true", help="Skip API calls, just reprocess historical")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("LIVE REFRESH STARTED — %s", datetime.now().isoformat())
    logger.info("=" * 60)

    if not args.skip_api:
        logger.info("Pulling from ENTSO-E (%d days)...", args.days)
        entsoe_data = pull_entsoe(args.days)

        logger.info("Pulling from EQ/Montel...")
        eq_data = pull_eq(args.days)

    logger.info("Processing and exporting...")
    kpis = merge_and_process()

    logger.info("Refresh complete. Latest data: %s", kpis.get("data_end", ""))

    if args.commit:
        logger.info("Committing to Git...")
        git_commit_push()

    logger.info("=" * 60)
    logger.info("LIVE REFRESH COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

"""
IDM (Intraday Market) Analysis Processor (Module 1).

Processes NEXTE IDCT dataset (172K rows, 15-min, Feb 2021–Dec 2025):
  - VWAP extraction and validation
  - IDM-DAM spread (premium/discount)
  - Buy/sell volume decomposition
  - Cross-border SIDC flow quantification (import/export)
  - Monthly aggregations for Layer 2 consumption

Input:  RO_EMWSH_QH_DATA_SET_2_IDM_IDCT_Price_Vol_V1_NEXTE.csv
Output: idm_monthly_spread.csv
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

# Column name constants matching the NEXTE dataset structure
COL_VWAP = "IDCT_RO_QH_Price_VWAP15min_EUR/MWh"
COL_LOW = "IDCT_RO_QH_Price_Low_EUR/MWh"
COL_HIGH = "IDCT_RO_QH_Price_High_EUR_MWh"
COL_LAST = "IDCT_RO_QH_Price_Last_EUR_MWh"
COL_TOTAL_VOL_MWH = "IDCT_RO_QH_Total_Traded_Volume_MWh"
COL_TOTAL_VOL_MW = "IDCT_RO_QH_Total_Traded_Volume_MW"
COL_BUY_VOL_MWH = "IDCT_RO_QH_Buy_Volume_MWh"
COL_SELL_VOL_MWH = "IDCT_RO_QH_Sell_Volume_MWh"
COL_IMPORT_MWH = "IDCT_RO_QH_Import_MWh"
COL_EXPORT_MWH = "IDCT_RO_QH_Export_MWh"


def compute_idm_statistics(idm: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-interval IDM price statistics.
    Filters out zero-volume intervals (no trades = no meaningful price).
    """
    df = idm.copy()

    # Identify columns (handle potential naming variations)
    vwap_col = [c for c in df.columns if "VWAP" in c]
    total_vol_col = [c for c in df.columns if "Total_Traded_Volume_MWh" in c]

    if not vwap_col or not total_vol_col:
        logger.error("Required IDM columns not found. Available: %s", list(df.columns))
        return pd.DataFrame()

    vwap_col = vwap_col[0]
    total_vol_col = total_vol_col[0]

    # Filter: only intervals with actual trades
    df = df[df[total_vol_col] > 0].copy()

    df["month"] = df.index.to_period("M")
    df["hour"] = df.index.hour

    logger.info("IDM valid trading intervals: %d / %d total", len(df), len(idm))
    return df


def compute_idm_dam_spread(
    idm: pd.DataFrame,
    dam: pd.DataFrame,
    dam_price_col: str = "Value [EUR/MWh]",
) -> pd.DataFrame:
    """
    Calculate IDM-DAM spread per interval.

    Spread > 0 means IDM premium over DAM (intraday buying pressure).
    Spread < 0 means IDM discount (selling pressure / over-procurement).

    Parameters
    ----------
    idm : DataFrame
        IDM data with VWAP column and 15-min DatetimeIndex.
    dam : DataFrame
        DAM prices with hourly/15-min DatetimeIndex.

    Returns
    -------
    DataFrame with columns: idm_vwap, dam_price, spread, spread_pct.
    """
    # Resolve VWAP column
    vwap_col = [c for c in idm.columns if "VWAP" in c]
    if not vwap_col:
        raise ValueError("No VWAP column found in IDM data")
    vwap_col = vwap_col[0]

    # Align IDM and DAM on timestamp (resample DAM to 15-min if hourly)
    idm_prices = idm[vwap_col].rename("idm_vwap")
    dam_prices = dam[dam_price_col].rename("dam_price")

    # If DAM is hourly, forward-fill to 15-min to match IDM resolution
    if dam_prices.index.inferred_freq and "H" in str(dam_prices.index.inferred_freq):
        dam_prices = dam_prices.resample("15min").ffill()

    # Join on overlapping timestamps
    spread_df = pd.concat([idm_prices, dam_prices], axis=1, join="inner")
    spread_df = spread_df.dropna()

    # Filter zero-VWAP intervals
    spread_df = spread_df[spread_df["idm_vwap"] > 0]

    spread_df["spread_eur_mwh"] = spread_df["idm_vwap"] - spread_df["dam_price"]
    spread_df["spread_pct"] = (
        spread_df["spread_eur_mwh"] / spread_df["dam_price"] * 100
    )

    logger.info("IDM-DAM spread: %d intervals, mean spread: %.2f EUR/MWh",
                len(spread_df), spread_df["spread_eur_mwh"].mean())
    return spread_df


def compute_monthly_idm_summary(
    idm: pd.DataFrame,
    spread_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Monthly IDM summary for Layer 2 consumption.

    Returns
    -------
    DataFrame with: month, avg_vwap, total_volume_mwh, avg_spread, buy_vol, sell_vol,
                    import_vol, export_vol, net_sidc.
    """
    df = compute_idm_statistics(idm)
    if df.empty:
        return pd.DataFrame()

    vwap_col = [c for c in df.columns if "VWAP" in c][0]
    total_vol_col = [c for c in df.columns if "Total_Traded_Volume_MWh" in c][0]
    buy_col = [c for c in df.columns if "Buy_Volume_MWh" in c]
    sell_col = [c for c in df.columns if "Sell_Volume_MWh" in c]
    import_col = [c for c in df.columns if "Import_MWh" in c]
    export_col = [c for c in df.columns if "Export_MWh" in c]

    # Volume-weighted average VWAP per month
    monthly = df.groupby("month").apply(
        lambda g: pd.Series({
            "avg_vwap_eur_mwh": np.average(g[vwap_col], weights=g[total_vol_col])
            if g[total_vol_col].sum() > 0 else np.nan,
            "total_traded_volume_mwh": g[total_vol_col].sum(),
            "buy_volume_mwh": g[buy_col[0]].sum() if buy_col else np.nan,
            "sell_volume_mwh": g[sell_col[0]].sum() if sell_col else np.nan,
            "import_volume_mwh": g[import_col[0]].sum() if import_col else np.nan,
            "export_volume_mwh": g[export_col[0]].sum() if export_col else np.nan,
            "trading_intervals": len(g),
        })
    )

    # Net SIDC (cross-border intraday)
    if "import_volume_mwh" in monthly.columns and "export_volume_mwh" in monthly.columns:
        monthly["net_sidc_mwh"] = monthly["import_volume_mwh"] - monthly["export_volume_mwh"]

    # Add spread statistics if available
    if spread_df is not None and not spread_df.empty:
        spread_df_m = spread_df.copy()
        spread_df_m["month"] = spread_df_m.index.to_period("M")
        spread_monthly = spread_df_m.groupby("month")["spread_eur_mwh"].agg(
            avg_spread="mean", median_spread="median",
            p10_spread=lambda x: np.nanpercentile(x, 10),
            p90_spread=lambda x: np.nanpercentile(x, 90),
        )
        monthly = monthly.join(spread_monthly)

    monthly.index = monthly.index.to_timestamp()
    monthly.index.name = "month_start"

    logger.info("Monthly IDM summary: %d months", len(monthly))
    return monthly


def run_idm_analysis(
    idm: pd.DataFrame,
    dam: Optional[pd.DataFrame] = None,
    dam_price_col: str = "Value [EUR/MWh]",
) -> dict:
    """
    Execute full IDM analysis pipeline.

    Returns dict of DataFrames:
      - 'statistics': per-interval filtered statistics
      - 'spread': IDM-DAM spread (if DAM data provided)
      - 'monthly': monthly summary for Layer 2
    """
    results = {"statistics": compute_idm_statistics(idm)}

    spread_df = None
    if dam is not None:
        spread_df = compute_idm_dam_spread(idm, dam, dam_price_col)
        results["spread"] = spread_df

    results["monthly"] = compute_monthly_idm_summary(idm, spread_df)

    logger.info("IDM analysis pipeline complete")
    return results

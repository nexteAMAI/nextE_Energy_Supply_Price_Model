"""
Forward Curve Processor (Module 1 → Layer 3).

Constructs and analyzes the Romanian power forward curve:
  - EEX RO Futures Power Base (M+1, Q+1, Cal+1, Cal+2)
  - Contango / backwardation analysis
  - Forward vs. Aurora Central/Low/High comparison
  - Gas-SRMC overlay on forward curve
  - Forward curve arbitrage check: Forward ≈ expected spot + risk premium

Input:  EQ OHLC data (live), Aurora forecast CSV (static)
Output: forward_curve_latest.csv, streamlit_forward_curve.parquet
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


def construct_forward_curve(
    forward_data: pd.DataFrame,
    field: str = "settlement",
) -> pd.DataFrame:
    """
    Construct a term-structure forward curve from EEX settlement data.

    Parameters
    ----------
    forward_data : DataFrame
        EQ OHLC data with delivery_start, delivery_end, settlement columns.

    Returns
    -------
    DataFrame indexed by delivery_start with settlement price and tenor label.
    """
    if forward_data.empty:
        logger.warning("Empty forward data — cannot construct curve")
        return pd.DataFrame()

    df = forward_data.copy()
    if field in df.columns:
        df = df[df[field].notna()]

    # Sort by delivery period
    df = df.sort_index()

    # Add tenor labels
    if "delivery_start" in df.columns and "delivery_end" in df.columns:
        df["tenor_days"] = (
            pd.to_datetime(df["delivery_end"]) - pd.to_datetime(df["delivery_start"])
        ).dt.days
        df["tenor_type"] = df["tenor_days"].apply(
            lambda d: "month" if d <= 32 else ("quarter" if d <= 93 else "year")
        )

    logger.info("Forward curve constructed: %d tenors", len(df))
    return df


def analyze_contango_backwardation(
    curve: pd.DataFrame,
    price_col: str = "settlement",
) -> dict:
    """
    Determine if the forward curve is in contango or backwardation.

    Contango: forward > spot (upward sloping)
    Backwardation: forward < spot (downward sloping)
    """
    if curve.empty or price_col not in curve.columns:
        return {"shape": "unknown", "slope": None, "spread": None}

    prices = curve[price_col].dropna()
    if len(prices) < 2:
        return {"shape": "unknown", "slope": None, "spread": None}

    front = prices.iloc[0]
    back = prices.iloc[-1]
    spread = back - front

    if spread > 0:
        shape = "contango"
    elif spread < 0:
        shape = "backwardation"
    else:
        shape = "flat"

    # Annualized slope (EUR/MWh per year)
    if hasattr(curve.index, 'to_series'):
        days_span = (curve.index[-1] - curve.index[0]).days
        if days_span > 0:
            slope = spread / (days_span / 365.25)
        else:
            slope = 0
    else:
        slope = None

    return {
        "shape": shape,
        "front_price": front,
        "back_price": back,
        "spread": spread,
        "annualized_slope": slope,
    }


def compare_forward_to_aurora(
    forward_curve: pd.DataFrame,
    aurora_forecast: pd.DataFrame,
    forward_price_col: str = "settlement",
) -> pd.DataFrame:
    """
    Compare EEX forward curve against Aurora Central/Low/High forecasts.

    Returns DataFrame with:
      - forward_price
      - aurora_central, aurora_low, aurora_high
      - forward_vs_central_spread
      - implied_risk_premium (forward - aurora_central)
    """
    # Aurora columns
    central_col = [c for c in aurora_forecast.columns if "Baseload_Central" in c and "Nominal" in c]
    low_col = [c for c in aurora_forecast.columns if "Baseload_Low" in c and "Nominal" in c]
    high_col = [c for c in aurora_forecast.columns if "Baseload_High" in c and "Nominal" in c]

    if not central_col:
        logger.warning("Aurora Central column not found")
        return pd.DataFrame()

    aurora = pd.DataFrame({
        "aurora_central": aurora_forecast[central_col[0]],
        "aurora_low": aurora_forecast[low_col[0]] if low_col else np.nan,
        "aurora_high": aurora_forecast[high_col[0]] if high_col else np.nan,
    })

    # Align forward curve to monthly frequency for comparison
    if forward_price_col in forward_curve.columns:
        fwd = forward_curve[[forward_price_col]].rename(
            columns={forward_price_col: "forward_price"}
        )
        # Resample to monthly if needed
        if not isinstance(fwd.index, pd.PeriodIndex):
            fwd = fwd.resample("MS").mean()
    else:
        logger.warning("Forward price column '%s' not in curve", forward_price_col)
        return pd.DataFrame()

    # Join
    result = fwd.join(aurora, how="outer")
    result["forward_vs_central_spread"] = result["forward_price"] - result["aurora_central"]
    result["implied_risk_premium"] = result["forward_vs_central_spread"]

    # Flag convergence (within ±5%)
    result["convergent"] = (
        result["forward_vs_central_spread"].abs()
        / result["aurora_central"] * 100 < 5
    )

    logger.info("Forward vs Aurora comparison: %d months", len(result.dropna(subset=["forward_price"])))
    return result


def overlay_srmc_on_forward(
    forward_curve: pd.DataFrame,
    srmc_gas: pd.DataFrame,
    forward_price_col: str = "settlement",
) -> pd.DataFrame:
    """
    Overlay gas-SRMC on the forward curve to identify
    periods where forwards are above/below marginal generation cost.
    """
    if forward_curve.empty or srmc_gas.empty:
        return pd.DataFrame()

    # Monthly SRMC average
    srmc_col = [c for c in srmc_gas.columns if "ccgt" in c.lower()]
    if not srmc_col:
        return pd.DataFrame()

    srmc_monthly = srmc_gas[srmc_col[0]].resample("MS").mean().rename("gas_srmc_ccgt")

    fwd = forward_curve[[forward_price_col]].rename(
        columns={forward_price_col: "forward_price"}
    )
    if not isinstance(fwd.index, pd.PeriodIndex):
        fwd = fwd.resample("MS").mean()

    result = fwd.join(srmc_monthly, how="outer")
    result["fwd_minus_srmc"] = result["forward_price"] - result["gas_srmc_ccgt"]

    return result


def run_forward_curve_analysis(
    forward_data: pd.DataFrame,
    aurora_forecast: Optional[pd.DataFrame] = None,
    srmc_gas: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Execute full forward curve analysis pipeline.

    Returns dict:
      - 'curve': constructed forward curve
      - 'shape': contango/backwardation analysis
      - 'aurora_comparison': forward vs Aurora (if provided)
      - 'srmc_overlay': forward vs gas-SRMC (if provided)
    """
    curve = construct_forward_curve(forward_data)
    shape = analyze_contango_backwardation(curve)

    results = {
        "curve": curve,
        "shape": shape,
    }

    if aurora_forecast is not None:
        results["aurora_comparison"] = compare_forward_to_aurora(curve, aurora_forecast)

    if srmc_gas is not None:
        results["srmc_overlay"] = overlay_srmc_on_forward(curve, srmc_gas)

    logger.info("Forward curve analysis complete: shape=%s", shape.get("shape"))
    return results

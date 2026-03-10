"""
Statistical Analysis Processor.

Cross-cutting statistical utilities used by multiple modules:
  - Annualized / monthly volatility calculation
  - Correlation matrix (DAM vs TTF, CO2, wind, load, etc.)
  - Distribution fitting (normal, log-normal, GEV) for price data
  - Percentile distribution tables

These outputs feed Module 6 (Risk Premium) and Layer 3 visualizations.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


def compute_volatility(
    prices: pd.Series,
    window: int = 30,
    annualize: bool = True,
    intervals_per_year: int = 365,
) -> pd.DataFrame:
    """
    Compute rolling volatility of price returns.

    Parameters
    ----------
    prices : Series
        Daily (or hourly) price series
    window : int
        Rolling window size (days)
    annualize : bool
        If True, annualize the volatility
    intervals_per_year : int
        Trading days/intervals per year for annualization

    Returns
    -------
    DataFrame with log_returns, rolling_vol, annualized_vol columns.
    """
    df = pd.DataFrame({"price": prices})
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    df["rolling_vol"] = df["log_return"].rolling(window).std()

    if annualize:
        df["annualized_vol"] = df["rolling_vol"] * np.sqrt(intervals_per_year)

    logger.info("Volatility computed: window=%d, latest: %.2f%%",
                window, df["annualized_vol"].iloc[-1] * 100 if len(df) > 0 else 0)
    return df


def compute_correlation_matrix(
    data: Dict[str, pd.Series],
    method: str = "pearson",
    min_periods: int = 30,
) -> pd.DataFrame:
    """
    Compute correlation matrix between multiple price/fundamental series.

    Parameters
    ----------
    data : Dict[str, Series]
        Named series to correlate (e.g., {'DAM': dam_prices, 'TTF': ttf_prices, ...})

    Returns
    -------
    Correlation matrix DataFrame.
    """
    aligned = pd.DataFrame(data)
    corr = aligned.corr(method=method, min_periods=min_periods)
    logger.info("Correlation matrix: %d × %d series", len(corr), len(corr.columns))
    return corr


def compute_rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int = 90,
    name_a: str = "A",
    name_b: str = "B",
) -> pd.Series:
    """Compute rolling correlation between two series."""
    aligned = pd.concat([series_a, series_b], axis=1, join="inner")
    aligned.columns = [name_a, name_b]
    rolling_corr = aligned[name_a].rolling(window).corr(aligned[name_b])
    rolling_corr.name = f"corr_{name_a}_{name_b}_{window}d"
    return rolling_corr


def compute_percentile_table(
    prices: pd.Series,
    percentiles: Tuple[int, ...] = (5, 10, 25, 50, 75, 90, 95),
    group_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute percentile distribution table.

    Parameters
    ----------
    group_by : str, optional
        Grouping ('month', 'quarter', 'year', 'hour'). None = full dataset.
    """
    df = pd.DataFrame({"price": prices})

    if group_by == "month":
        df["group"] = df.index.to_period("M")
    elif group_by == "quarter":
        df["group"] = df.index.to_period("Q")
    elif group_by == "year":
        df["group"] = df.index.year
    elif group_by == "hour":
        df["group"] = df.index.hour
    else:
        df["group"] = "all"

    result = df.groupby("group")["price"].describe(
        percentiles=[p / 100 for p in percentiles]
    )

    logger.info("Percentile table: %d groups × %d percentiles",
                len(result), len(percentiles))
    return result


def fit_distribution(
    prices: pd.Series,
    distributions: List[str] = None,
) -> Dict[str, dict]:
    """
    Fit statistical distributions to price data and rank by goodness-of-fit.

    Returns dict of {distribution_name: {params, ks_stat, p_value, aic}}.
    """
    if distributions is None:
        distributions = ["norm", "lognorm", "gumbel_r", "genextreme"]

    clean = prices.dropna()
    if len(clean) < 30:
        logger.warning("Too few data points (%d) for distribution fitting", len(clean))
        return {}

    results = {}
    for dist_name in distributions:
        try:
            dist = getattr(sp_stats, dist_name)
            params = dist.fit(clean)
            ks_stat, p_value = sp_stats.kstest(clean, dist_name, args=params)

            # AIC approximation
            log_likelihood = np.sum(dist.logpdf(clean, *params))
            k = len(params)
            aic = 2 * k - 2 * log_likelihood

            results[dist_name] = {
                "params": params,
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "aic": aic,
                "log_likelihood": log_likelihood,
            }
        except Exception as e:
            logger.warning("Distribution fitting failed for %s: %s", dist_name, e)

    # Rank by AIC (lower is better)
    if results:
        best = min(results.items(), key=lambda x: x[1]["aic"])
        logger.info("Best fit distribution: %s (AIC=%.1f)", best[0], best[1]["aic"])

    return results


def compute_var_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Compute Value-at-Risk and Conditional VaR (Expected Shortfall).
    Useful for risk premium calibration (Module 6).

    Parameters
    ----------
    returns : Series
        Daily log returns or price changes
    confidence : float
        Confidence level (e.g., 0.95 for 95% VaR)

    Returns
    -------
    Dict with var, cvar, and confidence level.
    """
    clean = returns.dropna()
    if len(clean) < 30:
        return {"var": np.nan, "cvar": np.nan, "confidence": confidence}

    alpha = 1 - confidence
    var = np.percentile(clean, alpha * 100)
    cvar = clean[clean <= var].mean()

    return {
        "var": var,
        "cvar": cvar,
        "confidence": confidence,
        "n_observations": len(clean),
    }

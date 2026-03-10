"""
Validation & Quality Assurance Module.

Cross-check protocol (Section 14.1):
  1. DAM price validation: model vs OPCOM published monthly average (±3%)
  2. SRMC validation: Gas-SRMC vs published clean spark spread
  3. Forward curve arbitrage check
  4. Tariff validation: sum of regulated components matches ANRE total
  5. Contract register integrity: contracted ≤ delivery obligation
  6. Final price benchmarking vs Romania-Insider ~108.28 EUR/MWh YTD H1 2025

Flags data staleness (>7 days) and missing values.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


class ValidationResult:
    """Structured validation result."""

    def __init__(self, check_name: str, status: str, detail: str,
                 expected: Any = None, actual: Any = None, tolerance: float = None):
        self.check_name = check_name
        self.status = status  # 'PASS', 'WARN', 'FAIL'
        self.detail = detail
        self.expected = expected
        self.actual = actual
        self.tolerance = tolerance
        self.timestamp = datetime.now()

    def __repr__(self):
        return f"[{self.status}] {self.check_name}: {self.detail}"

    def to_dict(self):
        return {
            "check": self.check_name,
            "status": self.status,
            "detail": self.detail,
            "expected": self.expected,
            "actual": self.actual,
            "tolerance": self.tolerance,
            "timestamp": self.timestamp.isoformat(),
        }


def validate_dam_prices(
    model_monthly: pd.DataFrame,
    benchmarks: Optional[Dict[str, float]] = None,
    tolerance: float = 0.03,
) -> List[ValidationResult]:
    """
    Validate model's DAM monthly averages against OPCOM published benchmarks.
    Tolerance: ±3% per Section 14.1.
    """
    results = []
    if benchmarks is None:
        benchmarks = settings.get_assumption("validation", "benchmark_prices", default={})

    # Map benchmark keys to dates for comparison
    benchmark_map = {
        "jan_2025_ron_mwh": ("2025-01", "RON"),
        "mar_2025_eur_mwh": ("2025-03", "EUR"),
        "jul_2025_eur_mwh": ("2025-07", "EUR"),
        "oct_2025_eur_mwh": ("2025-10", "EUR"),
        "feb_2026_ron_mwh": ("2026-02", "RON"),
    }

    price_col = "base_avg"  # From dam_analysis monthly summary
    if price_col not in model_monthly.columns:
        results.append(ValidationResult(
            "DAM_price_validation", "FAIL",
            f"Column '{price_col}' not found in monthly summary"
        ))
        return results

    for bm_key, (period, currency) in benchmark_map.items():
        if bm_key not in benchmarks:
            continue

        expected = benchmarks[bm_key]
        # Find matching month in model data
        matching = model_monthly[
            model_monthly.index.strftime("%Y-%m") == period
        ]

        if matching.empty:
            results.append(ValidationResult(
                f"DAM_{period}", "WARN",
                f"No model data for {period}",
                expected=expected
            ))
            continue

        actual = matching[price_col].iloc[0]
        # Convert RON benchmarks to EUR for comparison if needed
        if currency == "RON":
            # Skip direct comparison for RON benchmarks (need FX rate)
            results.append(ValidationResult(
                f"DAM_{period}", "INFO",
                f"RON benchmark: expected {expected} RON/MWh, model base avg (EUR): {actual:.2f}",
                expected=expected, actual=actual
            ))
            continue

        deviation = abs(actual - expected) / expected
        status = "PASS" if deviation <= tolerance else "FAIL"
        results.append(ValidationResult(
            f"DAM_{period}", status,
            f"Expected: {expected:.2f}, Actual: {actual:.2f}, "
            f"Deviation: {deviation*100:.1f}%",
            expected=expected, actual=actual, tolerance=tolerance
        ))

    return results


def validate_data_freshness(
    datasets: Dict[str, pd.DataFrame],
    max_staleness_days: int = 7,
) -> List[ValidationResult]:
    """
    Check that all datasets are within acceptable freshness thresholds.
    Flag any dataset with data older than max_staleness_days from today.
    """
    results = []
    today = pd.Timestamp.now(tz="Europe/Bucharest")

    for name, df in datasets.items():
        if df is None or df.empty:
            results.append(ValidationResult(
                f"freshness_{name}", "FAIL",
                f"Dataset '{name}' is empty or None"
            ))
            continue

        latest = df.index.max()
        if hasattr(latest, 'tz') and latest.tz is None:
            latest = latest.tz_localize("Europe/Bucharest")

        age_days = (today - latest).days
        status = "PASS" if age_days <= max_staleness_days else "WARN"
        results.append(ValidationResult(
            f"freshness_{name}", status,
            f"Latest data: {latest}, age: {age_days} days",
            expected=f"≤{max_staleness_days} days",
            actual=f"{age_days} days"
        ))

    return results


def validate_srmc(
    gas_srmc: pd.DataFrame,
    coal_srmc: Optional[pd.DataFrame] = None,
) -> List[ValidationResult]:
    """Sanity-check SRMC values against expected ranges."""
    results = []

    # Gas CCGT SRMC should be in 40–200 EUR/MWh range
    if "srmc_gas_ccgt_eur_mwh" in gas_srmc.columns:
        latest = gas_srmc["srmc_gas_ccgt_eur_mwh"].iloc[-1]
        if 40 <= latest <= 200:
            results.append(ValidationResult(
                "SRMC_gas_CCGT", "PASS",
                f"Latest: {latest:.2f} EUR/MWh (within 40–200 range)"
            ))
        else:
            results.append(ValidationResult(
                "SRMC_gas_CCGT", "WARN",
                f"Latest: {latest:.2f} EUR/MWh (outside typical 40–200 range)"
            ))

    return results


def validate_completeness(
    df: pd.DataFrame,
    name: str,
    max_gap_hours: int = 4,
) -> List[ValidationResult]:
    """Check for data gaps exceeding threshold in a time-indexed DataFrame."""
    results = []

    if df.empty:
        results.append(ValidationResult(
            f"completeness_{name}", "FAIL", "Empty dataset"
        ))
        return results

    # Check for NaN values
    null_pct = df.isnull().mean()
    for col in null_pct.index:
        pct = null_pct[col] * 100
        status = "PASS" if pct < 5 else ("WARN" if pct < 15 else "FAIL")
        results.append(ValidationResult(
            f"nulls_{name}_{col}", status,
            f"Null rate: {pct:.1f}%"
        ))

    # Check for time gaps
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        diffs = df.index.to_series().diff()
        expected_freq = diffs.median()
        large_gaps = diffs[diffs > expected_freq * max_gap_hours]

        if len(large_gaps) > 0:
            max_gap = large_gaps.max()
            results.append(ValidationResult(
                f"gaps_{name}", "WARN",
                f"{len(large_gaps)} gaps > {max_gap_hours}× median frequency; "
                f"largest: {max_gap}"
            ))
        else:
            results.append(ValidationResult(
                f"gaps_{name}", "PASS",
                f"No significant gaps detected (max gap threshold: {max_gap_hours}×)"
            ))

    return results


def run_full_validation(
    datasets: Dict[str, pd.DataFrame],
    dam_monthly: Optional[pd.DataFrame] = None,
    gas_srmc: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Execute all validation checks and return results as a DataFrame.

    Returns
    -------
    DataFrame with columns: check, status, detail, expected, actual, timestamp.
    """
    all_results = []

    # Data freshness
    all_results.extend(validate_data_freshness(datasets))

    # Completeness for each dataset
    for name, df in datasets.items():
        if df is not None and not df.empty:
            all_results.extend(validate_completeness(df, name))

    # DAM price benchmarks
    if dam_monthly is not None:
        all_results.extend(validate_dam_prices(dam_monthly))

    # SRMC sanity
    if gas_srmc is not None:
        all_results.extend(validate_srmc(gas_srmc))

    # Summary
    results_df = pd.DataFrame([r.to_dict() for r in all_results])
    pass_count = (results_df["status"] == "PASS").sum()
    warn_count = (results_df["status"] == "WARN").sum()
    fail_count = (results_df["status"] == "FAIL").sum()

    logger.info("Validation complete: %d PASS, %d WARN, %d FAIL (of %d checks)",
                pass_count, warn_count, fail_count, len(results_df))

    return results_df

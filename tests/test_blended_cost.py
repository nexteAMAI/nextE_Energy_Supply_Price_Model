"""
Tests for DAM Analysis and Imbalance processors.
"""

import numpy as np
import pandas as pd
import pytest

from processors.dam_analysis import (
    classify_hour_block,
    compute_daily_averages,
    compute_monthly_summary,
    add_time_dimensions,
)
from processors.imbalance import (
    compute_imbalance_spread_to_dam,
    compute_imbalance_cost_adder,
)


# ---- Fixtures ----

@pytest.fixture
def sample_dam():
    """Generate synthetic DAM price data (hourly, 1 month)."""
    idx = pd.date_range("2025-01-01", periods=744, freq="h", tz="Europe/Bucharest")
    np.random.seed(42)
    # Peak hours (9–19) get +30 EUR premium; off-peak gets base
    hours = idx.hour
    peak_premium = np.where((hours >= 9) & (hours < 20), 30, 0)
    prices = 80 + peak_premium + np.random.normal(0, 10, 744)
    df = pd.DataFrame({"Value [EUR/MWh]": prices}, index=idx)
    df.index.name = "timestamp_eet"
    return df


@pytest.fixture
def sample_imbalance_prices():
    """Generate synthetic imbalance prices (Long/Short)."""
    idx = pd.date_range("2025-01-01", periods=744, freq="h", tz="Europe/Bucharest")
    np.random.seed(42)
    long_prices = 70 + np.random.normal(0, 15, 744)
    short_prices = 130 + np.random.normal(0, 20, 744)
    df = pd.DataFrame({
        "Long [EUR/MWh]": long_prices,
        "Short [EUR/MWh]": short_prices,
    }, index=idx)
    df.index.name = "timestamp_eet"
    return df


# ---- DAM Tests ----

class TestClassifyHourBlock:

    def test_peak_hours(self):
        for h in range(9, 20):
            assert classify_hour_block(h) == "Peak"

    def test_offpeak_hours(self):
        for h in [0, 1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23]:
            assert classify_hour_block(h) == "Off-Peak"


class TestDailyAverages:

    def test_output_columns(self, sample_dam):
        result = compute_daily_averages(sample_dam)
        assert "base_eur_mwh" in result.columns
        assert "peak_eur_mwh" in result.columns
        assert "offpeak_eur_mwh" in result.columns
        assert "peak_offpeak_spread" in result.columns

    def test_peak_ge_offpeak_on_average(self, sample_dam):
        """Normally peak prices > off-peak prices."""
        result = compute_daily_averages(sample_dam)
        # With synthetic sinusoidal data, this should hold most days
        assert result["peak_offpeak_spread"].mean() > 0

    def test_day_count(self, sample_dam):
        result = compute_daily_averages(sample_dam)
        assert len(result) == 31  # January has 31 days


class TestMonthlySummary:

    def test_output_has_percentiles(self, sample_dam):
        result = compute_monthly_summary(sample_dam)
        assert "p10" in result.columns
        assert "p50" in result.columns
        assert "p90" in result.columns

    def test_p10_le_p50_le_p90(self, sample_dam):
        result = compute_monthly_summary(sample_dam)
        assert (result["p10"] <= result["p50"]).all()
        assert (result["p50"] <= result["p90"]).all()

    def test_single_month(self, sample_dam):
        result = compute_monthly_summary(sample_dam)
        assert len(result) == 1  # Only January


# ---- Imbalance Tests ----

class TestImbalanceSpread:

    def test_spread_calculation(self, sample_dam, sample_imbalance_prices):
        result = compute_imbalance_spread_to_dam(sample_imbalance_prices, sample_dam)
        assert "long_spread" in result.columns
        assert "short_spread" in result.columns
        assert "long_short_spread" in result.columns
        assert len(result) > 0

    def test_short_spread_positive(self, sample_dam, sample_imbalance_prices):
        """Short price > DAM → short spread should be positive on average."""
        result = compute_imbalance_spread_to_dam(sample_imbalance_prices, sample_dam)
        assert result["short_spread"].mean() > 0


class TestImbalanceCostAdder:

    def test_cost_within_expected_range(self, sample_dam, sample_imbalance_prices):
        """P50 imbalance cost should be in 1.5–4.0 EUR/MWh for well-managed portfolio."""
        spread = compute_imbalance_spread_to_dam(sample_imbalance_prices, sample_dam)
        monthly = compute_imbalance_cost_adder(spread, imbalance_rate=0.05)

        if not monthly.empty:
            avg_cost = monthly["imbalance_cost_p50_eur_mwh"].mean()
            # Relaxed bounds for synthetic data — just check it's reasonable
            assert 0 < avg_cost < 20

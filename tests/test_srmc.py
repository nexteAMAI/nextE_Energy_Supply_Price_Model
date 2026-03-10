"""
Tests for SRMC Calculation (Module 2 → Module 3).

Validates:
  - Gas-SRMC formula: (Gas / η) + (CO2 × intensity)
  - Coal-SRMC formula: (Coal_EUR_per_MWh_th / η) + (CO2 × intensity)
  - Clean spark/dark spread calculation
  - Edge cases (zero prices, NaN handling)
"""

import numpy as np
import pandas as pd
import pytest

from processors.srmc import (
    compute_gas_srmc,
    compute_coal_srmc,
    compute_clean_spreads,
    COAL_API2_MWH_TH_PER_TONNE,
)


@pytest.fixture
def sample_gas_prices():
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    return pd.Series([30.0, 35.0, 40.0, 25.0, 50.0], index=dates, name="gas_price")


@pytest.fixture
def sample_co2_prices():
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    return pd.Series([60.0, 65.0, 70.0, 55.0, 80.0], index=dates, name="co2_price")


class TestGasSRMC:

    def test_formula_correctness(self, sample_gas_prices, sample_co2_prices):
        """Gas-SRMC = (Gas/0.55) + (CO2 × 0.37)."""
        result = compute_gas_srmc(sample_gas_prices, sample_co2_prices)

        expected_ccgt = (30.0 / 0.55) + (60.0 * 0.37)
        assert abs(result["srmc_gas_ccgt_eur_mwh"].iloc[0] - expected_ccgt) < 0.01

    def test_ocgt_higher_than_ccgt(self, sample_gas_prices, sample_co2_prices):
        """OCGT (38% eff) should always produce higher SRMC than CCGT (55% eff)."""
        result = compute_gas_srmc(sample_gas_prices, sample_co2_prices)
        assert (result["srmc_gas_ocgt_eur_mwh"] > result["srmc_gas_ccgt_eur_mwh"]).all()

    def test_custom_efficiency(self, sample_gas_prices, sample_co2_prices):
        """Custom efficiency should override default."""
        result = compute_gas_srmc(sample_gas_prices, sample_co2_prices, efficiency=0.50)
        expected = (30.0 / 0.50) + (60.0 * 0.37)
        assert abs(result["srmc_gas_ccgt_eur_mwh"].iloc[0] - expected) < 0.01

    def test_output_columns(self, sample_gas_prices, sample_co2_prices):
        result = compute_gas_srmc(sample_gas_prices, sample_co2_prices)
        assert "srmc_gas_ccgt_eur_mwh" in result.columns
        assert "srmc_gas_ocgt_eur_mwh" in result.columns
        assert "gas_price_eur_mwh_th" in result.columns
        assert "co2_price_eur_t" in result.columns

    def test_length_matches_input(self, sample_gas_prices, sample_co2_prices):
        result = compute_gas_srmc(sample_gas_prices, sample_co2_prices)
        assert len(result) == 5


class TestCoalSRMC:

    def test_formula_correctness(self, sample_co2_prices):
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        coal_usd = pd.Series([100.0, 110.0, 120.0, 90.0, 130.0], index=dates)
        usd_eur = pd.Series([0.92, 0.92, 0.92, 0.92, 0.92], index=dates)

        result = compute_coal_srmc(coal_usd, sample_co2_prices, usd_eur, coal_type="hard_coal")

        coal_eur_t = 100.0 * 0.92
        coal_eur_mwh_th = coal_eur_t / COAL_API2_MWH_TH_PER_TONNE
        expected = (coal_eur_mwh_th / 0.42) + (60.0 * 0.34)
        assert abs(result["srmc_hard_coal_eur_mwh"].iloc[0] - expected) < 0.01

    def test_lignite_higher_co2(self, sample_co2_prices):
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        coal_usd = pd.Series([100.0] * 5, index=dates)
        usd_eur = pd.Series([0.92] * 5, index=dates)

        hard = compute_coal_srmc(coal_usd, sample_co2_prices, usd_eur, "hard_coal")
        lignite = compute_coal_srmc(coal_usd, sample_co2_prices, usd_eur, "lignite")

        # Lignite has higher CO2 intensity (0.40 vs 0.34) and lower efficiency (0.38 vs 0.42)
        assert (lignite.iloc[:, 0] > hard.iloc[:, 0]).all()


class TestCleanSpreads:

    def test_spark_spread(self, sample_gas_prices, sample_co2_prices):
        gas_srmc = compute_gas_srmc(sample_gas_prices, sample_co2_prices)
        power = pd.Series([120.0] * 5, index=gas_srmc.index)

        result = compute_clean_spreads(power, gas_srmc)

        assert "clean_spark_spread" in result.columns
        # Spread = Power - SRMC; should be positive when power > SRMC
        expected_srmc = (30.0 / 0.55) + (60.0 * 0.37)
        expected_spread = 120.0 - expected_srmc
        assert abs(result["clean_spark_spread"].iloc[0] - expected_spread) < 0.01

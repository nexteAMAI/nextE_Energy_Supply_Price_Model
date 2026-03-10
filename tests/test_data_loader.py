"""
Integration tests for the data loader module.

Validates that all pre-loaded CSV datasets can be correctly parsed
with the right format handling (encoding, delimiter, decimal, timezone).

These tests require the project data files to be present in /mnt/project.
"""

import pandas as pd
import pytest
import os

# Check if project data files are available
DATA_AVAILABLE = os.path.exists("/mnt/project/RO_day_ahead_prices_ENTSOE.csv")
skip_no_data = pytest.mark.skipif(not DATA_AVAILABLE, reason="Project data files not available")


@skip_no_data
class TestENTSOELoaders:
    """Test standard ENTSO-E format CSV loading."""

    def test_dam_prices(self):
        from extractors.data_loader import load_entsoe_csv
        df = load_entsoe_csv("RO_day_ahead_prices_ENTSOE.csv")
        assert len(df) > 100000
        assert "Value [EUR/MWh]" in df.columns
        assert df.index.tz is not None
        assert df.index.is_monotonic_increasing

    def test_generation(self):
        from extractors.data_loader import load_entsoe_csv
        df = load_entsoe_csv("RO_generation_ENTSOE.csv")
        assert len(df) > 200000
        assert "Nuclear [MW]" in df.columns
        assert "Wind Onshore [MW]" in df.columns

    def test_load(self):
        from extractors.data_loader import load_entsoe_csv
        df = load_entsoe_csv("RO_load_ENTSOE.csv")
        assert len(df) > 200000
        assert "Actual Load [MW]" in df.columns

    def test_imbalance_prices(self):
        from extractors.data_loader import load_entsoe_csv
        df = load_entsoe_csv("RO_imbalance_prices_ENTSOE.csv")
        assert "Long [EUR/MWh]" in df.columns
        assert "Short [EUR/MWh]" in df.columns

    def test_imports(self):
        from extractors.data_loader import load_entsoe_csv
        df = load_entsoe_csv("RO_import_ENTSOE.csv")
        assert "HU [MW]" in df.columns
        assert "BG [MW]" in df.columns


@skip_no_data
class TestSpecialFormatLoaders:

    def test_idm_nexte(self):
        from extractors.data_loader import load_idm_nexte
        df = load_idm_nexte()
        assert len(df) > 150000
        # Check VWAP column exists (may have prefix variation)
        vwap_cols = [c for c in df.columns if "VWAP" in c]
        assert len(vwap_cols) > 0

    def test_aurora_forecast(self):
        from extractors.data_loader import load_aurora_forecast
        df = load_aurora_forecast()
        assert len(df) > 200  # ~300 months (2026–2050+)
        central_cols = [c for c in df.columns if "Baseload_Central" in c]
        assert len(central_cols) > 0
        # Validate Jan 2026 benchmark: 133.62 EUR/MWh
        jan_2026 = df.loc["2026-01-01"]
        baseload_central = jan_2026[central_cols[0]]
        assert abs(baseload_central - 133.62) < 0.1

    def test_fx_eur_ron(self):
        from extractors.data_loader import load_fx_eur_ron
        df = load_fx_eur_ron()
        assert len(df) > 5000
        assert "eur_ron" in df.columns
        # Rate should be in realistic range (4.0–5.5)
        assert 3.5 < df["eur_ron"].mean() < 6.0

    def test_sensitivity_scenarios(self):
        from extractors.data_loader import load_sensitivity_scenarios
        df = load_sensitivity_scenarios()
        assert len(df) > 30000
        # Should have ~20 scenario columns
        assert len(df.columns) >= 15

    def test_residual_load_montel(self):
        from extractors.data_loader import load_montel_backcast
        df = load_montel_backcast(
            "RO_Residual_Load_MWh_h_15min_Backcast_MONTEL.csv",
            value_col_name="residual_load_mwh"
        )
        assert len(df) > 300000


@skip_no_data
class TestBalancingServicesLoaders:

    def test_imbalance_prices_bs(self):
        from extractors.data_loader import load_balancing_services_csv
        df = load_balancing_services_csv("RO_imbalance_prices_NA_BAL_SERV.csv")
        assert len(df) > 50000
        assert "direction" in df.columns

    def test_imbalance_volumes_bs(self):
        from extractors.data_loader import load_balancing_services_csv
        df = load_balancing_services_csv("RO_imbalance_totalvolumes_NA_BAL_SERV.csv")
        assert len(df) > 50000
        assert "direction" in df.columns

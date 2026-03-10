"""Tests for the validation module."""

import pandas as pd
import numpy as np
import pytest

from outputs.validation import (
    ValidationResult,
    validate_data_freshness,
    validate_completeness,
)


@pytest.fixture
def sample_dataset():
    idx = pd.date_range("2025-01-01", periods=100, freq="h", tz="Europe/Bucharest")
    return pd.DataFrame({"value": np.random.randn(100)}, index=idx)


class TestDataFreshness:

    def test_stale_data_warns(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="Europe/Bucharest")
        df = pd.DataFrame({"v": range(10)}, index=idx)
        results = validate_data_freshness({"old_data": df}, max_staleness_days=7)
        assert any(r.status == "WARN" for r in results)

    def test_empty_fails(self):
        results = validate_data_freshness({"empty": pd.DataFrame()})
        assert any(r.status == "FAIL" for r in results)


class TestCompleteness:

    def test_no_nulls_passes(self, sample_dataset):
        results = validate_completeness(sample_dataset, "test")
        null_checks = [r for r in results if "nulls" in r.check_name]
        assert all(r.status == "PASS" for r in null_checks)

    def test_many_nulls_fails(self):
        idx = pd.date_range("2025-01-01", periods=100, freq="h", tz="Europe/Bucharest")
        data = [np.nan] * 50 + list(range(50))
        df = pd.DataFrame({"value": data}, index=idx)
        results = validate_completeness(df, "test")
        null_checks = [r for r in results if "nulls" in r.check_name]
        assert any(r.status in ("WARN", "FAIL") for r in null_checks)

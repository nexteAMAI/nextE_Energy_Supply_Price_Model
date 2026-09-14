"""esb.monthly - helpers for the monthly tables (workbook columns B:M = months, O = year).

Year-column conventions of the workbook:
    volumes and values   -> SUM(B:M)
    average power rows   -> AVERAGEIFS over the whole year of quarter-hours > 0
    ratios / prices      -> recomputed from the year totals
    cumulative rows      -> December value
    outstanding amounts  -> MAX(B:M)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = list(range(1, 13))
YEAR = "Y"
COLUMNS = MONTHS + [YEAR]


class MonthlyTable:
    """Ordered rows of 12 monthly values + a year value. Rows are numpy arrays of length 13."""

    def __init__(self) -> None:
        self.rows: dict[str, np.ndarray] = {}
        self.order: list[str] = []

    def put(self, key: str, months: np.ndarray, year: float | None = None, year_rule: str = "sum") -> np.ndarray:
        m = np.asarray(months, dtype=float)
        if m.shape != (12,):
            raise ValueError(f"{key}: expected 12 monthly values, got {m.shape}")
        if year is None:
            if year_rule == "sum":
                year = float(m.sum())
            elif year_rule == "last":
                year = float(m[-1])
            elif year_rule == "max":
                year = float(m.max())
            elif year_rule == "mean":
                year = float(m.mean())
            else:
                raise ValueError(year_rule)
        row = np.concatenate([m, [float(year)]])
        if key not in self.rows:
            self.order.append(key)
        self.rows[key] = row
        return row

    def __getitem__(self, key: str) -> np.ndarray:
        return self.rows[key]

    def __contains__(self, key: str) -> bool:
        return key in self.rows

    def m(self, key: str) -> np.ndarray:
        return self.rows[key][:12]

    def y(self, key: str) -> float:
        return float(self.rows[key][12])

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({k: self.rows[k] for k in self.order}, index=COLUMNS).T


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """IFERROR(num/den, 0) elementwise (13-vectors)."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.zeros_like(num)
    ok = den != 0
    out[ok] = num[ok] / den[ok]
    return out


def ratio_row(t: MonthlyTable, key: str, num_key: str, den_key: str) -> np.ndarray:
    """IFERROR(num/den, 0) per month and for the year from the year totals."""
    r = safe_div(t[num_key], t[den_key])
    return t.put(key, r[:12], float(r[12]))


class QHAggregator:
    """SUMIFS / AVERAGEIFS by month over the quarter-hourly frame."""

    def __init__(self, month_of_row: np.ndarray) -> None:
        self.month = np.asarray(month_of_row, dtype=int)
        self.masks = [self.month == m for m in MONTHS]

    def sum(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        return np.array([v[mask].sum() for mask in self.masks])

    def mean_positive(self, values: np.ndarray) -> tuple[np.ndarray, float]:
        """(monthly means of values > 0, yearly mean of values > 0); 0 where nothing is > 0."""
        v = np.asarray(values, dtype=float)
        months = []
        for mask in self.masks:
            x = v[mask]
            x = x[x > 0]
            months.append(x.mean() if len(x) else 0.0)
        y = v[v > 0]
        return np.array(months), (float(y.mean()) if len(y) else 0.0)

    def mean(self, values: np.ndarray) -> np.ndarray:
        """AVERAGEIFS by month without a value condition (0 when the month is empty)."""
        v = np.asarray(values, dtype=float)
        return np.array([v[mask].mean() if mask.any() else 0.0 for mask in self.masks])

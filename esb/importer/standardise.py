"""Standardisation of a raw delivery onto the engine grid (days x 96), applying the daylight-
saving rules of the STD_EET_QH contract for `local_clock` deliveries.

Rules (registry-declared per slot, applied only on the two transition days of a local_clock
delivery):

* Injected (spring, the four 03:00-03:45 EET intervals that do not exist on the clock):
  Extensive -> 0; Intensive `interpolate` -> linear between the last value before and the first
  value after the gap, v_prev + (v_next - v_prev) * j / 5, j = 1..4; Intensive `locf` -> last
  value before the gap; Categorical `carry_forward` -> last value before the gap.
* Merged (autumn, the four 03:00-03:45 EET intervals that occur twice): Extensive -> sum of the
  two rows; Intensive `mean` -> simple mean; `volume_weighted` -> weighted by the paired volume
  slot's two rows (falls back to the simple mean when the weights sum to 0); Categorical
  `first` -> the first occurrence.

The output has exactly one row per (date, interval) of the spine year. Blank input cells stay
NaN; nothing else is ever filled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from esb import grid
from esb.importer.contract import Control, Registry

SPRING_LABELS = ("Injected",)
AUTUMN_LABELS = ("Merged",)


@dataclass
class LayoutReport:
    """Where the delivery deviates from the expected (date, interval) layout."""

    expected_rows: int
    delivered_rows: int
    per_day_expected: pd.Series
    per_day_delivered: pd.Series
    missing: list[tuple] = field(default_factory=list)  # (date, interval) expected once, absent
    duplicated: list[tuple] = field(default_factory=list)  # (date, interval) delivered more than expected
    out_of_year: int = 0
    unparseable_rows: int = 0
    non_monotonic: bool = False

    @property
    def ok(self) -> bool:
        return not (self.missing or self.duplicated or self.out_of_year or self.unparseable_rows)


def expected_multiplicity(control: Control) -> pd.DataFrame:
    """Expected number of raw rows per (date, interval) for the spine year."""
    g = grid.make_grid(int(control.spine_year))
    mult = np.ones(len(g), dtype=int)
    if control.time_basis == "local_clock":
        mult[g["dst_status"].values == "Injected"] = 0
        mult[g["dst_status"].values == "Merged"] = 2
    return pd.DataFrame({"date": g["date"].dt.date.values, "interval": g["interval"].values, "expected": mult})


def label_rows(raw: pd.DataFrame, control: Control) -> tuple[pd.DataFrame, LayoutReport]:
    """Attach (date, interval, occurrence) to every raw row and compare with the expected layout."""
    year = int(control.spine_year)
    df = raw.copy()
    bad = df["Date_EET"].isna() | df["Start_EET"].isna()
    unparseable = int(bad.sum())
    df = df[~bad].copy()
    df["interval"] = [grid.interval_of(t) for t in df["Start_EET"]]
    df["date"] = df["Date_EET"]
    in_year = pd.Series([d.year == year for d in df["date"]], index=df.index)
    out_of_year = int((~in_year).sum())
    df = df[in_year].copy()
    order = np.lexsort((df["interval"].values, pd.to_datetime(df["date"]).values))
    non_monotonic = not np.array_equal(order, np.arange(len(df)))
    df = df.iloc[order].reset_index(drop=True)
    df["occurrence"] = df.groupby(["date", "interval"]).cumcount() + 1

    exp = expected_multiplicity(control)
    delivered = df.groupby(["date", "interval"]).size().rename("delivered")
    cmp = exp.set_index(["date", "interval"]).join(delivered, how="left").fillna({"delivered": 0})
    cmp["delivered"] = cmp["delivered"].astype(int)
    missing = cmp.index[cmp["delivered"] < cmp["expected"]].tolist()
    duplicated = cmp.index[cmp["delivered"] > cmp["expected"]].tolist()
    per_day_exp = cmp.groupby(level=0)["expected"].sum()
    per_day_del = cmp.groupby(level=0)["delivered"].sum()
    report = LayoutReport(
        expected_rows=int(cmp["expected"].sum()),
        delivered_rows=int(len(df)),
        per_day_expected=per_day_exp,
        per_day_delivered=per_day_del,
        missing=missing[:200],
        duplicated=duplicated[:200],
        out_of_year=out_of_year,
        unparseable_rows=unparseable,
        non_monotonic=non_monotonic,
    )
    return df, report


def _interpolate_gap(series: pd.Series, first: int, n: int) -> None:
    """Fill positions first..first+n-1 linearly between series[first-1] and series[first+n] (in place)."""
    prev_i, next_i = first - 1, first + n
    v_prev = series.iat[prev_i] if prev_i >= 0 else np.nan
    v_next = series.iat[next_i] if next_i < len(series) else np.nan
    for j in range(1, n + 1):
        if np.isnan(v_prev) and np.isnan(v_next):
            val = np.nan
        elif np.isnan(v_prev):
            val = v_next
        elif np.isnan(v_next):
            val = v_prev
        else:
            val = v_prev + (v_next - v_prev) * j / (n + 1)
        series.iat[first + j - 1] = val


def standardise(labelled: pd.DataFrame, control: Control, registry: Registry) -> pd.DataFrame:
    """Map labelled raw rows onto the grid and apply the DST rules. Returns a frame indexed like
    esb.grid.make_grid (seq 1..N) with one column per slot *name* plus the grid columns."""
    g = grid.make_grid(int(control.spine_year))
    key = pd.MultiIndex.from_arrays([g["date"].dt.date.values, g["interval"].values])
    out = g.copy()
    first = labelled[labelled["occurrence"] == 1].set_index(["date", "interval"])
    second = labelled[labelled["occurrence"] == 2].set_index(["date", "interval"])
    is_injected = g["dst_status"].values == "Injected"
    is_merged = g["dst_status"].values == "Merged"
    local = control.time_basis == "local_clock"

    by_slot = registry.by_slot()

    def aligned(src: pd.DataFrame, slot: str, categorical: bool) -> pd.Series:
        # raw frames carry slot ids (reader) or slot names (frames built in code) - accept both
        col = slot if slot in src.columns else (by_slot[slot].name if slot in by_slot and by_slot[slot].name in src.columns else None)
        s = src[col].reindex(key) if col is not None else pd.Series(index=key, dtype=object)
        if categorical:
            return pd.Series(s.values, dtype=object)
        return pd.Series(pd.to_numeric(s.values, errors="coerce"), dtype=float)

    for spec in registry.slots:
        cat = spec.cls == "Categorical"
        a = aligned(first, spec.slot, cat)
        if not local:
            out[spec.name] = a.values
            continue
        b = aligned(second, spec.slot, cat)
        if spec.cls == "Extensive":
            a[is_injected] = 0.0
            a[is_merged] = a[is_merged].add(b[is_merged], fill_value=0.0)  # NaN only if both blank
        elif spec.cls == "Intensive":
            if spec.autumn_rule == "volume_weighted" and spec.paired_volume_slot:
                wa = aligned(first, spec.paired_volume_slot, False)
                wb = aligned(second, spec.paired_volume_slot, False)
                num = (a * wa.fillna(0)).where(~a.isna(), 0) + (b * wb.fillna(0)).where(~b.isna(), 0)
                den = wa.fillna(0).where(~a.isna(), 0) + wb.fillna(0).where(~b.isna(), 0)
                mean = pd.concat([a, b], axis=1).mean(axis=1)
                vw = np.where(den.values != 0, num.values / np.where(den.values != 0, den.values, 1), mean.values)
                a[is_merged] = vw[is_merged]
            else:
                mean = pd.concat([a, b], axis=1).mean(axis=1)
                a[is_merged] = mean[is_merged]
            # spring: gaps are the injected positions (values absent in the delivery)
            inj_idx = np.flatnonzero(is_injected)
            if len(inj_idx):
                if spec.spring_rule == "interpolate":
                    _interpolate_gap(a, int(inj_idx[0]), len(inj_idx))
                else:  # locf
                    prev = a.iat[int(inj_idx[0]) - 1] if inj_idx[0] > 0 else np.nan
                    for i in inj_idx:
                        a.iat[int(i)] = prev
        else:  # Categorical
            inj_idx = np.flatnonzero(is_injected)
            if len(inj_idx):
                prev = a.iat[int(inj_idx[0]) - 1] if inj_idx[0] > 0 else None
                for i in inj_idx:
                    a.iat[int(i)] = prev
            # merged: first occurrence already in `a`
        out[spec.name] = a.values
    return out

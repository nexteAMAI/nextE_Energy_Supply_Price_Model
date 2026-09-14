"""esb.sources - Forward Source 1 (PV) and Forward Source 2 (Baseload) per quarter-hour
(workbook sheets `FW_Source_Purch_Volume` and `FW_Purch_Sell_Price`).

PV (FW_Source_Purch_Volume!AH:AW): forecast generation + forecast deviation + imbalance deviation
    -> DSO metered / notified (sign -), BRP metered / notified (sign +), imbalance settlement,
    specific settlement per notified MWh. Rule book in esb.imbalance.

Baseload (FW_Source_Purch_Volume!BA:BW): per-off-taker monthly strips in MW for three products
    (Baseload 24, Peak, Off-Peak), converted to MWh per quarter-hour (MW / 4), gated by the
    off-taker Active flag (D77); the portfolio strip is their sum; deviations (BC, BJ) apply.

Product prices per quarter-hour (FW_Purch_Sell_Price!N / O per off-taker, verbatim logic):
    if strip(BL24 MW + product MW of the row's Peak/Off-Peak flag) == 0 -> BL24 price of the month
    else MW-weighted: (BL24 MW x BL24 price + product MW x product price) / (BL24 MW + product MW)
    IFERROR -> 0 (undated rows). Budgeted table AK7:AV18, forecast table AK27:AV38.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.schema import Offtaker, Parameters
from esb import imbalance
from esb.grid import PEAK_LABEL


def month_index(grid: pd.DataFrame) -> np.ndarray:
    """0-based month per row."""
    return grid["month"].to_numpy(dtype=int) - 1


def strip_mwh_per_qh(o: Offtaker, grid: pd.DataFrame) -> np.ndarray:
    """FW_Source_Purch_Volume!BT..BW: IF(OR(date="", off-taker inactive), 0,
    (BL24 MW of month + (Peak MW if Peak else OffPeak MW)) / 4)."""
    if not o.active:
        return np.zeros(len(grid))
    m = month_index(grid)
    bl24 = np.asarray(o.strip_mw["BL24"], dtype=float)[m]
    peak = np.asarray(o.strip_mw["Peak"], dtype=float)[m]
    off = np.asarray(o.strip_mw["OffPeak"], dtype=float)[m]
    is_peak = grid["peak"].to_numpy() == PEAK_LABEL
    return (bl24 + np.where(is_peak, peak, off)) / 4.0


def product_price_per_qh(o: Offtaker, grid: pd.DataFrame, which: str) -> np.ndarray:
    """FW_Purch_Sell_Price!N (which='budget') or O (which='forecast') for one off-taker."""
    tbl = o.product_price_budget if which == "budget" else o.product_price_forecast
    m = month_index(grid)
    is_peak = grid["peak"].to_numpy() == PEAK_LABEL
    mw_bl = np.asarray(o.strip_mw["BL24"], dtype=float)[m]
    mw_prod = np.where(is_peak, np.asarray(o.strip_mw["Peak"], dtype=float)[m], np.asarray(o.strip_mw["OffPeak"], dtype=float)[m])
    p_bl = np.asarray(tbl["BL24"], dtype=float)[m]
    p_prod = np.where(is_peak, np.asarray(tbl["Peak"], dtype=float)[m], np.asarray(tbl["OffPeak"], dtype=float)[m])
    total = mw_bl + mw_prod
    weighted = np.zeros_like(total)
    nz = total != 0
    weighted[nz] = (mw_bl[nz] * p_bl[nz] + mw_prod[nz] * p_prod[nz]) / total[nz]
    return np.where(total == 0, p_bl, weighted)


@dataclass
class SourceResult:
    pv: imbalance.GenerationImbalance
    baseload: imbalance.GenerationImbalance
    strips: dict[str, np.ndarray]  # per off-taker code: BT..BW (MWh / QH)
    bl_price_budget: dict[str, np.ndarray]  # FW_Purch_Sell_Price!N per off-taker
    bl_price_forecast: dict[str, np.ndarray]  # FW_Purch_Sell_Price!O per off-taker

    def frame(self, index) -> pd.DataFrame:
        out = pd.DataFrame(index=index)
        for name, blk in (("pv", self.pv), ("bl", self.baseload)):
            for k, v in blk.__dict__.items():
                out[f"{name}_{k}"] = v
        for code, v in self.strips.items():
            out[f"{code}_strip_mwh"] = v
        for code, v in self.bl_price_budget.items():
            out[f"{code}_bl_price_budget"] = v
        for code, v in self.bl_price_forecast.items():
            out[f"{code}_bl_price_forecast"] = v
        return out


def build_sources(series: pd.DataFrame, grid: pd.DataFrame, prices: pd.DataFrame, params: Parameters, pv_code: str = "PV1") -> SourceResult:
    cp_pv = params.counterparties["pv"]
    cp_bl = params.counterparties["baseload"]
    sur = prices["surplus_price"].to_numpy(dtype=float)
    dfc = prices["deficit_price"].to_numpy(dtype=float)

    pv = imbalance.generation_block(
        series[f"{pv_code}_forecast_generation_uncurtailed_MWh"].to_numpy(dtype=float),
        series[f"{pv_code}_forecast_generation_deviation_pct"].to_numpy(dtype=float),
        series[f"{pv_code}_imbalance_deviation_pct"].to_numpy(dtype=float),
        sur,
        dfc,
        k=cp_pv.k if cp_pv.k is not None else imbalance.K_DSO,
    )
    strips = {o.code: strip_mwh_per_qh(o, grid) for o in params.offtakers}
    total_strip = np.sum(np.vstack(list(strips.values())), axis=0) if strips else np.zeros(len(grid))
    n = len(grid)
    bl = imbalance.generation_block(
        total_strip,
        np.full(n, float(cp_bl.deviation_pct)),
        np.full(n, float(cp_bl.imbalance_deviation_pct)),
        sur,
        dfc,
        k=cp_bl.k if cp_bl.k is not None else imbalance.K_DSO,
    )
    return SourceResult(
        pv=pv,
        baseload=bl,
        strips=strips,
        bl_price_budget={o.code: product_price_per_qh(o, grid, "budget") for o in params.offtakers},
        bl_price_forecast={o.code: product_price_per_qh(o, grid, "forecast") for o in params.offtakers},
    )

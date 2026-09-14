"""esb.engine - one run of the whole model in the fixed evaluation order (ruling G0-D6).

    grid -> quarter-hourly engine (scenarios, sources, imbalance, merit order)
         -> monthly P&L stage 1 (volumes, margins, guarantees)
         -> monthly cash flow stage A (settlements, VAT, injection, interest)
         -> monthly P&L stage 2 (interest, net margins, tax, legs)
         -> monthly cash flow stage B (tax lines) -> daily ledger
         -> pricing calculator (every off-taker) -> overview and reconciliation

There is no iteration: every stage reads only stages before it. `RunResult.trace` records the
order and timing of the stages (the calculation-order trace of ruling G0-D6).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from config.schema import Parameters, load_parameters
from esb import grid
from esb.cashflow import CashflowResult, build_cashflow, build_daily_ledger, finalize_cashflow
from esb.merit_order import QHResult, run_qh
from esb.pnl import PnLResult, build_pnl, finalize_pnl
from esb.pricing import PricingResult, price_offtaker
from esb.reporting import OverviewResult, build_overview


@dataclass
class RunResult:
    params: Parameters
    grid: pd.DataFrame
    qh: QHResult
    pnl: PnLResult
    cashflow: CashflowResult
    pricing: dict[str, PricingResult]
    overview: OverviewResult
    selected_offtaker: str
    trace: list[tuple[str, float]] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        P = self.pnl.portfolio
        return {
            "retail_revenue": P.y("revenue"),
            "resell_revenue": P.y("rs_revenue"),
            "total_gm2_budget": P.y("t_gm2_budget"),
            "total_gm2_forecast": P.y("t_gm2_forecast"),
            "nm_budget": P.y("nm_budget"),
            "nm_forecast": P.y("nm_forecast"),
            "cit_budget": P.y("cit_budget"),
            "cit_forecast": P.y("cit_forecast"),
            "guarantees_outstanding_max": P.y("guarantees_outstanding"),
            "interest": P.y("interest"),
            "peak_funding_monthly": self.cashflow.y("peak_funding"),
            "peak_funding_daily": self.cashflow.daily_summary.get("peak_funding", float("nan")),
        }


def run(series: pd.DataFrame, params: Parameters | None = None, selected_offtaker: str | None = None,
        pricing_as_cached: bool = False) -> RunResult:
    """Run the model on standardised input series (columns as in data/reference/rc_v03_series.parquet)."""
    params = params or load_parameters()
    trace: list[tuple[str, float]] = []
    t0 = time.perf_counter()

    def mark(name: str) -> None:
        trace.append((name, round(time.perf_counter() - t0, 3)))

    g = grid.make_grid(params.spine_year)
    if len(series) != len(g):
        raise ValueError(f"series has {len(series)} rows; the {params.spine_year} grid has {len(g)}")
    mark("grid")
    qh = run_qh(series, g, params)
    mark("qh_engine")
    pnl = build_pnl(qh, params)
    mark("pnl_stage1")
    cf = build_cashflow(pnl, params)
    mark("cashflow_stage_a")
    pnl = finalize_pnl(pnl, cf.m("interest"), params)
    mark("pnl_stage2")
    cf = finalize_cashflow(cf, pnl, params)
    cf = build_daily_ledger(cf, pnl, qh, params)
    mark("cashflow_stage_b_daily")
    pricing = {o.code: price_offtaker(pnl, params, o.code, as_cached=pricing_as_cached) for o in params.offtakers}
    sel = selected_offtaker or (params.offtakers[0].code if params.offtakers else "")
    mark("pricing")
    overview = build_overview(qh, pnl, cf, pricing[sel], params)
    mark("overview")
    return RunResult(params=params, grid=g, qh=qh, pnl=pnl, cashflow=cf, pricing=pricing, overview=overview, selected_offtaker=sel, trace=trace)

"""
Supply P&L Engine — nextE Energy Supply
=========================================
Per-contract and portfolio-level profitability tracking with budget vs. actuals,
margin decomposition, variance analysis, and monthly/quarterly/annual aggregation.

Architecture:
  - ContractBudget: planned revenue/cost at contract signing (the "budget")
  - ContractActuals: realized revenue/cost from market settlement data
  - ContractPnL: computed margin = revenue - costs, with variance decomposition
  - PortfolioPnL: aggregation across all contracts with overhead allocation

Margin Decomposition (per EUR/MWh):
  Revenue (offer price)
  - Energy procurement cost (PV + forward blend)
  - GC quota cost
  - Balancing / imbalance cost
  - Risk premium consumed (realized losses)
  = Gross margin
  - Corporate overhead allocation
  = Net contribution margin

Author: nextE AI Workstation
Version: 1.0.0
Date: 2026-03-20
Dependencies: pandas, numpy
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# ENUMS
# ============================================================================

class PnLPeriod(Enum):
    """Reporting period granularity."""
    DAILY = "daily"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class VarianceType(Enum):
    """Variance decomposition categories."""
    PRICE = "price"        # Change in procurement cost vs budget
    VOLUME = "volume"      # Change in delivered volume vs budget
    MIX = "mix"            # Change in procurement channel mix
    FX = "fx"              # EUR/RON exchange rate movement
    REGULATORY = "regulatory"  # GC quota, tariff changes
    OTHER = "other"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ContractBudget:
    """
    Budget (planned) figures for a supply contract at signing.

    All monetary values in EUR. Volume in MWh.
    Period: full contract term, broken down monthly.
    """
    contract_id: str
    customer_name: str
    customer_category: str  # "commercial" or "industrial"

    # Contract terms
    contract_start: date
    contract_end: date
    annual_volume_mwh: float

    # Budgeted price components (EUR/MWh)
    offer_price_ex_vat: float          # Customer-facing price
    budgeted_energy_cost: float        # Blended PV + forward procurement
    budgeted_pv_cost: float            # PV procurement component
    budgeted_forward_cost: float       # Forward procurement component
    budgeted_gc_cost: float            # Green Certificate quota
    budgeted_balancing_cost: float     # BRP imbalance reserve
    budgeted_risk_premium: float       # Risk buffer
    budgeted_margin: float             # Target commercial margin

    # Procurement mix (% allocation)
    budgeted_solar_share: float        # 0-1
    budgeted_procurement_mix: Dict[str, float] = field(default_factory=dict)

    # Monthly volume profile (optional: flat if not provided)
    monthly_volume_profile: Optional[Dict[str, float]] = None

    # Budget metadata
    budget_date: date = field(default_factory=date.today)
    budget_version: str = "v1.0"
    approved_by: str = ""
    notes: str = ""

    def monthly_volume_mwh(self, month: date) -> float:
        """Get budgeted volume for a given month."""
        if self.monthly_volume_profile:
            key = month.strftime("%Y-%m")
            return self.monthly_volume_profile.get(key, self.annual_volume_mwh / 12)
        return self.annual_volume_mwh / 12

    def monthly_revenue_eur(self, month: date) -> float:
        """Budgeted monthly revenue."""
        return self.monthly_volume_mwh(month) * self.offer_price_ex_vat

    def monthly_cost_eur(self, month: date) -> float:
        """Budgeted monthly total cost."""
        vol = self.monthly_volume_mwh(month)
        return vol * (self.budgeted_energy_cost + self.budgeted_gc_cost +
                      self.budgeted_balancing_cost)

    def monthly_gross_margin_eur(self, month: date) -> float:
        """Budgeted monthly gross margin."""
        return self.monthly_revenue_eur(month) - self.monthly_cost_eur(month)


@dataclass
class ContractActuals:
    """
    Actual (realized) figures for a supply contract in a given period.
    Populated from market settlement data, metering, and invoicing.
    """
    contract_id: str
    period_start: date
    period_end: date

    # Realized volumes
    delivered_volume_mwh: float        # Actual metered delivery
    pv_volume_mwh: float               # PV generation consumed
    forward_volume_mwh: float          # Forward procurement utilized
    spot_volume_mwh: float             # Spot (DAM/IDM) procurement

    # Realized revenue
    invoiced_revenue_eur: float        # Actual invoiced amount (ex-VAT)
    effective_price_eur_per_mwh: float # Revenue / Volume

    # Realized costs (EUR)
    pv_procurement_cost_eur: float     # PV generation cost
    forward_procurement_cost_eur: float  # Forward contract cost
    spot_procurement_cost_eur: float   # Spot market purchases
    gc_cost_eur: float                 # Green Certificate cost
    balancing_cost_eur: float          # Imbalance settlement cost
    other_costs_eur: float = 0.0       # Admin, legal, etc.

    # Realized costs (EUR/MWh equivalent)
    avg_pv_cost_per_mwh: float = 0.0
    avg_forward_cost_per_mwh: float = 0.0
    avg_spot_cost_per_mwh: float = 0.0
    avg_gc_cost_per_mwh: float = 0.0
    avg_balancing_cost_per_mwh: float = 0.0

    # Settlement metadata
    settlement_date: Optional[date] = None
    settlement_status: str = "provisional"  # "provisional" | "confirmed" | "final"
    data_source: str = ""

    def __post_init__(self):
        """Compute per-MWh averages."""
        vol = max(self.delivered_volume_mwh, 1)
        if self.avg_pv_cost_per_mwh == 0 and self.pv_volume_mwh > 0:
            self.avg_pv_cost_per_mwh = self.pv_procurement_cost_eur / max(self.pv_volume_mwh, 1)
        if self.avg_forward_cost_per_mwh == 0 and self.forward_volume_mwh > 0:
            self.avg_forward_cost_per_mwh = self.forward_procurement_cost_eur / max(self.forward_volume_mwh, 1)
        if self.avg_spot_cost_per_mwh == 0 and self.spot_volume_mwh > 0:
            self.avg_spot_cost_per_mwh = self.spot_procurement_cost_eur / max(self.spot_volume_mwh, 1)
        if self.avg_gc_cost_per_mwh == 0:
            self.avg_gc_cost_per_mwh = self.gc_cost_eur / vol
        if self.avg_balancing_cost_per_mwh == 0:
            self.avg_balancing_cost_per_mwh = self.balancing_cost_eur / vol

    @property
    def total_procurement_cost_eur(self) -> float:
        return (self.pv_procurement_cost_eur + self.forward_procurement_cost_eur +
                self.spot_procurement_cost_eur)

    @property
    def total_cost_eur(self) -> float:
        return (self.total_procurement_cost_eur + self.gc_cost_eur +
                self.balancing_cost_eur + self.other_costs_eur)

    @property
    def gross_margin_eur(self) -> float:
        return self.invoiced_revenue_eur - self.total_cost_eur

    @property
    def gross_margin_per_mwh(self) -> float:
        return self.gross_margin_eur / max(self.delivered_volume_mwh, 1)

    @property
    def total_cost_per_mwh(self) -> float:
        return self.total_cost_eur / max(self.delivered_volume_mwh, 1)


@dataclass
class ContractPnL:
    """
    Computed P&L for a single contract in a given period,
    including budget vs actual variance decomposition.
    """
    contract_id: str
    customer_name: str
    period_start: date
    period_end: date
    period_type: PnLPeriod

    # Budget figures
    budget_volume_mwh: float
    budget_revenue_eur: float
    budget_cost_eur: float
    budget_margin_eur: float
    budget_margin_per_mwh: float

    # Actual figures
    actual_volume_mwh: float
    actual_revenue_eur: float
    actual_cost_eur: float
    actual_margin_eur: float
    actual_margin_per_mwh: float

    # Variances (actual - budget)
    volume_variance_mwh: float         # Volume delivered vs plan
    revenue_variance_eur: float        # Revenue vs plan
    cost_variance_eur: float           # Total cost vs plan
    margin_variance_eur: float         # Margin vs plan
    margin_variance_per_mwh: float     # Margin/MWh vs plan

    # Variance decomposition
    price_variance_eur: float = 0.0     # Due to procurement cost change
    volume_effect_eur: float = 0.0      # Due to volume change
    mix_variance_eur: float = 0.0       # Due to channel mix change
    gc_variance_eur: float = 0.0        # Due to GC cost change
    balancing_variance_eur: float = 0.0 # Due to imbalance change
    other_variance_eur: float = 0.0     # Residual / unexplained

    # Performance indicators
    margin_floor_status: str = ""       # "ABOVE" / "BELOW" min floor
    budget_achievement_pct: float = 0.0 # actual margin / budget margin

    # Metadata
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    notes: str = ""


@dataclass
class PortfolioPnL:
    """
    Portfolio-level aggregation of all contract P&Ls.
    """
    period_start: date
    period_end: date
    period_type: PnLPeriod

    # Aggregated figures
    total_contracts: int
    active_contracts: int
    total_volume_mwh: float
    total_revenue_eur: float
    total_cost_eur: float
    total_margin_eur: float
    avg_margin_per_mwh: float

    # Budget aggregates
    budget_volume_mwh: float
    budget_revenue_eur: float
    budget_cost_eur: float
    budget_margin_eur: float

    # Portfolio variances
    portfolio_margin_variance_eur: float
    portfolio_volume_variance_mwh: float
    portfolio_budget_achievement_pct: float

    # Overhead allocation
    corporate_overhead_eur: float = 0.0
    net_contribution_margin_eur: float = 0.0

    # Per-contract breakdown
    contract_pnls: List[ContractPnL] = field(default_factory=list)

    # Risk-adjusted metrics
    raroc_pct: float = 0.0              # Risk-adjusted return on capital
    capital_deployed_eur: float = 0.0   # Collateral + working capital

    # Metadata
    calculation_timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# P&L CALCULATION FUNCTIONS
# ============================================================================

def compute_contract_pnl(
    budget: ContractBudget,
    actuals: ContractActuals,
    period_type: PnLPeriod = PnLPeriod.MONTHLY,
    min_margin_floor: float = 8.0,
) -> ContractPnL:
    """
    Compute P&L for a single contract by comparing budget to actuals.

    Args:
        budget: ContractBudget with planned figures
        actuals: ContractActuals with realized figures
        period_type: Reporting period granularity
        min_margin_floor: Minimum acceptable margin (EUR/MWh)

    Returns:
        ContractPnL with full variance decomposition
    """
    # Budget figures for the actuals period
    b_vol = budget.monthly_volume_mwh(actuals.period_start)
    b_rev = budget.monthly_revenue_eur(actuals.period_start)
    b_cost = budget.monthly_cost_eur(actuals.period_start)
    b_margin = b_rev - b_cost
    b_margin_per_mwh = b_margin / max(b_vol, 1)

    # Actual figures
    a_vol = actuals.delivered_volume_mwh
    a_rev = actuals.invoiced_revenue_eur
    a_cost = actuals.total_cost_eur
    a_margin = actuals.gross_margin_eur
    a_margin_per_mwh = actuals.gross_margin_per_mwh

    # Variances
    vol_var = a_vol - b_vol
    rev_var = a_rev - b_rev
    cost_var = a_cost - b_cost
    margin_var = a_margin - b_margin
    margin_var_per_mwh = a_margin_per_mwh - b_margin_per_mwh

    # Variance decomposition
    # Price variance: (actual cost/MWh - budget cost/MWh) * actual volume
    b_cost_per_mwh = b_cost / max(b_vol, 1)
    a_cost_per_mwh = actuals.total_cost_per_mwh
    price_var = -(a_cost_per_mwh - b_cost_per_mwh) * a_vol  # negative cost increase = negative margin

    # Volume effect: (actual vol - budget vol) * budget margin/MWh
    volume_eff = vol_var * b_margin_per_mwh

    # GC variance: (actual GC cost - budget GC cost)
    b_gc_cost = b_vol * budget.budgeted_gc_cost
    gc_var = -(actuals.gc_cost_eur - b_gc_cost)

    # Balancing variance
    b_bal_cost = b_vol * budget.budgeted_balancing_cost
    bal_var = -(actuals.balancing_cost_eur - b_bal_cost)

    # Mix variance: residual attributed to procurement channel mix
    # (total variance - price - volume - gc - balancing)
    mix_var = margin_var - price_var - volume_eff - gc_var - bal_var
    # If mix_var is very small, attribute to "other"
    if abs(mix_var) < 0.01 * abs(margin_var) if margin_var != 0 else True:
        other_var = mix_var
        mix_var = 0.0
    else:
        other_var = 0.0

    # Margin floor status
    floor_status = "ABOVE" if a_margin_per_mwh >= min_margin_floor else "BELOW"

    # Budget achievement
    budget_ach = (a_margin / b_margin * 100) if b_margin != 0 else 0.0

    return ContractPnL(
        contract_id=budget.contract_id,
        customer_name=budget.customer_name,
        period_start=actuals.period_start,
        period_end=actuals.period_end,
        period_type=period_type,
        budget_volume_mwh=b_vol,
        budget_revenue_eur=b_rev,
        budget_cost_eur=b_cost,
        budget_margin_eur=b_margin,
        budget_margin_per_mwh=b_margin_per_mwh,
        actual_volume_mwh=a_vol,
        actual_revenue_eur=a_rev,
        actual_cost_eur=a_cost,
        actual_margin_eur=a_margin,
        actual_margin_per_mwh=a_margin_per_mwh,
        volume_variance_mwh=vol_var,
        revenue_variance_eur=rev_var,
        cost_variance_eur=cost_var,
        margin_variance_eur=margin_var,
        margin_variance_per_mwh=margin_var_per_mwh,
        price_variance_eur=price_var,
        volume_effect_eur=volume_eff,
        mix_variance_eur=mix_var,
        gc_variance_eur=gc_var,
        balancing_variance_eur=bal_var,
        other_variance_eur=other_var,
        margin_floor_status=floor_status,
        budget_achievement_pct=budget_ach,
    )


def compute_portfolio_pnl(
    contract_pnls: List[ContractPnL],
    period_start: date,
    period_end: date,
    period_type: PnLPeriod = PnLPeriod.MONTHLY,
    corporate_overhead_eur: float = 0.0,
    capital_deployed_eur: float = 0.0,
) -> PortfolioPnL:
    """
    Aggregate individual contract P&Ls into portfolio-level metrics.

    Args:
        contract_pnls: List of ContractPnL for the period
        period_start: Period start date
        period_end: Period end date
        period_type: Reporting granularity
        corporate_overhead_eur: Allocated overhead for the period
        capital_deployed_eur: Total capital deployed (for RAROC)

    Returns:
        PortfolioPnL with full aggregation
    """
    n = len(contract_pnls)
    if n == 0:
        return PortfolioPnL(
            period_start=period_start, period_end=period_end, period_type=period_type,
            total_contracts=0, active_contracts=0, total_volume_mwh=0,
            total_revenue_eur=0, total_cost_eur=0, total_margin_eur=0,
            avg_margin_per_mwh=0, budget_volume_mwh=0, budget_revenue_eur=0,
            budget_cost_eur=0, budget_margin_eur=0,
            portfolio_margin_variance_eur=0, portfolio_volume_variance_mwh=0,
            portfolio_budget_achievement_pct=0,
        )

    # Aggregate
    total_vol = sum(p.actual_volume_mwh for p in contract_pnls)
    total_rev = sum(p.actual_revenue_eur for p in contract_pnls)
    total_cost = sum(p.actual_cost_eur for p in contract_pnls)
    total_margin = sum(p.actual_margin_eur for p in contract_pnls)
    avg_margin = total_margin / max(total_vol, 1)

    b_vol = sum(p.budget_volume_mwh for p in contract_pnls)
    b_rev = sum(p.budget_revenue_eur for p in contract_pnls)
    b_cost = sum(p.budget_cost_eur for p in contract_pnls)
    b_margin = sum(p.budget_margin_eur for p in contract_pnls)

    margin_var = total_margin - b_margin
    vol_var = total_vol - b_vol
    budget_ach = (total_margin / b_margin * 100) if b_margin != 0 else 0.0

    net_margin = total_margin - corporate_overhead_eur
    raroc = (net_margin / capital_deployed_eur * 100) if capital_deployed_eur > 0 else 0.0

    active = sum(1 for p in contract_pnls if p.actual_volume_mwh > 0)

    return PortfolioPnL(
        period_start=period_start,
        period_end=period_end,
        period_type=period_type,
        total_contracts=n,
        active_contracts=active,
        total_volume_mwh=total_vol,
        total_revenue_eur=total_rev,
        total_cost_eur=total_cost,
        total_margin_eur=total_margin,
        avg_margin_per_mwh=avg_margin,
        budget_volume_mwh=b_vol,
        budget_revenue_eur=b_rev,
        budget_cost_eur=b_cost,
        budget_margin_eur=b_margin,
        portfolio_margin_variance_eur=margin_var,
        portfolio_volume_variance_mwh=vol_var,
        portfolio_budget_achievement_pct=budget_ach,
        corporate_overhead_eur=corporate_overhead_eur,
        net_contribution_margin_eur=net_margin,
        contract_pnls=contract_pnls,
        raroc_pct=raroc,
        capital_deployed_eur=capital_deployed_eur,
    )


# ============================================================================
# VARIANCE ANALYSIS FUNCTIONS
# ============================================================================

def decompose_margin_variance(
    pnl: ContractPnL,
) -> pd.DataFrame:
    """
    Create a structured variance waterfall DataFrame for a single contract.

    Returns DataFrame with columns: [component, budget, actual, variance, pct_of_total]
    """
    rows = [
        {"component": "Revenue", "budget": pnl.budget_revenue_eur,
         "actual": pnl.actual_revenue_eur, "variance": pnl.revenue_variance_eur},
        {"component": "Energy Procurement", "budget": -(pnl.budget_cost_eur - pnl.budget_volume_mwh * 0),
         "actual": -pnl.actual_cost_eur, "variance": -pnl.cost_variance_eur},
        {"component": "Gross Margin", "budget": pnl.budget_margin_eur,
         "actual": pnl.actual_margin_eur, "variance": pnl.margin_variance_eur},
    ]
    df = pd.DataFrame(rows)

    total_var = abs(pnl.margin_variance_eur) if pnl.margin_variance_eur != 0 else 1
    df["pct_of_total"] = df["variance"].abs() / total_var * 100

    return df


def build_variance_bridge(
    pnl: ContractPnL,
) -> pd.DataFrame:
    """
    Build a variance bridge from budget margin to actual margin.

    Returns DataFrame suitable for waterfall chart:
      Budget Margin → Price Effect → Volume Effect → GC Effect →
      Balancing Effect → Mix Effect → Other → Actual Margin
    """
    bridge = [
        {"item": "Budget Margin", "value": pnl.budget_margin_eur, "type": "absolute"},
        {"item": "Price Effect", "value": pnl.price_variance_eur, "type": "relative"},
        {"item": "Volume Effect", "value": pnl.volume_effect_eur, "type": "relative"},
        {"item": "GC Quota Effect", "value": pnl.gc_variance_eur, "type": "relative"},
        {"item": "Balancing Effect", "value": pnl.balancing_variance_eur, "type": "relative"},
        {"item": "Mix Effect", "value": pnl.mix_variance_eur, "type": "relative"},
        {"item": "Other", "value": pnl.other_variance_eur, "type": "relative"},
        {"item": "Actual Margin", "value": pnl.actual_margin_eur, "type": "total"},
    ]
    return pd.DataFrame(bridge)


def generate_monthly_pnl_series(
    budgets: List[ContractBudget],
    actuals_by_month: Dict[str, List[ContractActuals]],
    start_date: date,
    end_date: date,
    min_margin_floor: float = 8.0,
    corporate_overhead_monthly_eur: float = 0.0,
) -> pd.DataFrame:
    """
    Generate a monthly P&L time series across the portfolio.

    Args:
        budgets: List of contract budgets
        actuals_by_month: Dict mapping "YYYY-MM" to list of ContractActuals
        start_date: First month (YYYY-MM-01)
        end_date: Last month (YYYY-MM-01)
        min_margin_floor: Minimum margin threshold (EUR/MWh)
        corporate_overhead_monthly_eur: Monthly overhead allocation

    Returns:
        DataFrame indexed by month with columns:
          budget_volume, actual_volume, budget_revenue, actual_revenue,
          budget_cost, actual_cost, budget_margin, actual_margin,
          margin_variance, budget_achievement_pct, active_contracts
    """
    months = pd.date_range(start=start_date, end=end_date, freq="MS")
    records = []

    budget_map = {b.contract_id: b for b in budgets}

    for month in months:
        month_key = month.strftime("%Y-%m")
        month_actuals = actuals_by_month.get(month_key, [])

        contract_pnls = []
        for actual in month_actuals:
            budget = budget_map.get(actual.contract_id)
            if budget is None:
                logger.warning(f"No budget found for contract {actual.contract_id}, skipping")
                continue
            pnl = compute_contract_pnl(budget, actual, PnLPeriod.MONTHLY, min_margin_floor)
            contract_pnls.append(pnl)

        portfolio = compute_portfolio_pnl(
            contract_pnls,
            period_start=month.date(),
            period_end=(month + pd.DateOffset(months=1) - pd.Timedelta(days=1)).date(),
            period_type=PnLPeriod.MONTHLY,
            corporate_overhead_eur=corporate_overhead_monthly_eur,
        )

        records.append({
            "month": month,
            "active_contracts": portfolio.active_contracts,
            "budget_volume_mwh": portfolio.budget_volume_mwh,
            "actual_volume_mwh": portfolio.total_volume_mwh,
            "volume_variance_mwh": portfolio.portfolio_volume_variance_mwh,
            "budget_revenue_eur": portfolio.budget_revenue_eur,
            "actual_revenue_eur": portfolio.total_revenue_eur,
            "budget_cost_eur": portfolio.budget_cost_eur,
            "actual_cost_eur": portfolio.total_cost_eur,
            "budget_margin_eur": portfolio.budget_margin_eur,
            "actual_margin_eur": portfolio.total_margin_eur,
            "margin_variance_eur": portfolio.portfolio_margin_variance_eur,
            "budget_achievement_pct": portfolio.portfolio_budget_achievement_pct,
            "avg_margin_per_mwh": portfolio.avg_margin_per_mwh,
            "overhead_eur": corporate_overhead_monthly_eur,
            "net_contribution_eur": portfolio.net_contribution_margin_eur,
        })

    return pd.DataFrame(records).set_index("month")


# ============================================================================
# DEMO DATA GENERATOR (for Streamlit development)
# ============================================================================

def generate_demo_pnl_data(
    n_contracts: int = 5,
    n_months: int = 12,
    base_dam_price: float = 85.0,
) -> Tuple[List[ContractBudget], Dict[str, List[ContractActuals]], List[ContractPnL]]:
    """
    Generate realistic demo P&L data for Streamlit dashboard development.

    Returns:
        Tuple of (budgets, actuals_by_month, all_pnls)
    """
    np.random.seed(42)

    # Define demo contracts
    contracts = [
        {"id": "SC-2026-001", "name": "CUST-ALPHA", "cat": "industrial",
         "vol": 35000, "price": 108.50, "solar": 0.30, "margin": 12.0},
        {"id": "SC-2026-002", "name": "CUST-BETA", "cat": "commercial",
         "vol": 12000, "price": 115.20, "solar": 0.40, "margin": 14.5},
        {"id": "SC-2026-003", "name": "CUST-GAMMA", "cat": "industrial",
         "vol": 55000, "price": 102.80, "solar": 0.20, "margin": 10.0},
        {"id": "SC-2026-004", "name": "CUST-DELTA", "cat": "commercial",
         "vol": 8000, "price": 122.00, "solar": 0.45, "margin": 16.0},
        {"id": "SC-2026-005", "name": "CUST-EPSILON", "cat": "industrial",
         "vol": 25000, "price": 106.30, "solar": 0.35, "margin": 11.5},
    ][:n_contracts]

    budgets = []
    all_pnls = []
    actuals_by_month = {}

    start = date(2026, 4, 1)

    for c in contracts:
        energy_cost = c["price"] - c["margin"] - 14.50 - 3.0  # price - margin - gc - balancing
        pv_cost = energy_cost * c["solar"]
        fwd_cost = energy_cost * (1 - c["solar"])

        budget = ContractBudget(
            contract_id=c["id"],
            customer_name=c["name"],
            customer_category=c["cat"],
            contract_start=start,
            contract_end=date(2027, 3, 31),
            annual_volume_mwh=c["vol"],
            offer_price_ex_vat=c["price"],
            budgeted_energy_cost=energy_cost,
            budgeted_pv_cost=pv_cost / max(c["solar"], 0.01),  # per-MWh for PV portion
            budgeted_forward_cost=fwd_cost / max(1 - c["solar"], 0.01),  # per-MWh for fwd portion
            budgeted_gc_cost=14.50,
            budgeted_balancing_cost=3.0,
            budgeted_risk_premium=5.0,
            budgeted_margin=c["margin"],
            budgeted_solar_share=c["solar"],
        )
        budgets.append(budget)

    # Generate monthly actuals with realistic variance
    for month_idx in range(n_months):
        month = date(2026, 4 + month_idx if 4 + month_idx <= 12 else month_idx + 4 - 12,
                     1)
        if 4 + month_idx > 12:
            month = date(2027, month_idx + 4 - 12, 1)

        month_key = month.strftime("%Y-%m")
        month_actuals_list = []

        # Seasonal DAM price variation
        seasonal_factor = 1.0
        if month.month in (6, 7, 8):
            seasonal_factor = 0.85  # Summer: lower prices, more PV
        elif month.month in (12, 1, 2):
            seasonal_factor = 1.15  # Winter: higher prices

        dam_actual = base_dam_price * seasonal_factor * (1 + np.random.normal(0, 0.06))

        for budget in budgets:
            b_vol = budget.annual_volume_mwh / 12
            # Volume variance: ±8%
            vol_actual = b_vol * (1 + np.random.normal(0, 0.08))

            # PV volume varies with season
            pv_share_actual = budget.budgeted_solar_share
            if month.month in (5, 6, 7, 8):
                pv_share_actual *= 1.15  # More sun
            elif month.month in (11, 12, 1, 2):
                pv_share_actual *= 0.70  # Less sun
            pv_share_actual = min(pv_share_actual, 0.65)

            pv_vol = vol_actual * pv_share_actual
            fwd_vol = vol_actual * 0.60 * (1 - pv_share_actual)
            spot_vol = vol_actual - pv_vol - fwd_vol

            # Costs with variance
            pv_cost_rate = budget.budgeted_pv_cost * (1 + np.random.normal(0, 0.03))
            fwd_cost_rate = dam_actual * (1 + np.random.normal(0.01, 0.02))
            spot_cost_rate = dam_actual * (1 + np.random.normal(0.02, 0.04))
            gc_rate = 14.50 * (1 + np.random.normal(0, 0.02))
            bal_rate = 3.0 * (1 + np.random.normal(0.1, 0.3))

            actual = ContractActuals(
                contract_id=budget.contract_id,
                period_start=month,
                period_end=date(month.year, month.month, 28),
                delivered_volume_mwh=round(vol_actual, 1),
                pv_volume_mwh=round(pv_vol, 1),
                forward_volume_mwh=round(fwd_vol, 1),
                spot_volume_mwh=round(spot_vol, 1),
                invoiced_revenue_eur=round(vol_actual * budget.offer_price_ex_vat, 2),
                effective_price_eur_per_mwh=budget.offer_price_ex_vat,
                pv_procurement_cost_eur=round(pv_vol * pv_cost_rate, 2),
                forward_procurement_cost_eur=round(fwd_vol * fwd_cost_rate, 2),
                spot_procurement_cost_eur=round(spot_vol * spot_cost_rate, 2),
                gc_cost_eur=round(vol_actual * gc_rate, 2),
                balancing_cost_eur=round(vol_actual * bal_rate, 2),
                settlement_status="final" if month_idx < n_months - 2 else "provisional",
            )
            month_actuals_list.append(actual)

            # Compute P&L
            pnl = compute_contract_pnl(budget, actual)
            all_pnls.append(pnl)

        actuals_by_month[month_key] = month_actuals_list

    return budgets, actuals_by_month, all_pnls


# ============================================================================
# REPORTING UTILITIES
# ============================================================================

def pnl_to_dataframe(pnls: List[ContractPnL]) -> pd.DataFrame:
    """Convert list of ContractPnL to a flat DataFrame for reporting."""
    records = []
    for p in pnls:
        records.append({
            "contract_id": p.contract_id,
            "customer": p.customer_name,
            "period": p.period_start.strftime("%Y-%m"),
            "budget_vol": p.budget_volume_mwh,
            "actual_vol": p.actual_volume_mwh,
            "vol_var": p.volume_variance_mwh,
            "budget_rev": p.budget_revenue_eur,
            "actual_rev": p.actual_revenue_eur,
            "rev_var": p.revenue_variance_eur,
            "budget_cost": p.budget_cost_eur,
            "actual_cost": p.actual_cost_eur,
            "cost_var": p.cost_variance_eur,
            "budget_margin": p.budget_margin_eur,
            "actual_margin": p.actual_margin_eur,
            "margin_var": p.margin_variance_eur,
            "margin_per_mwh": p.actual_margin_per_mwh,
            "budget_ach_pct": p.budget_achievement_pct,
            "price_var": p.price_variance_eur,
            "volume_eff": p.volume_effect_eur,
            "gc_var": p.gc_variance_eur,
            "bal_var": p.balancing_variance_eur,
            "mix_var": p.mix_variance_eur,
            "floor_status": p.margin_floor_status,
        })
    return pd.DataFrame(records)


def portfolio_summary_table(portfolio: PortfolioPnL) -> pd.DataFrame:
    """Create a summary table for portfolio-level reporting."""
    rows = [
        {"Metric": "Active Contracts", "Value": f"{portfolio.active_contracts}"},
        {"Metric": "Total Volume (MWh)", "Value": f"{portfolio.total_volume_mwh:,.0f}"},
        {"Metric": "Total Revenue (EUR)", "Value": f"€{portfolio.total_revenue_eur:,.0f}"},
        {"Metric": "Total Cost (EUR)", "Value": f"€{portfolio.total_cost_eur:,.0f}"},
        {"Metric": "Gross Margin (EUR)", "Value": f"€{portfolio.total_margin_eur:,.0f}"},
        {"Metric": "Avg Margin (EUR/MWh)", "Value": f"€{portfolio.avg_margin_per_mwh:.2f}"},
        {"Metric": "Budget Margin (EUR)", "Value": f"€{portfolio.budget_margin_eur:,.0f}"},
        {"Metric": "Margin Variance (EUR)", "Value": f"€{portfolio.portfolio_margin_variance_eur:+,.0f}"},
        {"Metric": "Budget Achievement", "Value": f"{portfolio.portfolio_budget_achievement_pct:.1f}%"},
        {"Metric": "Overhead (EUR)", "Value": f"€{portfolio.corporate_overhead_eur:,.0f}"},
        {"Metric": "Net Contribution (EUR)", "Value": f"€{portfolio.net_contribution_margin_eur:,.0f}"},
    ]
    if portfolio.capital_deployed_eur > 0:
        rows.append({"Metric": "RAROC", "Value": f"{portfolio.raroc_pct:.1f}%"})
    return pd.DataFrame(rows).set_index("Metric")

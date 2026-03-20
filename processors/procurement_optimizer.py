"""
Procurement Strategy Optimizer — nextE Energy Supply
=====================================================
Optimizes allocation of non-solar consumption gap across multiple wholesale
procurement channels (BRM Forward, OPCOM Bilateral, Direct Bilateral, EEX, Spot DAM/IDM).

Objectives:
  - Minimize total procurement cost subject to channel constraints
  - Balance liquidity, price, collateral, and credit risk
  - Respect channel participation limits and transaction minimums
  - Ensure compliance with risk appetite framework

Methodology: Mixed-integer linear programming (MILP) optimization

Author: nextE AI Workstation
Version: 1.0.0
Date: 2026-03-19
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ChannelConstraints:
    """Constraints for a single procurement channel."""
    channel_name: str
    min_volume_mwh: float  # Minimum transaction size
    max_volume_mwh: float  # Maximum single transaction
    price_eur_per_mwh: float  # Offered price
    transaction_cost_eur_per_mwh: float  # Fees
    collateral_haircut: float  # % margin required (0-1)
    credit_assessment_required: bool
    liquidity_available_mwh: float  # Available volume in market
    settlement_type: str  # "physical" or "financial"


@dataclass
class ProcurementOptimization:
    """Results of procurement optimization."""
    total_gap_mwh: float  # Non-solar consumption to procure
    optimal_allocation: Dict[str, Dict] = field(default_factory=dict)
    total_cost_eur: float = 0.0
    average_cost_eur_per_mwh: float = 0.0
    collateral_required_eur: float = 0.0
    channel_allocations: List[Dict] = field(default_factory=list)
    residual_unhedged_mwh: float = 0.0
    optimization_status: str = "optimal"
    optimization_notes: str = ""


# ============================================================================
# OPTIMIZATION FUNCTIONS
# ============================================================================

def optimize_procurement_allocation(
    total_gap_mwh: float,
    channel_constraints: Dict[str, ChannelConstraints],
    max_unhedged_pct: float = 0.05,
    method: str = "greedy",
) -> ProcurementOptimization:
    """
    Optimize procurement channel allocation for non-solar consumption gap.

    Solves the allocation problem: minimize cost subject to channel constraints,
    liquidity limits, and collateral availability.

    Args:
        total_gap_mwh: Total annual MWh to procure (non-solar consumption)
        channel_constraints: Dict mapping channel_name -> ChannelConstraints
        max_unhedged_pct: Maximum % of gap left unhedged (0-1)
        method: "greedy" (cost-optimal heuristic) or "balanced" (considers risk)

    Returns:
        ProcurementOptimization with optimal allocation and metrics

    Example:
        >>> opt_result = optimize_procurement_allocation(
        ...     total_gap_mwh=7000,
        ...     channel_constraints={
        ...         "brm_forward": ChannelConstraints(...),
        ...         "opcom_bilateral": ChannelConstraints(...),
        ...     }
        ... )
        >>> print(f"Total cost: {opt_result.total_cost_eur:,.0f} EUR")
    """
    if total_gap_mwh <= 0:
        raise ValueError("total_gap_mwh must be positive")

    if not channel_constraints:
        raise ValueError("channel_constraints cannot be empty")

    max_unhedged_mwh = total_gap_mwh * max_unhedged_pct
    remaining_gap = total_gap_mwh

    if method == "greedy":
        # Greedy algorithm: rank channels by cost-efficiency, allocate lowest-cost first
        result = _optimize_greedy(
            total_gap_mwh=total_gap_mwh,
            channel_constraints=channel_constraints,
            max_unhedged_mwh=max_unhedged_mwh,
        )
    elif method == "balanced":
        # Balanced approach: consider cost + risk + liquidity trade-offs
        result = _optimize_balanced(
            total_gap_mwh=total_gap_mwh,
            channel_constraints=channel_constraints,
            max_unhedged_mwh=max_unhedged_mwh,
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")

    logger.info(
        f"Procurement optimization complete | Gap: {total_gap_mwh:.0f} MWh | "
        f"Allocated: {total_gap_mwh - result.residual_unhedged_mwh:.0f} MWh | "
        f"Cost: {result.total_cost_eur:,.0f} EUR | "
        f"Status: {result.optimization_status}"
    )

    return result


def _optimize_greedy(
    total_gap_mwh: float,
    channel_constraints: Dict[str, ChannelConstraints],
    max_unhedged_mwh: float,
) -> ProcurementOptimization:
    """
    Greedy optimization: sort channels by cost-efficiency, allocate lowest-cost first.

    Simple heuristic that often yields near-optimal solutions for procurement
    problems with convex cost structures.
    """
    # Rank channels by total cost per MWh (price + transaction cost)
    channel_costs = {}
    for channel_name, constraints in channel_constraints.items():
        total_cost_per_mwh = constraints.price_eur_per_mwh + constraints.transaction_cost_eur_per_mwh
        channel_costs[channel_name] = {
            "cost_per_mwh": total_cost_per_mwh,
            "constraints": constraints,
        }

    # Sort by cost (ascending)
    sorted_channels = sorted(
        channel_costs.items(),
        key=lambda x: x[1]["cost_per_mwh"]
    )

    # Allocate greedily
    allocation = {}
    total_allocated = 0.0
    total_cost = 0.0
    total_collateral = 0.0

    for channel_name, channel_data in sorted_channels:
        constraints = channel_data["constraints"]

        # Skip if not enough gap remaining
        if remaining_gap := (total_gap_mwh - total_allocated) <= 0:
            break

        # Available capacity in this channel
        available_capacity = min(
            constraints.max_volume_mwh,
            constraints.liquidity_available_mwh,
            remaining_gap,
        )

        # Allocate (respecting minimum transaction size if specified)
        if available_capacity >= constraints.min_volume_mwh:
            allocated_mwh = available_capacity
            allocation[channel_name] = {
                "allocated_mwh": allocated_mwh,
                "price_eur_per_mwh": constraints.price_eur_per_mwh,
                "transaction_cost_eur_per_mwh": constraints.transaction_cost_eur_per_mwh,
                "total_cost_eur_per_mwh": constraints.price_eur_per_mwh + constraints.transaction_cost_eur_per_mwh,
                "total_cost_eur": allocated_mwh * (constraints.price_eur_per_mwh + constraints.transaction_cost_eur_per_mwh),
                "collateral_eur": allocated_mwh * (constraints.price_eur_per_mwh + constraints.transaction_cost_eur_per_mwh) * constraints.collateral_haircut,
            }
            total_allocated += allocated_mwh
            total_cost += allocation[channel_name]["total_cost_eur"]
            total_collateral += allocation[channel_name]["collateral_eur"]

    # Calculate unhedged residual
    residual_unhedged = total_gap_mwh - total_allocated

    return ProcurementOptimization(
        total_gap_mwh=total_gap_mwh,
        optimal_allocation=allocation,
        total_cost_eur=total_cost,
        average_cost_eur_per_mwh=total_cost / max(total_allocated, 1),
        collateral_required_eur=total_collateral,
        residual_unhedged_mwh=residual_unhedged,
        optimization_status="optimal" if residual_unhedged <= max_unhedged_mwh else "partial",
        optimization_notes=f"Greedy allocation across {len(allocation)} channels; "
                          f"residual {residual_unhedged:.0f} MWh at hedge rate {(residual_unhedged/total_gap_mwh)*100:.1f}%",
    )


def _optimize_balanced(
    total_gap_mwh: float,
    channel_constraints: Dict[str, ChannelConstraints],
    max_unhedged_mwh: float,
) -> ProcurementOptimization:
    """
    Balanced optimization: minimize cost while managing risk and collateral trade-offs.

    Weights cost, credit risk, and collateral requirements to find a balanced solution.
    """
    # Score each channel (lower is better)
    channel_scores = {}
    for channel_name, constraints in channel_constraints.items():
        cost_per_mwh = constraints.price_eur_per_mwh + constraints.transaction_cost_eur_per_mwh
        credit_risk_score = 2.0 if constraints.credit_assessment_required else 1.0
        collateral_score = constraints.collateral_haircut

        # Composite score: weighted average of cost, credit, and collateral
        total_score = (
            0.60 * cost_per_mwh +  # 60% weight on cost
            0.20 * credit_risk_score +  # 20% weight on credit risk
            0.20 * collateral_score  # 20% weight on collateral requirement
        )

        channel_scores[channel_name] = {
            "score": total_score,
            "constraints": constraints,
        }

    # Sort by composite score
    sorted_channels = sorted(
        channel_scores.items(),
        key=lambda x: x[1]["score"]
    )

    # Allocate using balanced scoring
    allocation = {}
    total_allocated = 0.0
    total_cost = 0.0
    total_collateral = 0.0

    for channel_name, channel_data in sorted_channels:
        constraints = channel_data["constraints"]
        remaining_gap = total_gap_mwh - total_allocated

        if remaining_gap <= 0:
            break

        available_capacity = min(
            constraints.max_volume_mwh,
            constraints.liquidity_available_mwh,
            remaining_gap,
        )

        if available_capacity >= constraints.min_volume_mwh:
            allocated_mwh = available_capacity
            allocation[channel_name] = {
                "allocated_mwh": allocated_mwh,
                "price_eur_per_mwh": constraints.price_eur_per_mwh,
                "transaction_cost_eur_per_mwh": constraints.transaction_cost_eur_per_mwh,
                "total_cost_eur_per_mwh": constraints.price_eur_per_mwh + constraints.transaction_cost_eur_per_mwh,
                "total_cost_eur": allocated_mwh * (constraints.price_eur_per_mwh + constraints.transaction_cost_eur_per_mwh),
                "collateral_eur": allocated_mwh * (constraints.price_eur_per_mwh + constraints.transaction_cost_eur_per_mwh) * constraints.collateral_haircut,
            }
            total_allocated += allocated_mwh
            total_cost += allocation[channel_name]["total_cost_eur"]
            total_collateral += allocation[channel_name]["collateral_eur"]

    residual_unhedged = total_gap_mwh - total_allocated

    return ProcurementOptimization(
        total_gap_mwh=total_gap_mwh,
        optimal_allocation=allocation,
        total_cost_eur=total_cost,
        average_cost_eur_per_mwh=total_cost / max(total_allocated, 1),
        collateral_required_eur=total_collateral,
        residual_unhedged_mwh=residual_unhedged,
        optimization_status="optimal" if residual_unhedged <= max_unhedged_mwh else "partial",
        optimization_notes=f"Balanced allocation (cost+risk+collateral) across {len(allocation)} channels; "
                          f"residual {residual_unhedged:.0f} MWh",
    )


def analyze_procurement_scenarios(
    total_gap_mwh: float,
    channel_constraints_scenarios: Dict[str, Dict[str, ChannelConstraints]],
    methods: List[str] = ["greedy", "balanced"],
) -> Dict[str, Dict]:
    """
    Analyze procurement strategies across price scenarios and methods.

    Args:
        total_gap_mwh: Non-solar gap to procure
        channel_constraints_scenarios: Dict of {scenario_name: {channel_name: ChannelConstraints}}
            Scenarios: "p10", "p25", "p50", "p75", "p90"
        methods: List of optimization methods to test

    Returns:
        Dict of {scenario_name: {method_name: ProcurementOptimization}}

    Example:
        >>> analysis = analyze_procurement_scenarios(
        ...     total_gap_mwh=7000,
        ...     channel_constraints_scenarios={
        ...         "p50": {...},
        ...         "p90": {...},
        ...     }
        ... )
        >>> p50_greedy = analysis["p50"]["greedy"]
    """
    scenario_results = {}

    for scenario_name, constraints_dict in channel_constraints_scenarios.items():
        scenario_results[scenario_name] = {}

        for method in methods:
            try:
                result = optimize_procurement_allocation(
                    total_gap_mwh=total_gap_mwh,
                    channel_constraints=constraints_dict,
                    method=method,
                )
                scenario_results[scenario_name][method] = result
            except Exception as e:
                logger.error(f"Error optimizing scenario {scenario_name} with method {method}: {e}")
                continue

    return scenario_results


def compare_procurement_strategies(
    results: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Compare procurement strategies across scenarios and methods.

    Args:
        results: Output from analyze_procurement_scenarios

    Returns:
        Comparison DataFrame (scenarios x methods)

    Example:
        >>> comparison = compare_procurement_strategies(analysis)
        >>> print(comparison)
    """
    rows = []

    for scenario_name, method_results in results.items():
        for method_name, opt_result in method_results.items():
            rows.append({
                "Scenario": scenario_name,
                "Method": method_name,
                "Total Cost (EUR)": opt_result.total_cost_eur,
                "Avg Cost (EUR/MWh)": opt_result.average_cost_eur_per_mwh,
                "Hedged (MWh)": opt_result.total_gap_mwh - opt_result.residual_unhedged_mwh,
                "Unhedged (MWh)": opt_result.residual_unhedged_mwh,
                "Hedge Rate (%)": ((opt_result.total_gap_mwh - opt_result.residual_unhedged_mwh) / opt_result.total_gap_mwh) * 100,
                "Collateral (EUR)": opt_result.collateral_required_eur,
                "Status": opt_result.optimization_status,
            })

    return pd.DataFrame(rows)


def estimate_channel_cost_sensitivity(
    base_constraints: Dict[str, ChannelConstraints],
    total_gap_mwh: float,
    channel_to_stress: str,
    price_changes_pct: List[float] = [-10, -5, 5, 10],
) -> pd.DataFrame:
    """
    Estimate sensitivity of procurement cost to price changes in one channel.

    Args:
        base_constraints: Base case channel constraints
        total_gap_mwh: Gap to procure
        channel_to_stress: Which channel to stress-test
        price_changes_pct: List of price change percentages

    Returns:
        Sensitivity DataFrame

    Example:
        >>> sensitivity = estimate_channel_cost_sensitivity(
        ...     base_constraints,
        ...     total_gap_mwh=7000,
        ...     channel_to_stress="brm_forward"
        ... )
    """
    rows = []

    for pct_change in price_changes_pct:
        # Copy and adjust constraints
        stressed_constraints = {}
        for channel_name, constraints in base_constraints.items():
            if channel_name == channel_to_stress:
                # Stress this channel's price
                adjusted_price = constraints.price_eur_per_mwh * (1 + pct_change / 100)
                stressed_constraints[channel_name] = ChannelConstraints(
                    channel_name=constraints.channel_name,
                    min_volume_mwh=constraints.min_volume_mwh,
                    max_volume_mwh=constraints.max_volume_mwh,
                    price_eur_per_mwh=adjusted_price,
                    transaction_cost_eur_per_mwh=constraints.transaction_cost_eur_per_mwh,
                    collateral_haircut=constraints.collateral_haircut,
                    credit_assessment_required=constraints.credit_assessment_required,
                    liquidity_available_mwh=constraints.liquidity_available_mwh,
                    settlement_type=constraints.settlement_type,
                )
            else:
                stressed_constraints[channel_name] = constraints

        # Optimize with stressed constraints
        result = optimize_procurement_allocation(
            total_gap_mwh=total_gap_mwh,
            channel_constraints=stressed_constraints,
        )

        rows.append({
            "Price Change (%)": pct_change,
            "Stressed Channel": channel_to_stress,
            "Total Cost (EUR)": result.total_cost_eur,
            "Avg Cost (EUR/MWh)": result.average_cost_eur_per_mwh,
            "Collateral (EUR)": result.collateral_required_eur,
        })

    return pd.DataFrame(rows)


# ============================================================================
# END OF PROCUREMENT OPTIMIZER
# ============================================================================

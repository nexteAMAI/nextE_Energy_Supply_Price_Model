"""Page 11: Supply P&L — Per-contract profitability, budget vs actuals, variance bridge."""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent))
from components.shared import init_page, load_kpis
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date

st.header("📊 Supply P&L — Budget vs Actuals")
st.caption("Per-Contract Profitability · Variance Decomposition · Portfolio Margin Tracking")
DATA_DIR = init_page()

kpis = load_kpis()

# ---------------------------------------------------------------------------
# Import P&L engine
# ---------------------------------------------------------------------------
try:
    from processors.supply_pnl import (
        generate_demo_pnl_data,
        compute_contract_pnl,
        compute_portfolio_pnl,
        build_variance_bridge,
        pnl_to_dataframe,
        portfolio_summary_table,
        PnLPeriod,
    )
    _PNL_AVAILABLE = True
except ImportError as e:
    _PNL_AVAILABLE = False
    st.error(f"Supply P&L module not available: {e}")
    st.info("Ensure `processors/supply_pnl.py` is deployed. Falling back to demo mode.")

# ---------------------------------------------------------------------------
# Generate demo data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600)
def load_pnl_data():
    """Load or generate P&L data."""
    if _PNL_AVAILABLE:
        budgets, actuals_by_month, all_pnls = generate_demo_pnl_data(
            n_contracts=5, n_months=12, base_dam_price=85.0
        )
        df_pnl = pnl_to_dataframe(all_pnls)
        return budgets, actuals_by_month, all_pnls, df_pnl
    return None, None, None, pd.DataFrame()

budgets, actuals_by_month, all_pnls, df_pnl = load_pnl_data()

if df_pnl.empty:
    st.warning("No P&L data available.")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Portfolio P&L", "📋 Contract Detail", "🌊 Variance Bridge", "📊 Trend Analysis"
])

# ---------------------------------------------------------------------------
# Tab 1: Portfolio P&L Summary
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Portfolio P&L Summary")

    # Aggregate by month
    monthly = df_pnl.groupby("period").agg({
        "actual_vol": "sum",
        "budget_vol": "sum",
        "actual_rev": "sum",
        "budget_rev": "sum",
        "actual_cost": "sum",
        "budget_cost": "sum",
        "actual_margin": "sum",
        "budget_margin": "sum",
        "margin_var": "sum",
    }).sort_index()

    monthly["margin_per_mwh"] = monthly["actual_margin"] / monthly["actual_vol"].clip(lower=1)
    monthly["budget_margin_per_mwh"] = monthly["budget_margin"] / monthly["budget_vol"].clip(lower=1)
    monthly["budget_ach_pct"] = monthly["actual_margin"] / monthly["budget_margin"].clip(lower=1) * 100

    # Top-level KPIs (full period)
    total_rev = monthly["actual_rev"].sum()
    total_cost = monthly["actual_cost"].sum()
    total_margin = monthly["actual_margin"].sum()
    total_budget_margin = monthly["budget_margin"].sum()
    total_vol = monthly["actual_vol"].sum()
    avg_margin_mwh = total_margin / max(total_vol, 1)
    total_var = total_margin - total_budget_margin
    ach_pct = total_margin / max(total_budget_margin, 1) * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue", f"€{total_rev / 1e6:.2f}M")
    k2.metric("Total Margin", f"€{total_margin / 1e6:.2f}M",
              delta=f"€{total_var / 1e6:+.2f}M vs budget")
    k3.metric("Avg Margin/MWh", f"€{avg_margin_mwh:.2f}")
    k4.metric("Budget Achievement", f"{ach_pct:.1f}%")

    st.divider()

    # Monthly P&L chart: revenue, cost, margin
    fig_pnl = go.Figure()

    fig_pnl.add_trace(go.Bar(
        x=monthly.index, y=monthly["actual_rev"],
        name="Revenue", marker_color="#27ae60", opacity=0.85,
    ))
    fig_pnl.add_trace(go.Bar(
        x=monthly.index, y=-monthly["actual_cost"],
        name="Cost", marker_color="#e74c3c", opacity=0.85,
    ))
    fig_pnl.add_trace(go.Scatter(
        x=monthly.index, y=monthly["actual_margin"],
        name="Actual Margin", mode="lines+markers",
        line=dict(color="#2c3e50", width=3),
    ))
    fig_pnl.add_trace(go.Scatter(
        x=monthly.index, y=monthly["budget_margin"],
        name="Budget Margin", mode="lines",
        line=dict(color="#3498db", width=2, dash="dash"),
    ))

    fig_pnl.update_layout(
        barmode="relative", height=450,
        yaxis_title="EUR",
        font=dict(family="DM Sans"),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=20),
        hovermode="x unified",
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    # Monthly margin variance
    st.subheader("Monthly Margin Variance (Actual − Budget)")
    fig_var = go.Figure(go.Bar(
        x=monthly.index,
        y=monthly["margin_var"],
        marker_color=["#27ae60" if v >= 0 else "#e74c3c" for v in monthly["margin_var"]],
        text=[f"€{v / 1000:+,.1f}k" for v in monthly["margin_var"]],
        textposition="outside",
    ))
    fig_var.add_hline(y=0, line_color="black", line_width=0.5)
    fig_var.update_layout(
        height=320, yaxis_title="EUR",
        font=dict(family="DM Sans"), margin=dict(t=20),
    )
    st.plotly_chart(fig_var, use_container_width=True)

    # Budget achievement trend
    st.subheader("Budget Achievement Trend (%)")
    fig_ach = go.Figure()
    fig_ach.add_trace(go.Scatter(
        x=monthly.index, y=monthly["budget_ach_pct"],
        mode="lines+markers", name="Achievement %",
        line=dict(color="#9b59b6", width=2),
        fill="tozeroy", fillcolor="rgba(155,89,182,0.1)",
    ))
    fig_ach.add_hline(y=100, line_dash="dash", line_color="#e74c3c",
                      annotation_text="100% Target")
    fig_ach.update_layout(
        height=300, yaxis_title="%",
        font=dict(family="DM Sans"), margin=dict(t=20),
        yaxis=dict(range=[max(0, monthly["budget_ach_pct"].min() - 10),
                          monthly["budget_ach_pct"].max() + 10]),
    )
    st.plotly_chart(fig_ach, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Per-Contract Detail
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Per-Contract P&L Detail")

    contracts = df_pnl["contract_id"].unique().tolist()
    selected_contract = st.selectbox("Select Contract", contracts)

    df_contract = df_pnl[df_pnl["contract_id"] == selected_contract].sort_values("period")

    if df_contract.empty:
        st.warning("No data for selected contract.")
    else:
        customer = df_contract["customer"].iloc[0]
        st.caption(f"Customer: **{customer}** · Contract: **{selected_contract}**")

        # Contract KPIs
        c_rev = df_contract["actual_rev"].sum()
        c_cost = df_contract["actual_cost"].sum()
        c_margin = df_contract["actual_margin"].sum()
        c_vol = df_contract["actual_vol"].sum()
        c_budget_margin = df_contract["budget_margin"].sum()
        c_avg_margin = c_margin / max(c_vol, 1)
        c_ach = c_margin / max(c_budget_margin, 1) * 100

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Revenue", f"€{c_rev / 1e6:.3f}M")
        cc2.metric("Margin", f"€{c_margin / 1e6:.3f}M", delta=f"{c_ach - 100:+.1f}% vs budget")
        cc3.metric("Margin/MWh", f"€{c_avg_margin:.2f}")
        cc4.metric("Volume", f"{c_vol:,.0f} MWh")

        st.divider()

        # Monthly budget vs actual margin
        fig_ctr = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ctr.add_trace(go.Bar(
            x=df_contract["period"], y=df_contract["budget_margin"],
            name="Budget Margin", marker_color="rgba(52,152,219,0.4)",
        ), secondary_y=False)
        fig_ctr.add_trace(go.Bar(
            x=df_contract["period"], y=df_contract["actual_margin"],
            name="Actual Margin", marker_color="rgba(39,174,96,0.7)",
        ), secondary_y=False)
        fig_ctr.add_trace(go.Scatter(
            x=df_contract["period"], y=df_contract["margin_per_mwh"],
            name="Margin/MWh", mode="lines+markers",
            line=dict(color="#e74c3c", width=2),
        ), secondary_y=True)

        fig_ctr.update_layout(
            barmode="group", height=400,
            font=dict(family="DM Sans"),
            legend=dict(orientation="h", y=-0.15),
            margin=dict(t=20),
        )
        fig_ctr.update_yaxes(title_text="EUR", secondary_y=False)
        fig_ctr.update_yaxes(title_text="EUR/MWh", secondary_y=True)
        st.plotly_chart(fig_ctr, use_container_width=True)

        # Detail table
        st.subheader("Monthly Detail")
        display_df = df_contract[["period", "budget_vol", "actual_vol", "vol_var",
                                   "budget_rev", "actual_rev", "budget_cost", "actual_cost",
                                   "budget_margin", "actual_margin", "margin_var",
                                   "margin_per_mwh", "budget_ach_pct", "floor_status"]].copy()
        st.dataframe(
            display_df.style.format({
                "budget_vol": "{:,.0f}", "actual_vol": "{:,.0f}", "vol_var": "{:+,.0f}",
                "budget_rev": "€{:,.0f}", "actual_rev": "€{:,.0f}",
                "budget_cost": "€{:,.0f}", "actual_cost": "€{:,.0f}",
                "budget_margin": "€{:,.0f}", "actual_margin": "€{:,.0f}",
                "margin_var": "€{:+,.0f}", "margin_per_mwh": "€{:.2f}",
                "budget_ach_pct": "{:.1f}%",
            }).applymap(
                lambda v: "color: green" if v == "ABOVE" else "color: red",
                subset=["floor_status"]
            ),
            use_container_width=True,
            height=450,
        )


# ---------------------------------------------------------------------------
# Tab 3: Variance Bridge
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Margin Variance Bridge")
    st.caption("Decomposition: Budget Margin → Price → Volume → GC → Balancing → Mix → Actual Margin")

    # Variance scope selection
    var_scope = st.radio("Scope", ["Full Portfolio", "Single Contract"], horizontal=True)

    if var_scope == "Single Contract":
        var_contract = st.selectbox("Contract", contracts, key="var_contract")
        df_var = df_pnl[df_pnl["contract_id"] == var_contract]
    else:
        df_var = df_pnl

    # Aggregate variances
    b_margin = df_var["budget_margin"].sum()
    a_margin = df_var["actual_margin"].sum()
    price_v = df_var["price_var"].sum()
    vol_v = df_var["volume_eff"].sum()
    gc_v = df_var["gc_var"].sum()
    bal_v = df_var["bal_var"].sum()
    mix_v = df_var["mix_var"].sum()
    other_v = a_margin - b_margin - price_v - vol_v - gc_v - bal_v - mix_v

    bridge_items = ["Budget\nMargin", "Price\nEffect", "Volume\nEffect", "GC Quota\nEffect",
                    "Balancing\nEffect", "Mix\nEffect", "Other", "Actual\nMargin"]
    bridge_values = [b_margin, price_v, vol_v, gc_v, bal_v, mix_v, other_v, a_margin]
    bridge_measures = ["absolute", "relative", "relative", "relative", "relative", "relative", "relative", "total"]

    fig_bridge = go.Figure(go.Waterfall(
        orientation="v",
        measure=bridge_measures,
        x=bridge_items,
        y=bridge_values,
        connector={"line": {"color": "rgba(100,100,100,0.3)", "width": 1}},
        increasing={"marker": {"color": "#27ae60"}},
        decreasing={"marker": {"color": "#e74c3c"}},
        totals={"marker": {"color": "#1F4E79", "line": {"color": "#2c3e50", "width": 2}}},
        text=[f"€{v / 1000:,.1f}k" for v in bridge_values],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig_bridge.update_layout(
        height=480,
        yaxis_title="EUR",
        font=dict(family="DM Sans"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=80),
    )
    st.plotly_chart(fig_bridge, use_container_width=True)

    # Variance summary table
    st.subheader("Variance Decomposition Summary")
    var_table = pd.DataFrame([
        {"Driver": "Price Effect (procurement cost delta)", "EUR": price_v, "% of Total": abs(price_v) / max(abs(a_margin - b_margin), 1) * 100},
        {"Driver": "Volume Effect (delivered vol delta)", "EUR": vol_v, "% of Total": abs(vol_v) / max(abs(a_margin - b_margin), 1) * 100},
        {"Driver": "GC Quota Effect", "EUR": gc_v, "% of Total": abs(gc_v) / max(abs(a_margin - b_margin), 1) * 100},
        {"Driver": "Balancing Effect", "EUR": bal_v, "% of Total": abs(bal_v) / max(abs(a_margin - b_margin), 1) * 100},
        {"Driver": "Mix / Channel Effect", "EUR": mix_v, "% of Total": abs(mix_v) / max(abs(a_margin - b_margin), 1) * 100},
        {"Driver": "Other / Residual", "EUR": other_v, "% of Total": abs(other_v) / max(abs(a_margin - b_margin), 1) * 100},
        {"Driver": "TOTAL VARIANCE", "EUR": a_margin - b_margin, "% of Total": 100.0},
    ]).set_index("Driver")
    st.dataframe(
        var_table.style.format({"EUR": "€{:+,.0f}", "% of Total": "{:.1f}%"}),
        use_container_width=True,
    )

    # Variance attribution by contract
    if var_scope == "Full Portfolio":
        st.subheader("Variance Attribution by Contract")
        contract_vars = df_pnl.groupby(["contract_id", "customer"]).agg({
            "margin_var": "sum",
            "price_var": "sum",
            "volume_eff": "sum",
            "actual_margin": "sum",
            "budget_margin": "sum",
        }).reset_index()
        contract_vars["budget_ach_pct"] = contract_vars["actual_margin"] / contract_vars["budget_margin"].clip(lower=1) * 100

        fig_attr = go.Figure()
        fig_attr.add_trace(go.Bar(
            x=contract_vars["contract_id"],
            y=contract_vars["margin_var"],
            marker_color=["#27ae60" if v >= 0 else "#e74c3c" for v in contract_vars["margin_var"]],
            text=[f"€{v / 1000:+,.1f}k" for v in contract_vars["margin_var"]],
            textposition="outside",
        ))
        fig_attr.add_hline(y=0, line_color="black", line_width=0.5)
        fig_attr.update_layout(
            height=320, yaxis_title="Margin Variance (EUR)",
            font=dict(family="DM Sans"), margin=dict(t=20),
        )
        st.plotly_chart(fig_attr, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: Trend Analysis
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Margin Trend Analysis")

    # Margin/MWh trend by contract
    fig_trend = go.Figure()
    colors = ["#2c3e50", "#3498db", "#e67e22", "#27ae60", "#9b59b6"]
    for i, cid in enumerate(contracts):
        df_c = df_pnl[df_pnl["contract_id"] == cid].sort_values("period")
        fig_trend.add_trace(go.Scatter(
            x=df_c["period"], y=df_c["margin_per_mwh"],
            name=cid, mode="lines+markers",
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig_trend.add_hline(y=8.0, line_dash="dash", line_color="red",
                        annotation_text="Min Floor €8/MWh")
    fig_trend.update_layout(
        height=420, yaxis_title="Margin (EUR/MWh)",
        font=dict(family="DM Sans"),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=20),
        hovermode="x unified",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Cumulative margin
    st.subheader("Cumulative Margin (Portfolio)")
    monthly_agg = df_pnl.groupby("period").agg({
        "actual_margin": "sum", "budget_margin": "sum"
    }).sort_index()
    monthly_agg["cum_actual"] = monthly_agg["actual_margin"].cumsum()
    monthly_agg["cum_budget"] = monthly_agg["budget_margin"].cumsum()

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=monthly_agg.index, y=monthly_agg["cum_actual"],
        name="Actual (Cumulative)", mode="lines",
        line=dict(color="#27ae60", width=3),
        fill="tozeroy", fillcolor="rgba(39,174,96,0.1)",
    ))
    fig_cum.add_trace(go.Scatter(
        x=monthly_agg.index, y=monthly_agg["cum_budget"],
        name="Budget (Cumulative)", mode="lines",
        line=dict(color="#3498db", width=2, dash="dash"),
    ))
    fig_cum.update_layout(
        height=380, yaxis_title="Cumulative EUR",
        font=dict(family="DM Sans"),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=20),
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Rolling margin per MWh
    st.subheader("Rolling 3-Month Avg Margin/MWh")
    monthly_margin = df_pnl.groupby("period").agg({
        "actual_margin": "sum", "actual_vol": "sum"
    }).sort_index()
    monthly_margin["margin_per_mwh"] = monthly_margin["actual_margin"] / monthly_margin["actual_vol"].clip(lower=1)
    monthly_margin["rolling_3m"] = monthly_margin["margin_per_mwh"].rolling(3, min_periods=1).mean()

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Bar(
        x=monthly_margin.index, y=monthly_margin["margin_per_mwh"],
        name="Monthly", marker_color="rgba(52,152,219,0.4)",
    ))
    fig_roll.add_trace(go.Scatter(
        x=monthly_margin.index, y=monthly_margin["rolling_3m"],
        name="3M Rolling Avg", mode="lines",
        line=dict(color="#e74c3c", width=3),
    ))
    fig_roll.add_hline(y=8.0, line_dash="dash", line_color="red",
                       annotation_text="Min Floor")
    fig_roll.update_layout(
        height=350, yaxis_title="EUR/MWh",
        font=dict(family="DM Sans"),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=20),
    )
    st.plotly_chart(fig_roll, use_container_width=True)

    # Scatter: Volume vs Margin
    st.subheader("Volume vs Margin Relationship")
    contract_agg = df_pnl.groupby(["contract_id", "customer"]).agg({
        "actual_vol": "sum", "actual_margin": "sum",
    }).reset_index()
    contract_agg["margin_per_mwh"] = contract_agg["actual_margin"] / contract_agg["actual_vol"].clip(lower=1)

    fig_scatter = go.Figure(go.Scatter(
        x=contract_agg["actual_vol"] / 1000,
        y=contract_agg["margin_per_mwh"],
        mode="markers+text",
        marker=dict(size=15, color=colors[:len(contract_agg)], line=dict(width=1, color="white")),
        text=contract_agg["contract_id"],
        textposition="top center",
        textfont=dict(size=9),
    ))
    fig_scatter.add_hline(y=8.0, line_dash="dash", line_color="red")
    fig_scatter.update_layout(
        height=350,
        xaxis_title="Total Volume (GWh)",
        yaxis_title="Avg Margin (EUR/MWh)",
        font=dict(family="DM Sans"),
        margin=dict(t=20),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

"""Page 12: Supply Pipeline Dashboard — End-to-end pipeline execution & monitoring."""
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
from datetime import datetime, date, timedelta
import logging

st.header("🔄 Supply Pipeline")
st.caption("End-to-End Pipeline Orchestrator · Generation & Consumption Forecasts · Gap Analysis · Risk")
DATA_DIR = init_page()

# ---------------------------------------------------------------------------
# Color palette (consistent with other pages)
# ---------------------------------------------------------------------------
NAVY = "#1B2A4A"
BLUE = "#2E86AB"
GREEN = "#28A745"
AMBER = "#FFC107"
RED = "#DC3545"
TEAL = "#17A2B8"
PURPLE = "#6F42C1"
GRAY = "#6C757D"

PROB_COLORS = {
    "P10": "#FF6B6B",
    "P25": "#FFA07A",
    "P50": NAVY,
    "P75": "#87CEEB",
    "P90": "#4ECDC4",
}

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def _run_pipeline(horizon: int, scenarios: int, seed: int):
    """Run supply pipeline with caching."""
    from processors.supply_pipeline import (
        run_supply_pipeline, PipelineConfig, PipelineMode, StageStatus,
    )
    config = PipelineConfig(
        mode=PipelineMode.FULL,
        forecast_horizon_days=horizon,
        n_scenarios=scenarios,
        seed=seed,
    )
    return run_supply_pipeline(config)


# Sidebar controls
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Pipeline Configuration")
horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 30, 7)
scenarios = st.sidebar.slider("Monte Carlo Scenarios", 10, 200, 50, step=10)
seed = st.sidebar.number_input("Random Seed", value=42, min_value=0)

run_button = st.sidebar.button("▶️ Run Pipeline", type="primary", use_container_width=True)

if run_button:
    st.cache_data.clear()

with st.spinner("Running supply pipeline..."):
    try:
        result = _run_pipeline(horizon, scenarios, seed)
    except Exception as e:
        st.error(f"Pipeline execution failed: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# KPI Header
# ---------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

summary = result.summary
pricing_data = result.stages.get("pricing")
risk_data = result.stages.get("risk")
pnl_data = result.stages.get("pnl")

with col1:
    status_emoji = "✅" if result.success else "❌"
    st.metric("Status", f"{status_emoji} {'OK' if result.success else 'FAIL'}")
with col2:
    st.metric("Duration", f"{summary.get('duration_seconds', 0):.1f}s")
with col3:
    if pricing_data and pricing_data.output_data:
        price = pricing_data.output_data["waterfall"]["final_price_ex_vat"]
        st.metric("Price (ex-VAT)", f"€{price:.2f}/MWh")
    else:
        st.metric("Price", "N/A")
with col4:
    if pnl_data and pnl_data.output_data:
        margin = pnl_data.output_data["margin_per_mwh"]
        st.metric("Margin", f"€{margin:.2f}/MWh")
    else:
        st.metric("Margin", "N/A")
with col5:
    alerts_count = len(result.alerts)
    high_alerts = summary.get("high_alerts", 0)
    if high_alerts > 0:
        st.metric("Alerts", f"🔴 {alerts_count}")
    elif alerts_count > 0:
        st.metric("Alerts", f"🟡 {alerts_count}")
    else:
        st.metric("Alerts", f"🟢 0")

# ---------------------------------------------------------------------------
# Alerts banner
# ---------------------------------------------------------------------------
if result.alerts:
    with st.expander(f"⚠️ Pipeline Alerts ({len(result.alerts)})", expanded=True):
        for alert in result.alerts:
            sev = alert["severity"]
            icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "GREEN": "🟢"}.get(sev, "⚪")
            st.markdown(
                f"**{icon} [{sev}] {alert['category'].upper()}:** "
                f"{alert['message']}  \n"
                f"*Action:* {alert['action']}"
            )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Forecasts", "🔀 Gap Analysis", "💰 Pricing", "⚡ Risk", "📋 Pipeline Log"
])

# ===== TAB 1: FORECASTS =====
with tab1:
    gen_stage = result.stages.get("generation")
    con_stage = result.stages.get("consumption")

    if gen_stage and gen_stage.output_data and con_stage and con_stage.output_data:
        gen_profiles = gen_stage.output_data["portfolio_profiles"]
        con_profiles = con_stage.output_data["portfolio_profiles"]

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Generation Forecast (PV + Wind)")
            fig_gen = go.Figure()
            # P10-P90 band
            fig_gen.add_trace(go.Scatter(
                x=gen_profiles["timestamp"], y=gen_profiles["P90"],
                mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig_gen.add_trace(go.Scatter(
                x=gen_profiles["timestamp"], y=gen_profiles["P10"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(46,134,171,0.15)",
                name="P10–P90 band",
            ))
            # P25-P75 band
            fig_gen.add_trace(go.Scatter(
                x=gen_profiles["timestamp"], y=gen_profiles["P75"],
                mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig_gen.add_trace(go.Scatter(
                x=gen_profiles["timestamp"], y=gen_profiles["P25"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(46,134,171,0.3)",
                name="P25–P75 band",
            ))
            # P50
            fig_gen.add_trace(go.Scatter(
                x=gen_profiles["timestamp"], y=gen_profiles["P50"],
                mode="lines", line=dict(color=NAVY, width=2),
                name="P50 (expected)",
            ))
            fig_gen.update_layout(
                height=350, margin=dict(l=40, r=20, t=30, b=40),
                yaxis_title="MW", xaxis_title="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_gen, use_container_width=True)

            # Tech breakdown
            tech_bk = gen_stage.output_data.get("technology_breakdown", {})
            if tech_bk:
                st.caption("Technology Breakdown (P50, first day)")
                tech_cols = st.columns(len(tech_bk))
                for idx, (tech, df) in enumerate(tech_bk.items()):
                    day1 = df.head(96)
                    peak = day1["P50"].max()
                    energy = day1["P50"].sum() * 0.25
                    with tech_cols[idx]:
                        st.metric(f"{tech.upper()} Peak", f"{peak:.1f} MW")
                        st.metric(f"{tech.upper()} Energy D1", f"{energy:.0f} MWh")

        with c2:
            st.subheader("Consumption Forecast (Portfolio)")
            fig_con = go.Figure()
            fig_con.add_trace(go.Scatter(
                x=con_profiles["timestamp"], y=con_profiles["P90"],
                mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig_con.add_trace(go.Scatter(
                x=con_profiles["timestamp"], y=con_profiles["P10"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(111,66,193,0.15)",
                name="P10–P90 band",
            ))
            fig_con.add_trace(go.Scatter(
                x=con_profiles["timestamp"], y=con_profiles["P75"],
                mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig_con.add_trace(go.Scatter(
                x=con_profiles["timestamp"], y=con_profiles["P25"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(111,66,193,0.3)",
                name="P25–P75 band",
            ))
            fig_con.add_trace(go.Scatter(
                x=con_profiles["timestamp"], y=con_profiles["P50"],
                mode="lines", line=dict(color=PURPLE, width=2),
                name="P50 (expected)",
            ))
            fig_con.update_layout(
                height=350, margin=dict(l=40, r=20, t=30, b=40),
                yaxis_title="MW", xaxis_title="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_con, use_container_width=True)

            # Customer stats
            con_fc = con_stage.output_data.get("portfolio_forecast")
            if con_fc:
                st.caption(f"{con_fc.total_annual_mwh:,.0f} MWh/yr · {con_fc.total_peak_mw:.1f} MW peak · {len(con_fc.customer_forecasts)} customers")
    else:
        st.info("Forecast data not available. Run the pipeline with generation + consumption stages.")


# ===== TAB 2: GAP ANALYSIS =====
with tab2:
    gap_stage = result.stages.get("gap")

    if gap_stage and gap_stage.output_data:
        gap_data = gap_stage.output_data
        gap_profiles = gap_data["gap_profiles"]
        gap_stats = gap_data["gap_statistics"]

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        p50_stats = gap_stats.get("P50", {})
        with c1:
            st.metric("Self-Supply Ratio", f"{gap_data['self_supply_ratio']*100:.1f}%")
        with c2:
            st.metric("Procurement Need (P50)", f"{p50_stats.get('total_mwh', 0):,.0f} MWh")
        with c3:
            st.metric("Avg Gap (P50)", f"{p50_stats.get('mean_mw', 0):.1f} MW")
        with c4:
            st.metric("Max Gap (P50)", f"{p50_stats.get('max_mw', 0):.1f} MW")

        # Gap chart
        fig_gap = go.Figure()
        fig_gap.add_trace(go.Scatter(
            x=gap_profiles["timestamp"], y=gap_profiles["P90"],
            mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig_gap.add_trace(go.Scatter(
            x=gap_profiles["timestamp"], y=gap_profiles["P10"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(220,53,69,0.12)",
            name="P10–P90",
        ))
        fig_gap.add_trace(go.Scatter(
            x=gap_profiles["timestamp"], y=gap_profiles["P50"],
            mode="lines", line=dict(color=RED, width=2),
            name="Gap P50",
        ))
        fig_gap.add_hline(y=0, line=dict(color="gray", dash="dash"), annotation_text="Zero gap")
        fig_gap.update_layout(
            title="Generation–Consumption Gap (positive = buy, negative = sell)",
            height=400, margin=dict(l=40, r=20, t=50, b=40),
            yaxis_title="MW",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_gap, use_container_width=True)

        # Hourly average gap profile
        hourly = gap_data.get("hourly_average_gap")
        if hourly is not None and not hourly.empty:
            fig_hourly = go.Figure()
            colors_fill = [RED if v > 0 else GREEN for v in hourly["P50"]]
            fig_hourly.add_trace(go.Bar(
                x=hourly["hour"], y=hourly["P50"],
                marker_color=colors_fill,
                name="Avg Gap (P50)",
            ))
            fig_hourly.update_layout(
                title="Average Hourly Gap Profile",
                height=300, margin=dict(l=40, r=20, t=50, b=40),
                xaxis_title="Hour", yaxis_title="MW",
                xaxis=dict(dtick=1),
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
    else:
        st.info("Gap analysis not available.")


# ===== TAB 3: PRICING =====
with tab3:
    if pricing_data and pricing_data.output_data:
        pd_data = pricing_data.output_data
        waterfall = pd_data["waterfall"]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Final Price (ex-VAT)", f"€{waterfall['final_price_ex_vat']:.2f}/MWh")
        with c2:
            st.metric("Solar Share", f"{pd_data['solar_share_pct']:.1f}%")
        with c3:
            st.metric("Total Volume (P50)", f"{pd_data['total_volume_mwh_p50']:,.0f} MWh")

        # Waterfall chart
        wf_labels = [
            "Blended Energy", "GC Quota", "Balancing", "Transport",
            "Risk Premium", "Margin", "Final (ex-VAT)"
        ]
        wf_values = [
            waterfall["blended_energy_cost"],
            waterfall["gc_quota_cost"],
            waterfall["balancing_cost"],
            waterfall["transport_admin"],
            waterfall["risk_premium"],
            waterfall["nexte_margin"],
            waterfall["final_price_ex_vat"],
        ]
        wf_measure = ["relative"] * 6 + ["total"]

        fig_wf = go.Figure(go.Waterfall(
            x=wf_labels, y=wf_values, measure=wf_measure,
            connector_line_color="rgba(0,0,0,0.3)",
            increasing_marker_color=BLUE,
            totals_marker_color=NAVY,
            text=[f"€{v:.2f}" for v in wf_values],
            textposition="outside",
        ))
        fig_wf.update_layout(
            title="Supply Price Waterfall (EUR/MWh)",
            height=400, margin=dict(l=40, r=20, t=50, b=60),
            yaxis_title="EUR/MWh",
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        # Scenario prices
        st.subheader("Multi-Scenario Prices")
        scenario_prices = pd_data.get("scenario_prices", {})
        if scenario_prices:
            fig_sc = go.Figure(go.Bar(
                x=list(scenario_prices.keys()),
                y=list(scenario_prices.values()),
                marker_color=[PROB_COLORS.get(k, GRAY) for k in scenario_prices.keys()],
                text=[f"€{v:.2f}" for v in scenario_prices.values()],
                textposition="outside",
            ))
            fig_sc.update_layout(
                height=300, margin=dict(l=40, r=20, t=30, b=40),
                yaxis_title="EUR/MWh",
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        # P&L projection
        if pnl_data and pnl_data.output_data:
            st.subheader("P&L Projection")
            pnl = pnl_data.output_data
            pc1, pc2, pc3, pc4 = st.columns(4)
            with pc1:
                st.metric("Revenue", f"€{pnl['revenue_eur']:,.0f}")
            with pc2:
                st.metric("Cost", f"€{pnl['total_cost_eur']:,.0f}")
            with pc3:
                st.metric("Margin", f"€{pnl['gross_margin_eur']:,.0f}")
            with pc4:
                st.metric("Margin %", f"{pnl['margin_pct']:.1f}%")

            # Cost breakdown pie
            breakdown = pnl.get("cost_breakdown", {})
            if breakdown:
                fig_pie = go.Figure(go.Pie(
                    labels=list(breakdown.keys()),
                    values=list(breakdown.values()),
                    hole=0.4,
                    marker_colors=[BLUE, NAVY, TEAL, AMBER, GRAY, PURPLE],
                ))
                fig_pie.update_layout(
                    title="Cost Breakdown",
                    height=350, margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Pricing data not available.")


# ===== TAB 4: RISK =====
with tab4:
    if risk_data and risk_data.output_data:
        rd = risk_data.output_data

        c1, c2, c3, c4 = st.columns(4)
        var_color = {"GREEN": GREEN, "YELLOW": AMBER, "RED": RED}.get(rd["var_95_status"], GRAY)
        with c1:
            st.metric("VaR 95%", f"€{rd['var_95_eur']:,.0f}")
        with c2:
            st.metric("VaR Status", rd["var_95_status"])
        with c3:
            st.metric("Imbalance Cost/Day", f"€{rd['imbalance_cost_daily_eur']:,.0f}")
        with c4:
            st.metric("Imbalance Status", rd["imbalance_status"])

        # Risk decomposition
        decomp = rd.get("risk_decomposition", {})
        if decomp:
            fig_risk = go.Figure(go.Bar(
                x=list(decomp.keys()),
                y=list(decomp.values()),
                marker_color=[BLUE, TEAL, AMBER],
                text=[f"{v:.1f}%" for v in decomp.values()],
                textposition="outside",
            ))
            fig_risk.update_layout(
                title="Risk Decomposition (% of total VaR)",
                height=300, margin=dict(l=40, r=20, t=50, b=40),
                yaxis_title="%",
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        # Detailed risk metrics
        st.subheader("Detailed Risk Metrics")
        risk_table = pd.DataFrame([
            {"Metric": "Volume Risk (EUR)", "Value": f"€{rd['volume_risk_eur']:,.0f}"},
            {"Metric": "Shape Risk (EUR)", "Value": f"€{rd['shape_risk_eur']:,.0f}"},
            {"Metric": "Price Risk (EUR)", "Value": f"€{rd['price_risk_eur']:,.0f}"},
            {"Metric": "Shape Risk (MW std)", "Value": f"{rd['shape_risk_mw']:.2f} MW"},
            {"Metric": "Gen Volume Spread", "Value": f"{rd['gen_volume_spread_mwh']:,.0f} MWh"},
            {"Metric": "Imbalance Exposure", "Value": f"{rd['imbalance_exposure_mw']:.1f} MW"},
        ])
        st.dataframe(risk_table, use_container_width=True, hide_index=True)

        # Procurement allocation
        proc_stage = result.stages.get("procurement")
        if proc_stage and proc_stage.output_data:
            st.subheader("Procurement Allocation")
            alloc = proc_stage.output_data.get("allocation", {})
            if alloc:
                alloc_df = pd.DataFrame([
                    {
                        "Channel": ch,
                        "Volume (MWh)": f"{d['volume_mwh']:,.0f}",
                        "Price (€/MWh)": f"{d['price_eur_per_mwh']:.2f}",
                        "Cost (€)": f"{d['cost_eur']:,.0f}",
                        "Collateral (€)": f"{d['collateral_eur']:,.0f}",
                        "Share": f"{d['share_pct']:.0f}%",
                    }
                    for ch, d in alloc.items()
                ])
                st.dataframe(alloc_df, use_container_width=True, hide_index=True)

                wavg = proc_stage.output_data.get("weighted_avg_price_eur_per_mwh", 0)
                total_coll = proc_stage.output_data.get("total_collateral_eur", 0)
                st.caption(f"Weighted Avg Price: €{wavg:.2f}/MWh · Total Collateral: €{total_coll:,.0f}")
    else:
        st.info("Risk data not available.")


# ===== TAB 5: PIPELINE LOG =====
with tab5:
    st.subheader("Pipeline Execution Log")

    # Stage summary table
    stage_rows = []
    for name, sr in result.stages.items():
        status_icon = {
            "completed": "✅", "failed": "❌", "skipped": "⏭️", "running": "🔄", "pending": "⏳"
        }.get(sr.status.value, "❓")
        stage_rows.append({
            "Stage": name.capitalize(),
            "Status": f"{status_icon} {sr.status.value}",
            "Duration": f"{sr.duration_seconds:.1f}s" if sr.duration_seconds else "—",
            "Records": sr.records_processed if sr.records_processed else "—",
            "Error": sr.error_message or "—",
        })

    stage_df = pd.DataFrame(stage_rows)
    st.dataframe(stage_df, use_container_width=True, hide_index=True)

    # Summary JSON
    with st.expander("Pipeline Summary (JSON)"):
        st.json(result.summary)

    # Configuration
    with st.expander("Pipeline Configuration"):
        st.json({
            "mode": result.config.mode.value,
            "forecast_horizon_days": result.config.forecast_horizon_days,
            "n_scenarios": result.config.n_scenarios,
            "seed": result.config.seed,
            "min_margin_floor": result.config.min_margin_floor_eur_per_mwh,
        })

st.sidebar.markdown("---")
st.sidebar.caption(f"Pipeline v{result.pipeline_id}")

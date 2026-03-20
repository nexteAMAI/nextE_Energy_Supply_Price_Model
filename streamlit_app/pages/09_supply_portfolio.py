"""Page 09: Supply Portfolio Dashboard — Contract register, P&L, risk metrics."""
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
from datetime import datetime, timedelta

st.header("📂 Supply Portfolio Dashboard")
st.caption("Contract Register · Margin Tracking · Risk Exposure · P&L Attribution")
DATA_DIR = init_page()

kpis = load_kpis()
eur_ron = kpis.get("eur_ron_latest", 4.977)
dam_base = kpis.get("trailing_6m_avg", 85.0)

# ---------------------------------------------------------------------------
# Load supply config
# ---------------------------------------------------------------------------
_CFG = {}
try:
    import yaml
    _cfg_path = _P(__file__).resolve().parent.parent.parent / "config" / "supply_config.yaml"
    if _cfg_path.exists():
        with open(_cfg_path, "r") as f:
            _CFG = yaml.safe_load(f) or {}
except Exception:
    pass

# ---------------------------------------------------------------------------
# Demo contract portfolio (will be replaced by Supabase data in production)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600)
def get_demo_portfolio():
    """Generate a realistic demo portfolio for the dashboard."""
    np.random.seed(42)
    contracts = [
        {"id": "SC-2026-001", "customer": "CUST-ALPHA", "category": "industrial",
         "volume_mwh": 35000, "price_ex_vat": 108.50, "margin": 12.0,
         "start": "2026-04-01", "end": "2027-03-31", "status": "active",
         "pv_mechanism": "HYBRID", "solar_share": 30, "credit_rating": "BBB"},
        {"id": "SC-2026-002", "customer": "CUST-BETA", "category": "commercial",
         "volume_mwh": 12000, "price_ex_vat": 115.20, "margin": 14.5,
         "start": "2026-04-01", "end": "2026-12-31", "status": "active",
         "pv_mechanism": "FIXED", "solar_share": 40, "credit_rating": "A"},
        {"id": "SC-2026-003", "customer": "CUST-GAMMA", "category": "industrial",
         "volume_mwh": 55000, "price_ex_vat": 102.80, "margin": 10.0,
         "start": "2026-05-01", "end": "2027-04-30", "status": "pending",
         "pv_mechanism": "DAM_INDEXED", "solar_share": 20, "credit_rating": "BBB"},
        {"id": "SC-2026-004", "customer": "CUST-DELTA", "category": "commercial",
         "volume_mwh": 8000, "price_ex_vat": 122.00, "margin": 16.0,
         "start": "2026-06-01", "end": "2027-05-31", "status": "negotiation",
         "pv_mechanism": "HYBRID", "solar_share": 45, "credit_rating": "AA"},
        {"id": "SC-2026-005", "customer": "CUST-EPSILON", "category": "industrial",
         "volume_mwh": 25000, "price_ex_vat": 106.30, "margin": 11.5,
         "start": "2026-04-01", "end": "2027-03-31", "status": "active",
         "pv_mechanism": "HYBRID", "solar_share": 35, "credit_rating": "A"},
    ]
    return pd.DataFrame(contracts)

portfolio = get_demo_portfolio()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📋 Contract Register", "💰 Margin & P&L", "⚠️ Risk Exposure", "📈 Portfolio Analytics"])

# ---------------------------------------------------------------------------
# Tab 1: Contract Register
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Active & Pipeline Contracts")

    # Status filter
    status_filter = st.multiselect(
        "Filter by Status",
        options=["active", "pending", "negotiation"],
        default=["active", "pending", "negotiation"],
    )
    filtered = portfolio[portfolio["status"].isin(status_filter)]

    # KPI cards
    active = portfolio[portfolio["status"] == "active"]
    total_vol = active["volume_mwh"].sum()
    total_rev = (active["volume_mwh"] * active["price_ex_vat"]).sum()
    avg_margin = (active["volume_mwh"] * active["margin"]).sum() / max(total_vol, 1)
    pipeline_vol = portfolio[portfolio["status"].isin(["pending", "negotiation"])]["volume_mwh"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Contracts", f"{len(active)}")
    c2.metric("Contracted Volume", f"{total_vol:,.0f} MWh/yr")
    c3.metric("Wtd Avg Margin", f"€{avg_margin:.2f}/MWh")
    c4.metric("Pipeline Volume", f"{pipeline_vol:,.0f} MWh/yr")

    st.divider()

    # Contract table
    display_cols = ["id", "customer", "category", "volume_mwh", "price_ex_vat", "margin",
                    "start", "end", "status", "pv_mechanism", "solar_share", "credit_rating"]
    st.dataframe(
        filtered[display_cols].style.format({
            "volume_mwh": "{:,.0f}",
            "price_ex_vat": "€{:.2f}",
            "margin": "€{:.2f}",
            "solar_share": "{:.0f}%",
        }).applymap(
            lambda v: "background-color: #d4edda" if v == "active"
            else ("background-color: #fff3cd" if v == "pending"
                  else ("background-color: #f8d7da" if v == "negotiation" else "")),
            subset=["status"]
        ),
        use_container_width=True,
        height=250,
    )

    # Volume by category
    col_cat, col_status = st.columns(2)
    with col_cat:
        cat_vol = portfolio.groupby("category")["volume_mwh"].sum()
        fig_cat = go.Figure(go.Pie(
            labels=cat_vol.index.str.title(), values=cat_vol.values,
            hole=0.5, textinfo="label+percent",
            marker=dict(colors=["#2c3e50", "#3498db"]),
        ))
        fig_cat.update_layout(height=280, title="Volume by Category", font=dict(family="DM Sans", size=10), margin=dict(t=40, b=10))
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_status:
        stat_vol = portfolio.groupby("status")["volume_mwh"].sum()
        fig_stat = go.Figure(go.Pie(
            labels=stat_vol.index.str.title(), values=stat_vol.values,
            hole=0.5, textinfo="label+percent",
            marker=dict(colors=["#27ae60", "#f39c12", "#e74c3c"]),
        ))
        fig_stat.update_layout(height=280, title="Volume by Status", font=dict(family="DM Sans", size=10), margin=dict(t=40, b=10))
        st.plotly_chart(fig_stat, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Margin & P&L
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Margin Attribution & P&L Forecast")

    # Monthly P&L projection (demo: 12 months forward)
    months = pd.date_range(start="2026-04-01", periods=12, freq="MS")
    np.random.seed(123)

    monthly_data = []
    for m in months:
        vol = total_vol / 12 * (1 + np.random.normal(0, 0.05))
        dam_m = dam_base * (1 + np.random.normal(0, 0.08))
        procurement_cost = dam_m * 0.98 * vol  # Blended procurement
        revenue = active["price_ex_vat"].mean() * vol
        gc_cost = 14.50 * vol
        bal_cost = 3.0 * vol
        margin_val = revenue - procurement_cost - gc_cost - bal_cost
        monthly_data.append({
            "month": m,
            "volume_mwh": vol,
            "revenue_eur": revenue,
            "procurement_cost_eur": procurement_cost,
            "gc_cost_eur": gc_cost,
            "balancing_cost_eur": bal_cost,
            "gross_margin_eur": margin_val,
            "margin_per_mwh": margin_val / max(vol, 1),
        })

    df_pnl = pd.DataFrame(monthly_data)

    # P&L summary
    total_rev_proj = df_pnl["revenue_eur"].sum()
    total_cost_proj = df_pnl["procurement_cost_eur"].sum() + df_pnl["gc_cost_eur"].sum() + df_pnl["balancing_cost_eur"].sum()
    total_margin_proj = df_pnl["gross_margin_eur"].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Projected Revenue", f"€{total_rev_proj / 1e6:.2f}M")
    k2.metric("Total Cost", f"€{total_cost_proj / 1e6:.2f}M")
    k3.metric("Gross Margin", f"€{total_margin_proj / 1e6:.2f}M")
    k4.metric("Avg Margin/MWh", f"€{df_pnl['margin_per_mwh'].mean():.2f}")

    st.divider()

    # Revenue vs Cost chart
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Bar(
        x=df_pnl["month"], y=df_pnl["revenue_eur"],
        name="Revenue", marker_color="#27ae60",
    ))
    fig_pnl.add_trace(go.Bar(
        x=df_pnl["month"], y=-df_pnl["procurement_cost_eur"],
        name="Procurement", marker_color="#e74c3c",
    ))
    fig_pnl.add_trace(go.Bar(
        x=df_pnl["month"], y=-df_pnl["gc_cost_eur"],
        name="GC Cost", marker_color="#f39c12",
    ))
    fig_pnl.add_trace(go.Bar(
        x=df_pnl["month"], y=-df_pnl["balancing_cost_eur"],
        name="Balancing", marker_color="#e67e22",
    ))
    fig_pnl.add_trace(go.Scatter(
        x=df_pnl["month"], y=df_pnl["gross_margin_eur"],
        name="Gross Margin", mode="lines+markers",
        line=dict(color="#2c3e50", width=3),
        yaxis="y2",
    ))

    fig_pnl.update_layout(
        height=420,
        barmode="relative",
        yaxis_title="EUR",
        yaxis2=dict(title="Margin (EUR)", overlaying="y", side="right"),
        font=dict(family="DM Sans"),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=20),
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    # Margin per contract
    st.subheader("Margin by Contract")
    fig_margin = go.Figure(go.Bar(
        x=active["id"],
        y=active["margin"],
        marker_color=["#27ae60" if m >= 8 else "#e74c3c" for m in active["margin"]],
        text=[f"€{m:.1f}" for m in active["margin"]],
        textposition="outside",
    ))
    fig_margin.add_hline(y=8.0, line_dash="dash", line_color="red", annotation_text="Min Floor (€8)")
    fig_margin.update_layout(
        height=300, yaxis_title="Margin (EUR/MWh)",
        font=dict(family="DM Sans"), margin=dict(t=20),
    )
    st.plotly_chart(fig_margin, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Risk Exposure
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Portfolio Risk Dashboard")

    # Risk metrics (illustrative)
    gc_quota_coeff = _CFG.get("regulatory", {}).get("gc_quota_coefficient", 0.499387)
    var_limit = _CFG.get("risk_limits", {}).get("var_95_30d_eur_million", 2.5)
    max_customer_gwh = _CFG.get("risk_limits", {}).get("max_customer_exposure_gwh", 50.0)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Portfolio VaR (95%, 30d)", f"€{1.85:.2f}M", delta=f"-€{var_limit - 1.85:.2f}M to limit")
    r2.metric("Max Customer Exposure", f"{portfolio['volume_mwh'].max() / 1000:.1f} GWh", delta=f"Limit: {max_customer_gwh:.0f} GWh")
    r3.metric("Unhedged Exposure", "0%", delta="Policy: 0%")
    r4.metric("GC Quota Coefficient", f"{gc_quota_coeff:.6f}")

    st.divider()

    # Risk decomposition
    st.subheader("Risk Decomposition (EUR/MWh)")

    risk_components = {
        "Shape Risk": 2.80,
        "Volume Risk": 1.50,
        "Price Risk": 3.20,
        "Credit Risk": 0.85,
        "Basis Risk": 1.10,
        "Liquidity Risk": 0.45,
    }

    fig_risk = go.Figure(go.Bar(
        x=list(risk_components.keys()),
        y=list(risk_components.values()),
        marker_color=["#e74c3c", "#e67e22", "#f39c12", "#9b59b6", "#3498db", "#1abc9c"],
        text=[f"€{v:.2f}" for v in risk_components.values()],
        textposition="outside",
    ))
    total_risk = sum(risk_components.values())
    fig_risk.update_layout(
        height=350, yaxis_title="EUR/MWh",
        font=dict(family="DM Sans"), margin=dict(t=20),
        annotations=[
            dict(x=0.95, y=0.95, xref="paper", yref="paper", showarrow=False,
                 text=f"<b>Total Risk: €{total_risk:.2f}/MWh</b>",
                 font=dict(size=13, color="#e74c3c"),
                 bordercolor="#e74c3c", borderwidth=1, borderpad=6, bgcolor="rgba(255,255,255,0.9)")
        ],
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    # Credit exposure by rating
    st.subheader("Credit Exposure by Rating")
    credit_exp = portfolio.groupby("credit_rating")["volume_mwh"].sum().sort_index()
    fig_credit = go.Figure(go.Bar(
        x=credit_exp.index, y=credit_exp.values / 1000,
        marker_color=["#27ae60" if r in ("AAA", "AA", "A") else "#f39c12" if r == "BBB" else "#e74c3c"
                       for r in credit_exp.index],
        text=[f"{v / 1000:.1f} GWh" for v in credit_exp.values],
        textposition="outside",
    ))
    fig_credit.update_layout(
        height=300, yaxis_title="Exposure (GWh)",
        font=dict(family="DM Sans"), margin=dict(t=20),
    )
    st.plotly_chart(fig_credit, use_container_width=True)

    # Imbalance risk gauge
    st.subheader("Imbalance Risk Monitor")
    max_daily_imb = _CFG.get("risk_limits", {}).get("max_daily_imbalance_cost_eur", 50000)
    current_imb = 18500  # demo value

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_imb,
        delta={"reference": max_daily_imb, "decreasing": {"color": "#27ae60"}},
        gauge={
            "axis": {"range": [0, max_daily_imb * 1.5]},
            "bar": {"color": "#3498db"},
            "steps": [
                {"range": [0, max_daily_imb * 0.6], "color": "#d4edda"},
                {"range": [max_daily_imb * 0.6, max_daily_imb], "color": "#fff3cd"},
                {"range": [max_daily_imb, max_daily_imb * 1.5], "color": "#f8d7da"},
            ],
            "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.8, "value": max_daily_imb},
        },
        title={"text": "Daily Imbalance Cost (EUR)"},
        number={"prefix": "€", "valueformat": ",.0f"},
    ))
    fig_gauge.update_layout(height=300, font=dict(family="DM Sans"), margin=dict(t=60, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: Portfolio Analytics
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Portfolio Analytics")

    # Solar share distribution
    st.markdown("**Solar Share Distribution Across Contracts**")
    fig_solar = go.Figure(go.Bar(
        x=portfolio["id"],
        y=portfolio["solar_share"],
        marker_color=["#f39c12" if s < 30 else "#27ae60" for s in portfolio["solar_share"]],
        text=[f"{s}%" for s in portfolio["solar_share"]],
        textposition="outside",
    ))
    fig_solar.add_hline(y=portfolio["solar_share"].mean(), line_dash="dash", line_color="#2c3e50",
                        annotation_text=f"Portfolio Avg: {portfolio['solar_share'].mean():.0f}%")
    fig_solar.update_layout(
        height=300, yaxis_title="Solar Share (%)",
        font=dict(family="DM Sans"), margin=dict(t=20),
    )
    st.plotly_chart(fig_solar, use_container_width=True)

    # PV mechanism breakdown
    col_mech, col_timeline = st.columns(2)
    with col_mech:
        mech_vol = portfolio.groupby("pv_mechanism")["volume_mwh"].sum()
        fig_mech = go.Figure(go.Pie(
            labels=mech_vol.index, values=mech_vol.values,
            hole=0.5, textinfo="label+percent",
            marker=dict(colors=["#2c3e50", "#3498db", "#e67e22"]),
        ))
        fig_mech.update_layout(height=280, title="Volume by PV Mechanism", font=dict(family="DM Sans", size=10), margin=dict(t=40, b=10))
        st.plotly_chart(fig_mech, use_container_width=True)

    with col_timeline:
        # Contract timeline (Gantt-style)
        fig_gantt = go.Figure()
        colors_gantt = ["#2c3e50", "#3498db", "#e67e22", "#27ae60", "#9b59b6"]
        for i, row in portfolio.iterrows():
            fig_gantt.add_trace(go.Bar(
                x=[pd.Timestamp(row["end"]) - pd.Timestamp(row["start"])],
                y=[row["id"]],
                base=pd.Timestamp(row["start"]),
                orientation="h",
                marker_color=colors_gantt[i % len(colors_gantt)],
                name=row["customer"],
                text=f"{row['volume_mwh'] / 1000:.0f} GWh",
                textposition="inside",
                showlegend=False,
            ))
        fig_gantt.update_layout(
            height=280, title="Contract Timeline",
            xaxis_title="Date", font=dict(family="DM Sans", size=10),
            margin=dict(t=40, b=10, l=80),
            barmode="stack",
        )
        st.plotly_chart(fig_gantt, use_container_width=True)

    # Concentration analysis
    st.subheader("Concentration Analysis")
    portfolio_sorted = portfolio.sort_values("volume_mwh", ascending=False)
    cumulative_pct = portfolio_sorted["volume_mwh"].cumsum() / portfolio_sorted["volume_mwh"].sum() * 100

    fig_conc = go.Figure()
    fig_conc.add_trace(go.Bar(
        x=portfolio_sorted["customer"], y=portfolio_sorted["volume_mwh"] / 1000,
        name="Volume (GWh)", marker_color="#3498db",
    ))
    fig_conc.add_trace(go.Scatter(
        x=portfolio_sorted["customer"], y=cumulative_pct,
        name="Cumulative %", mode="lines+markers",
        line=dict(color="#e74c3c", width=2), yaxis="y2",
    ))
    fig_conc.update_layout(
        height=350,
        yaxis_title="Volume (GWh)",
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
        font=dict(family="DM Sans"),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=20),
    )
    st.plotly_chart(fig_conc, use_container_width=True)

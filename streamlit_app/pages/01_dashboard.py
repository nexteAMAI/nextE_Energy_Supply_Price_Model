"""Page 01: Dashboard — AI market commentary, KPI cards, 2x2 fundamental grid, waterfall."""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from components.shared import init_page, load_csv, load_parquet, load_kpis, load_contract_summary
import streamlit as st, plotly.graph_objects as go
import pandas as pd
import numpy as np

st.header("📊 Dashboard — Market Overview & KPIs")

DATA_DIR = init_page()

kpis = load_kpis()
if not kpis:
    st.warning("KPI data not found. Run Layer 1 pipeline first.")
    st.stop()

eur_ron = kpis.get("eur_ron_latest", 4.977)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Live Price Banner + AI Market Commentary
# ═══════════════════════════════════════════════════════════════════════════════
live_price = kpis.get("live_dam_latest_eur_mwh")
live_ts = kpis.get("live_dam_latest_timestamp", "")
if live_price:
    st.success(f"🔴 **LIVE DAM SPOT: €{live_price:.2f}/MWh** · {live_ts[:19]} · Source: EQ/Montel API")

# AI Market Commentary (auto-generated from available data)
monthly = load_csv("dam_monthly_summary.csv")
gen_monthly = load_csv("generation_monthly.csv")
imb = load_csv("imbalance_monthly_stats.csv")

def generate_market_commentary(kpis, monthly_df, gen_df):
    """Generate automated market narrative from available data."""
    lines = []

    # Price trend analysis
    dam_latest = kpis.get("dam_base_avg_latest_month", 0)
    dam_6m = kpis.get("trailing_6m_avg", 0)
    dam_12m = kpis.get("trailing_12m_avg", 0)

    if dam_latest and dam_6m:
        pct_vs_6m = (dam_latest - dam_6m) / dam_6m * 100 if dam_6m else 0
        if pct_vs_6m > 10:
            signal = "🔴 BULLISH"
            trend = f"Latest month base price (€{dam_latest:.1f}/MWh) is **{pct_vs_6m:+.1f}%** above the 6-month trailing average (€{dam_6m:.1f}), indicating upward price pressure."
        elif pct_vs_6m < -10:
            signal = "🟢 BEARISH"
            trend = f"Latest month base price (€{dam_latest:.1f}/MWh) is **{pct_vs_6m:+.1f}%** below the 6-month trailing average (€{dam_6m:.1f}), indicating downward price pressure."
        else:
            signal = "⚪ SIDEWAYS"
            trend = f"Latest month base price (€{dam_latest:.1f}/MWh) is **{pct_vs_6m:+.1f}%** vs. the 6-month average (€{dam_6m:.1f}), within normal range."
        lines.append(("Signal", signal))
        lines.append(("Price Trend", trend))

    # Peak-offpeak spread
    peak = kpis.get("dam_peak_avg_latest_month", 0)
    offpeak = kpis.get("dam_offpeak_avg_latest_month", 0)
    if peak and offpeak:
        spread = peak - offpeak
        lines.append(("Peak/Off-Peak", f"Peak-offpeak spread: €{spread:.1f}/MWh. {'Wide spread favors flexibility assets (BESS, hydro).' if spread > 30 else 'Narrow spread reduces arbitrage opportunity.'}"))

    # Generation mix insight
    if not gen_df.empty:
        latest_gen = gen_df.iloc[-1]
        solar_cols = [c for c in gen_df.columns if "solar" in c.lower() and "share" in c.lower()]
        wind_cols = [c for c in gen_df.columns if "wind" in c.lower() and "share" in c.lower()]
        nuclear_cols = [c for c in gen_df.columns if "nuclear" in c.lower() and "share" in c.lower()]

        mix_parts = []
        for cols, name in [(nuclear_cols, "Nuclear"), (wind_cols, "Wind"), (solar_cols, "Solar")]:
            if cols:
                val = latest_gen[cols[0]]
                mix_parts.append(f"{name} {val:.1f}%")
        if mix_parts:
            lines.append(("Generation Mix", f"Latest month: {', '.join(mix_parts)} of total generation."))

    # Imbalance cost insight
    imb_p50 = kpis.get("imbalance_cost_p50", 0)
    if imb_p50:
        lines.append(("Imbalance Risk", f"Imbalance cost P50: €{imb_p50:.1f}/MWh. {'Elevated — check forecast accuracy and consider hedging.' if imb_p50 > 5 else 'Within normal range.'}"))

    return lines

commentary = generate_market_commentary(kpis, monthly, gen_monthly)
if commentary:
    with st.container():
        st.subheader("🤖 AI Market Commentary")
        signal_line = [l for l in commentary if l[0] == "Signal"]
        if signal_line:
            st.markdown(f"**Market Signal: {signal_line[0][1]}**")

        for label, text in commentary:
            if label != "Signal":
                st.markdown(f"**{label}:** {text}")

        st.caption(f"Auto-generated from latest pipeline data · {kpis.get('last_updated', '')[:16]}")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: KPI Cards (compact 2-row layout)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Wholesale Market Snapshot")
c1, c2, c3, c4 = st.columns(4)
c1.metric("DAM Base (Latest Month)", f"€{kpis.get('dam_base_avg_latest_month',0):.2f}/MWh")
c2.metric("DAM Peak", f"€{kpis.get('dam_peak_avg_latest_month',0):.2f}/MWh")
c3.metric("Trailing 6M Avg", f"€{kpis.get('trailing_6m_avg',0):.2f}/MWh")
c4.metric("Trailing 12M Avg", f"€{kpis.get('trailing_12m_avg',0):.2f}/MWh")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Imbalance P50", f"€{kpis.get('imbalance_cost_p50',0):.2f}/MWh")
c6.metric("Imbalance P90", f"€{kpis.get('imbalance_cost_p90',0):.2f}/MWh")
c7.metric("EUR/RON", f"{eur_ron:.4f}")
c8.metric("Off-Peak Avg", f"€{kpis.get('dam_offpeak_avg_latest_month',0):.2f}/MWh")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: 2x2 Fundamental Chart Grid (Montel EQ-style)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Fundamental Drivers (Last 12 Months)")

dam_ts = load_parquet("streamlit_dam_timeseries.parquet")
gen_stack = load_parquet("streamlit_generation_stack.parquet")
cross = load_parquet("streamlit_cross_border.parquet")
imb_ts = load_parquet("streamlit_imbalance.parquet")

# Filter last 12 months
cutoff_12m = pd.Timestamp.now(tz="Europe/Bucharest") - pd.DateOffset(months=12)

grid_col1, grid_col2 = st.columns(2)

# Chart 1: DAM Price (daily average)
with grid_col1:
    if not dam_ts.empty:
        price_col = [c for c in dam_ts.columns if "EUR/MWh" in c or "Value" in c]
        if price_col:
            recent_dam = dam_ts[dam_ts.index >= cutoff_12m][price_col[0]].resample("D").mean()
            fig_dam = go.Figure()
            fig_dam.add_trace(go.Scatter(x=recent_dam.index, y=recent_dam.values,
                                          mode="lines", name="DAM Base",
                                          line=dict(color="#2c3e50", width=1.2)))
            # Add 30-day MA
            ma30 = recent_dam.rolling(30).mean()
            fig_dam.add_trace(go.Scatter(x=ma30.index, y=ma30.values,
                                          mode="lines", name="30d MA",
                                          line=dict(color="#e74c3c", width=1.5, dash="dot")))
            fig_dam.update_layout(height=280, title="DAM Spot Price", yaxis_title="EUR/MWh",
                                   font=dict(family="DM Sans", size=10),
                                   legend=dict(orientation="h", y=-0.2, font=dict(size=9)),
                                   margin=dict(t=35, b=40, l=50, r=10))
            st.plotly_chart(fig_dam, use_container_width=True)

# Chart 2: Generation mix (stacked area, monthly)
with grid_col2:
    if not gen_stack.empty:
        recent_gen = gen_stack[gen_stack.index >= cutoff_12m]
        gen_monthly_avg = recent_gen.resample("MS").mean()
        solar_col = [c for c in gen_monthly_avg.columns if "Solar" in c]
        wind_col = [c for c in gen_monthly_avg.columns if "Wind" in c]
        nuclear_col = [c for c in gen_monthly_avg.columns if "Nuclear" in c]
        hydro_cols = [c for c in gen_monthly_avg.columns if "Hydro" in c]

        fig_gen = go.Figure()
        gen_colors = {"Nuclear": "#FF6B35", "Solar": "#FFD166", "Wind": "#06D6A0"}
        for cols, name, color in [
            (nuclear_col, "Nuclear", "#FF6B35"),
            (hydro_cols, "Hydro", "#3A86FF"),
            (wind_col, "Wind", "#06D6A0"),
            (solar_col, "Solar", "#FFD166"),
        ]:
            for c in cols:
                fig_gen.add_trace(go.Scatter(
                    x=gen_monthly_avg.index, y=gen_monthly_avg[c],
                    name=name if c == cols[0] else None, stackgroup="one",
                    line=dict(width=0.5, color=color), fillcolor=color,
                    showlegend=(c == cols[0]),
                ))

        fig_gen.update_layout(height=280, title="Generation Mix", yaxis_title="MW",
                               font=dict(family="DM Sans", size=10),
                               legend=dict(orientation="h", y=-0.2, font=dict(size=9)),
                               margin=dict(t=35, b=40, l=50, r=10))
        st.plotly_chart(fig_gen, use_container_width=True)

grid_col3, grid_col4 = st.columns(2)

# Chart 3: Cross-border flows
with grid_col3:
    if not cross.empty:
        recent_cross = cross[cross.index >= cutoff_12m]
        net_cols = [c for c in recent_cross.columns if "net" in c.lower() or "import" in c.lower() or "export" in c.lower()]
        if net_cols:
            cross_monthly = recent_cross[net_cols[0]].resample("MS").mean()
            colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in cross_monthly.values]
            fig_cross = go.Figure(go.Bar(x=cross_monthly.index, y=cross_monthly.values,
                                          marker_color=colors))
            fig_cross.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_cross.update_layout(height=280, title="Net Import/Export", yaxis_title="MW",
                                     font=dict(family="DM Sans", size=10),
                                     margin=dict(t=35, b=40, l=50, r=10))
            st.plotly_chart(fig_cross, use_container_width=True)
        else:
            st.info("Cross-border flow data not available")
    else:
        st.info("Cross-border data not loaded")

# Chart 4: Imbalance costs
with grid_col4:
    if not imb_ts.empty:
        recent_imb = imb_ts[imb_ts.index >= cutoff_12m]
        price_cols = [c for c in recent_imb.columns if "price" in c.lower() or "eur" in c.lower()]
        if price_cols:
            imb_daily = recent_imb[price_cols[0]].resample("D").mean()
            fig_imb = go.Figure()
            fig_imb.add_trace(go.Scatter(x=imb_daily.index, y=imb_daily.values,
                                          mode="lines", name="Imbalance Price",
                                          line=dict(color="#9b59b6", width=1)))
            ma30_imb = imb_daily.rolling(30).mean()
            fig_imb.add_trace(go.Scatter(x=ma30_imb.index, y=ma30_imb.values,
                                          mode="lines", name="30d MA",
                                          line=dict(color="#e74c3c", width=1.5, dash="dot")))
            fig_imb.update_layout(height=280, title="Imbalance Price Trend", yaxis_title="EUR/MWh",
                                   font=dict(family="DM Sans", size=10),
                                   legend=dict(orientation="h", y=-0.2, font=dict(size=9)),
                                   margin=dict(t=35, b=40, l=50, r=10))
            st.plotly_chart(fig_imb, use_container_width=True)
        else:
            st.info("Imbalance price columns not found")
    else:
        st.info("Imbalance data not loaded")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Contract Register Summary
# ═══════════════════════════════════════════════════════════════════════════════
contracts = load_contract_summary()
if contracts:
    st.subheader("📋 Contract Register Summary (Layer 2)")
    st.caption("Source: Excel CONTRACT_REGISTER sheet — summary-level export only")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Hedging Ratio", f"{contracts.get('hedging_ratio_pct', 0):.1f}%")
    k2.metric("Contracted Volume", f"{contracts.get('total_contracted_volume_mwh', 0):,.0f} MWh/mo")
    k3.metric("Wtd PPA Price", f"€{contracts.get('portfolio_weighted_price_eur', 0):.2f}/MWh")
    k4.metric("Active Contracts", f"{contracts.get('contract_count_active', 0)}")

    col_type, col_fuel = st.columns(2)
    with col_type:
        bt = contracts.get("breakdown_by_type", {})
        if bt:
            fig_bt = go.Figure(go.Pie(
                labels=list(bt.keys()), values=list(bt.values()),
                hole=0.5, textinfo="label+percent",
                marker=dict(colors=["#2c3e50", "#3498db", "#e67e22", "#27ae60", "#9b59b6"]),
            ))
            fig_bt.update_layout(height=300, title="By Contract Type", font=dict(family="DM Sans", size=10), margin=dict(t=40, b=10))
            st.plotly_chart(fig_bt, use_container_width=True)

    with col_fuel:
        bf = contracts.get("breakdown_by_fuel", {})
        if bf:
            fig_bf = go.Figure(go.Pie(
                labels=list(bf.keys()), values=list(bf.values()),
                hole=0.5, textinfo="label+percent",
                marker=dict(colors=["#004E98", "#FF6B35", "#06D6A0", "#FFD166", "#95a5a6"]),
            ))
            fig_bf.update_layout(height=300, title="By Fuel Source", font=dict(family="DM Sans", size=10), margin=dict(t=40, b=10))
            st.plotly_chart(fig_bf, use_container_width=True)

    residual = contracts.get("residual_unhedged_volume_mwh", 0)
    total = contracts.get("total_delivery_obligation_mwh", 1)
    if residual > 0:
        st.info(f"⚠️ **Residual unhedged exposure:** {residual:,.0f} MWh/month ({residual/total*100:.1f}% of delivery obligation) — priced at DAM spot forecast")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Supply Price Build-Up Waterfall
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Supply Price Build-Up Waterfall (EUR/MWh)")
st.caption("Commercial MV customer · DEER 20kV · Mode A procurement weights")

dam_base = kpis.get("trailing_6m_avg", 110)

components = {
    "Wholesale\n(blended)": round(dam_base * 0.95, 1),
    "Imbalance": 2.5,
    "Profile\ncost": 2.5,
    "Transmission\n(TG)": round(7.28 / eur_ron, 1),
    "Distribution\n(TDc)": round(42.1 / eur_ron, 1),
    "System\nservices": round(6.62 / eur_ron, 1),
    "Cogeneration": round(13.6 / eur_ron, 1),
    "CfD": round(0.206 / eur_ron, 2),
    "Green\ncerts": round(72.54 / eur_ron, 1),
    "Excise": round(2.5 / eur_ron, 1),
    "Risk\npremium": 11.5,
    "Supplier\nmargin": 4.5,
}

names = list(components.keys()) + ["Subtotal\n(ex-VAT)"]
values = list(components.values())
subtotal = sum(values)

fig = go.Figure(go.Waterfall(
    orientation="v",
    measure=["relative"] * len(values) + ["total"],
    x=names,
    y=values + [subtotal],
    connector={"line": {"color": "rgba(100,100,100,0.3)", "width": 1}},
    increasing={"marker": {"color": "#e74c3c"}},
    decreasing={"marker": {"color": "#27ae60"}},
    totals={"marker": {"color": "#2c3e50", "line": {"color": "#e74c3c", "width": 2}}},
    text=[f"€{v:.1f}" for v in values] + [f"€{subtotal:.1f}"],
    textposition="outside",
    textfont=dict(size=10),
))

vat = subtotal * 0.21
total_price = subtotal + vat

fig.update_layout(
    height=450, showlegend=False,
    yaxis_title="EUR/MWh",
    font=dict(family="DM Sans"),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=30, b=80),
    annotations=[
        dict(x=0.95, y=0.95, xref="paper", yref="paper", showarrow=False,
             text=f"<b>+ VAT 21%: €{vat:.1f}</b><br><b>TOTAL: €{total_price:.1f}/MWh</b><br>(~{total_price*eur_ron:.0f} RON/MWh)",
             font=dict(size=13, color="#e74c3c"), align="right",
             bordercolor="#e74c3c", borderwidth=1, borderpad=8, bgcolor="rgba(255,255,255,0.9)")
    ]
)
st.plotly_chart(fig, use_container_width=True)

# --- Monthly DAM Trend (moved to bottom) ---
if not monthly.empty:
    with st.expander("📈 DAM Monthly Price Trend (Last 36 Months)", expanded=False):
        last_36 = monthly.tail(36)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=last_36.index, y=last_36["base_avg"], name="Base", line=dict(color="#2c3e50", width=2)))
        fig2.add_trace(go.Scatter(x=last_36.index, y=last_36["peak_avg"], name="Peak", line=dict(color="#e74c3c", width=1.5, dash="dot")))
        fig2.add_trace(go.Scatter(x=last_36.index, y=last_36["offpeak_avg"], name="Off-Peak", line=dict(color="#3498db", width=1.5, dash="dot")))
        fig2.update_layout(height=350, yaxis_title="EUR/MWh", font=dict(family="DM Sans"),
                            legend=dict(orientation="h", y=-0.15), margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

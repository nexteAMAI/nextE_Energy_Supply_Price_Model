"""Page 03: Forward Curve — Aurora forecast vs market, SRMC analysis, capture prices."""
import json, sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from components.shared import init_page, load_csv, load_parquet, load_kpis
import streamlit as st, pandas as pd, numpy as np, plotly.graph_objects as go
from plotly.subplots import make_subplots

st.header("📉 Forward Curve & Fundamental Value")
st.caption("Aurora Oct-2025 Forecast · SRMC Analysis · Capture Price Trends · Market Comparison")
DATA_DIR = init_page()

aurora = load_csv("aurora_forecast.csv")
if aurora.empty:
    st.warning("Aurora forecast not found."); st.stop()

kpis = load_kpis()
eur_ron = kpis.get("eur_ron_latest", 4.977)

# ─── Resolution & Horizon Controls ────────────────────────────────────────────
ctrl_cols = st.columns([2, 2, 2, 2, 2])
with ctrl_cols[0]:
    resolution = st.selectbox("Resolution", ["Monthly", "Quarterly", "Yearly"], index=0)
with ctrl_cols[1]:
    horizon = st.slider("Horizon (years)", 2, 35, 10)
with ctrl_cols[2]:
    scenario = st.selectbox("Scenario", ["Central", "Low", "High"], index=0)
with ctrl_cols[3]:
    currency = st.selectbox("Currency", ["EUR", "RON"], index=0)
with ctrl_cols[4]:
    nominal_real = st.selectbox("Price basis", ["Nominal"], index=0)

end_date = pd.Timestamp("2026-01-01") + pd.DateOffset(years=horizon)
aurora_f = aurora[aurora.index <= end_date].copy()

# Apply currency conversion
fx_mult = eur_ron if currency == "RON" else 1.0
unit = f"{currency}/MWh"

# ─── Tab Structure ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Baseload Forecast",
    "📊 Forward Curve vs. Fundamental",
    "☀️ Capture Prices",
    "⚡ SRMC & Spreads",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Baseload Forecast (enhanced from original)
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    central = [c for c in aurora_f.columns if "Baseload_Central" in c and "Nominal" in c]
    low = [c for c in aurora_f.columns if "Baseload_Low" in c and "Nominal" in c]
    high = [c for c in aurora_f.columns if "Baseload_High" in c and "Nominal" in c]

    if central:
        fig = go.Figure()

        # Confidence band (Low-High)
        if high and low:
            fig.add_trace(go.Scatter(
                x=aurora_f.index, y=aurora_f[high[0]] * fx_mult,
                name="High", line=dict(color="rgba(231,76,60,0.3)", width=0),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=aurora_f.index, y=aurora_f[low[0]] * fx_mult,
                name="Low-High Band", line=dict(color="rgba(39,174,96,0.3)", width=0),
                fill="tonexty", fillcolor="rgba(52,152,219,0.12)",
            ))

        # Central line
        fig.add_trace(go.Scatter(
            x=aurora_f.index, y=aurora_f[central[0]] * fx_mult,
            name="Central", line=dict(color="#2c3e50", width=2.5),
        ))

        # High/Low dashed lines
        if high:
            fig.add_trace(go.Scatter(
                x=aurora_f.index, y=aurora_f[high[0]] * fx_mult,
                name="High", line=dict(color="#e74c3c", width=1, dash="dot"),
            ))
        if low:
            fig.add_trace(go.Scatter(
                x=aurora_f.index, y=aurora_f[low[0]] * fx_mult,
                name="Low", line=dict(color="#27ae60", width=1, dash="dot"),
            ))

        fig.update_layout(
            height=500, yaxis_title=f"{unit} (nominal)",
            title="Aurora Oct 2025 — Romania Baseload Power Price Forecast",
            font=dict(family="DM Sans"), legend=dict(orientation="h", y=-0.12),
            margin=dict(t=50), hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Near-term summary table with resolution control
        st.subheader(f"Price Forecast ({resolution})")

        if resolution == "Monthly":
            display_df = aurora_f.copy()
            display_df["Period"] = display_df.apply(
                lambda r: f"{r.get('Month_Name', '')[:3]}-{int(r.get('Calendar_Year', 0))}" if 'Month_Name' in r.index else "", axis=1
            )
        elif resolution == "Quarterly":
            display_df = aurora_f.copy()
            display_df["_Q"] = display_df["Quarter"].astype(str).str.replace("Q", "", regex=False)
            display_df["_Y"] = display_df["Calendar_Year"]
            num_cols = [c for c in display_df.columns if "Nominal" in c and "€" in c]
            display_df = display_df.groupby(["_Y", "_Q"])[num_cols].mean().reset_index()
            display_df["Period"] = display_df.apply(lambda r: f"Q{r['_Q']}-{int(r['_Y'])}", axis=1)
            display_df.index = range(len(display_df))
        else:  # Yearly
            display_df = aurora_f.copy()
            display_df["_Y"] = display_df["Calendar_Year"]
            num_cols = [c for c in display_df.columns if "Nominal" in c and "€" in c]
            display_df = display_df.groupby("_Y")[num_cols].mean().reset_index()
            display_df["Period"] = display_df["_Y"].astype(int).astype(str)
            display_df.index = range(len(display_df))

        if central and not display_df.empty:
            tbl = pd.DataFrame()
            tbl["Period"] = display_df["Period"]
            tbl[f"Baseload Central ({unit})"] = (display_df[central[0]] * fx_mult).round(2)
            if low:
                tbl[f"Baseload Low ({unit})"] = (display_df[low[0]] * fx_mult).round(2)
            if high:
                tbl[f"Baseload High ({unit})"] = (display_df[high[0]] * fx_mult).round(2)
            if low and high:
                tbl["Spread (High-Low)"] = (tbl[f"Baseload High ({unit})"] - tbl[f"Baseload Low ({unit})"]).round(2)

            st.dataframe(
                tbl.set_index("Period").head(36).style.format("{:.2f}"),
                use_container_width=True, height=400,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Forward Curve vs. Fundamental Value (NEW — Montel EQ-style)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Forward Curve vs. Fundamental Forecast")
    st.caption("Compares Aurora fundamental forecast (Central) against benchmark market forward prices to identify over/under-valued periods")

    central_col = [c for c in aurora.columns if "Baseload_Central" in c and "Nominal" in c]
    if not central_col:
        st.warning("Central forecast column not found."); st.stop()

    # Build forward curve comparison table
    # Group Aurora data by contract periods (Monthly, Quarterly, Yearly)
    aurora_full = aurora[aurora.index >= "2026-03-01"].copy()
    aurora_full = aurora_full[aurora_full.index <= end_date]

    # Monthly contracts (next 12 months)
    monthly = aurora_full.head(12).copy()
    monthly["Contract"] = monthly.apply(
        lambda r: f"{r.get('Month_Name', '')[:3]}-{int(r.get('Calendar_Year', 0))}", axis=1
    )
    monthly["Type"] = "Monthly"
    monthly["Fundamental"] = monthly[central_col[0]]

    # Quarterly contracts
    aurora_q = aurora_full.copy()
    aurora_q["_Q"] = aurora_q["Quarter"].astype(str).str.replace("Q", "", regex=False)
    aurora_q["_Y"] = aurora_q["Calendar_Year"]
    quarterly = aurora_q.groupby(["_Y", "_Q"]).agg({central_col[0]: "mean"}).reset_index()
    quarterly["Contract"] = quarterly.apply(lambda r: f"Q{r['_Q']}-{int(r['_Y'])}", axis=1)
    quarterly["Type"] = "Quarterly"
    quarterly["Fundamental"] = quarterly[central_col[0]]

    # Yearly contracts
    aurora_y = aurora_full.copy()
    aurora_y["_Y"] = aurora_y["Calendar_Year"]
    yearly = aurora_y.groupby("_Y").agg({central_col[0]: "mean"}).reset_index()
    yearly["Contract"] = yearly["_Y"].astype(int).astype(str)
    yearly["Type"] = "Yearly"
    yearly["Fundamental"] = yearly[central_col[0]]

    # Market benchmark forward prices — loaded from EQ/Montel EEX forward curve data
    # Updated automatically by live_refresh.py; falls back to static defaults if unavailable
    market_forwards = {}
    fwd_source = "hardcoded defaults"
    fwd_file = DATA_DIR / "forward_prices.json"
    if fwd_file.exists():
        try:
            with open(fwd_file) as _f:
                fwd_data = json.load(_f)
            market_forwards = fwd_data.get("prices", {})
            fwd_source = fwd_data.get("last_updated", "unknown date")
        except Exception:
            market_forwards = {}

    if not market_forwards:
        # Fallback static values (last known good — updated 2026-03-24)
        market_forwards = {
            "Apr-2026": 102.82, "May-2026": 96.03, "Jun-2026": 112.67,
            "Jul-2026": 130.86, "Aug-2026": 125.00, "Sep-2026": 105.00,
            "Oct-2026": 109.33, "Nov-2026": 130.00, "Dec-2026": 145.00,
            "Jan-2027": 155.00, "Feb-2027": 140.00,
            "Q2-2026": 103.75, "Q3-2026": 134.30, "Q4-2026": 157.85,
            "Q1-2027": 146.39, "2027": 112.52, "2028": 90.07,
            "2029": 84.37, "2030": 82.12,
        }
        fwd_source = "static fallback (2026-03-24)"

    st.caption(f"_Market forward source: {fwd_source}_")

    # Build comparison table
    rows = []

    # Monthly rows
    for _, r in monthly.iterrows():
        contract = r["Contract"]
        fund = r["Fundamental"]
        market = market_forwards.get(contract, None)
        diff = (fund - market) if market else None
        pct_diff = ((fund - market) / market * 100) if market else None
        rows.append({
            "Contract": contract,
            "Type": "Monthly",
            f"Fundamental ({unit})": round(fund * fx_mult, 2),
            f"Market Forward ({unit})": round(market * fx_mult, 2) if market else None,
            f"Difference ({unit})": round(diff * fx_mult, 2) if diff is not None else None,
            "Diff %": round(pct_diff, 1) if pct_diff is not None else None,
            "Signal": "🟢 Undervalued" if (diff is not None and diff < -3) else ("🔴 Overvalued" if (diff is not None and diff > 3) else "⚪ Fair value") if diff is not None else "—",
        })

    # Quarterly rows
    for _, r in quarterly.iterrows():
        contract = r["Contract"]
        fund = r["Fundamental"]
        market = market_forwards.get(contract, None)
        diff = (fund - market) if market else None
        pct_diff = ((fund - market) / market * 100) if market else None
        rows.append({
            "Contract": contract,
            "Type": "Quarterly",
            f"Fundamental ({unit})": round(fund * fx_mult, 2),
            f"Market Forward ({unit})": round(market * fx_mult, 2) if market else None,
            f"Difference ({unit})": round(diff * fx_mult, 2) if diff is not None else None,
            "Diff %": round(pct_diff, 1) if pct_diff is not None else None,
            "Signal": "🟢 Undervalued" if (diff is not None and diff < -3) else ("🔴 Overvalued" if (diff is not None and diff > 3) else "⚪ Fair value") if diff is not None else "—",
        })

    # Yearly rows
    for _, r in yearly.head(6).iterrows():
        contract = r["Contract"]
        fund = r["Fundamental"]
        market = market_forwards.get(contract, None)
        diff = (fund - market) if market else None
        pct_diff = ((fund - market) / market * 100) if market else None
        rows.append({
            "Contract": contract,
            "Type": "Yearly",
            f"Fundamental ({unit})": round(fund * fx_mult, 2),
            f"Market Forward ({unit})": round(market * fx_mult, 2) if market else None,
            f"Difference ({unit})": round(diff * fx_mult, 2) if diff is not None else None,
            "Diff %": round(pct_diff, 1) if pct_diff is not None else None,
            "Signal": "🟢 Undervalued" if (diff is not None and diff < -3) else ("🔴 Overvalued" if (diff is not None and diff > 3) else "⚪ Fair value") if diff is not None else "—",
        })

    comparison_df = pd.DataFrame(rows)

    # Filter by contract type
    type_filter = st.multiselect("Contract type", ["Monthly", "Quarterly", "Yearly"], default=["Monthly", "Quarterly", "Yearly"])
    filtered = comparison_df[comparison_df["Type"].isin(type_filter)]

    # KPI summary
    has_market = filtered[f"Difference ({unit})"].notna()
    if has_market.any():
        avg_diff = filtered.loc[has_market, f"Difference ({unit})"].mean()
        max_under = filtered.loc[has_market, f"Difference ({unit})"].min()
        max_over = filtered.loc[has_market, f"Difference ({unit})"].max()

        k1, k2, k3 = st.columns(3)
        k1.metric("Avg Fundamental vs Market", f"{avg_diff:+.1f} {unit}",
                   delta=f"{'Fundamental below' if avg_diff < 0 else 'Fundamental above'} market")
        k2.metric("Max Undervaluation", f"{max_under:+.1f} {unit}")
        k3.metric("Max Overvaluation", f"{max_over:+.1f} {unit}")

    # Display table with conditional formatting
    st.dataframe(
        filtered.drop(columns=["Type"]).set_index("Contract"),
        use_container_width=True, height=500,
    )

    # Visual comparison chart
    chart_data = filtered[filtered[f"Market Forward ({unit})"].notna()].copy()
    if not chart_data.empty:
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=chart_data["Contract"], y=chart_data[f"Fundamental ({unit})"],
            name="Aurora Fundamental", marker_color="#3498db", opacity=0.8,
        ))
        fig_comp.add_trace(go.Bar(
            x=chart_data["Contract"], y=chart_data[f"Market Forward ({unit})"],
            name="Market Forward", marker_color="#e74c3c", opacity=0.8,
        ))
        fig_comp.update_layout(
            barmode="group", height=450,
            yaxis_title=f"{unit}", font=dict(family="DM Sans"),
            legend=dict(orientation="h", y=-0.15),
            margin=dict(t=30), hovermode="x unified",
            title="Fundamental Forecast vs. Market Forward Prices",
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Difference chart (bar showing over/under-valuation)
        diff_colors = ["#27ae60" if v < 0 else "#e74c3c" for v in chart_data[f"Difference ({unit})"]]
        fig_diff = go.Figure(go.Bar(
            x=chart_data["Contract"], y=chart_data[f"Difference ({unit})"],
            marker_color=diff_colors,
            text=[f"{v:+.1f}" for v in chart_data[f"Difference ({unit})"]],
            textposition="outside",
        ))
        fig_diff.update_layout(
            height=300, yaxis_title=f"Difference ({unit})",
            title="Fundamental - Market (negative = market overvalued vs. fundamentals)",
            font=dict(family="DM Sans"), margin=dict(t=50),
        )
        fig_diff.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_diff, use_container_width=True)

    st.info("""
    **How to read this table:**
    - **🟢 Undervalued** = Market forward is higher than Aurora fundamental → market is overpricing risk, selling opportunity
    - **🔴 Overvalued** = Market forward is lower than Aurora fundamental → market is underpricing, buying opportunity
    - **⚪ Fair value** = Within ±3 EUR/MWh band

    *Note: Market forward prices are benchmark references from OPCOM/broker data. Update regularly for live comparison.*
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Capture Prices (NEW — inspired by Montel EQ)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Technology Capture Prices vs. Baseload")
    st.caption("What renewable technologies actually earn relative to baseload — critical for PPA pricing and Route-to-Market")

    # Resolution for capture price view
    cp_resolution = st.radio("Resolution", ["Monthly", "Quarterly", "Yearly"], horizontal=True, key="cp_res")

    base_col = [c for c in aurora.columns if "Baseload_Central" in c and "Nominal" in c]
    wind_col = [c for c in aurora.columns if "Onshore_Wind_Curtailed_Central" in c and "Nominal" in c]
    solar_fixed_col = [c for c in aurora.columns if "Fixed_Solar_PV_Curtailed_Central" in c and "Nominal" in c]
    solar_track_col = [c for c in aurora.columns if "Tracking_Solar_PV_Curtailed_Central" in c and "Nominal" in c]
    offshore_col = [c for c in aurora.columns if "Offshore_Wind_Curtailed_Central" in c and "Nominal" in c]

    aurora_cp = aurora[aurora.index <= end_date].copy()

    # Aggregate by resolution
    if cp_resolution == "Quarterly":
        aurora_cp["_grp"] = aurora_cp["Quarter"].astype(str) + "-" + aurora_cp["Calendar_Year"].astype(int).astype(str)
        agg_cols = base_col + wind_col + solar_fixed_col + solar_track_col + offshore_col
        agg_cols = [c for c in agg_cols if c in aurora_cp.columns]
        aurora_cp = aurora_cp.groupby(["Calendar_Year", "Quarter"])[agg_cols].mean()
        aurora_cp["Period"] = [f"{q}-{int(y)}" for y, q in aurora_cp.index]
        aurora_cp = aurora_cp.reset_index(drop=True)
    elif cp_resolution == "Yearly":
        agg_cols = base_col + wind_col + solar_fixed_col + solar_track_col + offshore_col
        agg_cols = [c for c in agg_cols if c in aurora_cp.columns]
        aurora_cp = aurora_cp.groupby("Calendar_Year")[agg_cols].mean()
        aurora_cp["Period"] = aurora_cp.index.astype(int).astype(str)
        aurora_cp = aurora_cp.reset_index(drop=True)
    else:
        aurora_cp["Period"] = aurora_cp.apply(
            lambda r: f"{r.get('Month_Name', '')[:3]}-{int(r.get('Calendar_Year', 0))}", axis=1
        )

    # Chart 1: Absolute capture prices
    fig_cp = go.Figure()
    if base_col and base_col[0] in aurora_cp.columns:
        fig_cp.add_trace(go.Scatter(x=aurora_cp["Period"], y=aurora_cp[base_col[0]] * fx_mult,
                                     name="Baseload", line=dict(color="#2c3e50", width=2.5, dash="dash")))
    if wind_col and wind_col[0] in aurora_cp.columns:
        fig_cp.add_trace(go.Scatter(x=aurora_cp["Period"], y=aurora_cp[wind_col[0]] * fx_mult,
                                     name="Onshore Wind", line=dict(color="#3498db", width=2)))
    if solar_fixed_col and solar_fixed_col[0] in aurora_cp.columns:
        fig_cp.add_trace(go.Scatter(x=aurora_cp["Period"], y=aurora_cp[solar_fixed_col[0]] * fx_mult,
                                     name="Fixed Solar PV", line=dict(color="#f39c12", width=2)))
    if solar_track_col and solar_track_col[0] in aurora_cp.columns:
        fig_cp.add_trace(go.Scatter(x=aurora_cp["Period"], y=aurora_cp[solar_track_col[0]] * fx_mult,
                                     name="Tracking Solar PV", line=dict(color="#e67e22", width=2)))
    if offshore_col and offshore_col[0] in aurora_cp.columns:
        fig_cp.add_trace(go.Scatter(x=aurora_cp["Period"], y=aurora_cp[offshore_col[0]] * fx_mult,
                                     name="Offshore Wind", line=dict(color="#9b59b6", width=2)))

    fig_cp.update_layout(
        height=450, yaxis_title=f"Capture Price ({unit})",
        title=f"Technology Capture Prices — {scenario} Scenario ({cp_resolution})",
        font=dict(family="DM Sans"), legend=dict(orientation="h", y=-0.15),
        margin=dict(t=50), hovermode="x unified",
    )
    st.plotly_chart(fig_cp, use_container_width=True)

    # Chart 2: Capture rate as % of baseload
    st.subheader("Capture Rate (% of Baseload)")
    if base_col and base_col[0] in aurora_cp.columns:
        fig_pct = go.Figure()
        fig_pct.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="100% = Baseload")

        for col_list, name, color in [
            (wind_col, "Onshore Wind", "#3498db"),
            (solar_fixed_col, "Fixed Solar PV", "#f39c12"),
            (solar_track_col, "Tracking Solar PV", "#e67e22"),
            (offshore_col, "Offshore Wind", "#9b59b6"),
        ]:
            if col_list and col_list[0] in aurora_cp.columns:
                pct = (aurora_cp[col_list[0]] / aurora_cp[base_col[0]] * 100)
                fig_pct.add_trace(go.Scatter(
                    x=aurora_cp["Period"], y=pct, name=name,
                    line=dict(color=color, width=2),
                ))

        fig_pct.update_layout(
            height=400, yaxis_title="Capture Rate (% of Baseload)",
            title="Technology Capture Rate vs. Baseload",
            font=dict(family="DM Sans"), legend=dict(orientation="h", y=-0.15),
            margin=dict(t=50), hovermode="x unified",
        )
        st.plotly_chart(fig_pct, use_container_width=True)

        # Insight: average capture rates
        st.subheader("Average Capture Rates (Forecast Period)")
        cap_summary = {}
        for col_list, name in [
            (wind_col, "Onshore Wind"),
            (solar_fixed_col, "Fixed Solar PV"),
            (solar_track_col, "Tracking Solar PV"),
            (offshore_col, "Offshore Wind"),
        ]:
            if col_list and col_list[0] in aurora_cp.columns:
                avg_rate = (aurora_cp[col_list[0]] / aurora_cp[base_col[0]] * 100).mean()
                cap_summary[name] = round(avg_rate, 1)

        if cap_summary:
            cols = st.columns(len(cap_summary))
            for i, (tech, rate) in enumerate(cap_summary.items()):
                delta_text = f"{'Above' if rate > 100 else 'Below'} baseload"
                cols[i].metric(tech, f"{rate:.1f}%", delta=delta_text)

    # Curtailment rates
    curt_cols = {
        "Onshore Wind": [c for c in aurora.columns if "Onshore_Wind_Curtailment_Central" in c],
        "Fixed Solar PV": [c for c in aurora.columns if "Fixed_Solar_PV_Curtailment_Central" in c],
        "Tracking Solar PV": [c for c in aurora.columns if "Tracking_Solar_PV_Curtailment_Central" in c],
    }
    has_curt = any(v for v in curt_cols.values())

    if has_curt:
        with st.expander("📉 Curtailment Rates"):
            fig_curt = go.Figure()
            curt_colors = {"Onshore Wind": "#3498db", "Fixed Solar PV": "#f39c12", "Tracking Solar PV": "#e67e22"}
            for tech, cols in curt_cols.items():
                if cols and cols[0] in aurora_cp.columns:
                    fig_curt.add_trace(go.Scatter(
                        x=aurora_cp["Period"], y=aurora_cp[cols[0]],
                        name=tech, line=dict(color=curt_colors.get(tech, "#999"), width=2),
                    ))
            fig_curt.update_layout(
                height=350, yaxis_title="Curtailment (%)",
                font=dict(family="DM Sans"), legend=dict(orientation="h", y=-0.15),
                margin=dict(t=20),
            )
            st.plotly_chart(fig_curt, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: SRMC & Spreads (NEW — inspired by Montel EQ)
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Short-Run Marginal Cost & Clean Spreads")
    st.caption("Fundamental price support levels based on fuel costs, efficiency, and carbon — key price floor/ceiling indicators")

    # SRMC parameters (user-adjustable)
    with st.expander("⚙️ SRMC Parameters", expanded=False):
        p_cols = st.columns(4)
        with p_cols[0]:
            gas_price = st.number_input("Gas TTF (EUR/MWh)", value=42.0, step=1.0, key="gas_ttf")
            gas_eff = st.number_input("Gas efficiency (%)", value=55.0, step=1.0, key="gas_eff") / 100
        with p_cols[1]:
            coal_price_usd = st.number_input("Coal API2 (USD/t)", value=135.0, step=5.0, key="coal_api2")
            coal_eff = st.number_input("Coal efficiency (%)", value=42.0, step=1.0, key="coal_eff") / 100
        with p_cols[2]:
            carbon_price = st.number_input("EUA (EUR/t CO2)", value=65.0, step=1.0, key="eua")
            eur_usd = st.number_input("EUR/USD", value=1.08, step=0.01, key="eur_usd")
        with p_cols[3]:
            gas_co2 = st.number_input("Gas CO2 (t/MWhth)", value=0.202, step=0.001, format="%.3f", key="gas_co2")
            coal_co2 = st.number_input("Coal CO2 (t/MWhth)", value=0.341, step=0.001, format="%.3f", key="coal_co2")

    # Convert coal to EUR/MWh
    coal_eur_per_mwh = (coal_price_usd / eur_usd) / 8.141  # 8.141 MWh/t thermal content for API2

    # Calculate SRMC
    srmc_gas = gas_price / gas_eff + carbon_price * gas_co2 / gas_eff
    srmc_coal = coal_eur_per_mwh / coal_eff + carbon_price * coal_co2 / coal_eff

    # Build forward SRMC curves (assume fuel prices stay flat for simplicity)
    aurora_srmc = aurora_f.copy()
    if central_col:
        baseload = aurora_srmc[central_col[0]]
        clean_spark = baseload - srmc_gas
        clean_dark = baseload - srmc_coal

        # Summary KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("SRMC Gas", f"€{srmc_gas:.1f}/MWh", help="Short-run marginal cost of gas-fired generation")
        k2.metric("SRMC Coal", f"€{srmc_coal:.1f}/MWh", help="Short-run marginal cost of coal-fired generation")
        k3.metric("Clean Spark Spread (avg)", f"€{clean_spark.mean():.1f}/MWh",
                   delta=f"{'Positive' if clean_spark.mean() > 0 else 'Negative'} — gas {'in' if clean_spark.mean() > 0 else 'out of'} merit")
        k4.metric("Clean Dark Spread (avg)", f"€{clean_dark.mean():.1f}/MWh",
                   delta=f"{'Positive' if clean_dark.mean() > 0 else 'Negative'} — coal {'in' if clean_dark.mean() > 0 else 'out of'} merit")

        st.divider()

        # Chart: Power price vs SRMC lines
        fig_srmc = go.Figure()

        # SRMC bands
        fig_srmc.add_hline(y=srmc_gas * fx_mult, line_dash="dash", line_color="#e74c3c",
                           annotation_text=f"SRMC Gas: €{srmc_gas:.1f}", annotation_position="top left")
        fig_srmc.add_hline(y=srmc_coal * fx_mult, line_dash="dash", line_color="#6C584C",
                           annotation_text=f"SRMC Coal: €{srmc_coal:.1f}", annotation_position="bottom left")

        # Baseload price
        fig_srmc.add_trace(go.Scatter(
            x=aurora_srmc.index, y=baseload * fx_mult,
            name="Baseload Central", line=dict(color="#2c3e50", width=2.5),
        ))

        fig_srmc.update_layout(
            height=450, yaxis_title=f"{unit}",
            title="Baseload Power Price vs. SRMC Floor/Ceiling",
            font=dict(family="DM Sans"), legend=dict(orientation="h", y=-0.12),
            margin=dict(t=50), hovermode="x unified",
        )
        st.plotly_chart(fig_srmc, use_container_width=True)

        # Chart: Clean spreads over time
        fig_spreads = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     subplot_titles=["Clean Spark Spread (Power - Gas SRMC)", "Clean Dark Spread (Power - Coal SRMC)"],
                                     vertical_spacing=0.12)

        spark_colors = ["#27ae60" if v > 0 else "#e74c3c" for v in clean_spark]
        dark_colors = ["#27ae60" if v > 0 else "#e74c3c" for v in clean_dark]

        fig_spreads.add_trace(go.Bar(
            x=aurora_srmc.index, y=clean_spark * fx_mult,
            marker_color=spark_colors, name="Clean Spark", showlegend=False,
        ), row=1, col=1)
        fig_spreads.add_trace(go.Bar(
            x=aurora_srmc.index, y=clean_dark * fx_mult,
            marker_color=dark_colors, name="Clean Dark", showlegend=False,
        ), row=2, col=1)

        fig_spreads.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig_spreads.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        fig_spreads.update_layout(
            height=600, font=dict(family="DM Sans"), margin=dict(t=40),
        )
        fig_spreads.update_yaxes(title_text=f"{unit}", row=1, col=1)
        fig_spreads.update_yaxes(title_text=f"{unit}", row=2, col=1)
        st.plotly_chart(fig_spreads, use_container_width=True)

        st.info("""
        **How to read SRMC analysis:**
        - **Clean Spark Spread > 0** → Gas-fired plants are profitable → gas sets the marginal price → price floor
        - **Clean Dark Spread > 0** → Coal/lignite plants are profitable → coal in the merit order
        - When spreads are negative, the fuel is out-of-merit and does not set prices
        - The crossing point between baseload price and SRMC indicates fuel switching dynamics
        """)

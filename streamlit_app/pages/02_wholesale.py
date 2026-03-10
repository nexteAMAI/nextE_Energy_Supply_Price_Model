"""Page 02: Wholesale Analysis — DAM price history, IDM spread, statistics."""
import streamlit as st, pandas as pd, plotly.graph_objects as go, numpy as np
from pathlib import Path

st.header("📈 Wholesale Energy Market Analysis")
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

# --- Load data ---
dam_pq = DATA_DIR / "streamlit_dam_timeseries.parquet"
monthly_csv = DATA_DIR / "dam_monthly_summary.csv"
idm_csv = DATA_DIR / "idm_monthly_spread.csv"

if not dam_pq.exists():
    st.warning("DAM data not found."); st.stop()

dam = pd.read_parquet(dam_pq)
dam.index = pd.to_datetime(dam.index)
price_col = [c for c in dam.columns if "EUR/MWh" in c or "Value" in c][0]

# --- Date filter ---
tab1, tab2, tab3 = st.tabs(["📉 Price History", "📊 Monthly Statistics", "🔄 IDM Analysis"])

with tab1:
    col1, col2 = st.columns(2)
    years = sorted(dam.index.year.unique())
    year_range = col1.select_slider("Year range", options=years, value=(max(years[-5], years[0]), years[-1]))
    resolution = col2.selectbox("Resolution", ["Daily", "Weekly", "Monthly"], index=0)

    mask = (dam.index.year >= year_range[0]) & (dam.index.year <= year_range[1])
    filtered = dam[mask]

    if resolution == "Daily":
        plot_data = filtered[price_col].resample("D").mean()
    elif resolution == "Weekly":
        plot_data = filtered[price_col].resample("W").mean()
    else:
        plot_data = filtered[price_col].resample("MS").mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data.values,
                              mode="lines", name="DAM Price",
                              line=dict(color="#2c3e50", width=1.5)))

    # Add P10/P90 bands (rolling)
    if resolution in ["Daily", "Weekly"]:
        rolling = filtered[price_col].resample("D").mean()
        p90 = rolling.rolling(90, min_periods=30).quantile(0.90)
        p10 = rolling.rolling(90, min_periods=30).quantile(0.10)
        fig.add_trace(go.Scatter(x=p90.index, y=p90.values, mode="lines",
                                  name="P90 (90d)", line=dict(color="#e74c3c", width=0.8, dash="dot")))
        fig.add_trace(go.Scatter(x=p10.index, y=p10.values, mode="lines",
                                  name="P10 (90d)", line=dict(color="#27ae60", width=0.8, dash="dot"),
                                  fill="tonexty", fillcolor="rgba(39,174,96,0.08)"))

    fig.update_layout(height=450, yaxis_title="EUR/MWh", font=dict(family="DM Sans"),
                       legend=dict(orientation="h", y=-0.12), margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"€{plot_data.mean():.2f}")
    c2.metric("Median", f"€{plot_data.median():.2f}")
    c3.metric("Min", f"€{plot_data.min():.2f}")
    c4.metric("Max", f"€{plot_data.max():.2f}")

with tab2:
    if monthly_csv.exists():
        monthly = pd.read_csv(monthly_csv, index_col=0, parse_dates=True)
        last_24 = monthly.tail(24)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=last_24.index, y=last_24["base_avg"], name="Base",
                               marker_color="#2c3e50"))
        fig2.add_trace(go.Scatter(x=last_24.index, y=last_24["p90"], name="P90",
                                   line=dict(color="#e74c3c", dash="dot", width=1.5)))
        fig2.add_trace(go.Scatter(x=last_24.index, y=last_24["p10"], name="P10",
                                   line=dict(color="#27ae60", dash="dot", width=1.5)))
        fig2.update_layout(height=400, yaxis_title="EUR/MWh", barmode="group",
                            font=dict(family="DM Sans"), legend=dict(orientation="h", y=-0.15), margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Monthly Data Table")
        display_cols = ["base_avg", "peak_avg", "offpeak_avg", "p10", "p50", "p90", "min_price", "max_price", "count"]
        available = [c for c in display_cols if c in last_24.columns]
        st.dataframe(last_24[available].style.format("{:.2f}"), use_container_width=True, height=400)

with tab3:
    if idm_csv.exists():
        idm = pd.read_csv(idm_csv, index_col=0, parse_dates=True)
        st.subheader("IDM-DAM Spread (Monthly)")

        spread_cols = [c for c in idm.columns if "spread" in c.lower()]
        vwap_cols = [c for c in idm.columns if "vwap" in c.lower()]
        vol_cols = [c for c in idm.columns if "volume" in c.lower() and "total" in c.lower()]

        if vwap_cols:
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=idm.index, y=idm[spread_cols[0]] if spread_cols else [],
                                   name="IDM-DAM Spread",
                                   marker_color=["#e74c3c" if v > 0 else "#27ae60" for v in (idm[spread_cols[0]] if spread_cols else [])]))
            fig3.update_layout(height=350, yaxis_title="EUR/MWh (premium/discount)",
                                font=dict(family="DM Sans"), margin=dict(t=20))
            st.plotly_chart(fig3, use_container_width=True)

        if vol_cols:
            st.subheader("IDM Traded Volume (Monthly)")
            st.bar_chart(idm[vol_cols[0]].tail(24))
    else:
        st.info("IDM data not available.")

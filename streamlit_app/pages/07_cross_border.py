"""Page 07: Cross-Border — FBMC flow monitor, capacity utilization, price convergence."""
import streamlit as st, pandas as pd, plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.header("🌍 Cross-Border Flow Monitor")
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

cb_csv = DATA_DIR / "cross_border_monthly.csv"
cb_pq = DATA_DIR / "streamlit_cross_border.parquet"

tab1, tab2, tab3 = st.tabs(["📊 Monthly Flows", "📈 High-Resolution", "ℹ️ FBMC Context"])

with tab1:
    if cb_csv.exists():
        cb = pd.read_csv(cb_csv, index_col=0, parse_dates=True)
        n_months = st.slider("Months", 12, 120, 36, key="cb_m")
        last = cb.tail(n_months)

        border_cols = [c for c in last.columns if "MW" in c and "sum" not in c.lower()]
        sum_col = [c for c in last.columns if "sum" in c.lower()]

        if border_cols:
            fig = go.Figure()
            colors = {"BG": "#E63946", "HU": "#457B9D", "RS": "#2A9D8F", "UA": "#E9C46A"}
            for col in border_cols:
                border = col.split("[")[0].strip().split()[-1] if "[" in col else col[:2]
                fig.add_trace(go.Bar(
                    x=last.index, y=last[col], name=border,
                    marker_color=colors.get(border, "#999"),
                ))

            if sum_col:
                fig.add_trace(go.Scatter(
                    x=last.index, y=last[sum_col[0]], name="Net Total",
                    line=dict(color="#2c3e50", width=2.5), mode="lines",
                ))

            fig.update_layout(
                height=450, yaxis_title="Average MW (>0 = import into RO)",
                barmode="relative", font=dict(family="DM Sans"),
                legend=dict(orientation="h", y=-0.15), margin=dict(t=20),
                hovermode="x unified",
            )
            fig.add_hline(y=0, line_color="grey", line_width=0.5)
            st.plotly_chart(fig, use_container_width=True)

            # Summary table
            st.subheader("Annual Net Imports by Border (GWh)")
            annual = cb.resample("YS").mean()
            annual_display = annual[border_cols + sum_col].tail(5)
            # Convert MW avg to GWh (× hours in year / 1000)
            annual_gwh = annual_display * 8760 / 1000
            st.dataframe(annual_gwh.style.format("{:.0f}"), use_container_width=True)
    else:
        st.warning("Cross-border monthly data not found.")

with tab2:
    if cb_pq.exists():
        st.subheader("High-Resolution Cross-Border Flows")
        cb_hr = pd.read_parquet(cb_pq)
        cb_hr.index = pd.to_datetime(cb_hr.index)

        # Last 30 days
        last_30 = cb_hr[cb_hr.index >= cb_hr.index.max() - pd.Timedelta(days=30)]
        border_cols_hr = [c for c in last_30.columns if "MW" in c and "sum" not in c.lower()]

        if border_cols_hr:
            fig2 = go.Figure()
            colors = {"BG": "#E63946", "HU": "#457B9D", "RS": "#2A9D8F", "UA": "#E9C46A"}
            for col in border_cols_hr:
                border = col.split("[")[0].strip().split()[-1] if "[" in col else col[:2]
                fig2.add_trace(go.Scatter(
                    x=last_30.index, y=last_30[col], name=border,
                    mode="lines", line=dict(width=1, color=colors.get(border, "#999")),
                ))

            fig2.update_layout(height=400, yaxis_title="MW (>0 = import)",
                                font=dict(family="DM Sans"), legend=dict(orientation="h", y=-0.15),
                                margin=dict(t=20), hovermode="x unified")
            fig2.add_hline(y=0, line_color="grey", line_width=0.5)
            st.plotly_chart(fig2, use_container_width=True)

            # Hourly profile by border
            st.subheader("Average Hourly Import Profile (Last 30 Days)")
            hourly = last_30.copy()
            hourly["hour"] = hourly.index.hour
            hourly_avg = hourly.groupby("hour")[border_cols_hr].mean()
            fig3 = go.Figure()
            for col in border_cols_hr:
                border = col.split("[")[0].strip().split()[-1] if "[" in col else col[:2]
                fig3.add_trace(go.Scatter(x=hourly_avg.index, y=hourly_avg[col], name=border,
                                           mode="lines+markers", line=dict(color=colors.get(border, "#999"))))
            fig3.update_layout(height=350, xaxis_title="Hour of Day", yaxis_title="Avg MW",
                                font=dict(family="DM Sans"), legend=dict(orientation="h", y=-0.15), margin=dict(t=20))
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("High-resolution cross-border data not available.")

with tab3:
    st.markdown("""
    ### CORE Flow-Based Market Coupling (FBMC)

    Romania has been part of the **Single Day-Ahead Coupling (SDAC)** under the
    **CORE Flow-Based Market Coupling** framework since **8 June 2022**.

    **Key Romanian borders:**

    | Border | Direction | Typical Pattern |
    |--------|-----------|-----------------|
    | **RO → HU** | Export dominant | RO exports to Hungary during high-wind/hydro periods |
    | **RO → BG** | Bidirectional | Depends on relative hydro/nuclear availability |
    | **RO → RS** | Export dominant | Serbia imports during peak demand |
    | **RO → UA** | Import shifted | Ukraine became net importer post-2022, affecting regional dynamics |

    **Impact on DAM pricing:**
    - When cross-border capacity is constrained, RO DAM prices decouple from neighbouring zones
    - Price convergence with HU typically occurs in 40–60% of hours
    - FBMC increases available capacity vs. NTC-based allocation, improving price convergence

    **Data sources:** ENTSO-E physical flows + JAO auction results + CORE publication tool
    """)

"""Page 05: Balancing — Imbalance cost rolling 30-day, Long/Short spread, volume."""
import streamlit as st, pandas as pd, plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.header("⚖️ Balancing & Imbalance Cost Tracker")
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

imb_monthly = DATA_DIR / "imbalance_monthly_stats.csv"
imb_rolling = DATA_DIR / "imbalance_rolling_30d.csv"
imb_pq = DATA_DIR / "streamlit_imbalance.parquet"

tab1, tab2, tab3 = st.tabs(["📈 Monthly Cost", "🔄 Rolling 30-Day", "📊 Long/Short Spread"])

with tab1:
    if imb_monthly.exists():
        monthly = pd.read_csv(imb_monthly, index_col=0, parse_dates=True)
        n = st.slider("Months to show", 12, 120, 36, key="imb_m")
        last = monthly.tail(n)

        cost_cols = [c for c in last.columns if "cost" in c.lower()]
        spread_cols = [c for c in last.columns if "spread" in c.lower() and "avg" in c.lower()]

        if cost_cols:
            fig = go.Figure()
            for col in cost_cols:
                name = col.replace("imbalance_cost_", "").replace("_eur_mwh", " EUR/MWh").upper()
                fig.add_trace(go.Scatter(x=last.index, y=last[col], name=name, mode="lines+markers"))
            fig.update_layout(height=400, yaxis_title="EUR/MWh", font=dict(family="DM Sans"),
                               legend=dict(orientation="h", y=-0.15), margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

        if spread_cols:
            st.subheader("Long/Short Spread to DAM (Monthly Avg)")
            fig2 = go.Figure()
            for col in spread_cols[:3]:
                name = col.replace("avg_", "").replace("_spread", " spread").title()
                fig2.add_trace(go.Bar(x=last.index, y=last[col], name=name))
            fig2.update_layout(height=350, yaxis_title="EUR/MWh", barmode="group",
                                font=dict(family="DM Sans"), legend=dict(orientation="h", y=-0.15), margin=dict(t=20))
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Imbalance monthly stats not found.")

with tab2:
    if imb_rolling.exists():
        rolling = pd.read_csv(imb_rolling, index_col=0, parse_dates=True)
        last_365 = rolling.tail(365)

        cost_col = [c for c in last_365.columns if "rolling_imbalance_cost" in c]
        if cost_col:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=last_365.index, y=last_365[cost_col[0]],
                                       name="Rolling 30-day Cost", fill="tozeroy",
                                       line=dict(color="#e74c3c", width=2),
                                       fillcolor="rgba(231,76,60,0.15)"))
            fig3.add_hline(y=2.5, line_dash="dash", line_color="#27ae60",
                            annotation_text="Well-managed target (2.5 EUR/MWh)")
            fig3.update_layout(height=400, yaxis_title="EUR/MWh", font=dict(family="DM Sans"), margin=dict(t=20))
            st.plotly_chart(fig3, use_container_width=True)

        ls_cols = [c for c in last_365.columns if "spread" in c.lower()]
        if ls_cols:
            st.subheader("Rolling 30-Day Long/Short Spread")
            st.line_chart(last_365[ls_cols])
    else:
        st.info("Rolling imbalance data not available.")

with tab3:
    if imb_pq.exists():
        st.subheader("Imbalance Price Distribution (Last 12 Months)")
        imb = pd.read_parquet(imb_pq)
        imb.index = pd.to_datetime(imb.index)
        last_12m = imb[imb.index >= imb.index.max() - pd.Timedelta(days=365)]

        long_col = [c for c in last_12m.columns if "Long" in c]
        short_col = [c for c in last_12m.columns if "Short" in c]

        if long_col and short_col:
            fig4 = make_subplots(rows=1, cols=2, subplot_titles=["Long Price Distribution", "Short Price Distribution"])
            fig4.add_trace(go.Histogram(x=last_12m[long_col[0]].dropna(), nbinsx=80,
                                         marker_color="#3498db", name="Long"), row=1, col=1)
            fig4.add_trace(go.Histogram(x=last_12m[short_col[0]].dropna(), nbinsx=80,
                                         marker_color="#e74c3c", name="Short"), row=1, col=2)
            fig4.update_layout(height=400, font=dict(family="DM Sans"), showlegend=False, margin=dict(t=40))
            fig4.update_xaxes(title_text="EUR/MWh")
            fig4.update_yaxes(title_text="Count")
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Imbalance parquet data not available.")

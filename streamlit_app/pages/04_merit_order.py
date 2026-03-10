"""Page 04: Merit Order — Generation stack, capacity factors, residual demand."""
import streamlit as st, pandas as pd, plotly.graph_objects as go
from pathlib import Path

st.header("🏭 Generation Stack & Merit Order")
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

gen_csv = DATA_DIR / "generation_monthly.csv"
cf_csv = DATA_DIR / "capacity_factors_monthly.csv"

if not gen_csv.exists():
    st.warning("Generation data not found."); st.stop()

gen = pd.read_csv(gen_csv, index_col=0, parse_dates=True)

tab1, tab2, tab3 = st.tabs(["📊 Generation Mix", "⚡ Capacity Factors", "📋 Data Table"])

with tab1:
    st.subheader("Monthly Generation Mix (Stacked Area)")
    n_months = st.slider("Months to display", 12, 120, 36, key="gen_months")
    last_n = gen.tail(n_months)

    share_cols = [c for c in last_n.columns if "share_pct" in c]
    avg_cols = [c for c in last_n.columns if "avg_mw" in c and "total" not in c.lower()]

    if avg_cols:
        fuel_order = ["nuclear", "hydro_ror", "hydro_reservoir", "wind", "solar", "biomass", "gas", "lignite", "hard_coal", "storage"]
        colors = {
            "nuclear": "#FF6B35", "hydro_ror": "#004E98", "hydro_reservoir": "#3A86FF",
            "wind": "#06D6A0", "solar": "#FFD166", "biomass": "#8338EC",
            "gas": "#E63946", "lignite": "#6C584C", "hard_coal": "#463F3A", "storage": "#A7C957"
        }

        fig = go.Figure()
        for fuel in fuel_order:
            col = f"{fuel}_avg_mw"
            if col in last_n.columns:
                fig.add_trace(go.Scatter(
                    x=last_n.index, y=last_n[col], name=fuel.replace("_", " ").title(),
                    stackgroup="one", mode="lines",
                    line=dict(width=0.5, color=colors.get(fuel, "#999")),
                    fillcolor=colors.get(fuel, "#999"),
                ))

        fig.update_layout(
            height=500, yaxis_title="Average MW",
            font=dict(family="DM Sans"),
            legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
            margin=dict(t=20), hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Latest month pie chart
    if share_cols:
        latest = gen.iloc[-1]
        valid_shares = {c.replace("_share_pct", "").replace("_", " ").title(): latest[c]
                        for c in share_cols if latest[c] > 0.5}

        fig_pie = go.Figure(go.Pie(
            labels=list(valid_shares.keys()), values=list(valid_shares.values()),
            hole=0.45, textinfo="label+percent",
            marker=dict(colors=["#FF6B35", "#004E98", "#3A86FF", "#06D6A0", "#FFD166",
                                "#8338EC", "#E63946", "#6C584C", "#463F3A", "#A7C957"]),
        ))
        fig_pie.update_layout(height=380, title=f"Generation Mix — {gen.index[-1].strftime('%b %Y')}",
                               font=dict(family="DM Sans"), margin=dict(t=50))
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    if cf_csv.exists():
        cf = pd.read_csv(cf_csv, index_col=0, parse_dates=True)
        cf_last = cf.tail(24)
        cf_cols = [c for c in cf_last.columns if "cf_pct" in c]

        if cf_cols:
            fig_cf = go.Figure()
            for col in cf_cols:
                name = col.replace("_cf_pct", "").replace("_", " ").title()
                fig_cf.add_trace(go.Scatter(x=cf_last.index, y=cf_last[col], name=name, mode="lines+markers"))

            fig_cf.update_layout(height=400, yaxis_title="Capacity Factor (%)",
                                  font=dict(family="DM Sans"), legend=dict(orientation="h", y=-0.15), margin=dict(t=20))
            st.plotly_chart(fig_cf, use_container_width=True)
    else:
        st.info("Capacity factor data not available.")

with tab3:
    display = gen.tail(24)
    display_cols = [c for c in display.columns if "avg_mw" in c or "share_pct" in c or "total" in c]
    if display_cols:
        st.dataframe(display[display_cols].style.format("{:.1f}"), use_container_width=True, height=500)

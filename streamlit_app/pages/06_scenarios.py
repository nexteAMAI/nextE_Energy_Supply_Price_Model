"""Page 06: Scenarios — 3-scenario comparison + sensitivity tornado chart."""
import streamlit as st, pandas as pd, plotly.graph_objects as go, json
from plotly.subplots import make_subplots
from pathlib import Path

st.header("🔀 Scenario Comparison & Sensitivity Analysis")
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

kpi_path = DATA_DIR / "streamlit_kpis.json"
tornado_csv = DATA_DIR / "tornado_inputs.csv"

kpis = {}
if kpi_path.exists():
    with open(kpi_path) as f: kpis = json.load(f)

tab1, tab2 = st.tabs(["📊 3-Scenario Comparison", "🌪️ Sensitivity Tornado"])

with tab1:
    st.subheader("Price Build-Up Across Three Scenarios")
    st.caption("Base Case · High-Price Stress Test · Low-Price / Oversupply")

    eur_ron = kpis.get("eur_ron_latest", 4.977)
    base_dam = kpis.get("trailing_6m_avg", 119.64)
    dam_p90 = base_dam * 1.40
    dam_p10 = base_dam * 0.65

    # Regulated components (fixed across scenarios)
    tg = 7.28 / eur_ron
    tdc = 42.1 / eur_ron
    ss = 6.62 / eur_ron
    cogen = 13.6 / eur_ron
    cfd = 0.206 / eur_ron
    gc = 72.54 / eur_ron
    excise = 2.5 / eur_ron

    scenarios = {
        "Base Case": {"wholesale": base_dam * 0.95, "imbalance": 2.5, "profile": 2.5, "risk": 11.5, "margin": 4.5},
        "High-Price\nStress": {"wholesale": dam_p90 * 0.95, "imbalance": 5.0, "profile": 4.0, "risk": 16.0, "margin": 4.5},
        "Low-Price\nOversupply": {"wholesale": dam_p10 * 0.95, "imbalance": 1.5, "profile": 1.5, "risk": 8.0, "margin": 4.5},
    }

    fig = make_subplots(rows=1, cols=3, subplot_titles=list(scenarios.keys()), shared_yaxes=True)
    colors = ["#2c3e50", "#e74c3c", "#27ae60", "#3498db", "#f39c12", "#9b59b6",
              "#1abc9c", "#95a5a6", "#e67e22", "#34495e", "#8e44ad", "#16a085"]
    labels = ["Wholesale", "Imbalance", "Profile", "TG", "TDc", "SS",
              "Cogen", "CfD", "GC", "Excise", "Risk", "Margin"]

    for idx, (sname, svals) in enumerate(scenarios.items(), 1):
        vals = [svals["wholesale"], svals["imbalance"], svals["profile"],
                tg, tdc, ss, cogen, cfd, gc, excise, svals["risk"], svals["margin"]]
        subtotal = sum(vals)

        fig.add_trace(go.Waterfall(
            orientation="v", measure=["relative"] * len(vals) + ["total"],
            x=labels + ["Total"], y=vals + [subtotal],
            connector=dict(line=dict(color="rgba(100,100,100,0.2)")),
            increasing=dict(marker=dict(color="#e74c3c")),
            totals=dict(marker=dict(color="#2c3e50")),
            text=[f"€{v:.0f}" if v > 1 else "" for v in vals] + [f"€{subtotal:.0f}"],
            textposition="outside", textfont=dict(size=8),
            showlegend=False,
        ), row=1, col=idx)

    fig.update_layout(height=550, font=dict(family="DM Sans"), margin=dict(t=50, b=100))
    fig.update_yaxes(title_text="EUR/MWh", row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.subheader("Scenario Summary")
    summary = []
    for sname, svals in scenarios.items():
        vals = [svals["wholesale"], svals["imbalance"], svals["profile"],
                tg, tdc, ss, cogen, cfd, gc, excise, svals["risk"], svals["margin"]]
        sub = sum(vals)
        vat = sub * 0.21
        summary.append({
            "Scenario": sname.replace("\n", " "),
            "Wholesale (€)": f"{svals['wholesale']:.1f}",
            "Regulated (€)": f"{tg+tdc+ss+cogen+cfd+gc+excise:.1f}",
            "Risk+Margin (€)": f"{svals['risk']+svals['margin']:.1f}",
            "Subtotal (€)": f"{sub:.1f}",
            "Total incl. VAT (€)": f"{sub+vat:.1f}",
            "Total (RON)": f"{(sub+vat)*eur_ron:.0f}",
        })
    st.dataframe(pd.DataFrame(summary).set_index("Scenario"), use_container_width=True)

with tab2:
    st.subheader("Price Sensitivity Tornado Chart")
    st.caption("Impact on final price from ± shocks to key variables (Section 14.2)")

    if tornado_csv.exists():
        tornado = pd.read_csv(tornado_csv)

        fig_t = go.Figure()
        fig_t.add_trace(go.Bar(
            y=tornado["variable"], x=tornado["low_impact_eur_mwh"],
            name="Downside", orientation="h", marker_color="#27ae60",
            text=[f"{v:+.1f}" for v in tornado["low_impact_eur_mwh"]], textposition="outside",
        ))
        fig_t.add_trace(go.Bar(
            y=tornado["variable"], x=tornado["high_impact_eur_mwh"],
            name="Upside", orientation="h", marker_color="#e74c3c",
            text=[f"{v:+.1f}" for v in tornado["high_impact_eur_mwh"]], textposition="outside",
        ))
        fig_t.update_layout(
            height=400, barmode="overlay",
            xaxis_title="Impact on Final Price (EUR/MWh)",
            font=dict(family="DM Sans"),
            legend=dict(orientation="h", y=-0.15),
            margin=dict(l=200, t=20),
        )
        fig_t.add_vline(x=0, line_color="black", line_width=1)
        st.plotly_chart(fig_t, use_container_width=True)

        st.dataframe(tornado.style.format({
            "low_shock": "{:+.1f}", "high_shock": "{:+.1f}",
            "low_impact_eur_mwh": "{:+.2f}", "high_impact_eur_mwh": "{:+.2f}",
            "impact_range": "{:.2f}",
        }), use_container_width=True)

    # Price elasticity
    elast_csv = DATA_DIR / "price_elasticity.csv"
    if elast_csv.exists():
        st.subheader("Price Elasticity to Load Shifts")
        elast = pd.read_csv(elast_csv, index_col=0)

        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=elast.index, y=elast["avg_price_change_eur_mwh"],
            mode="lines+markers", name="Avg Price Change",
            line=dict(color="#2c3e50", width=2),
        ))
        fig_e.add_trace(go.Scatter(
            x=elast.index, y=elast["p10_price_change"],
            mode="lines", name="P10", line=dict(color="#27ae60", dash="dot"),
        ))
        fig_e.add_trace(go.Scatter(
            x=elast.index, y=elast["p90_price_change"],
            mode="lines", name="P90", line=dict(color="#e74c3c", dash="dot"),
            fill="tonexty", fillcolor="rgba(231,76,60,0.08)",
        ))
        fig_e.add_hline(y=0, line_color="grey", line_width=0.5)
        fig_e.update_layout(height=400, xaxis_title="Load Shift (MW)", yaxis_title="Price Change (EUR/MWh)",
                             font=dict(family="DM Sans"), legend=dict(orientation="h", y=-0.15), margin=dict(t=20))
        st.plotly_chart(fig_e, use_container_width=True)

"""Page 10: Procurement Strategy — Channel allocation, cost comparison, execution monitor."""
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

st.header("🏗️ Procurement Strategy & Execution")
st.caption("Channel Allocation · Cost Analysis · Market Depth · Execution Monitor")
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

channels_cfg = _CFG.get("procurement_channels", {})

# ---------------------------------------------------------------------------
# Channel reference data
# ---------------------------------------------------------------------------
CHANNELS = {
    "BRM Forward": {
        "operator": "Nord Pool", "settlement": "Financial", "liquidity": "Medium",
        "tx_cost": 0.75, "spread_vs_dam": 1.50, "collateral": 0.15,
        "color": "#2c3e50",
    },
    "OPCOM Bilateral": {
        "operator": "OPCOM S.A.", "settlement": "Physical", "liquidity": "High",
        "tx_cost": 0.50, "spread_vs_dam": 1.00, "collateral": 0.20,
        "color": "#3498db",
    },
    "Direct Bilateral": {
        "operator": "OTC", "settlement": "Physical", "liquidity": "Variable",
        "tx_cost": 1.25, "spread_vs_dam": 2.00, "collateral": 0.25,
        "color": "#e67e22",
    },
    "EEX Financial": {
        "operator": "EEX", "settlement": "Financial", "liquidity": "Very High",
        "tx_cost": 0.25, "spread_vs_dam": 15.0, "collateral": 0.10,
        "color": "#27ae60",
    },
    "Spot DAM": {
        "operator": "OPCOM PZU", "settlement": "Financial", "liquidity": "Very High",
        "tx_cost": 0.10, "spread_vs_dam": 0.0, "collateral": 0.05,
        "color": "#9b59b6",
    },
    "Spot IDM": {
        "operator": "OPCOM PI", "settlement": "Financial", "liquidity": "Medium",
        "tx_cost": 0.30, "spread_vs_dam": 1.50, "collateral": 0.08,
        "color": "#1abc9c",
    },
}

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Channel Comparison", "🎯 Optimization", "📈 Market Monitor", "📋 Execution Log"])

# ---------------------------------------------------------------------------
# Tab 1: Channel Comparison
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Procurement Channel Comparison")

    # Channel comparison table
    ch_data = []
    for name, info in CHANNELS.items():
        effective_cost = dam_base + info["spread_vs_dam"] + info["tx_cost"]
        ch_data.append({
            "Channel": name,
            "Operator": info["operator"],
            "Settlement": info["settlement"],
            "Liquidity": info["liquidity"],
            "Tx Cost (€/MWh)": info["tx_cost"],
            "Spread vs DAM (€)": info["spread_vs_dam"],
            "Collateral (%)": f"{info['collateral'] * 100:.0f}%",
            "Eff. Cost (€/MWh)": effective_cost,
        })
    df_ch = pd.DataFrame(ch_data).set_index("Channel")
    st.dataframe(
        df_ch.style.format({"Tx Cost (€/MWh)": "{:.2f}", "Spread vs DAM (€)": "{:.2f}", "Eff. Cost (€/MWh)": "{:.2f}"}),
        use_container_width=True,
    )

    st.divider()

    # Cost comparison bar chart
    ch_names = list(CHANNELS.keys())
    eff_costs = [dam_base + CHANNELS[ch]["spread_vs_dam"] + CHANNELS[ch]["tx_cost"] for ch in ch_names]
    ch_colors = [CHANNELS[ch]["color"] for ch in ch_names]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=ch_names, y=[dam_base] * len(ch_names),
        name="DAM Base Price", marker_color="rgba(44,62,80,0.3)",
    ))
    fig_comp.add_trace(go.Bar(
        x=ch_names, y=[CHANNELS[ch]["spread_vs_dam"] for ch in ch_names],
        name="Spread vs DAM", marker_color=[CHANNELS[ch]["color"] for ch in ch_names],
    ))
    fig_comp.add_trace(go.Bar(
        x=ch_names, y=[CHANNELS[ch]["tx_cost"] for ch in ch_names],
        name="Transaction Cost", marker_color="rgba(231,76,60,0.6)",
    ))

    # Add effective cost annotations
    for i, (name, cost) in enumerate(zip(ch_names, eff_costs)):
        fig_comp.add_annotation(x=name, y=cost + 2, text=f"€{cost:.1f}", showarrow=False,
                                font=dict(size=10, color="#2c3e50", family="JetBrains Mono"))

    fig_comp.update_layout(
        barmode="stack", height=420,
        yaxis_title="EUR/MWh",
        font=dict(family="DM Sans"),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=20),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Collateral comparison
    st.subheader("Collateral Requirements")
    annual_vol = 100_000  # Reference: 100 GWh portfolio
    col_req = []
    for ch in ch_names:
        info = CHANNELS[ch]
        cost = dam_base + info["spread_vs_dam"] + info["tx_cost"]
        collateral = cost * annual_vol * info["collateral"] / 1e6
        col_req.append(collateral)

    fig_coll = go.Figure(go.Bar(
        x=ch_names, y=col_req,
        marker_color=ch_colors,
        text=[f"€{c:.2f}M" for c in col_req],
        textposition="outside",
    ))
    fig_coll.update_layout(
        height=300, yaxis_title="Collateral (EUR million)",
        font=dict(family="DM Sans"), margin=dict(t=20),
    )
    st.plotly_chart(fig_coll, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Optimization
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Procurement Allocation Optimizer")

    # Input: gap to procure
    total_gap = st.number_input("Non-Solar Gap to Procure (MWh/year)", value=50_000, step=5000)

    st.markdown("**Allocation Strategy**")
    strategy = st.radio(
        "Optimization Objective",
        ["Minimize Cost", "Minimize Collateral", "Balance Cost & Risk"],
        horizontal=True,
    )

    # Run optimization (simplified greedy approach matching the processor logic)
    if st.button("🔄 Run Optimization", use_container_width=True):
        np.random.seed(99)

        # Simplified allocation logic
        if strategy == "Minimize Cost":
            weights = {"BRM Forward": 0.30, "OPCOM Bilateral": 0.35, "Direct Bilateral": 0.05,
                       "EEX Financial": 0.00, "Spot DAM": 0.25, "Spot IDM": 0.05}
        elif strategy == "Minimize Collateral":
            weights = {"BRM Forward": 0.15, "OPCOM Bilateral": 0.20, "Direct Bilateral": 0.05,
                       "EEX Financial": 0.05, "Spot DAM": 0.45, "Spot IDM": 0.10}
        else:
            weights = {"BRM Forward": 0.35, "OPCOM Bilateral": 0.30, "Direct Bilateral": 0.10,
                       "EEX Financial": 0.00, "Spot DAM": 0.20, "Spot IDM": 0.05}

        results = []
        total_cost = 0
        total_coll = 0
        for ch, w in weights.items():
            vol = total_gap * w
            if vol == 0:
                continue
            info = CHANNELS[ch]
            cost_per_mwh = dam_base + info["spread_vs_dam"] + info["tx_cost"]
            total_ch_cost = cost_per_mwh * vol
            coll = total_ch_cost * info["collateral"]
            total_cost += total_ch_cost
            total_coll += coll
            results.append({
                "Channel": ch,
                "Allocation (%)": f"{w * 100:.0f}%",
                "Volume (MWh)": f"{vol:,.0f}",
                "Cost/MWh (€)": f"{cost_per_mwh:.2f}",
                "Total Cost (€)": f"{total_ch_cost:,.0f}",
                "Collateral (€)": f"{coll:,.0f}",
            })

        avg_cost = total_cost / max(total_gap, 1)

        # Results display
        o1, o2, o3 = st.columns(3)
        o1.metric("Avg. Procurement Cost", f"€{avg_cost:.2f}/MWh")
        o2.metric("Total Annual Cost", f"€{total_cost / 1e6:.2f}M")
        o3.metric("Collateral Required", f"€{total_coll / 1e6:.2f}M")

        st.dataframe(pd.DataFrame(results).set_index("Channel"), use_container_width=True)

        # Allocation pie
        alloc_labels = [r["Channel"] for r in results]
        alloc_vals = [float(r["Volume (MWh)"].replace(",", "")) for r in results]
        alloc_colors = [CHANNELS[ch]["color"] for ch in alloc_labels]

        fig_opt = go.Figure(go.Pie(
            labels=alloc_labels, values=alloc_vals,
            hole=0.5, textinfo="label+percent",
            marker=dict(colors=alloc_colors),
        ))
        fig_opt.update_layout(height=350, font=dict(family="DM Sans", size=10), margin=dict(t=20, b=20))
        st.plotly_chart(fig_opt, use_container_width=True)

        st.success(f"✅ Optimization complete ({strategy}). Average cost: €{avg_cost:.2f}/MWh")


# ---------------------------------------------------------------------------
# Tab 3: Market Monitor
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Market Price Monitor")
    st.caption("Simulated market data — will connect to live OPCOM/Nord Pool feeds in production")

    # Generate demo price history
    np.random.seed(42)
    dates = pd.date_range(start="2025-04-01", end="2026-03-20", freq="D")
    dam_prices = dam_base + np.cumsum(np.random.normal(0, 1.2, len(dates)))
    dam_prices = np.clip(dam_prices, 30, 200)
    idm_prices = dam_prices + np.random.normal(1.5, 2.0, len(dates))
    bilateral_prices = dam_prices + np.random.normal(1.0, 1.5, len(dates))

    df_prices = pd.DataFrame({
        "date": dates,
        "DAM": dam_prices,
        "IDM": idm_prices,
        "Bilateral": bilateral_prices,
    }).set_index("date")

    # Latest prices
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("DAM Spot", f"€{df_prices['DAM'].iloc[-1]:.2f}/MWh",
              delta=f"{df_prices['DAM'].iloc[-1] - df_prices['DAM'].iloc[-2]:+.2f}")
    p2.metric("IDM VWAP", f"€{df_prices['IDM'].iloc[-1]:.2f}/MWh",
              delta=f"{df_prices['IDM'].iloc[-1] - df_prices['IDM'].iloc[-2]:+.2f}")
    p3.metric("Bilateral", f"€{df_prices['Bilateral'].iloc[-1]:.2f}/MWh",
              delta=f"{df_prices['Bilateral'].iloc[-1] - df_prices['Bilateral'].iloc[-2]:+.2f}")
    p4.metric("DAM 30d Avg", f"€{df_prices['DAM'].tail(30).mean():.2f}/MWh")

    st.divider()

    # Price chart
    fig_mkt = go.Figure()
    fig_mkt.add_trace(go.Scatter(
        x=df_prices.index, y=df_prices["DAM"],
        name="DAM (PZU)", line=dict(color="#2c3e50", width=2),
    ))
    fig_mkt.add_trace(go.Scatter(
        x=df_prices.index, y=df_prices["IDM"],
        name="IDM (PI)", line=dict(color="#3498db", width=1, dash="dot"),
    ))
    fig_mkt.add_trace(go.Scatter(
        x=df_prices.index, y=df_prices["Bilateral"],
        name="Bilateral (PCCB)", line=dict(color="#e67e22", width=1, dash="dash"),
    ))

    # Add 30d MA
    ma30 = df_prices["DAM"].rolling(30).mean()
    fig_mkt.add_trace(go.Scatter(
        x=df_prices.index, y=ma30,
        name="DAM 30d MA", line=dict(color="#e74c3c", width=2, dash="dashdot"),
    ))

    fig_mkt.update_layout(
        height=420,
        yaxis_title="EUR/MWh",
        font=dict(family="DM Sans"),
        legend=dict(orientation="h", y=-0.12),
        margin=dict(t=20),
        hovermode="x unified",
    )
    st.plotly_chart(fig_mkt, use_container_width=True)

    # Spread analysis
    st.subheader("Spread Analysis: IDM − DAM")
    spread = df_prices["IDM"] - df_prices["DAM"]

    fig_spread = go.Figure()
    fig_spread.add_trace(go.Scatter(
        x=spread.index, y=spread.values,
        mode="lines", name="IDM−DAM Spread",
        line=dict(color="#9b59b6", width=1),
        fill="tozeroy", fillcolor="rgba(155,89,182,0.15)",
    ))
    fig_spread.add_hline(y=0, line_color="black", line_width=0.5)
    fig_spread.add_hline(y=spread.mean(), line_dash="dash", line_color="#e74c3c",
                         annotation_text=f"Avg: €{spread.mean():.2f}")

    fig_spread.update_layout(
        height=300, yaxis_title="EUR/MWh",
        font=dict(family="DM Sans"), margin=dict(t=20),
    )
    st.plotly_chart(fig_spread, use_container_width=True)

    # Monthly volatility
    st.subheader("Monthly DAM Price Volatility")
    monthly_vol = df_prices["DAM"].resample("MS").std()
    fig_vol = go.Figure(go.Bar(
        x=monthly_vol.index, y=monthly_vol.values,
        marker_color=["#e74c3c" if v > monthly_vol.mean() else "#27ae60" for v in monthly_vol.values],
        text=[f"€{v:.1f}" for v in monthly_vol.values],
        textposition="outside",
    ))
    fig_vol.add_hline(y=monthly_vol.mean(), line_dash="dash", line_color="#2c3e50",
                      annotation_text=f"Avg: €{monthly_vol.mean():.1f}")
    fig_vol.update_layout(
        height=300, yaxis_title="Std Dev (EUR/MWh)",
        font=dict(family="DM Sans"), margin=dict(t=20),
    )
    st.plotly_chart(fig_vol, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: Execution Log
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Procurement Execution Log")
    st.caption("Historical procurement transactions — demo data")

    # Demo execution log
    np.random.seed(77)
    n_trades = 25
    trade_dates = pd.date_range(start="2026-01-15", periods=n_trades, freq="5D")
    trade_channels = np.random.choice(
        ["BRM Forward", "OPCOM Bilateral", "Direct Bilateral", "Spot DAM", "Spot IDM"],
        size=n_trades, p=[0.25, 0.30, 0.10, 0.25, 0.10],
    )
    trade_volumes = np.random.choice([100, 200, 500, 1000, 2000], size=n_trades, p=[0.15, 0.25, 0.30, 0.20, 0.10])
    trade_prices = [dam_base + CHANNELS[ch]["spread_vs_dam"] + np.random.normal(0, 2) for ch in trade_channels]

    trades_df = pd.DataFrame({
        "Date": trade_dates,
        "Channel": trade_channels,
        "Volume (MWh)": trade_volumes,
        "Price (€/MWh)": [round(p, 2) for p in trade_prices],
        "Total (€)": [round(v * p, 0) for v, p in zip(trade_volumes, trade_prices)],
        "Status": np.random.choice(["Settled", "Confirmed", "Pending"], size=n_trades, p=[0.60, 0.25, 0.15]),
        "Delivery": [d + timedelta(days=np.random.randint(7, 90)) for d in trade_dates],
    })

    # Summary metrics
    settled = trades_df[trades_df["Status"] == "Settled"]
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Total Trades", f"{len(trades_df)}")
    t2.metric("Total Volume", f"{trades_df['Volume (MWh)'].sum():,.0f} MWh")
    t3.metric("Avg Price", f"€{trades_df['Price (€/MWh)'].mean():.2f}/MWh")
    t4.metric("Settled Value", f"€{settled['Total (€)'].sum() / 1e6:.2f}M")

    st.divider()

    # Execution log table
    st.dataframe(
        trades_df.sort_values("Date", ascending=False).style.format({
            "Volume (MWh)": "{:,.0f}",
            "Price (€/MWh)": "€{:.2f}",
            "Total (€)": "€{:,.0f}",
        }).applymap(
            lambda v: "background-color: #d4edda" if v == "Settled"
            else ("background-color: #fff3cd" if v == "Confirmed"
                  else ("background-color: #f8d7da" if v == "Pending" else "")),
            subset=["Status"]
        ),
        use_container_width=True,
        height=400,
    )

    # Volume by channel
    vol_by_ch = trades_df.groupby("Channel")["Volume (MWh)"].sum().sort_values(ascending=False)
    fig_exec = go.Figure(go.Bar(
        x=vol_by_ch.index, y=vol_by_ch.values,
        marker_color=[CHANNELS.get(ch, {}).get("color", "#95a5a6") for ch in vol_by_ch.index],
        text=[f"{v:,.0f}" for v in vol_by_ch.values],
        textposition="outside",
    ))
    fig_exec.update_layout(
        height=300, yaxis_title="Volume (MWh)",
        font=dict(family="DM Sans"), margin=dict(t=20),
    )
    st.plotly_chart(fig_exec, use_container_width=True)

    # Price distribution by channel
    st.subheader("Price Distribution by Channel")
    fig_box = go.Figure()
    for ch in sorted(trades_df["Channel"].unique()):
        ch_data = trades_df[trades_df["Channel"] == ch]["Price (€/MWh)"]
        fig_box.add_trace(go.Box(
            y=ch_data, name=ch,
            marker_color=CHANNELS.get(ch, {}).get("color", "#95a5a6"),
            boxmean="sd",
        ))
    fig_box.update_layout(
        height=350, yaxis_title="Price (EUR/MWh)",
        font=dict(family="DM Sans"), margin=dict(t=20),
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)

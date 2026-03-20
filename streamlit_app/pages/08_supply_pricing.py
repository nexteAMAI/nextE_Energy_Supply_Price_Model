"""Page 08: Supply Price Builder — Interactive B2B offer pricing with waterfall."""
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

st.header("💰 Supply Price Builder")
st.caption("B2B Energy Supply Offer Pricing Engine · Waterfall Visualization · Multi-Scenario")
DATA_DIR = init_page()

# ---------------------------------------------------------------------------
# Try to load supply config for defaults
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

kpis = load_kpis()
eur_ron = kpis.get("eur_ron_latest", 4.977)
dam_base = kpis.get("trailing_6m_avg", 85.0)

# ---------------------------------------------------------------------------
# Sidebar: Contract Parameters
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📝 Contract Parameters")

customer_name = st.sidebar.text_input("Customer ID (anonymized)", value="CUST-001")
customer_category = st.sidebar.selectbox("Customer Category", ["commercial", "industrial"], index=1)
annual_volume = st.sidebar.number_input("Annual Volume (MWh)", min_value=100, max_value=500_000, value=20_000, step=1000)
contract_months = st.sidebar.slider("Contract Duration (months)", 6, 36, 12)

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ PV Pricing Mechanism")
pv_mechanism = st.sidebar.selectbox("Mechanism", ["HYBRID", "FIXED", "DAM_INDEXED"])

pv_fixed_price = 50.0
pv_floor = 40.0
pv_indexed_share = 0.70
if pv_mechanism == "FIXED":
    pv_fixed_price = st.sidebar.number_input("Fixed PV Price (EUR/MWh)", value=50.0, step=1.0)
elif pv_mechanism == "HYBRID":
    pv_floor = st.sidebar.number_input("Floor Price (EUR/MWh)", value=40.0, step=1.0)
    pv_indexed_share = st.sidebar.slider("Indexed Share (%)", 0, 100, 70) / 100.0
# DAM_INDEXED has no extra params for sidebar

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Procurement Mix (%)")

# Channel allocation sliders
pct_brm = st.sidebar.slider("BRM Forward", 0, 100, 40)
pct_opcom = st.sidebar.slider("OPCOM Bilateral", 0, 100, 25)
pct_direct = st.sidebar.slider("Direct Bilateral", 0, 100, 15)
pct_eex = st.sidebar.slider("EEX Financial", 0, 100, 0)
pct_dam = st.sidebar.slider("Spot DAM", 0, 100, 15)
pct_idm = st.sidebar.slider("Spot IDM", 0, 100, 5)
total_pct = pct_brm + pct_opcom + pct_direct + pct_eex + pct_dam + pct_idm

if total_pct != 100:
    st.sidebar.warning(f"⚠️ Allocation sums to {total_pct}% (must be 100%)")

st.sidebar.markdown("---")
st.sidebar.subheader("💶 Margins & Risk")
target_margin = st.sidebar.number_input("Commercial Margin (EUR/MWh)", value=12.0, step=0.5)
risk_margin = st.sidebar.number_input("Risk Premium (EUR/MWh)", value=5.0, step=0.5)
include_gc = st.sidebar.checkbox("Include GC Quota Cost", value=True)
include_balancing = st.sidebar.checkbox("Include Balancing Cost", value=True)
apply_volume_discount = st.sidebar.checkbox("Apply Volume Discount", value=True)
apply_seasonal_adj = st.sidebar.checkbox("Apply Seasonal Margin Adj.", value=True)

# ---------------------------------------------------------------------------
# Main: Pricing Calculation
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Price Waterfall", "🔀 Multi-Scenario", "📋 Offer Summary"])

# --- Helper: compute pricing waterfall ---
def compute_waterfall(
    dam_price: float,
    solar_share: float = 0.35,
    pv_cost: float = 45.0,
    gc_cost: float = 14.50,
    balancing: float = 3.0,
    risk: float = 5.0,
    margin: float = 12.0,
    scenario_label: str = "Base",
) -> dict:
    """Compute supply price waterfall components."""
    # PV procurement cost
    if pv_mechanism == "FIXED":
        pv_eff = pv_fixed_price
    elif pv_mechanism == "HYBRID":
        pv_eff = max(pv_floor, dam_price * pv_indexed_share - 10.0)
    else:  # DAM_INDEXED
        pv_eff = max(dam_price - 3.50, 30.0)

    # Forward procurement cost (weighted by channel)
    channel_costs = {
        "BRM Forward": dam_price + 1.50,
        "OPCOM Bilateral": dam_price + 1.00,
        "Direct Bilateral": dam_price + 2.00,
        "EEX Financial": dam_price + 15.0 + 0.25,  # basis + fee
        "Spot DAM": dam_price + 0.10,
        "Spot IDM": dam_price * 1.02 + 0.30,
    }
    channel_weights = {
        "BRM Forward": pct_brm / 100.0,
        "OPCOM Bilateral": pct_opcom / 100.0,
        "Direct Bilateral": pct_direct / 100.0,
        "EEX Financial": pct_eex / 100.0,
        "Spot DAM": pct_dam / 100.0,
        "Spot IDM": pct_idm / 100.0,
    }
    forward_cost = sum(channel_costs[ch] * channel_weights[ch] for ch in channel_costs)

    # Blended energy cost
    energy_cost = solar_share * pv_eff + (1 - solar_share) * forward_cost

    # GC quota
    gc = gc_cost if include_gc else 0.0

    # Balancing
    bal = balancing if include_balancing else 0.0

    # Volume discount
    vol_discount = 0.0
    if apply_volume_discount and annual_volume >= 1000:
        gwh = annual_volume / 1000.0
        disc_curve = _CFG.get("margins", {}).get("volume_discount_curve", {0: 0, 1: 1.5, 5: 3.0, 20: 5.0})
        disc_map = {float(k): v for k, v in disc_curve.items()}
        thresholds = sorted(disc_map.keys(), reverse=True)
        for t in thresholds:
            if gwh >= t:
                vol_discount = disc_map[t] / 100.0 * margin
                break

    # Seasonal adjustment
    seasonal_adj = 0.0
    if apply_seasonal_adj:
        now_month = datetime.now().month
        if now_month in (12, 1, 2, 3):
            seasonal_adj = margin * 0.15
        elif now_month in (6, 7, 8, 9):
            seasonal_adj = -margin * 0.15

    eff_margin = margin - vol_discount + seasonal_adj

    subtotal = energy_cost + gc + bal + risk + eff_margin
    vat_rate = _CFG.get("regulatory", {}).get("vat_rate", 0.19)
    vat = subtotal * vat_rate
    total = subtotal + vat

    return {
        "scenario": scenario_label,
        "pv_cost": round(pv_eff * solar_share, 2),
        "forward_cost": round(forward_cost * (1 - solar_share), 2),
        "blended_energy": round(energy_cost, 2),
        "gc_cost": round(gc, 2),
        "balancing": round(bal, 2),
        "risk_premium": round(risk, 2),
        "vol_discount": round(-vol_discount, 2),
        "seasonal_adj": round(seasonal_adj, 2),
        "nexte_margin": round(eff_margin, 2),
        "subtotal_ex_vat": round(subtotal, 2),
        "vat": round(vat, 2),
        "total_inc_vat": round(total, 2),
        "annual_cost_eur": round(subtotal * annual_volume, 0),
        "solar_share_pct": round(solar_share * 100, 1),
    }


# ---------------------------------------------------------------------------
# Tab 1: Price Waterfall
# ---------------------------------------------------------------------------
with tab1:
    # Estimate solar share based on customer category
    solar_share_est = 0.35 if customer_category == "commercial" else 0.25
    solar_share = st.slider("Solar Share of Consumption (%)", 0, 80, int(solar_share_est * 100)) / 100.0

    wf = compute_waterfall(dam_base, solar_share=solar_share, risk=risk_margin, margin=target_margin)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Offer Price (ex-VAT)", f"€{wf['subtotal_ex_vat']:.2f}/MWh")
    c2.metric("Offer Price (inc-VAT)", f"€{wf['total_inc_vat']:.2f}/MWh")
    c3.metric("Annual Revenue", f"€{wf['annual_cost_eur']:,.0f}")
    c4.metric("Solar Share", f"{wf['solar_share_pct']:.0f}%")

    st.divider()

    # Waterfall chart
    labels = [
        "PV\nProcurement",
        "Forward\nProcurement",
        "GC Quota\nCost",
        "Balancing\nCost",
        "Risk\nPremium",
        "nextE\nMargin",
        "Subtotal\n(ex-VAT)",
    ]
    values = [
        wf["pv_cost"],
        wf["forward_cost"],
        wf["gc_cost"],
        wf["balancing"],
        wf["risk_premium"],
        wf["nexte_margin"],
    ]
    measures = ["relative"] * 6 + ["total"]
    y_vals = values + [wf["subtotal_ex_vat"]]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=labels,
        y=y_vals,
        connector={"line": {"color": "rgba(100,100,100,0.3)", "width": 1}},
        increasing={"marker": {"color": "#e74c3c"}},
        decreasing={"marker": {"color": "#27ae60"}},
        totals={"marker": {"color": "#1F4E79", "line": {"color": "#e74c3c", "width": 2}}},
        text=[f"€{v:.1f}" for v in values] + [f"€{wf['subtotal_ex_vat']:.1f}"],
        textposition="outside",
        textfont=dict(size=11),
    ))

    fig.update_layout(
        height=480,
        showlegend=False,
        yaxis_title="EUR/MWh",
        font=dict(family="DM Sans"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=80),
        annotations=[
            dict(
                x=0.97, y=0.95, xref="paper", yref="paper", showarrow=False,
                text=(
                    f"<b>Offer: €{wf['subtotal_ex_vat']:.2f}/MWh (ex-VAT)</b><br>"
                    f"<b>+ VAT 19%: €{wf['vat']:.2f}</b><br>"
                    f"<b>TOTAL: €{wf['total_inc_vat']:.2f}/MWh</b><br>"
                    f"(~{wf['total_inc_vat'] * eur_ron:.0f} RON/MWh)"
                ),
                font=dict(size=12, color="#1F4E79"),
                align="right",
                bordercolor="#1F4E79", borderwidth=1, borderpad=8,
                bgcolor="rgba(255,255,255,0.9)",
            )
        ],
    )
    st.plotly_chart(fig, use_container_width=True)

    # Breakdown table
    st.subheader("Cost Breakdown")
    breakdown = pd.DataFrame([
        {"Component": "PV Procurement", "EUR/MWh": wf["pv_cost"], "Share": f"{wf['pv_cost']/wf['subtotal_ex_vat']*100:.1f}%"},
        {"Component": "Forward Procurement", "EUR/MWh": wf["forward_cost"], "Share": f"{wf['forward_cost']/wf['subtotal_ex_vat']*100:.1f}%"},
        {"Component": "Green Certificate (GC)", "EUR/MWh": wf["gc_cost"], "Share": f"{wf['gc_cost']/wf['subtotal_ex_vat']*100:.1f}%"},
        {"Component": "Balancing Cost", "EUR/MWh": wf["balancing"], "Share": f"{wf['balancing']/wf['subtotal_ex_vat']*100:.1f}%"},
        {"Component": "Risk Premium", "EUR/MWh": wf["risk_premium"], "Share": f"{wf['risk_premium']/wf['subtotal_ex_vat']*100:.1f}%"},
        {"Component": "nextE Margin", "EUR/MWh": wf["nexte_margin"], "Share": f"{wf['nexte_margin']/wf['subtotal_ex_vat']*100:.1f}%"},
        {"Component": "TOTAL (ex-VAT)", "EUR/MWh": wf["subtotal_ex_vat"], "Share": "100.0%"},
    ]).set_index("Component")
    st.dataframe(breakdown.style.format({"EUR/MWh": "{:.2f}"}), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Multi-Scenario
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Multi-Scenario Price Comparison")
    st.caption("P10 (low) → P50 (base) → P90 (high) DAM price scenarios")

    # Scenario DAM prices
    dam_p10 = dam_base * 0.65
    dam_p25 = dam_base * 0.80
    dam_p50 = dam_base
    dam_p75 = dam_base * 1.20
    dam_p90 = dam_base * 1.40

    scenarios = {
        "P10\n(Low)": {"dam": dam_p10, "bal": 1.5, "risk": risk_margin * 0.7},
        "P25": {"dam": dam_p25, "bal": 2.0, "risk": risk_margin * 0.85},
        "P50\n(Base)": {"dam": dam_p50, "bal": 3.0, "risk": risk_margin},
        "P75": {"dam": dam_p75, "bal": 4.0, "risk": risk_margin * 1.20},
        "P90\n(High)": {"dam": dam_p90, "bal": 5.5, "risk": risk_margin * 1.50},
    }

    scenario_results = []
    for sname, sparams in scenarios.items():
        res = compute_waterfall(
            sparams["dam"],
            solar_share=solar_share_est,
            balancing=sparams["bal"],
            risk=sparams["risk"],
            margin=target_margin,
            scenario_label=sname,
        )
        scenario_results.append(res)

    # Bar chart comparison
    fig_sc = go.Figure()
    component_keys = ["pv_cost", "forward_cost", "gc_cost", "balancing", "risk_premium", "nexte_margin"]
    component_labels = ["PV Procurement", "Forward Procurement", "GC Quota", "Balancing", "Risk Premium", "nextE Margin"]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e67e22", "#e74c3c", "#9b59b6"]

    for i, (key, label) in enumerate(zip(component_keys, component_labels)):
        fig_sc.add_trace(go.Bar(
            name=label,
            x=[r["scenario"] for r in scenario_results],
            y=[r[key] for r in scenario_results],
            marker_color=colors[i],
            text=[f"€{r[key]:.1f}" for r in scenario_results],
            textposition="inside",
            textfont=dict(size=9),
        ))

    fig_sc.update_layout(
        barmode="stack",
        height=500,
        yaxis_title="EUR/MWh",
        font=dict(family="DM Sans"),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=30),
    )

    # Add total annotation
    for idx, r in enumerate(scenario_results):
        fig_sc.add_annotation(
            x=r["scenario"], y=r["subtotal_ex_vat"] + 3,
            text=f"€{r['subtotal_ex_vat']:.1f}",
            showarrow=False, font=dict(size=11, color="#2c3e50", family="JetBrains Mono"),
        )

    st.plotly_chart(fig_sc, use_container_width=True)

    # Scenario summary table
    st.subheader("Scenario Summary")
    summary_rows = []
    for r in scenario_results:
        summary_rows.append({
            "Scenario": r["scenario"].replace("\n", " "),
            "DAM Price (€)": f"{r['blended_energy'] / (1 - solar_share_est) if solar_share_est < 1 else 0:.1f}",
            "Blended Energy (€)": f"{r['blended_energy']:.2f}",
            "Total ex-VAT (€)": f"{r['subtotal_ex_vat']:.2f}",
            "Total inc-VAT (€)": f"{r['total_inc_vat']:.2f}",
            "Annual Revenue (€)": f"{r['annual_cost_eur']:,.0f}",
        })
    st.dataframe(pd.DataFrame(summary_rows).set_index("Scenario"), use_container_width=True)

    # Probability-weighted expected price
    weights = [0.10, 0.25, 0.30, 0.25, 0.10]
    expected_price = sum(w * r["subtotal_ex_vat"] for w, r in zip(weights, scenario_results))
    st.info(f"📊 **Probability-Weighted Expected Price:** €{expected_price:.2f}/MWh (ex-VAT)")

# ---------------------------------------------------------------------------
# Tab 3: Offer Summary
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("📋 Customer Offer Summary")

    wf_base = compute_waterfall(dam_base, solar_share=solar_share_est, risk=risk_margin, margin=target_margin)
    contract_start = datetime.now().replace(day=1) + timedelta(days=32)
    contract_start = contract_start.replace(day=1)
    contract_end = contract_start + timedelta(days=contract_months * 30)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Contract Details**")
        st.markdown(f"""
        | Parameter | Value |
        |:--|:--|
        | Customer | {customer_name} |
        | Category | {customer_category.title()} |
        | Volume | {annual_volume:,} MWh/year |
        | Duration | {contract_months} months |
        | Start | {contract_start.strftime('%Y-%m-%d')} |
        | End | {contract_end.strftime('%Y-%m-%d')} |
        | PV Mechanism | {pv_mechanism} |
        | Solar Share | {solar_share_est * 100:.0f}% |
        """)

    with col_r:
        st.markdown("**Pricing Summary**")
        st.markdown(f"""
        | Component | EUR/MWh |
        |:--|--:|
        | Energy Procurement | {wf_base['blended_energy']:.2f} |
        | Green Certificate | {wf_base['gc_cost']:.2f} |
        | Balancing Cost | {wf_base['balancing']:.2f} |
        | Risk Premium | {wf_base['risk_premium']:.2f} |
        | nextE Margin | {wf_base['nexte_margin']:.2f} |
        | **Offer (ex-VAT)** | **{wf_base['subtotal_ex_vat']:.2f}** |
        | VAT (19%) | {wf_base['vat']:.2f} |
        | **Offer (inc-VAT)** | **{wf_base['total_inc_vat']:.2f}** |
        """)

    st.divider()

    # Procurement allocation pie
    st.subheader("Procurement Channel Allocation")
    alloc_labels = ["BRM Forward", "OPCOM Bilateral", "Direct Bilateral", "EEX Financial", "Spot DAM", "Spot IDM"]
    alloc_values = [pct_brm, pct_opcom, pct_direct, pct_eex, pct_dam, pct_idm]
    alloc_colors = ["#2c3e50", "#3498db", "#e67e22", "#27ae60", "#9b59b6", "#1abc9c"]

    fig_alloc = go.Figure(go.Pie(
        labels=alloc_labels, values=alloc_values,
        hole=0.5, textinfo="label+percent",
        marker=dict(colors=alloc_colors),
    ))
    fig_alloc.update_layout(
        height=350,
        font=dict(family="DM Sans", size=10),
        margin=dict(t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_alloc, use_container_width=True)

    # Margin analysis
    st.subheader("Margin Analysis")
    min_margin = _CFG.get("margins", {}).get("minimum_margin_eur_per_mwh", 8.0)
    eff_margin_val = wf_base["nexte_margin"]
    margin_ok = eff_margin_val >= min_margin

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Effective Margin", f"€{eff_margin_val:.2f}/MWh")
    mc2.metric("Minimum Threshold", f"€{min_margin:.2f}/MWh")
    mc3.metric("Status", "✅ ABOVE FLOOR" if margin_ok else "🔴 BELOW FLOOR")

    if not margin_ok:
        st.error(f"⚠️ Effective margin (€{eff_margin_val:.2f}) is below the minimum floor (€{min_margin:.2f}). Consider increasing the target margin or reducing volume discount.")
    else:
        st.success(f"Margin headroom: €{eff_margin_val - min_margin:.2f}/MWh above floor.")

    # REMIT compliance flag
    if annual_volume >= 350:
        st.info("📋 **REMIT Reporting:** This contract exceeds 350 MWh/year threshold — ACER ARIS reporting required.")

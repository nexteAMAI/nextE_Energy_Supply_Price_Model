"""app.brand - presentation layer of the application (execution prompt section 10.2, nexte-brand).

Montserrat, navy and neutrals only, no third hue, radius 0, 1 px lines instead of shadows,
Romanian number and date formats on every surface, confidentiality marking, third-person voice.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

NAVY = "#1F3E66"
NAVY_DEEP = "#14304F"
NAVY_SOFT = "#2E5386"
INK = "#0E1C2E"
WHITE = "#FFFFFF"
PAPER = "#F7F6F3"
LINE = "#C8C8C6"
MUTED = "#6A6A6A"
CONFIDENTIAL = "CONFIDENTIAL · nextE"
SERIES = [NAVY, "rgba(31,62,102,0.6)", "rgba(31,62,102,0.3)", NAVY_SOFT, "rgba(46,83,134,0.6)", "rgba(46,83,134,0.3)"]

CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, .stApp, .stApp *:not([class*="material"]):not([data-testid="stIconMaterial"]) {{
  font-family: 'Montserrat', system-ui, Arial, sans-serif !important;
}}
.stApp {{ color: {INK}; }}
.stApp {{ background: {WHITE}; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer, [data-testid="stStatusWidget"] {{ visibility: hidden; height: 0; }}
.stAppDeployButton {{ display: none; }}
section[data-testid="stSidebar"] {{ background: {PAPER}; border-right: 1px solid {LINE}; }}
h1, h2, h3, h4 {{ font-family: 'Montserrat', system-ui, Arial, sans-serif; color: {NAVY}; font-weight: 600; letter-spacing: 0; }}
h1 {{ font-size: 1.5rem; }} h2 {{ font-size: 1.15rem; margin-top: 1.2rem; }} h3 {{ font-size: 1rem; }}
.stButton > button, .stDownloadButton > button, div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input,
.stFileUploader section, div[data-testid="stExpander"] details, .stTabs [data-baseweb="tab"] {{
  border-radius: 0 !important; box-shadow: none !important;
}}
.stButton > button, .stDownloadButton > button {{ border: 1px solid {NAVY}; color: {NAVY}; background: {WHITE}; font-weight: 600; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ background: {NAVY}; color: {WHITE}; border-color: {NAVY}; }}
.stButton > button[kind="primary"] {{ background: {NAVY}; color: {WHITE}; }}
div[data-testid="stExpander"] details {{ border: 1px solid {LINE}; }}
div[data-testid="stMetric"] {{ border: 1px solid {LINE}; padding: 0.6rem 0.8rem; }}
.esb-eyebrow {{ font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: {MUTED}; margin-bottom: 0.15rem; }}
.esb-kpi {{ border: 1px solid {LINE}; padding: 0.7rem 0.9rem; background: {WHITE}; height: 100%; }}
.esb-kpi .v {{ font-size: 1.45rem; font-weight: 700; color: {NAVY}; font-variant-numeric: tabular-nums; line-height: 1.2; }}
.esb-kpi .u {{ font-size: 0.85rem; font-weight: 500; color: {NAVY}; margin-left: 0.25rem; }}
.esb-kpi .rule {{ border-top: 2px solid {INK}; margin: 0.45rem 0 0.35rem 0; }}
.esb-kpi .c {{ font-size: 0.78rem; color: {MUTED}; }}
.esb-mark {{ font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; color: {MUTED}; text-transform: uppercase; }}
.esb-caption {{ font-size: 0.78rem; color: {MUTED}; margin-top: -0.4rem; margin-bottom: 0.8rem; }}
.esb-note {{ border: 1px solid {LINE}; border-left: 3px solid {NAVY}; padding: 0.5rem 0.8rem; background: {PAPER}; font-size: 0.86rem; margin: 0.4rem 0 0.8rem 0; }}
.esb-refusal {{ border: 1px solid {NAVY}; padding: 0.5rem 0.8rem; background: {WHITE}; font-size: 0.86rem; margin: 0.4rem 0; }}
.esb-status {{ display: inline-block; border: 1px solid {NAVY}; padding: 0.05rem 0.45rem; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; }}
.esb-status.pass {{ background: {NAVY}; color: {WHITE}; }}
.esb-status.fail {{ background: {WHITE}; color: {NAVY}; border: 2px solid {NAVY}; }}
table.esb {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; margin-bottom: 0.8rem; }}
table.esb th {{ background: {NAVY}; color: {WHITE}; font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
  padding: 0.35rem 0.5rem; text-align: right; border: none; white-space: nowrap; }}
table.esb th:first-child, table.esb td:first-child {{ text-align: left; }}
table.esb td {{ padding: 0.28rem 0.5rem; border-bottom: 1px solid {LINE}; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
table.esb tr:nth-child(even) td {{ background: {PAPER}; }}
table.esb tr.total td {{ font-weight: 700; border-top: 2px solid {INK}; }}
.esb-scroll {{ overflow-x: auto; max-height: 560px; overflow-y: auto; border: 1px solid {LINE}; margin-bottom: 0.8rem; }}
.esb-scroll table.esb {{ margin-bottom: 0; }}
.esb-scroll table.esb th {{ position: sticky; top: 0; }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---- Romanian formats -------------------------------------------------------------------------
def num(x, decimals: int = 2, unit: str = "") -> str:
    """84,50 €/MWh · 1.490 MW · (12.345,67) for negatives · '–' for missing."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "–"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    neg = v < 0
    s = f"{abs(v):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    s = f"({s})" if neg else s
    return f"{s} {unit}".strip()


def pct(x, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    v = float(x) * 100
    s = f"{abs(v):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"({s} %)" if v < 0 else f"{s} %"


def dmy(d) -> str:
    if d is None or (isinstance(d, float) and math.isnan(d)):
        return "–"
    if isinstance(d, str):
        try:
            d = date.fromisoformat(d[:10])
        except ValueError:
            return d
    if isinstance(d, (pd.Timestamp, datetime)):
        d = d.date()
    return d.strftime("%d.%m.%Y")


MONTH_SHORT = ["Ian", "Feb", "Mar", "Apr", "Mai", "Iun", "Iul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---- components ---------------------------------------------------------------------------------
def page_title(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="esb-mark">{CONFIDENTIAL}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    if caption:
        st.markdown(f'<div class="esb-caption">{caption}</div>', unsafe_allow_html=True)


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="esb-eyebrow">{text}</div>', unsafe_allow_html=True)


def caption(text: str) -> None:
    st.markdown(f'<div class="esb-caption">{text}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="esb-note">{text}</div>', unsafe_allow_html=True)


def refusal(text: str) -> None:
    st.markdown(f'<div class="esb-refusal">{text}</div>', unsafe_allow_html=True)


def status(ok: bool, label_ok: str = "PASS", label_fail: str = "FAIL") -> str:
    return f'<span class="esb-status {"pass" if ok else "fail"}">{label_ok if ok else label_fail}</span>'


def kpi(label: str, value, unit: str = "", caption_text: str = "", decimals: int | None = None) -> None:
    if decimals is None:
        decimals = 2 if ("/MWh" in unit or "%" in unit) else 0
    v = value if isinstance(value, str) else num(value, decimals)
    st.markdown(
        f'<div class="esb-kpi"><div class="esb-eyebrow">{label}</div>'
        f'<div><span class="v">{v}</span><span class="u">{unit}</span></div>'
        f'<div class="rule"></div><div class="c">{caption_text}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, it in zip(cols, items, strict=True):
        with col:
            kpi(*it)


def _fmt_cell(v, decimals: int, pct_row: bool) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (date, datetime, pd.Timestamp)):
        return dmy(v)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "Yes" if v else "No"
    if pct_row:
        return pct(v)
    return num(v, decimals)


def table(df: pd.DataFrame, decimals: int = 2, index_label: str = "", pct_rows: set[str] | None = None,
          total_rows: set[str] | None = None, scroll: bool = False, decimals_by_col: dict[str, int] | None = None,
          max_rows: int | None = None) -> None:
    """Brand table: navy header, alternating rows, right-aligned tabular numerals, RO formats."""
    pct_rows = pct_rows or set()
    total_rows = total_rows or set()
    decimals_by_col = decimals_by_col or {}
    d = df if max_rows is None else df.head(max_rows)
    head = "".join(f"<th>{c}</th>" for c in [index_label, *[str(c) for c in d.columns]])
    rows = []
    for idx, rec in d.iterrows():
        key = str(idx)
        is_pct = key in pct_rows or key.endswith("_pct") or key.endswith(" %")
        cells = "".join(f"<td>{_fmt_cell(v, decimals_by_col.get(str(c), decimals), is_pct)}</td>" for c, v in rec.items())
        cls = ' class="total"' if key in total_rows else ""
        rows.append(f"<tr{cls}><td>{key}</td>{cells}</tr>")
    html = f'<table class="esb"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    if scroll:
        html = f'<div class="esb-scroll">{html}</div>'
    st.markdown(html, unsafe_allow_html=True)
    if max_rows is not None and len(df) > max_rows:
        caption(f"First {max_rows} of {num(len(df), 0)} rows shown; the full table is in the exports.")


# ---- charts ---------------------------------------------------------------------------------------
def layout(fig: go.Figure, height: int = 320, y_title: str = "", x_title: str = "", legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font={"family": "Montserrat, system-ui, Arial", "color": INK, "size": 12},
        paper_bgcolor=WHITE, plot_bgcolor=WHITE, height=height,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "font": {"size": 11}} if legend else None,
        showlegend=legend, colorway=SERIES, hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, title=x_title, tickfont={"size": 11})
    fig.update_yaxes(gridcolor=LINE, griddash="dash", gridwidth=1, zeroline=True, zerolinecolor=LINE, title=y_title,
                     tickformat=",.0f", separatethousands=True, tickfont={"size": 11})
    return fig


def bars(x, series: dict[str, np.ndarray], y_title: str = "", stacked: bool = False, height: int = 320) -> go.Figure:
    fig = go.Figure()
    for i, (name, y) in enumerate(series.items()):
        fig.add_bar(x=list(x), y=list(y), name=name, marker_color=SERIES[i % len(SERIES)], marker_line_width=0)
    fig.update_layout(barmode="stack" if stacked else "group", bargap=0.25)
    return layout(fig, height=height, y_title=y_title)


def lines(x, series: dict[str, np.ndarray], y_title: str = "", height: int = 320) -> go.Figure:
    fig = go.Figure()
    for i, (name, y) in enumerate(series.items()):
        fig.add_scatter(x=list(x), y=list(y), name=name, mode="lines", line={"color": SERIES[i % len(SERIES)], "width": 2 if i == 0 else 1.5})
    return layout(fig, height=height, y_title=y_title)


def donut(labels: list[str], values: list[float], height: int = 280) -> go.Figure:
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62, marker={"colors": SERIES[: len(labels)], "line": {"color": WHITE, "width": 1}},
                           textinfo="percent", sort=False))
    return layout(fig, height=height, legend=True)


def waterfall_bars(labels: list[str], values: list[float], height: int = 340) -> go.Figure:
    """A build-up as a bar sequence (cumulative levels drawn as bars with opacity), no third hue."""
    fig = go.Figure()
    cum = np.cumsum(values)
    base = np.concatenate([[0.0], cum[:-1]])
    fig.add_bar(x=labels, y=list(values), base=list(base), marker_color=[NAVY if v >= 0 else "rgba(31,62,102,0.45)" for v in values],
                marker_line_width=0, name="component", text=[num(v, 2) for v in values], textposition="outside")
    fig.add_bar(x=["Total"], y=[float(cum[-1])], marker_color=NAVY_DEEP, marker_line_width=0, name="total", text=[num(float(cum[-1]), 2)], textposition="outside")
    return layout(fig, height=height, y_title="EUR/MWh", legend=False)

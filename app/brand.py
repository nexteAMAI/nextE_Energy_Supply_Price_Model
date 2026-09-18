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

from esb.export import UNIT_DECIMALS  # one catalogue for the workbook, the CSVs and the app (D113)

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
[data-testid="stDecoration"], footer, [data-testid="stAppDeployButton"], .stAppDeployButton, [data-testid="stMainMenu"], #MainMenu {{ display: none !important; }}
[data-testid="stExpandSidebarButton"] {{ visibility: visible !important; }}
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
table.esb th.txt, table.esb td.txt {{ text-align: left; white-space: normal; }}
table.esb td {{ padding: 0.28rem 0.5rem; border-bottom: 1px solid {LINE}; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
table.esb td {{ background: {WHITE}; }}
table.esb td.unit {{ text-align: left; color: {MUTED}; font-size: 0.72rem; white-space: nowrap; }}
table.esb tr.total td {{ font-weight: 700; border-top: 2px solid {INK}; }}
table.esb tr.subtotal td {{ font-weight: 700; }}
table.esb tr.memo td {{ font-style: italic; color: {MUTED}; }}
table.esb tr.check td {{ color: {MUTED}; }}
table.esb tr.section td {{ font-weight: 700; color: {NAVY}; text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.7rem; border-bottom: 2px solid {NAVY}; }}
.esb-scroll {{ overflow-x: auto; max-height: 560px; overflow-y: auto; border: 1px solid {LINE}; margin-bottom: 0.8rem; }}
.esb-scroll table.esb {{ margin-bottom: 0; }}
.esb-scroll table.esb th {{ position: sticky; top: 0; }}
.esb-wide {{ overflow-x: auto; margin-bottom: 0.8rem; }}
.esb-wide table.esb {{ margin-bottom: 0; }}
.esb-wide::-webkit-scrollbar {{ height: 8px; }}
.esb-wide::-webkit-scrollbar-thumb {{ background: {LINE}; }}
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
    s = f"{abs(v):,.{decimals}f}"
    neg = v < 0 and any(ch not in "0.," for ch in s)  # a value that rounds to zero is never shown as (0) (G5-1)
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    s = f"({s})" if neg else s
    return f"{s} {unit}".strip()


def pct(x, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    v = float(x) * 100
    s = f"{abs(v):,.{decimals}f}"
    neg = v < 0 and any(ch not in "0.," for ch in s)
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"({s} %)" if neg else f"{s} %"


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


def parse_num(text: str) -> float:
    """Parse a number typed in Romanian format: '.' groups thousands, ',' is the decimal separator.

    Rule for a string without a comma: exactly one dot NOT followed by exactly three digits is read as a
    decimal point ('0.21' -> 0,21); otherwise dots are thousands separators ('1.234' -> 1234).
    A blank string, '–' or '-' is a refusal (ValueError): blank is not zero.
    """
    t = (text or "").strip().replace(" ", "").replace("\u00a0", "")
    if t in ("", "–", "-", "(", ")"):
        raise ValueError("blank")
    neg = t.startswith("(") and t.endswith(")")
    if neg:
        t = t[1:-1]
    if "," in t:
        t = t.replace(".", "").replace(",", ".")
    elif t.count(".") == 1 and not (len(t) - t.index(".") - 1 == 3 and t[t.index(".") + 1 :].isdigit()):
        pass  # a single dot with other than three digits after it is a decimal point
    else:
        t = t.replace(".", "")
    v = float(t)
    return -v if neg else v


def num_input(label: str, value, decimals: int = 2, key: str | None = None, help: str | None = None,  # noqa: A002
              min_value: float | None = None, max_value: float | None = None, disabled: bool = False) -> float:
    """A numeric field in Romanian format (text-based). Shows 1.234,56; returns the parsed float.

    An unparseable entry or one outside [min_value, max_value] is refused in place and the previous value
    is returned, so a typing error never reaches the parameter register silently.
    """
    shown = num(value, decimals) if value is not None else ""
    text = st.text_input(label, value=shown, key=key, help=help, disabled=disabled)
    try:
        v = parse_num(text)
    except ValueError:
        refusal(f"'{text}' is not a number in Romanian format (example: 1.234,56). The previous value {shown} is kept.")
        return float(value) if value is not None else 0.0
    if min_value is not None and v < min_value:
        refusal(f"{num(v, decimals)} is below the minimum {num(min_value, decimals)}. The previous value {shown} is kept.")
        return float(value)
    if max_value is not None and v > max_value:
        refusal(f"{num(v, decimals)} is above the maximum {num(max_value, decimals)}. The previous value {shown} is kept.")
        return float(value)
    return float(v)


def grid_input(df: pd.DataFrame, key: str, decimals: int = 2) -> pd.DataFrame:
    """An editable grid in Romanian format: cells are shown as 1.234,56 text and parsed back.

    A cell that cannot be parsed keeps its previous value and is reported below the grid.
    """
    shown = df.map(lambda v: num(v, decimals))
    cfg = {str(c): st.column_config.TextColumn(str(c), width="small") for c in shown.columns}
    if df.index.name:
        cfg["_index"] = st.column_config.TextColumn(str(df.index.name))  # the row-label column carries its name (G5-6)
    edited = st.data_editor(shown, key=key, width="stretch", column_config=cfg)
    out = df.copy().astype(float)
    bad = []
    for c in df.columns:
        for i in df.index:
            try:
                out.loc[i, c] = parse_num(str(edited.loc[i, c]))
            except (ValueError, KeyError):
                bad.append(f"{i} / {c}: '{edited.loc[i, c] if c in edited.columns and i in edited.index else ''}'")
    if bad:
        refusal("Kept the previous value for cells that are not numbers in Romanian format: " + "; ".join(bad))
    return out


def dmy_hm(iso: str | None) -> str:
    """ISO-8601 UTC timestamp -> dd.mm.yyyy HH:MM; '–' when absent."""
    if not iso:
        return "–"
    try:
        t = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return str(iso)
    return t.strftime("%d.%m.%Y %H:%M")


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


ROLE_CLASSES = ("total", "subtotal", "memo", "check", "section")


def table(df: pd.DataFrame, decimals: int = 2, index_label: str = "", pct_rows: set[str] | None = None,
          total_rows: set[str] | None = None, scroll: bool = False, decimals_by_col: dict[str, int] | None = None,
          max_rows: int | None = None, units: list[str] | dict[str, str] | None = None,
          col_units: dict[str, str] | None = None, roles: list[str] | dict[str, str] | None = None,
          granularity: str = "monthly") -> None:
    """Brand table: navy header, white rows, right-aligned tabular numerals, RO formats.

    units: the unit of every row (list aligned with the rows, or dict by row label) - shown in a Unit column after
    the label so every number in the table carries its unit of measurement (G5 request 2). A row whose unit is "%"
    is formatted as a percentage, and the decimals of every row follow its unit and the table's granularity
    (UNIT_DECIMALS - the output-workbook catalogue; `decimals` is the fallback for units outside it).
    col_units: the unit of every column, shown in the header as "Column (unit)". roles: the role of every row
    (total, subtotal, memo, check, section, data) - rows stay white; a role changes weight, rule and tone only.
    """
    pct_rows = set(pct_rows or set())
    total_rows = total_rows or set()
    decimals_by_col = decimals_by_col or {}
    col_units = col_units or {}
    d = df if max_rows is None else df.head(max_rows)
    if isinstance(units, dict):
        unit_list = [units.get(str(i), "") for i in d.index]
    elif units is not None:
        unit_list = [str(u) for u in list(units)[: len(d)]]
    else:
        unit_list = None
    if isinstance(roles, dict):
        role_list = [roles.get(str(i), "data") for i in d.index]
    elif roles is not None:
        role_list = [str(x) for x in list(roles)[: len(d)]]
    else:
        role_list = None
    unit_dec = UNIT_DECIMALS.get(granularity, UNIT_DECIMALS["monthly"])
    if unit_list is not None:
        pct_rows |= {str(i) for i, u in zip(d.index, unit_list, strict=False) if u == "%"}
    text_cols = {c for c in d.columns if not pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_datetime64_any_dtype(d[c])
                 and all(isinstance(v, str) for v in d[c] if v is not None and v == v)}
    left = ' class="txt"'

    def _h(c) -> str:
        u = col_units.get(str(c), "")
        return f"{c} ({u})" if u else str(c)

    head = f"<th>{index_label}</th>" + ('<th class="txt">Unit</th>' if unit_list is not None else "")
    head += "".join(f"<th{left if c in text_cols else ''}>{_h(c)}</th>" for c in d.columns)
    rows = []
    for n, (idx, rec) in enumerate(d.iterrows()):
        key = str(idx)
        is_pct = key in pct_rows or key.endswith("_pct") or key.endswith(" %")
        row_dec = unit_dec.get(unit_list[n], decimals) if unit_list is not None and n < len(unit_list) else decimals
        cells = "".join(f"<td{left if c in text_cols else ''}>{_fmt_cell(v, decimals_by_col.get(str(c), row_dec), is_pct)}</td>" for c, v in rec.items())
        role = role_list[n] if role_list is not None and n < len(role_list) else ("total" if key in total_rows else "")
        cls = f' class="{role}"' if role in ROLE_CLASSES else ""
        ucell = f'<td class="unit">{unit_list[n] if n < len(unit_list) else ""}</td>' if unit_list is not None else ""
        rows.append(f"<tr{cls}><td>{key}</td>{ucell}{cells}</tr>")
    html = f'<table class="esb"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    if scroll:
        html = f'<div class="esb-scroll">{html}</div>'
    else:
        html = f'<div class="esb-wide">{html}</div>'  # a table wider than the page scrolls sideways instead of clipping (G5-7)
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
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "font": {"size": 11}, "traceorder": "normal"} if legend else None,
        showlegend=legend, colorway=SERIES, hovermode="x unified", separators=",.",
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

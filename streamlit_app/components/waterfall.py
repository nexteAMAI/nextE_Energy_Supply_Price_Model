"""
Waterfall Chart Component for price build-up visualization.
Used by Dashboard and Scenarios pages.
"""
import plotly.graph_objects as go
from typing import Dict


def create_waterfall_chart(
    components: Dict[str, float],
    title: str = "Price Build-Up Waterfall (EUR/MWh)",
) -> go.Figure:
    names = list(components.keys()) + ["Total"]
    values = list(components.values())
    total = sum(values)
    measures = ["relative"] * len(values) + ["total"]
    values_with_total = values + [total]

    fig = go.Figure(go.Waterfall(
        name="Price Build-Up",
        orientation="v",
        measure=measures,
        x=names,
        y=values_with_total,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#e74c3c"}},
        decreasing={"marker": {"color": "#2ecc71"}},
        totals={"marker": {"color": "#3498db"}},
    ))
    fig.update_layout(
        title=title, yaxis_title="EUR/MWh",
        showlegend=False, height=500,
    )
    return fig

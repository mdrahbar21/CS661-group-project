"""
viz_utils.py
All Plotly charts + PDF helper live here – keeps the Dash page thin.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

__all__ = (
    "dismissal_pie",
    "run_breakdown_pie",
    "gauge",
    "compare_bar",
    "export_summary_pdf",
)

# --------------------------- pie charts -----------------------------------


def dismissal_pie(df: pd.DataFrame, title: str):
    if df.empty:
        return go.Figure(layout=go.Layout(title=f"{title}<br>(none)"))
    fig = px.pie(df, names="Dismissal Type", values="Count",
                 title=title, hole=.35,
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=False, margin=dict(t=45, b=0, l=0, r=0))
    return fig


def run_breakdown_pie(r4: int, r6: int, ro: int, title: str):
    data = pd.DataFrame({
        "Run Type": ["4s", "6s", "Other"],
        "Runs": [r4, r6, ro]
    }).query("Runs > 0")
    if data.empty:
        return go.Figure(layout=go.Layout(title=f"{title}<br>(0)"))
    fig = px.pie(data, names="Run Type", values="Runs", hole=.35,
                 title=title, color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(textposition="outside", textinfo="percent+value")
    fig.update_layout(showlegend=False, margin=dict(t=45, b=0, l=0, r=0))
    return fig

# --------------------------- gauges & bars ---------------------------------


def gauge(value, reference, title, *, inverse=False):
    lo, hi = 0, max(value, reference or 0) * 1.5 + 1
    if inverse:
        value, reference = hi - value, hi - reference if reference else None
    fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if reference else ""),
        value=value,
        delta={"reference": reference} if reference else None,
        gauge={'axis': {'range': [lo, hi]}, 'bar': {'color': '#1f77b4'}},
        title={'text': title}
    ))
    fig.update_layout(height=250, margin=dict(t=45, b=0, l=20, r=20))
    return fig


def compare_bar(v1, v2, l1, l2, title):
    fig = go.Figure(go.Bar(x=[l1, l2], y=[v1, v2],
                           marker_color=['#1f77b4', '#ff7f0e'],
                           text=[f"{v1:,.2f}", f"{v2:,.2f}"],
                           textposition="auto"))
    fig.update_layout(title=title, yaxis_title="Value",
                      height=300, margin=dict(t=45, b=0, l=30, r=30))
    return fig

# --------------------------- PDF helper ------------------------------------


def export_summary_pdf(text: str) -> BytesIO:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    t = c.beginText(72, h-72)
    t.setFont("Helvetica", 10)
    for line in text.splitlines():
        t.textLine(line)
    c.drawText(t)
    c.save()
    buf.seek(0)
    return buf

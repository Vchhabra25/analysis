"""Plotly helpers: one dark theme, RR-inspired palette, restrained styling."""
from __future__ import annotations

import plotly.graph_objects as go

PINK, BLUE, SKY, GREY, LIGHT = "#EA1A85", "#5B84E6", "#9DB6F2", "#7F8AB0", "#C9D1EE"
TEXT, PANEL, BORDER, GRID = "#F2F4FA", "#101A3C", "#1E2B58", "rgba(154,165,200,0.14)"
COLORWAY = [PINK, BLUE, SKY, GREY, LIGHT, "#B5307F"]
FONT = "IBM Plex Sans, Segoe UI, system-ui, sans-serif"
CFG = {"displayModeBar": False, "responsive": True}


def style(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT, size=12), margin=dict(l=6, r=6, t=26, b=6),
        height=height, colorway=COLORWAY, showlegend=legend,
        hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color=TEXT, family=FONT)),
        legend=dict(orientation="h", y=1.14, x=0, bgcolor="rgba(0,0,0,0)", title_text=""),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=BORDER, title_font=dict(size=12))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=BORDER, title_font=dict(size=12))
    return fig


def radar(categories: list[str], series: dict[str, list[float]], height: int = 400) -> go.Figure:
    fig = go.Figure()
    for i, (name, vals) in enumerate(series.items()):
        v = [0 if x != x else x for x in vals]
        fig.add_trace(go.Scatterpolar(
            r=v + v[:1], theta=categories + categories[:1], name=name, fill="toself",
            opacity=0.55, line=dict(color=COLORWAY[i % len(COLORWAY)], width=2),
        ))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 100], gridcolor=GRID, linecolor=BORDER, tickfont=dict(size=10, color="#9AA5C8")),
        angularaxis=dict(gridcolor=GRID, linecolor=BORDER),
    ))
    return style(fig, height=height)

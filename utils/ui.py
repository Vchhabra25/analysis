"""Small HTML/CSS helpers so every page shares one visual language."""
from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from utils.data_loader import SCOPES, SCOPE_NOTES

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def inject_css() -> None:
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def sidebar_scope() -> str:
    st.sidebar.markdown('<div class="ai-side-title">Competition scope</div>', unsafe_allow_html=True)
    scope = st.sidebar.radio(
        "Competition scope", list(SCOPES), format_func=SCOPES.get,
        key="scope", label_visibility="collapsed",
    )
    st.sidebar.markdown(f'<div class="ai-side-note">{escape(SCOPE_NOTES[scope])}</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="ai-side-note" style="margin-top:1.4rem">Competitions are never pooled. '
        'Every figure on every page follows this selection.</div>',
        unsafe_allow_html=True,
    )
    return scope


def page_header(title: str, subtitle: str, lede: str | None = None) -> None:
    html = f'<h1 class="ai-title">{escape(title)}</h1><p class="ai-subtitle">{escape(subtitle)}</p><div class="ai-rule"></div>'
    if lede:
        html += f'<p class="ai-lede">{escape(lede)}</p>'
    st.markdown(html, unsafe_allow_html=True)


def section(title: str, sub: str | None = None) -> None:
    html = f'<div class="ai-section">{escape(title)}</div>'
    if sub:
        html += f'<div class="ai-section-sub">{escape(sub)}</div>'
    st.markdown(html, unsafe_allow_html=True)


def kpi(label: str, value, sub: str = "", pending: bool = False) -> None:
    cls = "ai-kpi pending" if pending else "ai-kpi"
    val = "Pending" if pending else value
    st.markdown(
        f'<div class="{cls}"><div class="ai-kpi-label">{escape(label)}</div>'
        f'<div class="ai-kpi-value">{escape(str(val))}</div>'
        f'<div class="ai-kpi-sub">{escape(sub)}</div></div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str) -> None:
    st.markdown(
        f'<div class="ai-empty"><div class="ai-empty-title">{escape(title)}</div>'
        f'<div class="ai-empty-body">{escape(body)}</div></div>',
        unsafe_allow_html=True,
    )


def chip(text: str, kind: str = "") -> str:
    return f'<span class="ai-chip {kind}">{escape(text)}</span>'


def status_rows(rows: list[tuple[str, str, str, str]]) -> None:
    """rows: (name, description, chip text, chip kind)"""
    html = '<div class="ai-card">'
    for name, desc, label, kind in rows:
        html += (
            f'<div class="ai-status-row"><div><div>{escape(name)}</div>'
            f'<div class="ai-status-sub">{escape(desc)}</div></div>{chip(label, kind)}</div>'
        )
    st.markdown(html + "</div>", unsafe_allow_html=True)


def fmt(x, nd: int = 1, suffix: str = "") -> str:
    try:
        if x is None or x != x:
            return "-"
        return f"{x:,.{nd}f}{suffix}"
    except (TypeError, ValueError):
        return "-"

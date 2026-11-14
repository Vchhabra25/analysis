"""Analysis Intelligence: entry point.

Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Analysis Intelligence", page_icon=":material/query_stats:",
                   layout="wide", initial_sidebar_state="expanded")

from views import (methodology, overview, player_comparison, player_intelligence,  # noqa: E402
                   pressure_analytics, reliability_rankings)
from utils.ui import ASSETS, inject_css, sidebar_scope  # noqa: E402

inject_css()
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")

nav = st.navigation([
    st.Page(overview.render, title="Overview", icon=":material/home:", url_path="overview", default=True),
    st.Page(player_intelligence.render, title="Player Intelligence", icon=":material/person_search:", url_path="players"),
    st.Page(pressure_analytics.render, title="Pressure Analytics", icon=":material/monitoring:", url_path="pressure"),
    st.Page(reliability_rankings.render, title="Reliability Rankings", icon=":material/leaderboard:", url_path="rankings"),
    st.Page(player_comparison.render, title="Player Comparison", icon=":material/compare_arrows:", url_path="compare"),
    st.Page(methodology.render, title="Methodology", icon=":material/menu_book:", url_path="methodology"),
])
sidebar_scope()
nav.run()

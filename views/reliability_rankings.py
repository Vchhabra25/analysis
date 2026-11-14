"""Reliability rankings built from the validated pressure engine (analytics/ outputs)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import analytics as an, ui, visualizations as viz
from utils.data_loader import SCOPES, current_scope

HEADLINE = {
    "IPL": "Primary ranking. IPL ball-by-ball data, out-of-fold models, IPL par tables.",
    "Emerging": "Prospect ranking. Scored on the IPL scale, never mixed with the IPL list. Samples are very small, so read it as a watchlist.",
    "T20I": "Separate analysis. T20I has its own par tables and thresholds and is not comparable with the IPL list.",
}


def _table(top: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Rank": top["rank"],
        "Player": top.player_name,
        "Pressure score": top.pressure_score,
        "80% range": top.score_lo80.map("{:.1f}".format) + " to " + top.score_hi80.map("{:.1f}".format),
        "Baseline": top.baseline,
        "Pressure vs baseline": top.gap,
        "Pressure balls": top.hp_balls,
        "Pressure innings": top.hp_innings,
        "Confidence": top.confidence,
        "P(top 10)": (top.p_top10 * 100).round(0),
        "Consistency %": top.consistency,
    })


def _interval_chart(top: pd.DataFrame) -> go.Figure:
    t = top.iloc[::-1]
    fig = go.Figure(go.Scatter(
        x=t.pressure_score, y=t.player_name, mode="markers",
        marker=dict(color=viz.PINK, size=9),
        error_x=dict(type="data", symmetric=False, array=(t.score_hi80 - t.pressure_score),
                     arrayminus=(t.pressure_score - t.score_lo80), color=viz.GREY, thickness=1.4, width=4),
        hovertemplate="%{y}<br>Score %{x:.1f}<extra></extra>"))
    fig.add_vline(x=0, line_color=viz.GREY, line_dash="dot")
    fig.update_xaxes(title="Pressure score (runs per 100 balls vs average), with 80% range")
    return viz.style(fig, 380, legend=False)


def _role_tab(scope: str, role: str, noun: str) -> None:
    r = an.ranking(scope, role)
    if r is None or r.empty:
        ui.empty_state("No ranking available", f"No {noun} met the minimum pressure sample in this scope.")
        return
    st.caption(an.eligibility_text(scope) + f" {len(r)} {noun} qualify.")
    n = 10 if len(r) >= 10 else len(r)
    top = r.head(n)
    ui.section(f"Top {n} {noun}", "Ranked by pressure score. Higher is better for every player, batter or bowler.")
    st.dataframe(_table(top), hide_index=True, width="stretch", column_config={
        "Pressure score": st.column_config.NumberColumn(format="%.1f", help="Runs per 100 pressure balls above an average player in the same innings and phase, after sample-size adjustment."),
        "Baseline": st.column_config.NumberColumn(format="%.1f", help="Same measure on the player's non-pressure balls (raw, unshrunk)."),
        "Pressure vs baseline": st.column_config.NumberColumn(format="%+.1f", help="Raw pressure score minus raw baseline. Noisy for small samples."),
        "P(top 10)": st.column_config.NumberColumn(format="%d%%", help="Posterior probability of belonging to the true top 10 of this eligible pool."),
        "Consistency %": st.column_config.NumberColumn(format="%.0f%%", help="Share of the player's pressure innings (3+ pressure balls) with above-average value."),
    })
    st.plotly_chart(_interval_chart(top), width="stretch", config=viz.CFG)
    st.caption("Overlapping ranges mean the order within the list is not statistically separable.")

    ui.section("Why a player ranks here", "Every figure traces back to the delivery-level data.")
    pick = st.selectbox("Player", top.player_name.tolist(), key=f"rr_pick_{scope}_{role}")
    p = r[r.player_name == pick].iloc[0]
    c = st.columns(4)
    with c[0]:
        ui.kpi("Pressure score", f"{p.pressure_score:+.1f}", f"80% range {p.score_lo80:+.1f} to {p.score_hi80:+.1f}")
    with c[1]:
        ui.kpi("Baseline", f"{p.baseline:+.1f}", f"{int(p.baseline_balls)} baseline balls")
    with c[2]:
        ui.kpi("Pressure vs baseline", f"{p.gap:+.1f}", f"z = {p.gap_z:+.2f} (noise-level if within +/-2)")
    with c[3]:
        ui.kpi("Pressure sample", f"{int(p.hp_balls)}", f"balls in {int(p.hp_innings)} innings; confidence {p.confidence}")
    st.markdown(f"**Strengths:** {p.strengths}")
    st.markdown(f"**Weaknesses:** {p.weaknesses}")
    st.markdown(f"**Risk factors:** {p.risks}")
    cl = an.clusters(scope, role)
    if cl is not None:
        m = cl[(cl.pid == p.player_id) & (cl.n_hp > 0)].sort_values("match_date", ascending=False)
        show = pd.DataFrame({
            "Date": m.match_date.dt.strftime("%d %b %Y"), "Competition": m.comp_name,
            "Pressure balls": m.n_hp, "Pressure value /100": (m.s_hp / m.n_hp * 100).round(1),
            "Baseline balls": m.n_nh,
            "Baseline value /100": (m.s_nh.where(m.n_nh > 0) / m.n_nh.where(m.n_nh > 0) * 100).round(1)})
        with st.expander(f"Innings behind this score ({len(show)} pressure innings)"):
            st.dataframe(show, hide_index=True, width="stretch", height=300)

    with st.expander(f"Full eligible list ({len(r)})"):
        full = _table(r).assign(**{"Rank (pressure evidence only)": r.rank_pressure_only})
        st.dataframe(full, hide_index=True, width="stretch", height=420)
        st.caption("The last column ranks players on pressure-ball evidence alone, ignoring baseline. "
                   "It is shown for transparency and is noisier than the main rank.")


def render() -> None:
    scope = current_scope()
    ui.page_header("Reliability Rankings", f"Top 10 under pressure - {SCOPES[scope]}",
                   "Separate batting and bowling lists. A score is runs per 100 pressure balls above an average player "
                   "in the same innings and phase, adjusted for sample size. Higher is better for both roles.")
    if not an.available():
        ui.empty_state("Analytics outputs not found", "Run `python -m engine.run` to generate the analytics folder.")
        return
    st.markdown(ui.chip(HEADLINE[scope], "live"), unsafe_allow_html=True)
    st.write("")
    v = an.validation()[an.SCOPE_FOR_VALIDATION[scope]]
    hb = v["gap_heterogeneity"]["batting"]
    hw = v["gap_heterogeneity"]["bowling"]
    st.info(
        f"Read this first: across eligible players, differences in how much each one changes under pressure are "
        f"not distinguishable from noise (heterogeneity test p = {hb['p_value']:.2f} batting, {hw['p_value']:.2f} bowling). "
        "The scores therefore combine pressure-ball evidence with each player's baseline, weighted by sample size. "
        "Treat the lists as best estimates with stated uncertainty, not as proof that a player 'rises' or 'chokes'.")
    tab_bat, tab_bowl = st.tabs(["Batting reliability", "Bowling reliability"])
    with tab_bat:
        _role_tab(scope, "bat", "batters")
    with tab_bowl:
        _role_tab(scope, "bowl", "bowlers")

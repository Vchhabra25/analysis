"""Situational splits plus the validated high-pressure vs ordinary results."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import analytics as an, ui, visualizations as viz
from utils.data_loader import SCOPES, batting_agg, bowling_agg, current_scope, get_balls, player_summary


def _names(scope: str) -> pd.DataFrame:
    return player_summary(scope)[["player_id", "label"]]


def _scatter(d: pd.DataFrame, x: str, y: str, size: str, xl: str, yl: str, hover: dict) -> go.Figure:
    fig = px.scatter(d, x=x, y=y, size=size, hover_name="label", hover_data=hover,
                     labels={x: xl, y: yl}, size_max=22)
    fig.update_traces(marker=dict(color=viz.BLUE, opacity=0.75, line=dict(width=0.5, color=viz.BORDER)))
    lo, hi = float(min(d[x].min(), d[y].min())), float(max(d[x].max(), d[y].max()))
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color=viz.GREY, dash="dot", width=1))
    return viz.style(fig, 420, legend=False)


def _batting(df: pd.DataFrame, names: pd.DataFrame) -> None:
    c1, c2 = st.columns(2)
    min_death = c1.slider("Minimum death-over balls faced (overs 17-20)", 20, 150, 50, step=10, key="pa_bd")
    min_chase = c2.slider("Minimum balls faced in each innings type", 40, 300, 100, step=20, key="pa_bc")

    ui.section("Death overs", "Strike rate in overs 17-20 against the same batter's strike rate in all other overs.")
    allb = batting_agg(df).set_index("player_id")
    dth = batting_agg(df[df.phase == "Death"]).set_index("player_id")
    oth = batting_agg(df[df.phase != "Death"]).set_index("player_id")
    d = dth[dth.balls >= min_death].join(oth[["sr", "balls"]], rsuffix="_other", how="inner").reset_index()
    d = d[d.balls_other >= 60].merge(names, on="player_id")
    if d.empty:
        ui.empty_state("No players meet the sample threshold", "Lower the minimum death-over balls or widen the season filter.")
    else:
        fig = _scatter(d, "sr_other", "sr", "balls", "Strike rate in overs 1-16", "Strike rate in overs 17-20",
                       {"balls": True, "outs": True})
        st.plotly_chart(fig, width="stretch", config=viz.CFG)
        st.caption(f"{len(d)} batters. Points above the dotted line score faster at the death. Bubble size is death-over balls faced.")

    ui.section("Chase situations", "Strike rate when chasing against when batting first.")
    f1 = batting_agg(df[df.innings == 1]).set_index("player_id")
    f2 = batting_agg(df[df.innings == 2]).set_index("player_id")
    d = f2[f2.balls >= min_chase].join(f1[["sr", "balls"]], rsuffix="_first", how="inner").reset_index()
    d = d[d.balls_first >= min_chase].merge(names, on="player_id")
    if d.empty:
        ui.empty_state("No players meet the sample threshold", "Lower the minimum balls or widen the season filter.")
    else:
        fig = _scatter(d, "sr_first", "sr", "balls", "Strike rate batting first", "Strike rate chasing",
                       {"balls": True, "balls_first": True})
        st.plotly_chart(fig, width="stretch", config=viz.CFG)
        st.caption(f"{len(d)} batters. Chasing strike rate is not pressure-adjusted: it mixes easy and difficult chases.")


def _bowling(df: pd.DataFrame, names: pd.DataFrame) -> None:
    c1, c2 = st.columns(2)
    min_death = c1.slider("Minimum death-over overs bowled", 5, 40, 12, step=1, key="pa_wd")
    min_inn = c2.slider("Minimum overs bowled in each innings type", 10, 80, 20, step=5, key="pa_wc")

    ui.section("Death overs", "Economy in overs 17-20 against the same bowler's economy in all other overs.")
    dth = bowling_agg(df[df.phase == "Death"]).set_index("player_id")
    oth = bowling_agg(df[df.phase != "Death"]).set_index("player_id")
    d = dth[dth.balls >= min_death * 6].join(oth[["econ", "balls"]], rsuffix="_other", how="inner").reset_index()
    d = d[d.balls_other >= 60].merge(names, on="player_id")
    if d.empty:
        ui.empty_state("No players meet the sample threshold", "Lower the minimum overs or widen the season filter.")
    else:
        fig = _scatter(d, "econ_other", "econ", "balls", "Economy in overs 1-16", "Economy in overs 17-20",
                       {"balls": True, "wkts": True})
        st.plotly_chart(fig, width="stretch", config=viz.CFG)
        st.caption(f"{len(d)} bowlers. Points below the dotted line are cheaper at the death. Bubble size is death-over balls bowled.")

    ui.section("Defending and chasing", "Economy when the opposition is chasing (bowling first) against when defending a total.")
    i1 = bowling_agg(df[df.innings == 1]).set_index("player_id")   # bowling in innings 1 = defending
    i2 = bowling_agg(df[df.innings == 2]).set_index("player_id")   # bowling in innings 2 = opposition chasing
    d = i1[i1.balls >= min_inn * 6].join(i2[["econ", "balls"]], rsuffix="_chase", how="inner").reset_index()
    d = d[d.balls_chase >= min_inn * 6].merge(names, on="player_id")
    if d.empty:
        ui.empty_state("No players meet the sample threshold", "Lower the minimum overs or widen the season filter.")
    else:
        fig = _scatter(d, "econ", "econ_chase", "balls", "Economy when bowling first", "Economy when bowling second",
                       {"balls": True, "balls_chase": True})
        st.plotly_chart(fig, width="stretch", config=viz.CFG)
        st.caption(f"{len(d)} bowlers. Not pressure-adjusted: a defended total can be small or large.")


def render() -> None:
    scope = current_scope()
    ui.page_header("Pressure Analytics", f"Situational performance - {SCOPES[scope]}",
                   "Death-over and chase splits are descriptive statistics from the data. The last tab shows how the "
                   "validated Pressure Index separates high-pressure from ordinary situations.")
    df = get_balls(scope)
    seasons = sorted(df.season.astype(str).unique())
    sel = st.multiselect("Seasons", seasons, default=seasons, key=f"pa_seasons_{scope}")
    if not sel:
        ui.empty_state("Select at least one season", "Choose a season above to load the charts.")
        return
    df = df[df.season.astype(str).isin(sel)]
    names = _names(scope)

    tab_bat, tab_bowl, tab_hp = st.tabs(["Batting under pressure", "Bowling under pressure", "High pressure vs ordinary"])
    with tab_bat:
        _batting(df, names)
    with tab_bowl:
        _bowling(df, names)
    with tab_hp:
        _high_vs_ordinary(scope)


def _high_vs_ordinary(scope: str) -> None:
    if not an.available():
        ui.empty_state("Analytics outputs not found", "Run `python -m engine.run` to generate the analytics folder.")
        return
    key = an.SCOPE_FOR_VALIDATION[scope]
    v = an.validation()[key]
    if scope == "Emerging":
        st.info("Emerging players are scored with the IPL pressure index, so the checks below are the IPL results.")
    ui.section("High-pressure vs ordinary balls",
               "League-wide rates. High pressure = top 20% of the Pressure Index for that role; quietest = bottom 50%.")
    t = pd.DataFrame(v["hp_vs_ordinary"]["table"]).rename(columns={
        "role": "Role", "group": "Balls", "balls": "Count", "runs_per_ball": "Runs per ball", "dot_pct": "Dot %",
        "boundary_pct": "Boundary %", "wicket_pct": "Wicket %"}).drop(columns=["value_per_100"])
    st.dataframe(t.round(2), hide_index=True, width="stretch", column_config={"Count": st.column_config.NumberColumn(format="localized")})
    st.caption("Raw rates, not adjusted for over or wickets. High-pressure balls sit later in the innings, so they "
               "naturally score faster. The player scores use context-adjusted values instead.")

    l, r = st.columns(2)
    d = v["difficulty"]
    with l:
        ui.section("Pressure rises as chases get harder", "Average by required run rate when chasing.")
        b = pd.DataFrame(d["by_rrr"])
        fig = go.Figure()
        fig.add_bar(x=b.rrr, y=b.pi_bat, name="Batter pressure", marker_color=viz.PINK)
        fig.add_bar(x=b.rrr, y=b.pi_bowl, name="Bowler pressure", marker_color=viz.BLUE)
        fig.add_scatter(x=b.rrr, y=b.win_prob, name="Chase win prob.", mode="lines+markers",
                        line=dict(color=viz.LIGHT, dash="dot"), yaxis="y2")
        fig.update_layout(barmode="group", yaxis2=dict(overlaying="y", side="right", range=[0, 1], showgrid=False, title="Win probability", tickformat=".1f"))
        fig.update_xaxes(title="Required run rate (runs per over)", type="category")
        fig.update_yaxes(title="Average Pressure Index")
        fig = viz.style(fig, 360)
        fig.update_layout(legend=dict(orientation="h", y=1.2, x=0))
        st.plotly_chart(fig, width="stretch", config=viz.CFG)
        st.caption("Batter pressure peaks at 10-15 an over: hard but still alive. At 15+ the chase is mostly lost and leverage falls.")
    with r:
        ui.section("Pressure builds towards the end of a chase", "Average Pressure Index by over band.")
        o = pd.DataFrame(d["by_over"])
        fig = go.Figure()
        for inn, col in [("Batting first", viz.BLUE), ("Chasing", viz.PINK)]:
            x = o[o.innings == inn]
            fig.add_scatter(x=x.overid, y=x.pi_bat, name=inn, mode="lines+markers", line=dict(color=col))
        fig.update_xaxes(title="Overs", type="category")
        fig.update_yaxes(title="Average Pressure Index (batter)")
        st.plotly_chart(viz.style(fig, 340), width="stretch", config=viz.CFG)
        st.caption("First-innings overs, including the death, are rarely flagged as high pressure. Pressure concentrates in chases.")

    ui.section("Does the index agree with intuition?", "Share of balls flagged as high pressure for the batter.")
    lift = pd.DataFrame(d["tag_lift"]).rename(columns={
        "situation": "Situation", "balls": "Balls", "high_pressure_share": "High-pressure share", "share_elsewhere": "Elsewhere"})
    lift["High-pressure share"] = (lift["High-pressure share"] * 100).round(1)
    lift["Elsewhere"] = (lift["Elsewhere"] * 100).round(1)
    st.dataframe(lift, hide_index=True, width="stretch", column_config={
        "High-pressure share": st.column_config.NumberColumn(format="%.1f%%"), "Elsewhere": st.column_config.NumberColumn(format="%.1f%%")})
    st.caption("Chases, close finishes and steep required rates are strongly over-represented. "
               "Death overs and wickets down are not, because they matter only when the match is also close.")

    ui.section("Do players differ in how they handle pressure?")
    hb, hw = v["gap_heterogeneity"]["batting"], v["gap_heterogeneity"]["bowling"]
    st.markdown(
        f"Heterogeneity test on each player's pressure-minus-baseline gap: batting Q = {hb['Q']:.0f} on {hb['df']} df "
        f"(p = {hb['p_value']:.2f}); bowling Q = {hw['Q']:.0f} on {hw['df']} df (p = {hw['p_value']:.2f}). "
        f"{hb['share_abs_z_gt_1.96'] * 100:.1f}% of batters and {hw['share_abs_z_gt_1.96'] * 100:.1f}% of bowlers have an "
        "individual gap beyond two standard errors, against 5% expected from chance alone. "
        "With this much data there is no reliable evidence that individual players systematically rise or fall under pressure.")

"""Side-by-side comparison using real descriptive metrics (percentiles vs the qualified pool)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import analytics as an, ui, visualizations as viz
from utils.data_loader import (SCOPES, batting_agg, bowling_agg, current_scope, get_balls,
                               percentile_in_pool, player_summary)

MIN_BAT, MIN_BAT_DEATH, MIN_BOWL, MIN_BOWL_DEATH = 100, 30, 120, 30


def _pct_table(rows: pd.DataFrame, pool: pd.DataFrame, spec: list[tuple[str, str, bool]]) -> pd.DataFrame:
    out = {}
    for _, r in rows.iterrows():
        out[r["label"]] = [percentile_in_pool(pool[col], r[col], hib) for _, col, hib in spec]
    return pd.DataFrame(out, index=[n for n, _, _ in spec])


def _strengths(pct: pd.DataFrame) -> None:
    for name in pct.columns:
        s = pct[name].dropna().sort_values(ascending=False)
        if len(s) >= 3:
            st.markdown(f"**{name}**: strongest on {s.index[0].lower()} ({s.iloc[0]:.0f}th percentile) and "
                        f"{s.index[1].lower()} ({s.iloc[1]:.0f}th). Weakest on {s.index[-1].lower()} ({s.iloc[-1]:.0f}th).")


def render() -> None:
    scope = current_scope()
    ui.page_header("Player Comparison", f"Compare two to four players - {SCOPES[scope]}",
                   "Percentiles are measured against players in the same competition scope who meet a minimum sample. "
                   "They describe volume and style, not reliability under pressure.")
    s = player_summary(scope)
    picks = st.multiselect("Select players (2 to 4)", s.label.tolist(), max_selections=4, key=f"pc_pick_{scope}")
    if len(picks) < 2:
        ui.empty_state("Select at least two players", "Search by name in the box above. Start typing to filter the list.")
        return
    df = get_balls(scope)
    sel = s[s.label.isin(picks)].set_index("label").loc[picks].reset_index()
    ids = sel.player_id.astype(int).tolist()
    lab = dict(zip(sel.player_id.astype(int), sel.label))

    tab_bat, tab_bowl, tab_pr = st.tabs(["Batting", "Bowling", "Under pressure"])

    with tab_bat:
        allb = batting_agg(df)
        death = batting_agg(df[df.phase == "Death"])[["player_id", "sr", "balls"]].rename(columns={"sr": "death_sr", "balls": "death_balls"})
        allb = allb.merge(death, on="player_id", how="left")
        allb["low_dot"] = allb.dot_pct
        pool = allb[allb.balls >= MIN_BAT]
        mine = allb[allb.player_id.isin(ids)].copy()
        mine["label"] = mine.player_id.map(lab)
        mine = mine[mine.balls > 0]
        if len(mine) < 2:
            ui.empty_state("Not enough batting data", "At least two of the selected players need to have batted in this scope.")
        else:
            order = [lab[i] for i in ids if lab[i] in set(mine.label)]
            tbl = mine.set_index("label").loc[order]
            show = pd.DataFrame({
                "Innings": tbl.inns, "Balls faced": tbl.balls, "Runs": tbl.runs, "Strike rate": tbl.sr.round(1),
                "Average": tbl.avg.round(1), "Dot ball %": tbl.dot_pct.round(1), "Boundary %": tbl.bnd_pct.round(1),
                "Death-over SR": tbl.death_sr.round(1), "Death balls": tbl.death_balls,
            }).T
            st.dataframe(show, width="stretch")
            spec = [("Strike rate", "sr", True), ("Average", "avg", True), ("Boundary %", "bnd_pct", True),
                    ("Few dot balls", "dot_pct", False), ("Death-over SR", "death_sr", True), ("Volume", "balls", True)]
            pct = _pct_table(tbl.reset_index(), pool, spec)
            fig = viz.radar(list(pct.index), {c: pct[c].tolist() for c in pct.columns})
            st.plotly_chart(fig, width="stretch", config=viz.CFG)
            st.caption(f"Percentile among {len(pool)} batters with at least {MIN_BAT} balls faced. "
                       f"Death-over SR is only meaningful with {MIN_BAT_DEATH}+ death balls; a missing axis plots as zero.")
            small = [n for n in order if tbl.loc[n, "balls"] < MIN_BAT]
            if small:
                st.info("Below the minimum sample, read with caution: " + ", ".join(small))
            _strengths(pct)

    with tab_bowl:
        allw = bowling_agg(df)
        deathw = bowling_agg(df[df.phase == "Death"])[["player_id", "econ", "balls"]].rename(columns={"econ": "death_econ", "balls": "death_balls"})
        allw = allw.merge(deathw, on="player_id", how="left")
        pool = allw[allw.balls >= MIN_BOWL]
        mine = allw[allw.player_id.isin(ids)].copy()
        mine["label"] = mine.player_id.map(lab)
        if len(mine) < 2:
            ui.empty_state("Not enough bowling data", "At least two of the selected players need to have bowled in this scope.")
        else:
            order = [lab[i] for i in ids if lab[i] in set(mine.label)]
            tbl = mine.set_index("label").loc[order]
            show = pd.DataFrame({
                "Overs": (tbl.balls / 6).round(1), "Wickets": tbl.wkts, "Economy": tbl.econ.round(2),
                "Average": tbl.avg.round(1), "Strike rate": tbl.sr.round(1), "Dot ball %": tbl.dot_pct.round(1),
                "Boundary conceded %": tbl.bnd_pct.round(1), "Death economy": tbl.death_econ.round(2),
                "Death overs": (tbl.death_balls / 6).round(1),
            }).T
            st.dataframe(show, width="stretch")
            spec = [("Economy", "econ", False), ("Wicket rate", "wkt_rate", True), ("Dot ball %", "dot_pct", True),
                    ("Few boundaries", "bnd_pct", False), ("Death economy", "death_econ", False), ("Volume", "balls", True)]
            pct = _pct_table(tbl.reset_index(), pool, spec)
            fig = viz.radar(list(pct.index), {c: pct[c].tolist() for c in pct.columns})
            st.plotly_chart(fig, width="stretch", config=viz.CFG)
            st.caption(f"Percentile among {len(pool)} bowlers with at least {MIN_BOWL // 6} overs. "
                       "For economy and boundaries conceded, a higher percentile means better (lower) numbers.")
            small = [n for n in order if tbl.loc[n, "balls"] < MIN_BOWL]
            if small:
                st.info("Below the minimum sample, read with caution: " + ", ".join(small))
            _strengths(pct)

    with tab_pr:
        _pressure_tab(scope, ids, lab)


def _pressure_tab(scope: str, ids: list[int], lab: dict) -> None:
    if not an.available():
        ui.empty_state("Analytics outputs not found", "Run `python -m engine.run` to generate the analytics folder.")
        return
    shown = False
    for role, noun in [("bat", "Batting"), ("bowl", "Bowling")]:
        r = an.ranking(scope, role)
        if r is None:
            continue
        rows = r[r.player_id.isin(ids)]
        missing = [lab[i] for i in ids if i not in set(rows.player_id)]
        if rows.empty:
            continue
        shown = True
        ui.section(f"{noun} under pressure", "Pressure score is runs per 100 pressure balls vs an average player; higher is better.")
        t = pd.DataFrame({
            "Player": rows.player_name, "Rank": rows["rank"], "Pressure score": rows.pressure_score,
            "80% range": rows.score_lo80.map("{:.1f}".format) + " to " + rows.score_hi80.map("{:.1f}".format),
            "Baseline": rows.baseline, "Pressure vs baseline": rows.gap, "Pressure balls": rows.hp_balls,
            "Confidence": rows.confidence})
        st.dataframe(t, hide_index=True, width="stretch", column_config={
            "Pressure score": st.column_config.NumberColumn(format="%.1f"), "Baseline": st.column_config.NumberColumn(format="%.1f"),
            "Pressure vs baseline": st.column_config.NumberColumn(format="%+.1f")})
        fig = go.Figure(go.Bar(
            x=rows.player_name, y=rows.pressure_score, marker_color=viz.PINK,
            error_y=dict(type="data", symmetric=False, array=rows.score_hi80 - rows.pressure_score,
                         arrayminus=rows.pressure_score - rows.score_lo80, color=viz.GREY)))
        fig.update_yaxes(title="Pressure score with 80% range")
        st.plotly_chart(viz.style(fig, 300, legend=False), width="stretch", config=viz.CFG)
        if missing:
            st.caption(f"Not ranked in this role (below the minimum pressure sample): {', '.join(missing)}.")
    if not shown:
        ui.empty_state("None of the selected players are ranked", an.eligibility_text(scope))

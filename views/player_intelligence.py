"""Player search and profile cards built from real ball-by-ball data."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import analytics as an, ui, visualizations as viz
from utils.data_loader import (PHASES, SCOPES, batting_agg, bowling_agg, current_scope,
                               get_balls, player_summary, player_teams)


def _initials(name: str) -> str:
    parts = [p for p in name.replace("(", " ").split() if p[:1].isalpha()]
    return "".join(p[0] for p in parts[:2]).upper() or "?"


def _pressure_line(scope: str, role: str, pid: int) -> None:
    if not an.available():
        return
    row = an.player_row(scope, role, pid)
    n = len(an.ranking(scope, role))
    if row is None:
        st.caption("Pressure ranking: not ranked in this scope (below the minimum pressure sample).")
        return
    st.markdown(
        ui.chip(f"Pressure rank {int(row['rank'])} of {n}", "live") + " "
        + ui.chip(f"Score {row.pressure_score:+.1f} per 100 balls (80% range {row.score_lo80:+.1f} to {row.score_hi80:+.1f})") + " "
        + ui.chip(f"{int(row.hp_balls)} pressure balls") + " " + ui.chip(f"Confidence {row.confidence}"),
        unsafe_allow_html=True)
    st.write("")


def render() -> None:
    scope = current_scope()
    ui.page_header("Player Intelligence", f"Search and profile players - {SCOPES[scope]}",
                   "Descriptive batting and bowling statistics from the ball-by-ball data. "
                   "The statistics are career-in-scope descriptive numbers; the pressure line shows the validated ranking where the player qualifies.")
    s = player_summary(scope)
    q = st.text_input("Search by player name", placeholder="Start typing, for example Kohli", key="pi_search")
    pool = s[s.label.str.contains(q.strip(), case=False, regex=False)] if q.strip() else s
    if pool.empty:
        ui.empty_state("No players match that search", f"Nothing in {SCOPES[scope]} matches \"{q}\". Try a shorter spelling or switch the competition scope.")
        return
    shown = min(len(pool), 150)
    st.caption(f"{len(pool):,} player(s) match; the {shown} most active are listed. Type a name to narrow the list.")
    options = pool.head(150).label.tolist()
    label = st.selectbox("Select a player", options, key=f"pi_select_{scope}")
    p = s[s.label == label].iloc[0]
    pid = int(p.player_id)
    df = get_balls(scope)
    teams = player_teams(pid, scope)

    meta = [p.usage]
    if isinstance(p.batting_hand, str):
        meta.append(f"{p.batting_hand} bat")
    if isinstance(p.bowling_type, str):
        meta.append(p.bowling_type)
    if p.typical_order == p.typical_order and p.bat_balls > 0:
        meta.append(f"usually bats {int(p.typical_order)}")
    st.markdown(
        f'<div class="ai-card"><div class="ai-profile"><div class="ai-avatar">{_initials(p.player_name)}</div>'
        f'<div><div class="ai-profile-name">{p.player_name}</div>'
        f'<div class="ai-profile-meta">{" | ".join(meta)}</div>'
        f'<div class="ai-profile-meta">Teams in {SCOPES[scope]}: {", ".join(teams) if teams else "not listed in mapping"}</div>'
        f'</div></div></div>', unsafe_allow_html=True)
    st.caption("Usage profile is derived: 60+ balls faced and 60+ balls bowled marks an all-rounder. "
               "The dataset has no official role field.")

    tab_bat, tab_bowl, tab_comp, tab_log = st.tabs(["Batting", "Bowling", "Competition-wise", "Innings log"])

    with tab_bat:
        if p.bat_balls == 0:
            ui.empty_state("No batting data", "This player has not faced a delivery in the selected scope.")
        else:
            mine = df[(df.striker_id == pid) | (df.dismissed_player_id == pid)]
            tot = batting_agg(mine).query("player_id == @pid").iloc[0]
            cols = st.columns(6)
            vals = [("Innings", ui.fmt(tot.inns, 0)), ("Runs", ui.fmt(tot.runs, 0)),
                    ("Strike rate", ui.fmt(tot.sr)), ("Average", ui.fmt(tot.avg)),
                    ("Dot ball %", ui.fmt(tot.dot_pct, 1, "%")), ("Boundary %", ui.fmt(tot.bnd_pct, 1, "%"))]
            for col, (a, b) in zip(cols, vals):
                with col:
                    ui.kpi(a, b, f"{int(tot.balls)} balls" if a == "Innings" else "")
            _pressure_line(scope, "bat", pid)
            ph = batting_agg(mine, ["phase"]).query("player_id == @pid")
            sp = batting_agg(mine, ["pace_or_spin"]).query("player_id == @pid")
            l, r = st.columns(2)
            with l:
                ui.section("Strike rate by phase")
                fig = px.bar(ph, x="phase", y="sr", category_orders={"phase": PHASES}, text=ph.balls.astype(str) + " balls",
                             labels={"phase": "", "sr": "Strike rate"})
                fig.update_traces(marker_color=viz.BLUE, textposition="outside")
                st.plotly_chart(viz.style(fig, 300, False), width="stretch", config=viz.CFG)
            with r:
                ui.section("Against pace and spin")
                fig = px.bar(sp, x="pace_or_spin", y="sr", text=sp.balls.astype(str) + " balls",
                             labels={"pace_or_spin": "", "sr": "Strike rate"})
                fig.update_traces(marker_color=viz.PINK, textposition="outside")
                st.plotly_chart(viz.style(fig, 300, False), width="stretch", config=viz.CFG)
            if tot.balls < 100:
                st.info("Fewer than 100 balls faced: treat these splits as indicative only.")

    with tab_bowl:
        if p.bowl_balls == 0:
            ui.empty_state("No bowling data", "This player has not bowled a legal delivery in the selected scope.")
        else:
            bt = bowling_agg(df[df.bowler_id == pid]).iloc[0]
            cols = st.columns(6)
            vals = [("Overs", ui.fmt(bt.balls / 6)), ("Wickets", ui.fmt(bt.wkts, 0)),
                    ("Economy", ui.fmt(bt.econ, 2)), ("Average", ui.fmt(bt.avg)),
                    ("Dot ball %", ui.fmt(bt.dot_pct, 1, "%")), ("Boundary conceded %", ui.fmt(bt.bnd_pct, 1, "%"))]
            for col, (a, b) in zip(cols, vals):
                with col:
                    ui.kpi(a, b)
            _pressure_line(scope, "bowl", pid)
            ph = bowling_agg(df[df.bowler_id == pid], ["phase"])
            inn = bowling_agg(df[df.bowler_id == pid], ["innings"])
            inn["innings"] = inn.innings.map({1: "Batting first", 2: "Chasing"})
            l, r = st.columns(2)
            with l:
                ui.section("Economy by phase")
                fig = px.bar(ph, x="phase", y="econ", category_orders={"phase": PHASES},
                             text=(ph.balls / 6).round(1).astype(str) + " ov", labels={"phase": "", "econ": "Economy"})
                fig.update_traces(marker_color=viz.BLUE, textposition="outside")
                st.plotly_chart(viz.style(fig, 300, False), width="stretch", config=viz.CFG)
            with r:
                ui.section("Economy by innings")
                fig = px.bar(inn, x="innings", y="econ", text=(inn.balls / 6).round(1).astype(str) + " ov",
                             labels={"innings": "Bowling in innings", "econ": "Economy"})
                fig.update_traces(marker_color=viz.PINK, textposition="outside")
                st.plotly_chart(viz.style(fig, 300, False), width="stretch", config=viz.CFG)

    with tab_comp:
        rows = []
        bat_c = batting_agg(df[(df.striker_id == pid) | (df.dismissed_player_id == pid)], ["comp_name"])
        bat_c = bat_c[bat_c.player_id == pid].set_index("comp_name")
        bowl_c = bowling_agg(df[df.bowler_id == pid], ["comp_name"]).set_index("comp_name")
        comps = sorted(set(bat_c.index) | set(bowl_c.index))
        for cname in comps:
            r = {"Competition": cname}
            if cname in bat_c.index:
                x = bat_c.loc[cname]
                r.update({"Bat inns": int(x.inns), "Runs": int(x.runs), "Balls": int(x.balls),
                          "SR": round(x.sr, 1), "Avg": None if pd.isna(x.avg) else round(x.avg, 1)})
            if cname in bowl_c.index:
                x = bowl_c.loc[cname]
                r.update({"Overs": round(x.balls / 6, 1), "Wkts": int(x.wkts), "Econ": round(x.econ, 2)})
            rows.append(r)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            ui.empty_state("No competition rows", "No deliveries recorded for this player in the selected scope.")

    with tab_log:
        d = df[(df.striker_id == pid) & df.faced]
        if d.empty:
            ui.empty_state("No innings to show", "This player has not batted in the selected scope.")
        else:
            g = d.groupby(["match_date", "comp_name", "mi"], observed=True).agg(
                Runs=("runs", "sum"), Balls=("faced", "size"), Fours=("is_four", "sum"), Sixes=("is_six", "sum")).reset_index()
            outs = set(df[(df.dismissed_player_id == pid) & df.batter_out].mi)
            g["Out"] = g.mi.isin(outs)
            g["SR"] = (g.Runs / g.Balls * 100).round(1)
            g = g.sort_values("match_date")
            fig = px.bar(g, x="match_date", y="Runs", color=g.Out.map({True: "Dismissed", False: "Not out"}),
                         color_discrete_map={"Dismissed": viz.BLUE, "Not out": viz.PINK},
                         hover_data={"Balls": True, "SR": True, "comp_name": True},
                         labels={"match_date": "Date", "color": ""})
            st.plotly_chart(viz.style(fig, 300), width="stretch", config=viz.CFG)
            show = g[["match_date", "comp_name", "Runs", "Balls", "SR", "Fours", "Sixes", "Out"]].rename(
                columns={"match_date": "Date", "comp_name": "Competition"}).iloc[::-1]
            show["Date"] = show.Date.dt.strftime("%d %b %Y")
            st.dataframe(show, hide_index=True, width="stretch", height=300)

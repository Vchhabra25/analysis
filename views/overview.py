"""Home / overview dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import analytics as an, ui, visualizations as viz
from utils.data_loader import PHASES, SCOPES, current_scope, dataset_counts, get_balls, load_tables


def render() -> None:
    scope = current_scope()
    c = dataset_counts()
    ui.page_header(
        "Analysis Intelligence",
        "Cricket Performance & Pressure Intelligence",
        "A scouting workbench that looks past averages and strike rates to ask which batters and bowlers "
        "hold their level when the match is tight. The data layer, descriptive analytics, Pressure Index and "
        "reliability rankings are live, and every number traces back to the delivery-level data.",
    )

    cols = st.columns(4)
    with cols[0]:
        ui.kpi("Matches analysed", f"{c['matches_analysed']:,}",
               f"of {c['matches_listed']:,} listed")
    with cols[1]:
        ui.kpi("Players in dataset", f"{c['players_mapped']:,}", "in team mappings")
    with cols[2]:
        ui.kpi("Competitions covered", f"{c['competitions_with_data']}",
               f"of {c['competitions_listed']} listed")
    with cols[3]:
        ui.kpi("Players profiled", f"{c['players_profiled']:,}",
               "with at least one delivery")

    if an.available():
        mt, v = an.meta()["scopes"]["IPL"], an.validation()["IPL"]
        hp_bat = v["hp_vs_ordinary"]["table"][0]["balls"]
        nb, nw = len(an.ranking("IPL", "bat")), len(an.ranking("IPL", "bowl"))
        ui.section("Pressure engine (IPL)", "Every figure comes from the analytics outputs, scored out-of-fold.")
        cols = st.columns(4)
        with cols[0]:
            ui.kpi("Matches in pressure model", f"{mt['exclusions']['usable']}",
                   f"of {mt['exclusions']['matches_with_ball_data']}; D/L, no-result, incomplete excluded")
        with cols[1]:
            ui.kpi("High-pressure balls (batting)", f"{hp_bat:,}", "top 20% of the Pressure Index")
        with cols[2]:
            ui.kpi("Players ranked", f"{nb + nw}", f"{nb} batters, {nw} bowlers")
        with cols[3]:
            ui.kpi("Chase win-probability AUC", f"{v['wp']['oof']['2']['auc']:.2f}", "out-of-fold, unseen matches")
    else:
        ui.section("Analytics in progress", "Run `python -m engine.run` to generate the pressure analytics.")
        cols = st.columns(4)
        for col, (lab, sub) in zip(cols, [("Pressure index", "not generated"), ("Players rated", "not generated"),
                                          ("Reliability rankings", "not generated"), ("Model validation", "not generated")]):
            with col:
                ui.kpi(lab, "", sub, pending=True)

    b = load_tables()["balls"]
    ui.section("Coverage by season", "Matches with ball-by-ball data, by competition group.")
    m = b.groupby(["season", "scope"], observed=True).match_id.nunique().reset_index(name="matches")
    m["scope"] = m.scope.map(SCOPES)
    fig = px.bar(m, x="season", y="matches", color="scope", barmode="stack",
                 labels={"season": "Season", "matches": "Matches", "scope": ""},
                 color_discrete_sequence=[viz.PINK, viz.BLUE, viz.GREY])
    st.plotly_chart(viz.style(fig, 300), width="stretch", config=viz.CFG)

    df = get_balls(scope)
    ui.section(f"Scoring trends: {SCOPES[scope]}", "Descriptive league-wide context. Not pressure-adjusted.")
    left, right = st.columns(2)
    with left:
        st.caption("Average first-innings total by season")
        inn = df[df.innings == 1].groupby(["season", "mi"], observed=True).total_runs.sum().reset_index()
        t = inn.groupby("season", observed=True).total_runs.agg(["mean", "size"]).reset_index()
        t.columns = ["season", "avg_total", "innings"]
        fig = px.bar(t, x="season", y="avg_total", text=t.avg_total.round(0).astype(int),
                     labels={"season": "Season", "avg_total": "Average total"},
                     hover_data={"innings": True})
        fig.update_traces(marker_color=viz.BLUE, textposition="outside")
        st.plotly_chart(viz.style(fig, 320, legend=False), width="stretch", config=viz.CFG)
    with right:
        st.caption("Run rate by over and innings")
        g = df.groupby(["innings", "overid"]).agg(r=("total_runs", "sum"), b=("legal", "sum")).reset_index()
        g["rr"] = g.r / g.b * 6
        g["innings"] = g.innings.map({1: "Batting first", 2: "Chasing"})
        fig = px.line(g, x="overid", y="rr", color="innings", markers=True,
                      labels={"overid": "Over", "rr": "Runs per over", "innings": ""},
                      color_discrete_sequence=[viz.BLUE, viz.PINK])
        fig.update_xaxes(dtick=1)
        st.plotly_chart(viz.style(fig, 320), width="stretch", config=viz.CFG)

    left, right = st.columns(2)
    with left:
        st.caption("Boundary share of balls faced, by phase")
        f = df[df.faced].groupby(["phase", "innings"], observed=True).agg(b=("is_bnd", "mean")).reset_index()
        f["b"] *= 100
        f["innings"] = f.innings.map({1: "Batting first", 2: "Chasing"})
        fig = px.bar(f, x="phase", y="b", color="innings", barmode="group",
                     category_orders={"phase": PHASES},
                     labels={"phase": "", "b": "Boundary %", "innings": ""},
                     color_discrete_sequence=[viz.BLUE, viz.PINK])
        st.plotly_chart(viz.style(fig, 300), width="stretch", config=viz.CFG)
    with right:
        st.caption("Wickets per innings by phase")
        w = df.groupby("phase", observed=True).agg(w=("is_wicket", "sum")).reset_index()
        n = df.mi.nunique()
        w["wpi"] = w.w / n
        fig = px.bar(w, x="phase", y="wpi", category_orders={"phase": PHASES},
                     labels={"phase": "", "wpi": "Wickets per innings"})
        fig.update_traces(marker_color=viz.SKY)
        st.plotly_chart(viz.style(fig, 300, legend=False), width="stretch", config=viz.CFG)

    ui.section("Platform status")
    live = an.available()
    ui.status_rows([
        ("Data layer", "Six CSV files loaded through one module with verified cleaning rules", "Live", "live"),
        ("Descriptive analytics", "Batting, bowling, phase and innings splits per player", "Live", "live"),
        ("Pressure index", "Advantage, win probability, leverage; out-of-fold, validated", "Live" if live else "Not generated", "live" if live else "pending"),
        ("Reliability scoring", "Empirical-Bayes pressure scores with uncertainty, per scope", "Live" if live else "Not generated", "live" if live else "pending"),
        ("Machine learning layer", "Logistic win probability and a monotone gradient-boosted par table", "Live" if live else "Not generated", "live" if live else "pending"),
    ])

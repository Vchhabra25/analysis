"""Central data layer for Analysis Intelligence.

All CSV access goes through this module. Cleaning rules follow the verified
schema notes (see views/methodology.py):

* ``runs`` is bat runs only; extras live in ``extras1_runs`` / ``extras_2_runs``.
* Legal delivery = not a wide, not a no-ball ("Overstep" / "other"), not a penalty row.
* Batter faces every delivery except wides.
* Bowler concedes bat runs + wide/no-ball extras (not byes / leg-byes).
* Bowler wickets exclude run-outs, retirements and obstructing the field.
* Batter dismissal comes from ``dismissed_player_id`` (not "Retire Out").
* Super overs (innings 3/4) are excluded from all statistics.

No pressure index is computed here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SCOPES = {
    "IPL": "IPL (primary)",
    "Emerging": "Emerging / Rising Stars",
    "T20I": "T20I (separate)",
}
SCOPE_NOTES = {
    "IPL": "Primary analysis population. Par tables and baselines are estimated here.",
    "Emerging": "Two small tournaments (30 matches). Shown as a prospect watchlist, never pooled with IPL.",
    "T20I": "International matches. Kept separate because opposition and scoring conditions differ.",
}
BOWLER_WICKETS = ["Bowled", "Caught", "LBW", "Stump", "Hit Wicket"]
PHASES = ["Powerplay", "Middle", "Death"]

_BBB_COLS = [
    "match_id", "innings", "overid", "innings_ball_num", "batting_team_id", "bowling_team_id",
    "striker_id", "striker_name", "bowler_id", "bowler_name",
    "batting_type", "bowling_type", "batting_order", "runs", "is_four", "is_six",
    "is_wicket", "dismissal_type", "dismissed_player_id", "extras_type",
    "extras1_runs", "extras_2_runs", "pace_or_spin",
]


def current_scope() -> str:
    return st.session_state.get("scope", "IPL")


@st.cache_data(show_spinner="Loading cricket data...")
def load_tables() -> dict[str, pd.DataFrame]:
    comps = pd.read_csv(DATA_DIR / "competitions.csv")
    comps["scope"] = np.where(
        comps.match_format.eq("T20I"), "T20I",
        np.where(comps.comp_name.str.startswith("IPL"), "IPL", "Emerging"),
    )
    matches = pd.read_csv(DATA_DIR / "matches.csv", parse_dates=["match_date"])
    matches = matches.merge(
        comps[["comp_id", "comp_name", "match_format", "season", "scope"]], on="comp_id", how="left"
    )
    teams = pd.read_csv(DATA_DIR / "teams.csv").drop_duplicates("team_id")[["team_id", "team_name"]]
    mapping = pd.read_csv(DATA_DIR / "competition_team_player_mapping.csv")
    mapping = mapping.merge(comps[["comp_id", "scope", "season"]], on="comp_id", how="left")
    xi = pd.read_csv(DATA_DIR / "playing_eleven.csv", usecols=["match_id", "team_id", "player_id", "is_played"])

    b = pd.read_csv(DATA_DIR / "ball_by_ball.csv", usecols=_BBB_COLS)
    b = b[b.innings <= 2].copy()  # super overs excluded
    ex = b.extras_type.fillna("")
    b["is_wide"] = ex.eq("wide")
    b["is_nb"] = ex.isin(["Overstep", "other"])
    b["is_pen"] = ex.eq("Penalty")          # 5 penalty runs: not a delivery
    b["legal"] = ~(b.is_wide | b.is_nb | b.is_pen)
    b["faced"] = ~(b.is_wide | b.is_pen)
    b["total_runs"] = b.runs + b.extras1_runs + b.extras_2_runs
    b["bowler_runs"] = b.runs + np.where(ex.isin(["wide", "Overstep", "other"]), b.extras1_runs, 0)
    b["is_bnd"] = (b.is_four | b.is_six) & (b.runs > 0)
    b["bowler_wkt"] = b.is_wicket & b.dismissal_type.isin(BOWLER_WICKETS)
    b["batter_out"] = b.is_wicket & b.dismissed_player_id.notna() & b.dismissal_type.ne("Retire Out")
    b["bat_dot"] = b.faced & b.runs.eq(0)
    b["bowl_dot"] = b.legal & b.bowler_runs.eq(0)
    b["phase"] = pd.cut(b.overid, [0, 6, 16, 20], labels=PHASES).astype(str)
    b["mi"] = b.match_id.astype("int64") * 10 + b.innings
    b = b.merge(
        matches[["match_id", "match_date", "comp_id", "comp_name", "season", "scope"]],
        on="match_id", how="left",
    )
    b = b[b.scope.notna()].copy()
    for c in ["striker_name", "bowler_name", "batting_type", "bowling_type",
              "pace_or_spin", "comp_name", "season", "scope", "phase"]:
        b[c] = b[c].astype("category")

    # player directory (names): mapping first, ball-by-ball names as fallback
    names = mapping.drop_duplicates("player_id", keep="last")[["player_id", "player_name"]]
    fb = pd.concat([
        b[["striker_id", "striker_name"]].set_axis(["player_id", "player_name"], axis=1),
        b[["bowler_id", "bowler_name"]].set_axis(["player_id", "player_name"], axis=1),
    ]).astype({"player_name": str}).drop_duplicates("player_id")
    names = pd.concat([names, fb[~fb.player_id.isin(names.player_id)]], ignore_index=True)
    return {"comps": comps, "matches": matches, "teams": teams, "mapping": mapping,
            "xi": xi, "balls": b, "players": names}


def get_balls(scope: str) -> pd.DataFrame:
    b = load_tables()["balls"]
    return b[b.scope == scope]


# ---------------------------------------------------------------- aggregates
def batting_agg(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Descriptive batting table per player (and optional extra keys)."""
    by = list(by or [])
    d = df[df.faced].rename(columns={"striker_id": "player_id"})
    g = d.groupby(["player_id"] + by, observed=True).agg(
        inns=("mi", "nunique"), balls=("faced", "size"), runs=("runs", "sum"),
        fours=("is_four", "sum"), sixes=("is_six", "sum"),
        dots=("bat_dot", "sum"), bnd=("is_bnd", "sum"),
    )
    o = df[df.batter_out].rename(columns={"dismissed_player_id": "player_id"})
    outs = o.groupby(["player_id"] + by, observed=True).size().rename("outs")
    t = g.join(outs).fillna({"outs": 0}).reset_index()
    t["player_id"] = t.player_id.astype(int)
    t["sr"] = t.runs / t.balls * 100
    t["avg"] = t.runs / t.outs.replace(0, np.nan)
    t["dot_pct"] = t.dots / t.balls * 100
    t["bnd_pct"] = t.bnd / t.balls * 100
    t["out_rate"] = t.outs / t.balls * 100
    return t


def bowling_agg(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    by = list(by or [])
    d = df.rename(columns={"bowler_id": "player_id"})
    g = d.groupby(["player_id"] + by, observed=True).agg(
        inns=("mi", "nunique"), balls=("legal", "sum"), runs=("bowler_runs", "sum"),
        wkts=("bowler_wkt", "sum"), dots=("bowl_dot", "sum"), bnd=("is_bnd", "sum"),
    ).reset_index()
    g["player_id"] = g.player_id.astype(int)
    b = g.balls.replace(0, np.nan)
    g["econ"] = g.runs / b * 6
    g["avg"] = g.runs / g.wkts.replace(0, np.nan)
    g["sr"] = b / g.wkts.replace(0, np.nan)
    g["dot_pct"] = g.dots / b * 100
    g["bnd_pct"] = g.bnd / b * 100
    g["wkt_rate"] = g.wkts / b * 100
    return g


@st.cache_data(show_spinner=False)
def player_summary(scope: str) -> pd.DataFrame:
    """One row per player in scope: names, usage profile, batting + bowling totals."""
    t = load_tables()
    df = get_balls(scope)
    bat = batting_agg(df).add_prefix("bat_").rename(columns={"bat_player_id": "player_id"})
    bowl = bowling_agg(df).add_prefix("bowl_").rename(columns={"bowl_player_id": "player_id"})
    s = bat.merge(bowl, on="player_id", how="outer").merge(t["players"], on="player_id", how="left")
    s["player_name"] = s.player_name.fillna("Unknown (id " + s.player_id.astype(str) + ")")
    for c in ["bat_balls", "bowl_balls", "bat_runs", "bowl_wkts", "bat_inns", "bowl_inns"]:
        s[c] = s[c].fillna(0)
    bats, bowls = s.bat_balls >= 60, s.bowl_balls >= 60
    s["usage"] = np.select(
        [bats & bowls, bats, bowls], ["All-rounder", "Batter", "Bowler"], default="Limited data"
    )
    hand = (df.dropna(subset=["batting_type"]).groupby("striker_id", observed=True)
            .batting_type.agg(lambda x: x.value_counts().idxmax()))
    btype = (df.dropna(subset=["bowling_type"]).groupby("bowler_id", observed=True)
             .bowling_type.agg(lambda x: x.value_counts().idxmax()))
    order = df[df.faced].groupby("striker_id").batting_order.agg(lambda x: int(x.median()))
    s["batting_hand"] = s.player_id.map(hand).astype("object")
    s["bowling_type"] = s.player_id.map(btype).astype("object")
    s["typical_order"] = s.player_id.map(order)
    s["label"] = s.player_name
    dup = s.label.duplicated(keep=False)
    s.loc[dup, "label"] = s.label + " (#" + s.player_id.astype(str) + ")"
    s["activity"] = s.bat_balls + s.bowl_balls
    return s.sort_values("activity", ascending=False).reset_index(drop=True)


def player_teams(player_id: int, scope: str) -> list[str]:
    m = load_tables()["mapping"]
    return sorted(m[(m.player_id == player_id) & (m.scope == scope)].team_name.unique())


def percentile_in_pool(series: pd.Series, value: float, higher_is_better: bool = True) -> float:
    s = series.dropna()
    if s.empty or pd.isna(value):
        return np.nan
    p = (s < value).mean() * 100 + (s == value).mean() * 50
    return p if higher_is_better else 100 - p


def dataset_counts() -> dict[str, int]:
    t = load_tables()
    b, m, c = t["balls"], t["matches"], t["comps"]
    return {
        "matches_listed": int(len(m)),
        "matches_analysed": int(b.match_id.nunique()),
        "deliveries": int(len(b)),
        "players_mapped": int(t["mapping"].player_id.nunique()),
        "players_profiled": int(pd.concat([b.striker_id, b.bowler_id]).nunique()),
        "competitions_listed": int(len(c)),
        "competitions_with_data": int(b.comp_id.nunique()),
    }

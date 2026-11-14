"""Step 1: rebuild the pre-delivery match state and decide which matches are usable.

Every state variable is computed from deliveries *before* the current one, so
nothing the ball itself produces leaks into its own pressure score.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from utils.data_loader import get_balls, load_tables

BALLS = 120
DLS_RE = re.compile(r"D/?L|DLS|Duckworth", re.I)


def build_states(scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (deliveries with state, per-match eligibility table) for a scope."""
    t = load_tables()
    m = t["matches"]
    df = get_balls(scope).sort_values(["match_id", "innings", "innings_ball_num"]).copy()
    df["category"] = df.scope.astype(str)

    g = df.groupby("mi", sort=False)
    df["S"] = g.total_runs.cumsum() - df.total_runs                    # score before the ball
    df["W"] = g.is_wicket.cumsum() - df.is_wicket.astype(int)          # wickets before the ball
    df["L"] = g.legal.cumsum() - df.legal.astype(int)                  # legal balls before
    df["B"] = (BALLS - df.L).clip(lower=1)                             # legal balls left
    df["W"] = df.W.clip(upper=9)

    inn = df.groupby(["match_id", "innings"]).agg(
        runs=("total_runs", "sum"), legal=("legal", "sum"), wkts=("is_wicket", "sum")).unstack("innings")
    inn.columns = [f"{a}{b}" for a, b in inn.columns]
    el = inn.reset_index().merge(
        m[["match_id", "match_date", "season", "comp_name", "result", "win_team_id", "revised_target",
           "first_inn_revised_overs", "second_inn_revised_overs"]], on="match_id", how="left")
    el["target"] = el.runs1 + 1
    res = el.result.fillna("")
    el["no_result"] = res.str.contains("No Result", case=False)
    el["dls"] = (res.map(lambda x: bool(DLS_RE.search(x))) | el.revised_target.notna()
                 | el.first_inn_revised_overs.notna() | el.second_inn_revised_overs.notna())
    el["missing_inn2"] = el.runs2.isna()
    el["inn1_incomplete"] = (el.legal1 < BALLS) & (el.wkts1 < 9)
    el["inn2_incomplete"] = (el.legal2 < BALLS) & (el.wkts2 < 9) & (el.runs2 < el.target)
    el["tied"] = res.str.contains("Tied", case=False)
    reasons = ["no_result", "dls", "missing_inn2", "inn1_incomplete", "inn2_incomplete"]
    el["excluded"] = el[reasons].any(axis=1)
    first = el[reasons].idxmax(axis=1)
    el["exclusion_reason"] = np.where(el.excluded, first, "")
    el["fit_label_ok"] = ~el.tied & el.win_team_id.notna()

    # innings-2 batting team and chase outcome (used only to calibrate win probability)
    bt = df[df.innings == 2].groupby("match_id").batting_team_id.first().rename("chase_team")
    bf = df[df.innings == 1].groupby("match_id").batting_team_id.first().rename("first_team")
    el = el.merge(bt, on="match_id", how="left").merge(bf, on="match_id", how="left")
    el["chase_won"] = (el.win_team_id == el.chase_team).astype(float)
    el["bat_first_won"] = (el.win_team_id == el.first_team).astype(float)

    keep = el[~el.excluded].match_id
    df = df[df.match_id.isin(keep)].merge(el[["match_id", "target"]], on="match_id", how="left")
    df.loc[df.innings == 1, "target"] = np.nan
    return df.reset_index(drop=True), el


def exclusion_summary(el: pd.DataFrame) -> dict:
    out = {"matches_with_ball_data": int(len(el)), "usable": int((~el.excluded).sum())}
    out.update({k: int(v) for k, v in el[el.excluded].exclusion_reason.value_counts().items()})
    out["tied_kept"] = int((el.tied & ~el.excluded).sum())
    return out


def target_consistency(df: pd.DataFrame, el: pd.DataFrame) -> dict:
    """Sanity check: does innings-2 total vs target agree with the recorded winner?"""
    k = el[~el.excluded & el.fit_label_ok]
    agree = ((k.runs2 >= k.target) == (k.chase_won == 1)).mean()
    return {"matches_checked": int(len(k)), "target_vs_winner_agreement": float(agree)}

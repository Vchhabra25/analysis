"""Orchestration helpers shared by run.py and validate.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.scoring import (cluster_table, risk_flags, role_frame, score_players, strengths_weaknesses)

CFG = {
    "IPL": dict(min_hp=40, min_inn=8, min_base=100),
    "T20I": dict(min_hp=40, min_inn=8, min_base=100),
    "Emerging": dict(min_hp=10, min_inn=2, min_base=30),
}


def prepare_role(o: pd.DataFrame, role: str, thr: dict, means: dict | None = None):
    r, means = role_frame(o, role, thr, means)
    c = cluster_table(r)
    floor = float(r[r.hp].val.var())
    return r, c, means, floor


def rank_role(c: pd.DataFrame, floor: float, cfg: dict, prior: dict | None = None):
    return score_players(c, floor, cfg["min_hp"], cfg["min_inn"], cfg["min_base"], prior)


def decorate(t: pd.DataFrame, r: pd.DataFrame, role: str, names: pd.Series) -> pd.DataFrame:
    sw = strengths_weaknesses(r, role, pd.Index(t.pid))
    out = pd.DataFrame({
        "rank": t["rank"], "player_id": t.pid.astype(int), "player_name": t.pid.map(names).values,
        "pressure_score": t.post * 100, "score_lo80": t.lo80 * 100, "score_hi80": t.hi80 * 100,
        "score_sd": t.post_sd * 100,
        "baseline": t.x_nh * 100, "baseline_se": t.se_nh * 100, "pressure_raw": t.x_hp * 100,
        "pressure_raw_se": t.se_hp * 100, "gap": t.gap * 100, "gap_z": t.z_gap,
        "hp_balls": t.N_hp.astype(int), "hp_innings": t.G_hp.astype(int), "baseline_balls": t.N_nh.astype(int),
        "data_weight": t.B, "p_top10": t.p_top10, "confidence": t.confidence,
        "rank_pressure_only": t.rank_pressure_only,
    })
    out["strengths"] = out.player_id.map(sw.strengths)
    out["weaknesses"] = out.player_id.map(sw.weaknesses)
    out["risks"] = risk_flags(t.assign(G_hp=t.G_hp)).values
    return out.round({"pressure_score": 2, "score_lo80": 2, "score_hi80": 2, "score_sd": 2, "baseline": 2,
                      "baseline_se": 2, "pressure_raw": 2, "pressure_raw_se": 2, "gap": 2, "gap_z": 2,
                      "data_weight": 3, "p_top10": 3})

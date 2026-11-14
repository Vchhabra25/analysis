"""Run the full analytical engine and write results to analytics/.

    python -m engine.run
"""
from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from engine import validate as V                                   # noqa: E402
from engine.core import FoldModels, finalize_pressure, run_oof, thresholds   # noqa: E402
from engine.pipeline import CFG, decorate, prepare_role, rank_role            # noqa: E402
from engine.prep import build_states, exclusion_summary, target_consistency   # noqa: E402
from utils.data_loader import load_tables                                      # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "analytics"
ROLES = {"bat": "batting", "bowl": "bowling"}
BALL_COLS = ["match_id", "innings", "innings_ball_num", "overid", "striker_id", "bowler_id", "S", "W", "B", "target",
             "A", "WP", "lam", "PI_bat", "PI_bowl", "v_bat", "v_bowl", "runs", "bowler_runs", "is_wicket"]


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    raise TypeError(type(o))


def write_scope(scope, o, el, thr, means, prior_floor, names, prior=None, key=None):
    key = key or scope
    cfg = CFG[scope]
    ranked, metas, traces = {}, {}, []
    for role in ROLES:
        r, c, _, floor = prepare_role(o, role, thr, means[role] if means else None)
        floor = prior_floor[role] if prior_floor else floor
        t, meta = rank_role(c, floor, cfg, None if prior is None else prior[role])
        dec = decorate(t, r, role, names)
        k = c[(c.n_hp >= 3) & c.pid.isin(t.pid)].assign(pos=lambda x: (x.s_hp / x.n_hp) >= 0)
        cons = k.groupby("pid").pos.agg(["mean", "size"])
        dec["consistency"] = (dec.player_id.map(cons["mean"]) * 100).round(1)
        dec["consistency_n"] = dec.player_id.map(cons["size"]).fillna(0).astype(int)
        dec.to_csv(OUT / f"ranking_{key}_{ROLES[role]}.csv", index=False)
        cl = c[c.pid.isin(t.pid)].merge(el[["match_id", "match_date", "comp_name"]], on="match_id", how="left")
        cl.to_csv(OUT / f"clusters_{key}_{ROLES[role]}.csv", index=False)
        ranked[role], metas[role] = t, dict(meta, floor=floor)
        print(f"  {key}/{ROLES[role]}: ranked {len(t)}; top5 =", ", ".join(dec.player_name.head(5)))
    return ranked, metas


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    names = load_tables()["players"].set_index("player_id").player_name
    meta = {"generated": datetime.now().isoformat(timespec="seconds"), "config": CFG, "scopes": {}}
    val = {}

    # ----------------------------------------------------------- IPL (primary, out-of-fold)
    print("IPL: building states and out-of-fold models")
    df, el = build_states("IPL")
    o = run_oof(df, el)
    o, lam_mean = finalize_pressure(o)
    thr = thresholds(o)
    means, floors = {}, {}
    for role in ROLES:
        _, _, m, fl = prepare_role(o, role, thr)
        means[role], floors[role] = m, fl
    ranked, metas = write_scope("IPL", o, el, thr, means, None, names)
    o[BALL_COLS].round(3).to_csv(OUT / "balls_IPL.csv.gz", index=False)
    meta["scopes"]["IPL"] = {"exclusions": exclusion_summary(el), "target_check": target_consistency(df, el),
                             "lam_mean": lam_mean, "thresholds": thr, "model": metas,
                             "deliveries": int(len(o)), "matches": int(o.match_id.nunique())}
    print("IPL: validation")
    val["IPL"] = {
        "wp": V.wp_validation(df, o, el), "difficulty": V.difficulty_checks(o, thr),
        "hp_vs_ordinary": V.hp_vs_ordinary(o, thr),
        "gap_heterogeneity": {ROLES[r]: V.gap_heterogeneity(ranked[r]) for r in ROLES},
        "sensitivity": V.sensitivity(o, ranked, CFG["IPL"]),
        "stability": V.stability(o, thr, CFG["IPL"]),
    }

    # ----------------------------------------------------------- Emerging (IPL-scale models)
    print("Emerging: IPL-fitted models applied to prospects")
    de, ee = build_states("Emerging")
    fm = FoldModels().fit(df, el)
    oe, _ = finalize_pressure(fm.annotate(de), lam_mean)
    prior = {r: {k: v for k, v in metas[r].items() if k not in ("k", "floor")} for r in ROLES}
    rk_e, mt_e = write_scope("Emerging", oe, ee, thr, means, floors, names, prior)
    oe[BALL_COLS].round(3).to_csv(OUT / "balls_Emerging.csv.gz", index=False)
    meta["scopes"]["Emerging"] = {"exclusions": exclusion_summary(ee), "note": "IPL models, thresholds and priors; separate prospect ranking",
                                  "deliveries": int(len(oe)), "matches": int(oe.match_id.nunique()),
                                  "players_ranked": {ROLES[r]: int(len(rk_e[r])) for r in ROLES}}

    # ----------------------------------------------------------- T20I (separate analysis)
    print("T20I: separate pipeline")
    dt, et = build_states("T20I")
    ot = run_oof(dt, et)
    ot, lam_t = finalize_pressure(ot)
    thr_t = thresholds(ot)
    rk_t, mt_t = write_scope("T20I", ot, et, thr_t, None, None, names)
    ot[BALL_COLS].round(3).to_csv(OUT / "balls_T20I.csv.gz", index=False)
    meta["scopes"]["T20I"] = {"exclusions": exclusion_summary(et), "target_check": target_consistency(dt, et),
                              "lam_mean": lam_t, "thresholds": thr_t, "model": mt_t,
                              "deliveries": int(len(ot)), "matches": int(ot.match_id.nunique())}
    val["T20I"] = {"wp": V.wp_validation(dt, ot, et), "difficulty": V.difficulty_checks(ot, thr_t),
                   "hp_vs_ordinary": V.hp_vs_ordinary(ot, thr_t),
                   "gap_heterogeneity": {ROLES[r]: V.gap_heterogeneity(rk_t[r]) for r in ROLES}}

    (OUT / "meta.json").write_text(json.dumps(meta, indent=1, default=_js))
    (OUT / "validation.json").write_text(json.dumps(val, indent=1, default=_js))
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()

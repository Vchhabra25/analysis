"""Step 4: validation of the pressure index and the rankings."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from engine.core import FoldModels, finalize_pressure, thresholds
from engine.pipeline import CFG, prepare_role, rank_role
from engine.scoring import cluster_table, role_frame


def _df(x: pd.DataFrame) -> list[dict]:
    return x.reset_index().round(4).astype(object).where(x.reset_index().notna(), None).to_dict("records")


# ---------------------------------------------------------------- V5 leakage / WP quality
def _wp_metrics(d: pd.DataFrame, el: pd.DataFrame, col: str) -> dict:
    lab = el.set_index("match_id")
    y = lab.loc[d.match_id, col].to_numpy()
    p = d.WP.clip(1e-4, 1 - 1e-4).to_numpy()
    return {"balls": int(len(d)), "auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, p)),
            "baseline_logloss": float(log_loss(y, np.full(len(y), y.mean()))), "brier": float(brier_score_loss(y, p))}


def wp_validation(df: pd.DataFrame, o: pd.DataFrame, el: pd.DataFrame) -> dict:
    ok = set(el[el.fit_label_ok].match_id)
    cols = {1: "bat_first_won", 2: "chase_won"}
    out = {"oof": {}, "in_sample": {}, "temporal": {}, "calibration": []}
    full = FoldModels().fit(df, el).annotate(df)
    cut = pd.Timestamp("2025-01-01")
    tr, te = df[df.match_date < cut], df[df.match_date >= cut]
    fm = FoldModels().fit(tr, el[el.match_id.isin(tr.match_id)])
    ann = fm.annotate(te)
    for inn, col in cols.items():
        out["oof"][inn] = _wp_metrics(o[(o.innings == inn) & o.match_id.isin(ok)], el, col)
        out["in_sample"][inn] = _wp_metrics(full[(full.innings == inn) & full.match_id.isin(ok)], el, col)
        d = ann[(ann.innings == inn) & ann.match_id.isin(ok)]
        out["temporal"][inn] = _wp_metrics(d, el, col)
        out["temporal"][inn]["train_matches"] = int(tr.match_id.nunique())
        out["temporal"][inn]["test_matches"] = int(te.match_id.nunique())
        d = o[(o.innings == inn) & o.match_id.isin(ok)]
        y = el.set_index("match_id").loc[d.match_id, col].to_numpy()
        q = pd.qcut(d.WP, 10, duplicates="drop")
        cal = pd.DataFrame({"q": q.to_numpy(), "p": d.WP.to_numpy(), "y": y}).groupby("q", observed=True).agg(
            predicted=("p", "mean"), observed=("y", "mean"), balls=("p", "size"))
        for r in cal.itertuples():
            out["calibration"].append({"innings": inn, "predicted": round(r.predicted, 3),
                                       "observed": round(r.observed, 3), "balls": int(r.balls)})
    cent = o.groupby(["innings", "phase"], observed=True).agg(bat=("v_bat", "mean"), bowl=("v_bowl", "mean"))
    out["value_means"] = _df(cent)
    return out


# ---------------------------------------------------------------- V1 pressure rises when it should
def difficulty_checks(o: pd.DataFrame, thr: dict) -> dict:
    o = o.copy()
    o["hp_bat"] = o.PI_bat >= thr["bat"][0]
    c2 = o[o.innings == 2].copy()
    c2["rrr"] = (c2.target - c2.S) / (c2.B / 6)
    bands = pd.cut(c2.rrr, [-1, 6, 8, 10, 12, 15, 100], labels=["<6", "6-8", "8-10", "10-12", "12-15", "15+"])
    by_rrr = c2.groupby(bands, observed=True).agg(balls=("WP", "size"), win_prob=("WP", "mean"),
                                                 leverage=("lam", "mean"), pi_bat=("PI_bat", "mean"),
                                                 pi_bowl=("PI_bowl", "mean"))
    ob = pd.cut(o.overid, [0, 6, 12, 16, 18, 20], labels=["1-6", "7-12", "13-16", "17-18", "19-20"])
    by_over = o.groupby([o.innings.map({1: "Batting first", 2: "Chasing"}), ob], observed=True).agg(
        balls=("PI_bat", "size"), leverage=("lam", "mean"), pi_bat=("PI_bat", "mean"), pi_bowl=("PI_bowl", "mean"))
    close = (o.innings == 2) & (o.overid >= 16) & (o.A.abs() < 1)
    tags = {"Death overs (17-20)": o.overid >= 17, "Chasing": o.innings == 2,
            "Chase needing 12+ per over": (o.innings == 2) & ((o.target - o.S) / (o.B / 6) >= 12),
            "5+ wickets down": o.W >= 5, "Close finish (over 16+, |A|<1)": close}
    base = float(o.hp_bat.mean())
    lift = [{"situation": k, "balls": int(m.sum()), "high_pressure_share": float(o.hp_bat[m].mean()),
             "share_elsewhere": float(o.hp_bat[~m].mean())} for k, m in tags.items()]
    return {"by_rrr": _df(by_rrr), "by_over": _df(by_over), "tag_lift": lift, "overall_hp_share": base}


# ---------------------------------------------------------------- V2 high vs ordinary
def hp_vs_ordinary(o: pd.DataFrame, thr: dict) -> dict:
    rows = []
    for role in ("bat", "bowl"):
        r, _ = role_frame(o, role, thr)
        if role == "bat":
            g = lambda m: dict(runs_per_ball=m.runs.mean(), dot_pct=m.bat_dot.mean() * 100, boundary_pct=m.is_bnd.mean() * 100,
                               wicket_pct=m.batter_out.mean() * 100, value_per_100=m.raw.mean() * 100)
            sets = {"High pressure": r[r.hp], "Quietest half": r[r.ordy], "All below threshold": r[~r.hp]}
        else:
            g = lambda m: dict(runs_per_ball=m.bowler_runs.mean(), dot_pct=m.bowl_dot.mean() * 100, boundary_pct=m.is_bnd.mean() * 100,
                               wicket_pct=m.bowler_wkt.mean() * 100, value_per_100=m.raw.mean() * 100)
            sets = {"High pressure": r[r.hp], "Quietest half": r[r.ordy], "All below threshold": r[~r.hp]}
        for name, m in sets.items():
            rows.append({"role": "Batting" if role == "bat" else "Bowling", "group": name, "balls": int(len(m)),
                         **{k: round(float(v), 3) for k, v in g(m).items()}})
    return {"table": rows}


def gap_heterogeneity(t: pd.DataFrame) -> dict:
    g, s2 = (t.x_hp - t.x_nh).to_numpy(), (t.se2_hp + t.se2_nh).to_numpy()
    w = 1 / s2
    mu = (w * g).sum() / w.sum()
    q = float((w * (g - mu) ** 2).sum())
    k = len(g)
    z = g / np.sqrt(s2)
    return {"players": int(k), "Q": q, "df": int(k - 1), "p_value": float(1 - stats.chi2.cdf(q, k - 1)),
            "share_abs_z_gt_1.96": float((np.abs(z) > 1.96).mean()), "expected_under_null": 0.05,
            "mean_gap_per_100": float(g.mean() * 100)}


# ---------------------------------------------------------------- V3 sensitivity
def sensitivity(o: pd.DataFrame, base_rank: dict, cfg: dict, scope_means=None) -> list[dict]:
    rows = []
    for role in ("bat", "bowl"):
        main = base_rank[role].set_index("pid")
        main10 = set(main.index[:10])
        for label, hi, mh in [("Top 30% threshold", 0.70, cfg["min_hp"]), ("Main: top 20%", 0.80, cfg["min_hp"]),
                              ("Top 10% threshold", 0.90, 25), ("Min 30 pressure balls", 0.80, 30),
                              ("Min 60 pressure balls", 0.80, 60)]:
            thr = thresholds(o, hi, 0.5)
            _, c, _, fl = prepare_role(o, role, thr)
            t, _ = rank_role(c, fl, dict(cfg, min_hp=mh))
            t = t.set_index("pid")
            common = main.index.intersection(t.index)
            rho = stats.spearmanr(main.loc[common, "post"], t.loc[common, "post"]).statistic
            rows.append({"role": "Batting" if role == "bat" else "Bowling", "variant": label,
                         "players": int(len(t)), "spearman_vs_main": float(rho),
                         "top10_overlap": int(len(main10 & set(t.index[:10])))})
    return rows


# ---------------------------------------------------------------- V4 stability
def stability(o: pd.DataFrame, thr: dict, cfg: dict, reps: int = 40, boots: int = 150, seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    mids = o.match_id.unique()
    half_cfg = dict(min_hp=max(cfg["min_hp"] // 2, 15), min_inn=max(cfg["min_inn"] // 2, 3), min_base=cfg["min_base"] // 2)
    seasons = o.drop_duplicates("match_id").set_index("match_id").season.astype(str)
    early = set(seasons[seasons.isin(["2022", "2023", "2024"])].index)
    for role in ("bat", "bowl"):
        r, c, _, fl = prepare_role(o, role, thr)
        full, _ = rank_role(c, fl, cfg)
        full_top = list(full.pid[:10])
        res = {"post_rho": [], "raw_rho": [], "top10_overlap": []}

        def compare(a_ids, b_ids):
            ta, _ = rank_role(c[c.match_id.isin(a_ids)], fl, half_cfg)
            tb, _ = rank_role(c[c.match_id.isin(b_ids)], fl, half_cfg)
            ta, tb = ta.set_index("pid"), tb.set_index("pid")
            cm = ta.index.intersection(tb.index)
            if len(cm) < 15:
                return None
            return (stats.spearmanr(ta.loc[cm, "post"], tb.loc[cm, "post"]).statistic,
                    stats.spearmanr(ta.loc[cm, "x_hp"], tb.loc[cm, "x_hp"]).statistic,
                    len(set(ta.index[:10]) & set(tb.index[:10])), len(cm))
        for _ in range(reps):
            a = set(rng.choice(mids, len(mids) // 2, replace=False))
            r_ = compare(a, set(mids) - a)
            if r_:
                res["post_rho"].append(r_[0]); res["raw_rho"].append(r_[1]); res["top10_overlap"].append(r_[2])
        seas = compare(early, set(mids) - early)
        # bootstrap over matches
        cnt = {pid: 0 for pid in full_top}
        ranks = {}
        for _ in range(boots):
            pick = pd.Series(rng.choice(mids, len(mids), replace=True)).value_counts()
            cc = c.merge(pick.rename("w"), left_on="match_id", right_index=True)
            cc = cc.loc[cc.index.repeat(cc.w)]
            tb, _ = rank_role(cc, fl, cfg)
            for pid in tb.pid[:10]:
                ranks[pid] = ranks.get(pid, 0) + 1
        out[role] = {
            "random_split_post_spearman": float(np.mean(res["post_rho"])),
            "random_split_raw_spearman": float(np.mean(res["raw_rho"])),
            "random_split_top10_overlap": float(np.mean(res["top10_overlap"])),
            "splits_used": len(res["post_rho"]),
            "season_split_post_spearman": None if seas is None else float(seas[0]),
            "season_split_raw_spearman": None if seas is None else float(seas[1]),
            "season_split_top10_overlap": None if seas is None else int(seas[2]),
            "season_split_players": None if seas is None else int(seas[3]),
            "bootstrap_top10_freq": {int(k): v / boots for k, v in ranks.items()},
            "bootstrap_reps": boots,
        }
    return out

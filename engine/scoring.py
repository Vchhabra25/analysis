"""Step 3: player pressure scores with sample-size adjustment.

Score units: runs per 100 balls versus an average player in the same innings and phase.
Higher is better for BOTH batters and bowlers (a bowler's value is runs saved).

Shrinkage: normal-normal empirical Bayes. Standard errors are cluster-robust (balls from
the same innings are correlated), so a player's pressure sample counts as *innings*, not
just balls. Players with thin samples are pulled toward the population mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Z80 = 1.2816


# ---------------------------------------------------------------- preparation
def role_frame(d: pd.DataFrame, role: str, thr: dict, means: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Rows relevant to a role, tagged high-pressure (hp) or baseline (nh), with class-centred values.

    ``val`` = ball value minus the league mean of balls in the same (class, innings, phase),
    so 0 means "an average player in the same kind of situation". ``means`` lets another
    population (Emerging) be scored against IPL reference values.
    ``ordy`` flags the quietest half of balls (bottom 50% of PI) for contrast checks.
    """
    if role == "bat":
        r = d[d.faced].copy()
        r["pid"], r["raw"], r["pi"] = r.striker_id, r.v_bat, r.PI_bat
    else:
        r = d[~d.is_pen].copy()
        r["pid"], r["raw"], r["pi"] = r.bowler_id, r.v_bowl, r.PI_bowl
    hi, lo = thr[role]
    r["hp"] = r.pi >= hi
    r["ordy"] = r.pi <= lo
    r["cls"] = np.where(r.hp, "hp", "nh")
    r["ph"] = r.phase.astype(str)
    if means is None:
        means = r.groupby(["cls", "innings", "ph"]).raw.mean().to_dict()
    key = list(zip(r.cls, r.innings, r.ph))
    r["val"] = r.raw - np.array([means.get(k, 0.0) for k in key])
    return r, means


def cluster_table(r: pd.DataFrame) -> pd.DataFrame:
    """One row per player x innings with high-pressure and baseline sums."""
    r = r.assign(hv=np.where(r.hp, r.val, 0.0), bv=np.where(r.hp, 0.0, r.val),
                 hn=r.hp.astype(int), bn=(~r.hp).astype(int))
    return r.groupby(["pid", "mi"], observed=True).agg(
        match_id=("match_id", "first"), n_hp=("hn", "sum"), s_hp=("hv", "sum"),
        n_nh=("bn", "sum"), s_nh=("bv", "sum")).reset_index()


# ------------------------------------------------------------------ statistics
def _mean_se(c: pd.DataFrame, n: str, s: str, floor_var: float) -> pd.DataFrame:
    g = c[c[n] > 0].copy()
    tot = g.groupby("pid").agg(N=(n, "sum"), S=(s, "sum"), G=(n, "size"))
    g = g.join(tot, on="pid")
    g["res2"] = (g[s] - g.S / g.N * g[n]) ** 2
    rob = g.groupby("pid").res2.sum() / tot.N ** 2 * (tot.G / (tot.G - 1).clip(lower=1))
    out = tot.assign(x=tot.S / tot.N)
    out["se2"] = np.maximum(rob, floor_var / out.N)          # never trust a tiny cluster-robust SE
    out["se"] = np.sqrt(out.se2)
    return out[["N", "G", "x", "se2", "se"]]


def dl_tau2(x: np.ndarray, se2: np.ndarray) -> tuple[float, float]:
    """DerSimonian-Laird between-player variance and the precision-weighted mean."""
    w = 1 / se2
    mu = (w * x).sum() / w.sum()
    q = (w * (x - mu) ** 2).sum()
    tau2 = max((q - (len(x) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()), 1e-8)
    w2 = 1 / (se2 + tau2)
    return float(tau2), float((w2 * x).sum() / w2.sum())


def estimate_prior(t: pd.DataFrame) -> dict:
    """Population model: pressure performance p = skill a + pressure-specific shift d.

    a ~ N(mu_a, tau_a^2) is estimated from baseline balls (the large sample);
    d ~ N(d_bar, tau_d^2) from the pressure-minus-baseline gap. Both by DerSimonian-Laird.
    """
    tau_a, mu_a = dl_tau2(t.x_nh.to_numpy(), t.se2_nh.to_numpy())
    g, sg2 = (t.x_hp - t.x_nh).to_numpy(), (t.se2_hp + t.se2_nh).to_numpy()
    tau_d, d_bar = dl_tau2(g, sg2)
    cov = np.array([[tau_a + tau_d, tau_a], [tau_a, tau_a]])
    return {"mu": [mu_a + d_bar, mu_a], "cov": cov.tolist(), "tau2_skill": tau_a, "tau2_pressure": tau_d,
            "mu_skill": mu_a, "pressure_shift": d_bar,
            "reliability_baseline": float(tau_a / (tau_a + t.se2_nh.mean())),
            "reliability_pressure_shift": float(tau_d / (tau_d + sg2.mean()))}


def posterior(t: pd.DataFrame, prior: dict) -> pd.DataFrame:
    """Bivariate normal-normal EB: baseline evidence informs the pressure estimate."""
    mu, S = np.array(prior["mu"]), np.array(prior["cov"])
    pm, pv = [], []
    for x1, x2, s1, s2 in zip(t.x_hp, t.x_nh, t.se2_hp, t.se2_nh):
        K = S @ np.linalg.inv(S + np.diag([s1, s2]))
        pm.append(mu + K @ (np.array([x1, x2]) - mu))
        pv.append(S - K @ S)
    pm = np.array(pm)
    t = t.copy()
    t["post"], t["post_base"] = pm[:, 0], pm[:, 1]
    t["post_sd"] = np.sqrt([v[0, 0] for v in pv])
    t["B"] = 1 - t.post_sd ** 2 / S[0, 0]               # share of prior variance resolved by the data
    t["lo80"], t["hi80"] = t.post - Z80 * t.post_sd, t.post + Z80 * t.post_sd
    rng = np.random.default_rng(11)
    draws = rng.standard_normal((4000, len(t))) * t.post_sd.to_numpy() + t.post.to_numpy()
    rank = (-draws).argsort(axis=1).argsort(axis=1) + 1
    t["p_top10"] = (rank <= 10).mean(axis=0)
    return t


def score_players(c: pd.DataFrame, floor_var: float, min_hp: int, min_inn: int, min_base: int,
                  prior: dict | None = None) -> tuple[pd.DataFrame, dict]:
    hp = _mean_se(c, "n_hp", "s_hp", floor_var).add_suffix("_hp")
    nh = _mean_se(c, "n_nh", "s_nh", floor_var).add_suffix("_nh")
    t = hp.join(nh, how="inner")
    t = t[(t.N_hp >= min_hp) & (t.G_hp >= min_inn) & (t.N_nh >= min_base)].copy()
    prior = estimate_prior(t) if prior is None else prior
    t = posterior(t, prior)
    t["gap"] = t.x_hp - t.x_nh
    t["se_gap"] = np.sqrt(t.se2_hp + t.se2_nh)
    t["z_gap"] = t.gap / t.se_gap
    # pressure-only alternative: shrink the raw pressure mean toward the population, ignoring baseline
    tau_h, mu_h = dl_tau2(t.x_hp.to_numpy(), t.se2_hp.to_numpy())
    t["post_hp_only"] = mu_h + tau_h / (tau_h + t.se2_hp) * (t.x_hp - mu_h)
    t["rank_pressure_only"] = t.post_hp_only.rank(ascending=False, method="first").astype(int)
    # confidence = how much of the population spread in true pressure performance the data resolve
    t["confidence"] = np.select([t.B >= 0.5, t.B >= 0.3], ["High", "Medium"], default="Low")
    t = t.sort_values("post", ascending=False)
    t["rank"] = np.arange(1, len(t) + 1)
    meta = dict(prior)
    meta["k"] = int(len(t))
    return t.reset_index(), meta


# ---------------------------------------------------- strengths and weaknesses
def _components(r: pd.DataFrame, role: str) -> pd.DataFrame:
    h = r[r.hp]
    if role == "bat":
        g = h.groupby("pid").agg(n=("hp", "size"), runs=("runs", "sum"), bnd=("is_bnd", "mean"),
                                 dts=("bat_dot", "mean"), ots=("batter_out", "mean"))
        comp = pd.DataFrame({
            "Strike rate": g.runs / g.n * 100, "Boundary %": g.bnd * 100,
            "Dot-ball %": g.dts * 100, "Dismissals per 100 balls": g.ots * 100})
        better = {"Strike rate": 1, "Boundary %": 1, "Dot-ball %": -1, "Dismissals per 100 balls": -1}
        subsets = {"in chases": h.innings == 2, "in the death overs": h.phase.astype(str) == "Death",
                   "with 3+ wickets down": h.W >= 3}
    else:
        h = h[h.legal]
        g = h.groupby("pid").agg(n=("hp", "size"), runs=("bowler_runs", "sum"), bnd=("is_bnd", "mean"),
                                 dts=("bowl_dot", "mean"), wk=("bowler_wkt", "mean"))
        comp = pd.DataFrame({
            "Economy": g.runs / g.n * 6, "Dot-ball %": g.dts * 100, "Boundary % conceded": g.bnd * 100,
            "Wickets per 100 balls": g.wk * 100})
        better = {"Economy": -1, "Dot-ball %": 1, "Boundary % conceded": -1, "Wickets per 100 balls": 1}
        subsets = {"when defending": h.innings == 2, "in the death overs": h.phase.astype(str) == "Death",
                   "in the powerplay": h.phase.astype(str) == "Powerplay"}
    sub = {}
    for name, mask in subsets.items():
        s = h[mask].groupby("pid").agg(n=("val", "size"), m=("val", "mean"))
        sub[name] = (s.m * 100).where(s.n >= 20)
        better[name] = 1
    comp = comp.join(pd.DataFrame(sub))
    ctx = {"n": g.n}
    league = {k: (h.val.mean() * 100 if k in subsets else None) for k in better}
    return comp, better, subsets, league


def _ord(x: float) -> str:
    n = int(round(x))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def strengths_weaknesses(r: pd.DataFrame, role: str, ranked_ids: pd.Index) -> pd.DataFrame:
    comp, better, subsets, _ = _components(r, role)
    peers = comp.loc[comp.index.intersection(ranked_ids)]
    pooled = {}
    h = r[r.hp]
    if role == "bat":
        hb = h
        pooled = {"Strike rate": hb.runs.mean() * 100, "Boundary %": hb.is_bnd.mean() * 100,
                  "Dot-ball %": hb.bat_dot.mean() * 100, "Dismissals per 100 balls": hb.batter_out.mean() * 100}
    else:
        hb = h[h.legal]
        pooled = {"Economy": hb.bowler_runs.mean() * 6, "Dot-ball %": hb.bowl_dot.mean() * 100,
                  "Boundary % conceded": hb.is_bnd.mean() * 100, "Wickets per 100 balls": hb.bowler_wkt.mean() * 100}
    for name in subsets:
        pooled[name] = 0.0
    rows = []
    for pid, row in peers.iterrows():
        scored = []
        for k, sign in better.items():
            v = row.get(k)
            if pd.isna(v):
                continue
            col = peers[k].dropna()
            pct = (col < v).mean() * 100 if sign == 1 else (col > v).mean() * 100
            scored.append((k, v, pct, sign))
        strong = sorted([x for x in scored if x[2] >= 75], key=lambda x: -x[2])[:3]
        weak = sorted([x for x in scored if x[2] <= 25], key=lambda x: x[2])[:2]

        def fmt(x):
            k, v, pct, _ = x
            if k in subsets:
                return f"{k.capitalize()}: {v:+.1f} runs/100 balls vs average ({_ord(pct)} percentile of ranked peers)"
            return f"{k}: {v:.1f} vs {pooled[k]:.1f} pressure-ball average ({_ord(pct)} percentile of ranked peers)"
        rows.append({"pid": pid, "strengths": " | ".join(fmt(x) for x in strong) or "No metric in top quartile of ranked peers",
                     "weaknesses": " | ".join(fmt(x) for x in weak) or "No metric in bottom quartile of ranked peers"})
    return pd.DataFrame(rows).set_index("pid")


def risk_flags(t: pd.DataFrame) -> pd.Series:
    def f(r):
        out = []
        if r.confidence == "Low":
            out.append("Limited pressure sample: score leans heavily on the population prior")
        if r.G_hp < 10:
            out.append(f"Pressure balls come from only {int(r.G_hp)} innings")
        if r.z_gap <= -1.0:
            out.append("Pressure performance sits below own ordinary baseline")
        return "; ".join(out) or "None flagged"
    return t.apply(f, axis=1)

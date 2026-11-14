"""Step 2: Advantage (A), Win probability (WP), Leverage (lambda), Pressure Index (PI).

Leakage control: every delivery's A / WP / lambda / value comes from models that were
fitted WITHOUT any delivery of that delivery's match (K-fold by match).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

BALLS = 120
OUTCOMES = [0, 1, 2, 3, 4, 6, "W"]
RUNS_ARR = np.array([0, 1, 2, 3, 4, 6])
PHASES = ["Powerplay", "Middle", "Death"]


class FoldModels:
    """Par table + win-probability model fitted on a training set of matches."""

    def fit(self, tr: pd.DataFrame, el: pd.DataFrame) -> "FoldModels":
        i1 = tr[tr.innings == 1]
        tot = i1.groupby("mi").total_runs.transform("sum")
        X = i1[["B", "W"]].to_numpy(float)
        rem = (tot - i1.S).to_numpy(float)
        self.erem_m = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=12, min_samples_leaf=150,
            monotonic_cst=[1, -1], random_state=0).fit(X, rem)
        res2 = (rem - self.erem_m.predict(X)) ** 2
        self.sig_m = HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.06, max_leaf_nodes=8, min_samples_leaf=200,
            monotonic_cst=[1, 0], random_state=0).fit(X, res2)
        totals = i1.groupby("mi").total_runs.sum()
        self.mu1, self.sd1 = float(totals.mean()), float(totals.std())
        # outcome probabilities by innings x phase (faced deliveries)
        self.p_out = {}
        f = tr[tr.faced]
        r = f.runs.where(f.runs != 5, 4).where(f.runs != 7, 6).clip(upper=6)
        for (inn, ph), idx in f.groupby(["innings", "phase"], observed=True).groups.items():
            rr, w = r.loc[idx], f.is_wicket.loc[idx]
            p = [float(((rr == k) & ~w).mean()) for k in RUNS_ARR] + [float(w.mean())]
            self.p_out[(int(inn), ph)] = np.array(p) / sum(p)
        # win-probability logistic regressions (one per innings)
        lab = el.set_index("match_id")
        self.wp_m = {}
        for inn, col in [(1, "bat_first_won"), (2, "chase_won")]:
            d = tr[(tr.innings == inn) & tr.match_id.isin(el[el.fit_label_ok].match_id)]
            A = self.advantage(inn, d.S.to_numpy(float), d.W.to_numpy(float), d.B.to_numpy(float),
                               d.target.to_numpy(float))
            y = lab.loc[d.match_id, col].to_numpy()
            self.wp_m[inn] = LogisticRegression(C=100.0, max_iter=1000).fit(self._feat(A, d.B.to_numpy(float)), y)
        return self

    # ---- par table ------------------------------------------------------
    def erem(self, B, W):
        B, W = np.asarray(B, float), np.asarray(W, float)
        out = np.zeros(B.shape)
        ok = (B > 0) & (W < 10)
        if ok.any():
            out[ok] = np.maximum(self.erem_m.predict(np.c_[B[ok], W[ok]]), 0)
        return out

    def sigma(self, B, W):
        B, W = np.asarray(B, float), np.asarray(W, float)
        out = np.full(B.shape, 5.0)
        ok = (B > 0) & (W < 10)
        if ok.any():
            out[ok] = np.sqrt(np.clip(self.sig_m.predict(np.c_[B[ok], W[ok]]), 25, None))
        return out

    # ---- advantage / win probability ---------------------------------------
    def advantage(self, inn, S, W, B, T, E=None, sg=None):
        E = self.erem(B, W) if E is None else E
        sg = self.sigma(B, W) if sg is None else sg
        if inn == 2:
            return (E - (T - S)) / sg
        return (S + E - self.mu1) / self.sd1

    @staticmethod
    def _feat(A, B):
        late = 1 - np.clip(B, 0, BALLS) / BALLS
        return np.c_[A, A * late, late]

    def wp(self, inn, S, W, B, T, E=None, sg=None):
        A = self.advantage(inn, S, W, B, T, E, sg)
        p = expit(self.wp_m[inn].decision_function(self._feat(A, B)))
        if inn == 2:
            done = S >= T
            over = ((W >= 10) | (B <= 0)) & ~done
            p = np.where(done, 1.0, p)
            p = np.where(over, np.where(S == T - 1, 0.5, 0.0), p)
        return p, A

    # ---- annotate deliveries ---------------------------------------------
    def annotate(self, d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        S, W, B = d.S.to_numpy(float), d.W.to_numpy(float), d.B.to_numpy(float)
        T = d.target.to_numpy(float)
        inn = d.innings.to_numpy()
        E0, sg0 = self.erem(B, W), self.sigma(B, W)
        wp0, A = np.zeros(len(d)), np.zeros(len(d))
        for i in (1, 2):
            m = inn == i
            if m.any():
                wp0[m], A[m] = self.wp(i, S[m], W[m], B[m], T[m], E0[m], sg0[m])
        # one-ball-ahead resources (as if the next delivery were legal)
        Bn = B - 1
        En_ok, En_w = self.erem(Bn, W), self.erem(Bn, W + 1)
        sg_ok, sg_w = self.sigma(Bn, W), self.sigma(Bn, W + 1)
        P = np.zeros((len(d), len(OUTCOMES)))
        for (i, ph), p in self.p_out.items():
            P[((inn == i) & (d.phase.to_numpy() == ph))] = p
        lam = np.zeros(len(d))
        for j, o in enumerate(OUTCOMES):
            wk = o == "W"
            Sn = S if wk else S + o
            Wn = W + 1 if wk else W
            En, sgn = (En_w, sg_w) if wk else (En_ok, sg_ok)
            wpn = np.zeros(len(d))
            for i in (1, 2):
                m = inn == i
                if m.any():
                    wpn[m] = self.wp(i, Sn[m], Wn[m], Bn[m], T[m], En[m], sgn[m])[0]
            lam += P[:, j] * np.abs(wpn - wp0)
        d["E0"], d["A"], d["WP"], d["lam_raw"] = E0, A, wp0, lam
        # ---- value of each delivery: runs + change in expected remaining runs --------
        Bp = B - d.legal.to_numpy(int)
        out_striker = (d.batter_out & (d.dismissed_player_id == d.striker_id)).to_numpy(int)
        bw = d.bowler_wkt.to_numpy(int)
        Eb = self.erem(Bp, np.minimum(W + out_striker, 10))
        Ew = self.erem(Bp, np.minimum(W + bw, 10))
        d["v_bat"] = d.runs.to_numpy(float) + Eb - E0
        d["v_bowl"] = -(d.bowler_runs.to_numpy(float) + Ew - E0)
        return d


def run_oof(df: pd.DataFrame, el: pd.DataFrame, k: int = 5, seed: int = 7) -> pd.DataFrame:
    """Out-of-fold annotation: each match is scored by models that never saw it."""
    rng = np.random.default_rng(seed)
    mids = df.match_id.unique()
    fold = dict(zip(mids, rng.permutation(len(mids)) % k))
    df = df.assign(fold=df.match_id.map(fold))
    parts = []
    for f in range(k):
        tr, te = df[df.fold != f], df[df.fold == f]
        fm = FoldModels().fit(tr, el[el.match_id.isin(tr.match_id)])
        parts.append(fm.annotate(te))
    return pd.concat(parts).sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)


def finalize_pressure(d: pd.DataFrame, lam_mean: float | None = None) -> tuple[pd.DataFrame, float]:
    lam_mean = float(d.lam_raw.mean()) if lam_mean is None else lam_mean
    d = d.copy()
    d["lam"] = d.lam_raw / lam_mean
    d["PI_bat"] = d.lam * (1 - d.WP)
    d["PI_bowl"] = d.lam * d.WP
    return d, lam_mean


def thresholds(d: pd.DataFrame, hi: float = 0.80, lo: float = 0.50) -> dict:
    fb = d[d.faced]
    return {"bat": (float(fb.PI_bat.quantile(hi)), float(fb.PI_bat.quantile(lo))),
            "bowl": (float(d.PI_bowl.quantile(hi)), float(d.PI_bowl.quantile(lo)))}

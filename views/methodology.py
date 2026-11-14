"""Methodology: what is verified, what is proposed, what is still pending."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import analytics as an, ui
from utils.data_loader import DATA_DIR, SCOPES, dataset_counts, load_tables


@st.cache_data(show_spinner=False)
def _exclusions() -> dict[str, int]:
    t = load_tables()
    m = t["matches"]
    have = set(t["balls"].match_id)
    analysed = m[m.match_id.isin(have)]
    raw = pd.read_csv(DATA_DIR / "ball_by_ball.csv", usecols=["match_id", "innings"])
    res = analysed.result.fillna("")
    return {
        "no_ball_data": int((~m.match_id.isin(have)).sum()),
        "no_result": int(res.str.contains("No Result", case=False).sum()),
        "dls": int((res.str.contains(r"D/?L", case=False, regex=True) | analysed.revised_target.notna()).sum()),
        "super_over_matches": int(raw[raw.innings > 2].match_id.nunique()),
    }


def render() -> None:
    ui.page_header("Methodology", "How the data is cleaned, how pressure is defined and scored, and what the validation shows",
                   "Every number in the rankings can be traced to delivery-level data. Sections are labelled by status, "
                   "and the validation results are shown with their weaknesses.")
    c, ex = dataset_counts(), _exclusions()

    st.markdown("### Data and cleaning rules  " + ui.chip("Verified", "live"), unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        ("Runs", "`runs` is bat runs only. Total ball runs = runs + extras1_runs + extras_2_runs."),
        ("Legal delivery", "Not a wide and not a no-ball (extras types Overstep and other). Verified: 30,622 of 30,626 non-final overs have exactly six legal balls."),
        ("Balls faced", "Every delivery except wides."),
        ("Runs conceded by bowler", "Bat runs plus wide and no-ball extras. Byes, leg-byes and penalty runs are excluded."),
        ("Bowler wickets", "Bowled, caught, LBW, stumped, hit wicket. Run-outs, retired out and obstructing the field are not credited."),
        ("Batter dismissals", "From dismissed_player_id, which correctly handles non-striker run-outs. Retired-hurt rows have no wicket flag and are ignored."),
        ("Super overs", f"Innings 3 and 4 ({ex['super_over_matches']} matches) are excluded from every statistic."),
    ], columns=["Item", "Rule"]), hide_index=True, width="stretch")

    st.markdown("### Competitions and comparability  " + ui.chip("Verified", "live"), unsafe_allow_html=True)
    st.markdown(
        "- **IPL** is the primary population. Par tables and baselines are estimated from IPL only.\n"
        "- **Emerging / Rising Stars** (two tournaments, 30 matches) form a separate prospect watchlist and are never pooled with IPL.\n"
        "- **T20I** is analysed separately because opposition quality and scoring conditions differ.\n"
        "- The sidebar scope selector drives every page. No figure combines competition groups.")
    st.markdown(
        f"Of {c['matches_listed']:,} matches listed, {c['matches_analysed']:,} have ball-by-ball data "
        f"({ex['no_ball_data']} abandoned matches have none). Exclusions for the pressure model are listed below: "
        f"{ex['no_result']} no-result matches, and {ex['dls']} matches with a revised target or a D/L result "
        "(excluded from chase features because balls remaining and target change).")

    if not an.available():
        st.warning("Analytics outputs not found. Run `python -m engine.run` to generate them; the sections below use those results.")
        return
    mt, val = an.meta(), an.validation()
    ipl = mt["scopes"]["IPL"]
    vi = val["IPL"]
    ex = ipl["exclusions"]
    st.markdown(
        f"**Pressure-model exclusions (IPL):** {ex['usable']} of {ex['matches_with_ball_data']} matches are usable. "
        f"Excluded: {ex.get('dls', 0)} D/L or revised-target, {ex.get('no_result', 0)} no-result, "
        f"{ex.get('inn1_incomplete', 0)} curtailed first innings, {ex.get('missing_inn2', 0)} with no second innings. "
        f"{ex.get('tied_kept', 0)} tied matches are kept (super overs are excluded). "
        f"Targets agree with the recorded winner in {ipl['target_check']['target_vs_winner_agreement'] * 100:.0f}% of "
        f"{ipl['target_check']['matches_checked']} checked matches.")

    st.markdown("### Pressure definition  " + ui.chip("Implemented and validated", "live"), unsafe_allow_html=True)
    st.markdown("Pressure on a side is **how much is at stake on this ball multiplied by how likely that side is to lose**, "
                "using only information available before the delivery. State: balls left *B*, wickets lost *W*, score *S*, target *T*.")
    st.markdown("**1. Par table.** A monotone gradient-boosted model gives the mean remaining runs *E(B,W)* and its spread *sigma(B,W)* "
                "from first-innings states of the training matches.")
    st.latex(r"A_{\text{chase}}=\frac{E(B,W)-(T-S)}{\sigma(B,W)}\qquad A_{\text{first}}=\frac{S+E(B,W)-\mu_1}{\sigma_1}")
    st.markdown("**2. Win probability.** One logistic regression per innings on *A*, lateness and their product. "
                "Terminal states (target reached, all out, innings over) are handled exactly.")
    st.markdown("**3. Leverage.** The expected one-ball swing in win probability over the outcomes 0, 1, 2, 3, 4, 6 and wicket, "
                "weighted by their observed frequency in that innings and phase, then scaled to mean 1.")
    st.latex(r"\lambda=\sum_{o}P(o\mid\text{phase})\,\bigl|WP(s_o)-WP(s)\bigr|")
    st.markdown("**4. Pressure Index.** A high-pressure ball is in the top 20% of the index for its role.")
    st.latex(r"PI_{\text{bat}}=\lambda\,(1-WP_{\text{bat}})\qquad PI_{\text{bowl}}=\lambda\,WP_{\text{bat}}")
    st.markdown("**Leakage control.** Matches are split into five folds. Each match is scored by par-table and win-probability models "
                "that never saw it. Match results only calibrate win probability and are never used to score a player. "
                "Emerging matches are scored with models fitted on all IPL matches.")

    st.markdown("### Performance measurement  " + ui.chip("Implemented", "live"), unsafe_allow_html=True)
    st.markdown("Each delivery is valued by the runs it produces plus the change in expected remaining runs, so dot balls and wickets "
                "are priced automatically in the same units.")
    st.latex(r"v_{\text{bat}}=r+E(B',W')-E(B,W)\qquad v_{\text{bowl}}=-\bigl(r_{\text{conceded}}+E(B',W'_{\text{bowler}})-E(B,W)\bigr)")
    st.markdown("- Batter wickets are charged only when the striker is out; non-striker run-outs and retirements are not. "
                "Bowlers are charged wides and no-balls, but not byes or leg-byes, and are credited only with bowler wickets.\n"
                "- Values are centred on the league mean of the same class (high-pressure or not), innings and phase, so 0 means an average "
                "player in the same kind of situation.\n"
                "- **Pressure score** = runs per 100 pressure balls above average. Higher is better for batters and bowlers.\n"
                "- **Baseline** = the same measure on all of the player's balls below the high-pressure threshold.")

    st.markdown("### Sample-size adjustment and ranking  " + ui.chip("Implemented", "live"), unsafe_allow_html=True)
    bm, wm = ipl["model"]["bat"], ipl["model"]["bowl"]
    st.markdown("True pressure performance is modelled as skill plus a pressure-specific shift. Skill variance comes from the large baseline sample and "
                "the shift variance from the pressure-minus-baseline gap, both by DerSimonian-Laird. A bivariate normal-normal empirical-Bayes "
                "update then shrinks each player toward the population, letting the baseline inform the pressure estimate. "
                "Standard errors are cluster-robust by innings, because balls in an innings are correlated.")
    st.latex(r"\hat p_i=\mu_p+K_i\,(y_i-\mu),\quad y_i=(x_i^{\text{pressure}},x_i^{\text{baseline}}),\quad K_i=\Sigma(\Sigma+D_i)^{-1}")
    st.dataframe(pd.DataFrame([
        {"Role": "Batting", "Eligible": bm["k"], "Skill SD (runs/100 balls)": (bm["tau2_skill"] ** 0.5) * 100,
         "Pressure-shift SD": (bm["tau2_pressure"] ** 0.5) * 100, "Baseline reliability": bm["reliability_baseline"]},
        {"Role": "Bowling", "Eligible": wm["k"], "Skill SD (runs/100 balls)": (wm["tau2_skill"] ** 0.5) * 100,
         "Pressure-shift SD": (wm["tau2_pressure"] ** 0.5) * 100, "Baseline reliability": wm["reliability_baseline"]},
    ]).round(2), hide_index=True, width="stretch")
    st.markdown("Confidence is the share of the population spread resolved by a player's data (High 50% or more, Medium 30-50%, Low below 30%). "
                "P(top 10) comes from 4,000 posterior draws. Consistency is the share of a player's pressure innings, with at least 3 pressure balls, "
                "with above-average value.")

    st.markdown("### Validation results  " + ui.chip("IPL", "live"), unsafe_allow_html=True)
    w = vi["wp"]
    rows = []
    for inn, name in [("1", "First innings"), ("2", "Chase")]:
        rows.append({"Innings": name, "AUC out-of-fold": w["oof"][inn]["auc"], "AUC in-sample": w["in_sample"][inn]["auc"],
                     "AUC train before 2025, test 2025+": w["temporal"][inn]["auc"],
                     "Log loss": w["oof"][inn]["logloss"], "Log loss (no model)": w["oof"][inn]["baseline_logloss"]})
    st.markdown("**Win-probability model, leakage check.** Out-of-fold and in-sample AUC are close, so the model is not memorising matches, "
                "and a strict train-before-2025 test holds up. The first-innings model adds little beyond chance, which is expected "
                "because a first innings has no target.")
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    st.markdown("**Pressure rises where it should.** Chases, close finishes and steep required rates are heavily over-represented among "
                "high-pressure balls. Pressure on the batter peaks when a chase is hard but alive and falls once it is nearly lost. "
                "First-innings death overs are not flagged, which is a real limitation of a win-probability definition.")
    lift = pd.DataFrame(vi["difficulty"]["tag_lift"])
    lift["high_pressure_share"] = (lift.high_pressure_share * 100).round(1)
    lift["share_elsewhere"] = (lift.share_elsewhere * 100).round(1)
    st.dataframe(lift.rename(columns={"situation": "Situation", "balls": "Balls", "high_pressure_share": "High-pressure %", "share_elsewhere": "Elsewhere %"}),
                 hide_index=True, width="stretch")
    st.markdown("**Sensitivity to the pressure threshold and minimum sample.** Rankings barely move, because they are dominated by "
                "each player's overall level, which the threshold does not change.")
    st.dataframe(pd.DataFrame(vi["sensitivity"]).rename(columns={"role": "Role", "variant": "Variant", "players": "Players",
                 "spearman_vs_main": "Rank correlation vs main", "top10_overlap": "Top-10 overlap"}).round(3),
                 hide_index=True, width="stretch")
    st.markdown("**Ranking stability across match-level splits.** Matches are split in half, each half is scored from scratch, and the two "
                "rankings are compared. Shrinkage multiplies stability several times over raw pressure averages, but half-sample agreement stays modest.")
    stab = []
    for role, name in [("bat", "Batting"), ("bowl", "Bowling")]:
        sx = vi["stability"][role]
        stab.append({"Role": name, "Random splits": sx["splits_used"], "Rank corr (shrunk score)": sx["random_split_post_spearman"],
                     "Rank corr (raw pressure mean)": sx["random_split_raw_spearman"],
                     "Avg top-10 overlap (of 10)": sx["random_split_top10_overlap"],
                     "2022-24 vs 2025-26 rank corr": sx["season_split_post_spearman"]})
    st.dataframe(pd.DataFrame(stab).round(2), hide_index=True, width="stretch")
    hb, hw = vi["gap_heterogeneity"]["batting"], vi["gap_heterogeneity"]["bowling"]
    st.markdown(f"**Key finding.** Individual pressure-minus-baseline gaps are statistically indistinguishable from noise "
                f"(p = {hb['p_value']:.2f} batting, {hw['p_value']:.2f} bowling). The data cannot separate players who rise from those who fade. "
                "The best available estimate of pressure performance is therefore overall situation-adjusted quality, weighted by how much "
                "evidence each player has, with pressure-ball evidence included and uncertainty reported.")

    st.markdown("### Limitations")
    st.markdown(
        "- Players shape the match state they play in, so the Pressure Index is partly endogenous.\n"
        "- The par table is fitted on first innings only (second innings are cut short by the target) and ignores venue, opposition and pitch.\n"
        "- First-innings win probability is weak, so pressure concentrates in chases.\n"
        "- A 40-ball pressure sample is small: top-10 membership is uncertain, and order within a list is often not separable.\n"
        "- Players with limited IPL data can appear above better-established players when their baseline is strong; check the confidence and pressure-sample columns.\n"
        "- Emerging players are scored on the IPL scale with very small samples and are a watchlist, not a ranking to act on.\n"
        "- Non-striker run-outs credit no cost to the dismissed non-striker.\n"
        "- The dataset has no role, age, nationality or auction information.")

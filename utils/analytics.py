"""Read-only access to the engine's output files in analytics/ (written by `python -m engine.run`)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ANALYTICS = Path(__file__).resolve().parent.parent / "analytics"
ROLE_FILE = {"bat": "batting", "bowl": "bowling"}
SCOPE_FOR_VALIDATION = {"IPL": "IPL", "Emerging": "IPL", "T20I": "T20I"}   # Emerging is scored on the IPL scale


def available() -> bool:
    return (ANALYTICS / "meta.json").exists() and (ANALYTICS / "ranking_IPL_batting.csv").exists()


@st.cache_data(show_spinner=False)
def ranking(scope: str, role: str) -> pd.DataFrame | None:
    p = ANALYTICS / f"ranking_{scope}_{ROLE_FILE[role]}.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def clusters(scope: str, role: str) -> pd.DataFrame | None:
    p = ANALYTICS / f"clusters_{scope}_{ROLE_FILE[role]}.csv"
    return pd.read_csv(p, parse_dates=["match_date"]) if p.exists() else None


@st.cache_data(show_spinner=False)
def meta() -> dict:
    return json.loads((ANALYTICS / "meta.json").read_text())


@st.cache_data(show_spinner=False)
def validation() -> dict:
    return json.loads((ANALYTICS / "validation.json").read_text())


def player_row(scope: str, role: str, pid: int) -> pd.Series | None:
    r = ranking(scope, role)
    if r is None:
        return None
    m = r[r.player_id == pid]
    return None if m.empty else m.iloc[0]


def eligibility_text(scope: str) -> str:
    cfg = meta()["config"][scope]
    return (f"Eligible: at least {cfg['min_hp']} high-pressure balls from {cfg['min_inn']}+ innings "
            f"and {cfg['min_base']}+ baseline balls.")

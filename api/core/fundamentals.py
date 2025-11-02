# api/core/fundamentals.py
from __future__ import annotations

import yfinance as yf
import pandas as pd
from functools import lru_cache


def _safe_info(tkr: str) -> dict:
    try:
        return yf.Ticker(tkr).info or {}
    except Exception:
        return {}


@lru_cache(maxsize=4096)
def get_fundamentals(tkr: str) -> dict:
    """Retourne un dict de fondamentaux de base (peut manquer selon le titre)."""
    info = _safe_info(tkr)
    # champs utiles (si dispo)
    rev_growth = info.get("revenueGrowth", None)  # ex: 0.08 = +8% YoY
    eps_growth = info.get("earningsQuarterlyGrowth", None)  # ou "earningsGrowth"
    profit_margin = info.get("profitMargins", None)  # ex: 0.12 = 12%
    roe = info.get("returnOnEquity", None)  # ex: 0.15
    debt_to_equity = info.get("debtToEquity", None)  # ex: 80.0 (ratio)

    # fallback minimal (si dispo ailleurs)
    if eps_growth is None:
        eps_growth = info.get("earningsGrowth", None)

    return {
        "eps_growth": eps_growth,
        "revenue_growth": rev_growth,
        "profit_margin": profit_margin,
        "roe": roe,
        "debt_to_equity": debt_to_equity,
    }


def compute_fscore_basic(metrics: dict) -> tuple[float, dict]:
    """Calcule un Fscore simple sur 0..5 puis normalisé 0..100.
       Règles:
       +1 si eps_growth > 0
       +1 si revenue_growth > 0
       +1 si profit_margin > 0.10
       +1 si roe > 0.10
       +1 si debt_to_equity is not None and < 100
    """
    epsg = metrics.get("eps_growth")
    rg = metrics.get("revenue_growth")
    pm = metrics.get("profit_margin")
    roe = metrics.get("roe")
    dte = metrics.get("debt_to_equity")

    PM_MIN = 0.12
    ROE_MIN = 0.12
    DTE_MAX = 150.0

    pts = 0
    if (epsg is not None) and (epsg > 0):
        pts += 1
    if (rg is not None) and (rg > 0):
        pts += 1
    if (pm is not None) and (pm >= PM_MIN):
        pts += 1
    if (roe is not None) and (roe >= ROE_MIN):
        pts += 1
    if (dte is not None) and (dte <= DTE_MAX):
        pts += 1

    raw5 = float(pts)
    f100 = (raw5 / 5.0) * 100.0
    details = {"points": raw5, "epsg": epsg, "rev": rg, "pm": pm, "roe": roe, "dte": dte}
    return f100, details

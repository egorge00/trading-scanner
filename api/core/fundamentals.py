import yfinance as yf
from functools import lru_cache
import math
import numpy as np

__all__ = [
    "get_fundamentals",
    "compute_fscore_continuous_safe",
]

# -------- utils robustes --------
def _scale(x, lo, hi):
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    return (x - lo) / (hi - lo)


def _scale_sym(x, neg, pos):
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    if x <= -neg:
        return 0.0
    if x >= +pos:
        return 1.0
    return (x + neg) / (neg + pos)


def _winsorize(x, lo, hi):
    if x is None:
        return None
    try:
        return float(min(max(x, lo), hi))
    except Exception:
        return None


# -------- récupération Yahoo (best-effort) --------
@lru_cache(maxsize=4096)
def get_fundamentals(tkr: str) -> dict:
    try:
        info = yf.Ticker(tkr).info or {}
    except Exception:
        info = {}

    epsg = info.get("earningsGrowth", info.get("earningsQuarterlyGrowth"))
    revg = info.get("revenueGrowth")
    pm = info.get("profitMargins")  # net margin (TTM)
    roe = info.get("returnOnEquity")
    dte = info.get("debtToEquity")

    dte = _winsorize(dte, 0, 400)  # borne haute douce

    return {
        "eps_growth": epsg,
        "revenue_growth": revg,
        "profit_margin": pm,
        "roe": roe,
        "debt_to_equity": dte,
    }


def compute_fscore_continuous_safe(metrics: dict):
    """
    Retourne (fscore_0_100 | None, details).
    None si < 2 métriques valides. Renormalise les poids disponibles.
    """
    # Normalisations 0..1
    eps_norm = _scale_sym(metrics.get("eps_growth"), 0.10, 0.20)  # [-10%..+20%]
    rev_norm = _scale_sym(metrics.get("revenue_growth"), 0.10, 0.20)
    pm_norm = _scale(metrics.get("profit_margin"), 0.05, 0.25)  # 5%..25%
    roe_norm = _scale(metrics.get("roe"), 0.08, 0.25)  # 8%..25%

    dte_raw = _scale(metrics.get("debt_to_equity"), 50.0, 200.0)  # 50..200
    dte_norm = None if dte_raw is None else max(0.0, min(1.0, 1.0 - dte_raw))  # pénalité

    parts = [
        ("eps", eps_norm, 0.22),
        ("rev", rev_norm, 0.18),
        ("pm", pm_norm, 0.22),
        ("roe", roe_norm, 0.22),
        ("dte", dte_norm, 0.16),
    ]
    valid = [
        (k, v, w)
        for (k, v, w) in parts
        if v is not None and not (isinstance(v, float) and (v != v))
    ]
    n_valid = len(valid)
    if n_valid < 2:
        return None, {"reason": "insufficient_data", "valid": n_valid}

    sum_w = sum(w for _, _, w in valid)
    score01 = sum((v * w) for _, v, w in valid) / sum_w
    score01 = max(0.0, min(1.0, float(score01)))

    # Lissage léger
    def _sigmoid(z, k=5.0):
        return 1.0 / (1.0 + math.exp(-k * z))

    score01_sm = _sigmoid(score01 - 0.5, k=5.0)
    score01_mix = 0.7 * score01 + 0.3 * score01_sm

    f100 = round(100.0 * score01_mix, 2)
    details = {
        "valid_metrics": n_valid,
        "eps_norm": eps_norm,
        "rev_norm": rev_norm,
        "pm_norm": pm_norm,
        "roe_norm": roe_norm,
        "dte_norm": dte_norm,
    }
    return f100, details

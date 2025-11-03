import yfinance as yf
from functools import lru_cache
import math
import numpy as np

# ---------- Normalisations 0..1 robustes -----------
def _clip(x, lo, hi):
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return None


def _scale(x, lo, hi):
    """Map [lo..hi] -> [0..1], clip; None si x None."""
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
    """Map [-neg..+pos] -> [0..1], centré 0."""
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
    """Renvoie x limité entre lo et hi (pour éviter outliers destructifs)."""
    if x is None:
        return None
    try:
        return float(min(max(x, lo), hi))
    except Exception:
        return None


# ---------- Récup Yahoo (best-effort) -----------
@lru_cache(maxsize=4096)
def get_fundamentals(tkr: str) -> dict:
    try:
        info = yf.Ticker(tkr).info or {}
    except Exception:
        info = {}

    # Certains champs peuvent manquer selon les titres:
    epsg = info.get("earningsGrowth", info.get("earningsQuarterlyGrowth"))   # YoY approx
    revg = info.get("revenueGrowth")
    pm   = info.get("profitMargins")          # net margin TTM (ex: 0.12 = 12%)
    roe  = info.get("returnOnEquity")         # (ex: 0.15 = 15%)
    dte  = info.get("debtToEquity")           # (ex: 80.0)

    # Winsorisation douce (les extrêmes ne tuent pas le score)
    dte  = _winsorize(dte, 0, 400)  # borne haute 400

    return {
        "eps_growth": epsg,
        "revenue_growth": revg,
        "profit_margin": pm,
        "roe": roe,
        "debt_to_equity": dte,
    }


def compute_fscore_continuous_safe(metrics: dict) -> tuple[float | None, dict]:
    """
    Fscore 0..100 CONTINU, avec règles de sûreté:
    - Renvoie None si < 2 métriques valides (on préfère '—' à du faux).
    - Renormalise les poids sur les métriques disponibles.
    - Scales robustes + winsorization.
    """

    # 1) Normalisations (0..1). Plages choisies pour l’investisseur LT.
    eps_norm = _scale_sym(metrics.get("eps_growth"),    0.10, 0.20)  # [-10%..+20%]
    rev_norm = _scale_sym(metrics.get("revenue_growth"),0.10, 0.20)
    pm_norm  = _scale(     metrics.get("profit_margin"),0.05, 0.25)  # 5%..25%
    roe_norm = _scale(     metrics.get("roe"),          0.08, 0.25)  # 8%..25%

    # D/E → pénalité: faible D/E = 1.0 ; très élevé = 0.0 (winsorisé)
    dte_raw  = _scale(metrics.get("debt_to_equity"), 50.0, 200.0)     # 50..200
    dte_norm = None if dte_raw is None else max(0.0, min(1.0, 1.0 - dte_raw))

    parts = [
        ("eps", eps_norm, 0.22),
        ("rev", rev_norm, 0.18),
        ("pm",  pm_norm,  0.22),
        ("roe", roe_norm, 0.22),
        ("dte", dte_norm, 0.16),
    ]

    # 2) Compter les métriques valides
    valid = [(k,v,w) for (k,v,w) in parts if v is not None and not np.isnan(v)]
    n_valid = len(valid)
    if n_valid < 2:
        return None, {"reason":"insufficient_data", "valid": n_valid}

    # 3) Renormaliser les poids sur les métriques disponibles
    sum_w = sum(w for _,_,w in valid)
    score01 = sum((v * w) for _,v,w in valid) / sum_w
    score01 = max(0.0, min(1.0, float(score01)))

    # 4) Lissage léger (stabilité autour de 0.5)
    def _sigmoid(z, k=5.0): return 1.0 / (1.0 + math.exp(-k*z))
    score01_sm  = _sigmoid(score01 - 0.5, k=5.0)
    score01_mix = 0.7*score01 + 0.3*score01_sm

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

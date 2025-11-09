from functools import lru_cache
import numpy as np
import pandas as pd
import yfinance as yf
import math

__all__ = ["get_fundamentals", "compute_fscore_continuous_safe"]

# ---------- helpers num ----------

def _nan(x):
    try:
        return np.nan if x is None else float(x)
    except Exception:
        return np.nan

def _last(series: pd.Series):
    try:
        return series.dropna().iloc[-1]
    except Exception:
        return np.nan

def _avg(series: pd.Series, n=2):
    try:
        s = series.dropna().tail(n)
        return np.nan if s.empty else float(s.mean())
    except Exception:
        return np.nan

def _pct_change(series: pd.Series, periods=4):
    # YoY sur trimestriel (periods=4) ou annuel (periods=1)
    try:
        s = series.dropna()
        if len(s) <= periods:
            return np.nan
        return float((s.iloc[-1] - s.iloc[-1-periods]) / abs(s.iloc[-1-periods]))
    except Exception:
        return np.nan

def _winsorize(x, lo, hi):
    try:
        return float(min(max(x, lo), hi))
    except Exception:
        return np.nan

# ---------- lecture statements ----------

@lru_cache(maxsize=4096)
def _load_statements(tkr: str):
    t = yf.Ticker(tkr)

    # états financiers (annual + quarterly) — best effort
    try:
        is_q = t.get_income_stmt(freq="quarterly")  # index = champs, colonnes = dates
    except Exception:
        is_q = pd.DataFrame()

    try:
        is_a = t.get_income_stmt(freq="annual")
    except Exception:
        is_a = pd.DataFrame()

    try:
        bs_a = t.get_balance_sheet(freq="annual")
    except Exception:
        bs_a = pd.DataFrame()

    # metadata .info uniquement en secours
    try:
        info = t.get_info()  # nouveau wrapper (évite .info legacy si possible)
    except Exception:
        try:
            info = t.info
        except Exception:
            info = {}

    return is_q, is_a, bs_a, (info or {})

def _series_from(df: pd.DataFrame, key: str):
    # Retourne une série chronologique (colonne -> index datetime croissant) pour le champ key
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    if key not in df.index:
        return pd.Series(dtype="float64")
    s = df.loc[key].T  # colonnes -> index
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.sort_index()
    return pd.to_numeric(s, errors="coerce")

def _safe_ratio(num, den):
    try:
        num = float(num); den = float(den)
        if den == 0 or np.isnan(num) or np.isnan(den):
            return np.nan
        return num / den
    except Exception:
        return np.nan

# ---------- extraction métriques fondamentales reconstruites ----------

@lru_cache(maxsize=4096)
def get_fundamentals(tkr: str) -> dict:
    """
    Reconstruit les métriques clés depuis les états financiers :
      - eps_growth (YoY) : ΔEPS dilué sur 4 trimestres
      - revenue_growth (YoY) : ΔCA sur 4 trimestres
      - profit_margin (TTM) : netIncomeTTM / revenueTTM
      - roe (TTM) : netIncomeTTM / equity moyen (2 derniers annuels)
      - debt_to_equity (annuel) : totalDebt / totalStockholderEquity

    Chacune peut renvoyer NaN si les données sont insuffisantes.
    """
    is_q, is_a, bs_a, info = _load_statements(tkr)

    # Séries trimestrielles
    s_revenue_q = _series_from(is_q, "TotalRevenue")
    s_eps_q     = _series_from(is_q, "DilutedEPS")
    s_net_q     = _series_from(is_q, "NetIncome")
    # Séries annuelles
    s_revenue_a = _series_from(is_a, "TotalRevenue")
    s_net_a     = _series_from(is_a, "NetIncome")
    s_equity_a  = _series_from(bs_a, "TotalStockholderEquity")
    s_debt_a    = _series_from(bs_a, "TotalDebt")

    # TTM approximé depuis trimestriel (somme des 4 derniers)
    def _ttm(series):
        s = series.dropna().tail(4)
        return np.nan if len(s) < 4 else float(s.sum())

    revenue_ttm = _ttm(s_revenue_q)
    net_ttm     = _ttm(s_net_q)

    # YoY growth depuis trimestriel (plus réactif que annuel)
    eps_growth_yoy = _pct_change(s_eps_q, periods=4)   # ΔEPS YoY
    rev_growth_yoy = _pct_change(s_revenue_q, periods=4)

    # profit margin TTM
    pm_ttm = _safe_ratio(net_ttm, revenue_ttm)

    # ROE = NetIncome TTM / Equity moyen (annuel, 2 derniers points)
    equity_avg = _avg(s_equity_a, n=2)
    roe_ttm = _safe_ratio(net_ttm, equity_avg)

    # Debt/Equity (dernier annuel)
    dte = _safe_ratio(_last(s_debt_a), _last(s_equity_a))
    # Yahoo peut renvoyer déjà un ratio "debtToEquity" (en %) — on prend notre calc en priorité
    if np.isnan(dte):
        dte = _nan(info.get("debtToEquity"))

    # Winsorize D/E pour éviter outliers extrêmes
    if not np.isnan(dte):
        dte = _winsorize(dte, 0.0, 5.0)  # ratio en fois (ex: 0.8 = 80%)

    # Fallback sur info pour marges/roe si TTM absent (rare)
    if np.isnan(pm_ttm):
        pm_ttm = _nan(info.get("profitMargins"))
    if np.isnan(roe_ttm):
        roe_ttm = _nan(info.get("returnOnEquity"))

    return {
        "eps_growth": eps_growth_yoy,     # ex: +0.12 = +12%
        "revenue_growth": rev_growth_yoy, # ex: +0.07 = +7%
        "profit_margin": pm_ttm,          # ex: 0.14 = 14%
        "roe": roe_ttm,                   # ex: 0.18 = 18%
        "debt_to_equity": dte,            # ex: 0.8 = D/E 0.8x (ou 80% si tu préfères)
        "diag": {
            "eps_q_len": int(len(s_eps_q.dropna())),
            "rev_q_len": int(len(s_revenue_q.dropna())),
            "net_q_len": int(len(s_net_q.dropna())),
            "equity_a_len": int(len(s_equity_a.dropna())),
            "debt_a_len": int(len(s_debt_a.dropna())),
            "used_info_fallback": bool(np.isnan(pm_ttm) or np.isnan(roe_ttm)),
        }
    }

# ---------- scoring fondamental (même API que ta version 'safe') ----------

def _scale(x, lo, hi):
    if x is None or np.isnan(x): return None
    if x <= lo: return 0.0
    if x >= hi: return 1.0
    return (x - lo) / (hi - lo)

def _scale_sym(x, neg, pos):
    if x is None or np.isnan(x): return None
    if x <= -neg: return 0.0
    if x >= +pos: return 1.0
    return (x + neg) / (neg + pos)

def compute_fscore_continuous_safe(metrics: dict):
    """
    Score fondamental 0..100 :
    - None si < 2 métriques valides
    - pondérations renormalisées
    """
    eps_norm = _scale_sym(metrics.get("eps_growth"),     0.10, 0.20)  # [-10%..+20%]
    rev_norm = _scale_sym(metrics.get("revenue_growth"), 0.10, 0.20)
    pm_norm  = _scale(     metrics.get("profit_margin"), 0.05, 0.25)  # 5%..25%
    roe_norm = _scale(     metrics.get("roe"),           0.08, 0.25)  # 8%..25%

    # D/E : plus petit = mieux. Admet ratio en "x" (0.8 = 80%)
    # mappe [0.5 .. 2.0] -> [1..0] (idéal ~0.5-1.0), puis clip 0..1
    dte = metrics.get("debt_to_equity")
    if dte is None or np.isnan(dte):
        dte_norm = None
    else:
        dte_lin = 1.0 - _scale(dte, 0.5, 2.0)  # 0.5x -> ~1, 2.0x -> ~0
        dte_norm = None if dte_lin is None else max(0.0, min(1.0, dte_lin))

    parts = [
        ("eps", eps_norm, 0.22),
        ("rev", rev_norm, 0.18),
        ("pm",  pm_norm,  0.22),
        ("roe", roe_norm, 0.22),
        ("dte", dte_norm, 0.16),
    ]
    valid = [(k,v,w) for (k,v,w) in parts if v is not None]
    if len(valid) < 2:
        return None, {"reason":"insufficient_data", "valid": len(valid), "diag": metrics.get("diag")}

    sw = sum(w for _,_,w in valid)
    s01 = sum((v*w) for _,v,w in valid) / sw
    s01 = max(0.0, min(1.0, float(s01)))

    # léger lissage
    z = s01 - 0.5
    s01_sm = 1.0 / (1.0 + math.exp(-5.0*z))
    s01_mix = 0.7*s01 + 0.3*s01_sm

    f100 = round(100.0 * s01_mix, 2)
    details = {"valid_metrics": len(valid), "diag": metrics.get("diag")}
    return f100, details

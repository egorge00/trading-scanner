"""Scoring logic with extended technical indicators and diagnostics."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Compute an exponential moving average with a modest warmup."""

    return series.ewm(span=span, adjust=False, min_periods=max(1, span // 2)).mean()


def _tanh_clip(values: pd.Series | np.ndarray | float, *, scale: float = 1.0, clip: float = 3.0) -> np.ndarray:
    """Helper to softly squash values into [-1, 1]."""

    arr = np.asarray(values, dtype="float64")
    arr = np.clip(arr, -clip, clip)
    return np.tanh(scale * arr)


def _rsi_ema(close: pd.Series, period: int = 14) -> pd.Series:
    """EMA-based RSI implementation without external dependencies."""

    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast = _ema(close, span=12)
    slow = _ema(close, span=26)
    macd_line = fast - slow
    signal = _ema(macd_line, span=9)
    hist = macd_line - signal
    return macd_line, signal, hist


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window // 2).mean()


def _bollinger_bbp(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    mid = close.rolling(window, min_periods=window // 2).mean()
    std = close.rolling(window, min_periods=window // 2).std(ddof=0)
    denom = 4.0 * std
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_b = (close - (mid - num_std * std)) / denom
    pct_b = pct_b.clip(lower=0.0, upper=1.0)
    score = (pct_b - 0.5) * 2.0
    return score


def _volume_z(volume: pd.Series, window: int = 20) -> pd.Series:
    vol_ma = volume.rolling(window, min_periods=window // 2).mean()
    vol_std = volume.rolling(window, min_periods=window // 2).std(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (volume - vol_ma) / vol_std
    z = z.clip(-3.0, 3.0) / 3.0
    return z


def _pct_to_hh52(close: pd.Series, window: int = 252) -> pd.Series:
    hh = close.rolling(window, min_periods=window // 4).max()
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = close / hh - 1.0
    pct = pct.clip(lower=-0.5, upper=0.05) / 0.5
    pct = pct.clip(-1.0, 1.0)
    return pct


def _mean_reversion_score(close: pd.Series, window: int = 20) -> pd.Series:
    ma = close.rolling(window, min_periods=window // 2).mean()
    std = close.rolling(window, min_periods=window // 2).std(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (close - ma) / std
    z = z.replace([np.inf, -np.inf], np.nan)
    z = z.clip(-3.0, 3.0)
    score = (-z) / 3.0
    score = score.clip(-1.0, 1.0)
    return score


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    high_shift = high.shift(1)
    low_shift = low.shift(1)
    close_shift = close.shift(1)

    up_move = high - high_shift
    down_move = low_shift - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - close_shift).abs()
    tr3 = (low - close_shift).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr

    with np.errstate(divide="ignore", invalid="ignore"):
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).abs()) * 100.0
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule des KPIs robustes pour les tickers Yahoo Finance.
    Colonnes: RSI, MACD_hist, SMA50, SMA200, %toHH52, VolZ20
    Corrigé pour gérer doublons et MultiIndex.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()

    # --- 1) Aplatir MultiIndex éventuel ---
    if isinstance(x.columns, pd.MultiIndex):
        lvl0 = x.columns.get_level_values(0)
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in set(lvl0)]
        if keep:
            x = x[keep]
            if isinstance(x.columns, pd.MultiIndex):
                x.columns = [c[0] if isinstance(c, tuple) else c for c in x.columns]
        else:
            x = x.droplevel(-1, axis=1)

    # --- 2) Coalescer les colonnes OHLCV en Series numériques alignées ---
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in x.columns]
    x = x[cols].copy().sort_index()

    def _best_series_from_dataframe(df_like: pd.DataFrame, idx, name: str) -> pd.Series:
        """
        Sélectionne la colonne la plus 'complète' d'un DataFrame (max non-NaN),
        la convertit en float et l'aligne sur idx.
        """
        if df_like.shape[1] == 0:
            return pd.Series(index=idx, dtype=float, name=name)
        best_s, best_cnt = None, -1
        for j in range(df_like.shape[1]):
            s = df_like.iloc[:, j]
            if not isinstance(s, pd.Series):
                s = pd.Series(s, index=df_like.index, name=name)
            s = pd.to_numeric(s, errors="coerce")
            cnt = int(s.notna().sum())
            if cnt > best_cnt:
                best_cnt, best_s = cnt, s
        if best_s is None:
            return pd.Series(index=idx, dtype=float, name=name)
        return best_s.reindex(idx).astype(float)

    def _force_series_any(obj, name: str, idx) -> pd.Series:
        """
        Retourne TOUJOURS une pd.Series[float] alignée sur idx, quelle que soit la forme d'entrée :
        - Series -> numeric float
        - DataFrame -> choisit la colonne la plus complète
        - ndarray 2D -> choisit la colonne la plus complète
        - ndarray 1D / list / scalar -> Series alignée (broadcast si scalaire)
        """
        # 1) DataFrame (doublons "Close", etc.)
        if isinstance(obj, pd.DataFrame):
            return _best_series_from_dataframe(obj, idx, name)

        # 2) Series
        if isinstance(obj, pd.Series):
            s = pd.to_numeric(obj, errors="coerce")
            return s.reindex(idx).astype(float)

        # 3) Numpy array / listes / scalaires
        import numpy as np

        arr = np.asarray(obj)
        # 3a) 2D: choisir la meilleure colonne
        if arr.ndim == 2:
            if arr.shape[0] != len(idx):
                # si transposé par erreur, on tente l'autre orientation
                if arr.shape[1] == len(idx):
                    arr = arr.T
                else:
                    # forme inexploitables -> Série vide
                    return pd.Series(index=idx, dtype=float, name=name)
            best_s, best_cnt = None, -1
            for j in range(arr.shape[1]):
                s = pd.Series(arr[:, j], index=idx, name=name)
                s = pd.to_numeric(s, errors="coerce")
                cnt = int(s.notna().sum())
                if cnt > best_cnt:
                    best_cnt, best_s = cnt, s
            if best_s is None:
                return pd.Series(index=idx, dtype=float, name=name)
            return best_s.astype(float)

        # 3b) 1D (ou scalaire broadcast)
        try:
            s = pd.Series(arr, index=idx, name=name)
        except Exception:
            s = pd.Series(index=idx, dtype=float, name=name)
        s = pd.to_numeric(s, errors="coerce")
        return s.astype(float)

    # Réécriture colonne par colonne (sans dict -> DataFrame)
    for c in cols:
        x[c] = _force_series_any(x[c], c, x.index)

    # Si Close est absent ou vide après nettoyage, on stoppe proprement
    if "Close" not in x.columns or x["Close"].dropna().empty:
        return pd.DataFrame()

    # --- 3) Close / Volume propres ---
    if "Close" not in x.columns:
        return pd.DataFrame()
    close = x["Close"].astype(float)
    vol = x["Volume"] if "Volume" in x.columns else pd.Series(index=close.index, dtype=float)

    # --- 4) RSI(14) ---
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(span=14, adjust=False, min_periods=5).mean()
    roll_down = down.ewm(span=14, adjust=False, min_periods=5).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # --- 5) MACD (12/26 EMA) + histogramme ---
    ema12 = close.ewm(span=12, adjust=False, min_periods=6).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=13).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False, min_periods=5).mean()
    macd_hist = macd - sig

    # --- 6) SMA50 / SMA200 ---
    sma50 = close.rolling(50, min_periods=25).mean()
    sma200 = close.rolling(200, min_periods=50).mean()

    # --- 7) Plus haut 52 semaines ---
    hh52 = close.rolling(252, min_periods=60).max()
    pct_to_hh52 = close / hh52 - 1.0

    # --- 8) Z-score Volume(20) ---
    if vol is not None and vol.notna().any():
        vma20 = vol.rolling(20, min_periods=10).mean()
        vstd20 = vol.rolling(20, min_periods=10).std(ddof=0)
        volz20 = (vol - vma20) / vstd20
    else:
        volz20 = pd.Series(index=close.index, dtype=float)

    out = pd.DataFrame(
        {
            "RSI": rsi,
            "MACD_hist": macd_hist,
            "SMA50": sma50,
            "SMA200": sma200,
            "%toHH52": pct_to_hh52,
            "VolZ20": volz20,
        },
        index=close.index,
    )

    return out

def _dynamic_weights(rv: float | pd.Series | None) -> Tuple[Dict[str, float], float, float]:
    """Allocate dynamic weights between momentum and mean-reversion blocks."""

    if isinstance(rv, pd.Series):
        rv_clean = rv.dropna()
        rv_value = float(rv_clean.iloc[-1]) if not rv_clean.empty else float("nan")
    elif rv is None:
        rv_value = float("nan")
    else:
        rv_value = float(rv)

    if np.isnan(rv_value):
        wmom = 0.5
    else:
        wmom = float(np.clip(rv_value / 0.25, 0.2, 0.8))
    wmr = 1.0 - wmom

    momentum_alloc = {
        "MACD": 0.25,
        "ADX": 0.20,
        "SMA_CT": 0.20,
        "SMA_LT": 0.15,
        "HH52": 0.10,
        "VOL_BBP": 0.10,
    }
    mr_alloc = {
        "RSI": 0.60,
        "MR": 0.40,
    }

    weights = {
        "MACD": momentum_alloc["MACD"] * wmom,
        "ADX": momentum_alloc["ADX"] * wmom,
        "SMA_CT": momentum_alloc["SMA_CT"] * wmom,
        "SMA_LT": momentum_alloc["SMA_LT"] * wmom,
        "HH52": momentum_alloc["HH52"] * wmom,
        "VOLZ": momentum_alloc["VOL_BBP"] * wmom * 0.5,
        "BBP": momentum_alloc["VOL_BBP"] * wmom * 0.5,
        "RSI": mr_alloc["RSI"] * wmr,
        "MR": mr_alloc["MR"] * wmr,
    }

    return weights, wmom, wmr


def _compute_subscores(row: pd.Series) -> Dict[str, float]:
    close = row.get("Close")
    sma50 = row.get("SMA50")
    sma200 = row.get("SMA200")
    rsi = row.get("RSI")
    macd_hist = row.get("MACD_hist")
    bbp = row.get("BBP")
    volz = row.get("VolZ20")
    pct_hh52 = row.get("pct_to_HH52")
    adx = row.get("ADX")
    mr = row.get("MR")

    subscores: Dict[str, float] = {}

    if pd.notna(rsi):
        subscores["RSI"] = float(np.clip((50.0 - rsi) / 25.0, -1.0, 1.0))
    else:
        subscores["RSI"] = np.nan

    if pd.notna(macd_hist) and pd.notna(close) and close not in (0.0, np.nan):
        macd_ratio = macd_hist / close
        subscores["MACD"] = float(_tanh_clip(macd_ratio, scale=20.0, clip=0.05))
    else:
        subscores["MACD"] = np.nan

    if pd.notna(adx):
        subscores["ADX"] = float(np.clip((adx - 25.0) / 25.0, -1.0, 1.0))
    else:
        subscores["ADX"] = np.nan

    if pd.notna(close) and pd.notna(sma50) and sma50 not in (0.0, np.nan):
        subscores["SMA_CT"] = float(np.tanh(((close - sma50) / sma50) * 5.0))
    else:
        subscores["SMA_CT"] = np.nan

    if pd.notna(sma50) and pd.notna(sma200) and sma200 not in (0.0, np.nan):
        subscores["SMA_LT"] = float(np.tanh(((sma50 - sma200) / sma200) * 3.0))
    else:
        subscores["SMA_LT"] = np.nan

    if pd.notna(pct_hh52):
        subscores["HH52"] = float(pct_hh52)
    else:
        subscores["HH52"] = np.nan

    if pd.notna(volz):
        subscores["VOLZ"] = float(np.clip(volz, -1.0, 1.0))
    else:
        subscores["VOLZ"] = np.nan

    if pd.notna(bbp):
        subscores["BBP"] = float(np.clip(bbp, -1.0, 1.0))
    else:
        subscores["BBP"] = np.nan

    if pd.notna(mr):
        subscores["MR"] = float(np.clip(mr, -1.0, 1.0))
    else:
        subscores["MR"] = np.nan

    return subscores


def _aggregate_score(subscores: Dict[str, float], weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    numerator = 0.0
    denom = 0.0
    used_weights: Dict[str, float] = {}

    for key, score in subscores.items():
        weight = weights.get(key)
        if weight is None or np.isnan(weight) or np.isnan(score):
            continue
        numerator += weight * score
        denom += weight
        used_weights[key] = weight

    if denom == 0.0:
        return float("nan"), {}

    normalized = {k: v / denom for k, v in used_weights.items()}
    score_norm = float(np.clip(numerator / denom, -1.0, 1.0))
    return score_norm, normalized


def compute_score(df: pd.DataFrame):
    """
    Retourne (score, action).
    - N'utilise que l'accès par crochets.
    - Ne dépend PAS d'une colonne 'close_above_sma50'.
    """

    k = compute_kpis(df)  # KPIs robustes

    # Helpers
    def last(series: pd.Series, default=np.nan) -> float:
        if series is None:
            return default
        s = series.dropna()
        if s.empty:
            return default
        try:
            return float(s.iloc[-1])
        except Exception:
            return default

    # Garanties minimales SMA
    sma50 = last(k["SMA50"]) if "SMA50" in k.columns else last(df["Close"].rolling(50, min_periods=25).mean())
    sma200 = last(k["SMA200"]) if "SMA200" in k.columns else np.nan

    # Dernières valeurs
    close = last(df["Close"])
    rsi = last(k["RSI"]) if "RSI" in k.columns else 50.0
    macd_h = last(k["MACD_hist"]) if "MACD_hist" in k.columns else 0.0
    pct_hh52 = last(k["%toHH52"]) if "%toHH52" in k.columns else -0.2
    volz20 = last(k["VolZ20"]) if "VolZ20" in k.columns else 0.0

    # --- Bonus SMA50 calculé inline (PAS de colonne close_above_sma50) ---
    above50 = int(not (np.isnan(close) or np.isnan(sma50)) and (sma50 != 0) and (close > sma50))
    bonus50 = 0.15 if above50 == 1 else -0.05

    # Normalisations [-1,1]
    rsi_s = np.clip(1 - 2 * abs(rsi - 50) / 50, -1, 1)
    macd_std = k["MACD_hist"].dropna().tail(100).std(ddof=0) if "MACD_hist" in k.columns else 0.0
    macd_s = np.tanh(macd_h / macd_std) if macd_std and macd_std > 0 else 0.0
    sma_ct = np.tanh(((close - sma50) / sma50) * 5) if (not np.isnan(close) and not np.isnan(sma50) and sma50) else 0.0
    sma_lt = np.tanh(((sma50 - sma200) / sma200) * 3) if (not np.isnan(sma50) and not np.isnan(sma200) and sma200) else 0.0
    hh52_s = -np.tanh((1 - (1 + pct_hh52)) * 2)
    vol_s = np.tanh(volz20 / 3)

    weights = {"rsi": 0.15, "macd": 0.25, "sma_ct": 0.20, "sma_lt": 0.20, "hh52": 0.10, "vol": 0.10}
    parts = {"rsi": rsi_s, "macd": macd_s, "sma_ct": sma_ct, "sma_lt": sma_lt, "hh52": hh52_s, "vol": vol_s}

    valid = {k_: v for k_, v in parts.items() if not np.isnan(v)}
    if valid:
        wsum = sum(weights[k_] for k_ in valid.keys())
        score_norm = sum(parts[k_] * (weights[k_] / wsum) for k_ in valid.keys())
    else:
        score_norm = 0.0

    score_norm += bonus50
    score = float(np.clip(score_norm * 5.0, -5.0, 5.0))

    if score >= 3.0:
        action = "BUY"
    elif score >= 1.5:
        action = "WATCH"
    elif score <= -3.0:
        action = "SELL"
    elif score <= -1.5:
        action = "REDUCE"
    else:
        action = "HOLD"

    return score, action


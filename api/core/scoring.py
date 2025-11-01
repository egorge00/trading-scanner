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

    # --- 2) Coalescer les colonnes OHLCV en Series numériques, colonnes uniques ---
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in x.columns]
    x = x[cols].copy().sort_index()

    def _best_series_from_dataframe(df_like: pd.DataFrame, idx, name: str) -> pd.Series:
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
        import numpy as np

        if isinstance(obj, pd.DataFrame):
            return _best_series_from_dataframe(obj, idx, name)
        if isinstance(obj, pd.Series):
            s = pd.to_numeric(obj, errors="coerce")
            return s.reindex(idx).astype(float)
        arr = np.asarray(obj)
        if arr.ndim == 2:
            if arr.shape[0] != len(idx) and arr.shape[1] == len(idx):
                arr = arr.T
            if arr.shape[0] != len(idx):
                return pd.Series(index=idx, dtype=float, name=name)
            best_s, best_cnt = None, -1
            for j in range(arr.shape[1]):
                s = pd.Series(arr[:, j], index=idx, name=name)
                s = pd.to_numeric(s, errors="coerce")
                cnt = int(s.notna().sum())
                if cnt > best_cnt:
                    best_cnt, best_s = cnt, s
            return (best_s if best_s is not None else pd.Series(index=idx, dtype=float, name=name)).astype(float)
        try:
            s = pd.Series(arr, index=idx, name=name)
        except Exception:
            s = pd.Series(index=idx, dtype=float, name=name)
        s = pd.to_numeric(s, errors="coerce")
        return s.astype(float)

    x_fix = pd.DataFrame(index=x.index)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in x.columns:
            x_fix[c] = _force_series_any(x[c], c, x.index)

    if "Close" not in x_fix.columns or x_fix["Close"].dropna().empty:
        return pd.DataFrame()

    close = x_fix["Close"].astype(float)
    vol = x_fix["Volume"].astype(float) if "Volume" in x_fix.columns else pd.Series(index=close.index, dtype=float)

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
    if "Volume" in x_fix.columns and not vol.dropna().empty:
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


# --- INVESTOR PROFILE (weekly KPIs & LT score) ---


def _ensure_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Aplati yfinance, garde OHLCV numériques, index DateTime trié et tz-naive."""

    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()

    # Aplatir MultiIndex éventuel
    if isinstance(x.columns, pd.MultiIndex):
        lvl0 = x.columns.get_level_values(0)
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in set(lvl0)]
        if keep:
            x = x[keep]
            if isinstance(x.columns, pd.MultiIndex):
                x.columns = [c[0] if isinstance(c, tuple) else c for c in x.columns]
        else:
            x = x.droplevel(-1, axis=1)

    # Garde OHLCV si présents
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in x.columns]
    x = x[cols].copy()

    # Index -> DateTimeIndex trié, tz-naive
    if not isinstance(x.index, pd.DatetimeIndex):
        x.index = pd.to_datetime(x.index, errors="coerce")
    x = x[~x.index.isna()].sort_index()
    try:
        if x.index.tz is not None:
            x.index = x.index.tz_localize(None)
    except Exception:
        pass

    # Cast numérique
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce").astype(float)

    return x


def _resample_weekly_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily OHLCV to weekly (Fri close) de façon ultra-robuste.
    - Open: last
    - High: max
    - Low:  min
    - Close: last
    - Volume: sum
    Retourne un DataFrame plat OHLCV hebdo, ou vide si rien d'exploitable.
    """

    x = _ensure_ohlcv_df(df)
    if x.empty:
        return pd.DataFrame()

    # Pour chaque colonne dispo, on résample séparément pour éviter tout scalaire
    def _res_last(name):
        return x[name].resample("W-FRI").last() if name in x.columns else pd.Series(dtype=float)

    def _res_max(name):
        return x[name].resample("W-FRI").max() if name in x.columns else pd.Series(dtype=float)

    def _res_min(name):
        return x[name].resample("W-FRI").min() if name in x.columns else pd.Series(dtype=float)

    def _res_sum(name):
        return x[name].resample("W-FRI").sum() if name in x.columns else pd.Series(dtype=float)

    open_w = _res_last("Open")
    high_w = _res_max("High")
    low_w = _res_min("Low")
    close_w = _res_last("Close")
    vol_w = _res_sum("Volume")

    # Concat sûre (Series → jamais scalaire), colonnes nommées
    parts = []
    if not open_w.empty:
        parts.append(open_w.rename("Open"))
    if not high_w.empty:
        parts.append(high_w.rename("High"))
    if not low_w.empty:
        parts.append(low_w.rename("Low"))
    if not close_w.empty:
        parts.append(close_w.rename("Close"))
    if not vol_w.empty:
        parts.append(vol_w.rename("Volume"))

    if not parts:
        return pd.DataFrame()

    w = pd.concat(parts, axis=1)

    # Nettoyage final
    if "Close" not in w.columns or w["Close"].dropna().empty:
        return pd.DataFrame()
    w = w.dropna(subset=["Close"]).sort_index()

    return w


def compute_kpis_investor(df: pd.DataFrame) -> pd.DataFrame:
    """
    KPIs long terme (hebdo):
    - SMA26w (~6 mois), SMA52w (~1 an)
    - % distance au plus haut 52w
    - Momentum 12-1 (ret 52w - ret 4w)
    - Volatilité réalisée 20w (annualisée)
    - Drawdown 26w (close vs rolling max)
    """

    w = _resample_weekly_ohlcv(df)
    if w.empty:
        return pd.DataFrame()

    close = w["Close"].astype(float)
    vol = w["Volume"].astype(float) if "Volume" in w.columns else pd.Series(index=close.index, dtype=float)

    sma26 = close.rolling(26, min_periods=8).mean()
    sma52 = close.rolling(52, min_periods=13).mean()

    hh52 = close.rolling(52, min_periods=13).max()
    pct_to_hh52 = close / hh52 - 1.0

    def _ret(series, win):
        s0 = series.shift(win)
        r = (series / s0) - 1.0
        return r

    mom52 = _ret(close, 52)
    mom4 = _ret(close, 4)
    mom_12m_minus_1m = mom52 - mom4

    ret_w = close.pct_change()
    rv20w = ret_w.rolling(20, min_periods=10).std(ddof=0) * np.sqrt(52)

    roll_max26 = close.rolling(26, min_periods=8).max()
    dd26 = close / roll_max26 - 1.0

    out = pd.DataFrame(
        {
            "W_Close": close,
            "SMA26w": sma26,
            "SMA52w": sma52,
            "%toHH52w": pct_to_hh52,
            "MOM_12m_minus_1m": mom_12m_minus_1m,
            "RV20w": rv20w,
            "DD26w": dd26,
        },
        index=close.index,
    )
    return out


def compute_score_investor(df: pd.DataFrame):
    """
    Score LT ∈ [-5, +5] + action (facultatif pour compat UI):
    - Trend (Close>SMA52w + pente SMA52w)
    - Momentum 12-1
    - %toHH52w
    - Drawdown 26w (moindre drawdown = mieux)
    - Volatilité 20w (plus faible = mieux)
    """

    k = compute_kpis_investor(df)
    if k.empty:
        return 0.0, "HOLD"

    def last(col, default=np.nan):
        if col in k.columns:
            s = k[col].dropna()
            if not s.empty:
                return float(s.iloc[-1])
        return default

    c = last("W_Close")
    sma26 = last("SMA26w")
    sma52 = last("SMA52w")
    pctH = last("%toHH52w", -0.2)
    mom = last("MOM_12m_minus_1m", 0.0)
    rv = last("RV20w", np.nan)
    dd = last("DD26w", -0.2)

    above52 = int(not (np.isnan(c) or np.isnan(sma52)) and sma52 and c > sma52)
    try:
        s52_tail = k["SMA52w"].dropna().tail(13)
        slope52 = (s52_tail.iloc[-1] / s52_tail.iloc[0] - 1.0) if len(s52_tail) >= 2 else 0.0
    except Exception:
        slope52 = 0.0

    trend_s = (0.4 if above52 else -0.2) + np.tanh(slope52 * 5.0) * 0.6
    trend_s = np.clip(trend_s, -1.0, 1.0)

    mom_s = np.tanh(mom / 0.6)

    hh_s = -np.tanh((1 - (1 + pctH)) * 2.0)

    dd_s = np.tanh((-dd) * 3.0)

    vol_s = -np.tanh(max(rv - 0.30, 0.0) * 3.0) if not np.isnan(rv) else 0.0

    parts = {
        "trend": trend_s,
        "mom12_1": mom_s,
        "hh52": hh_s,
        "dd26": dd_s,
        "vol20w": vol_s,
    }
    weights = {"trend": 0.35, "mom12_1": 0.25, "hh52": 0.15, "dd26": 0.15, "vol20w": 0.10}

    valid = {k_: v for k_, v in parts.items() if not np.isnan(v)}
    if not valid:
        score_norm = 0.0
    else:
        wsum = sum(weights[k_] for k_ in valid.keys())
        score_norm = sum(parts[k_] * (weights[k_] / wsum) for k_ in valid.keys())

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


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
    """Compute the indicator panel required for scoring.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV dataframe (daily) with columns Open, High, Low, Close, Volume.

    Returns
    -------
    pd.DataFrame
        DataFrame aligned on the input index including, at minimum, the columns
        Close, RSI, MACD_hist, SMA50, SMA200, BBP, VolZ20, pct_to_HH52, ADX,
        and MR (mean-reversion z-score).
    """

    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "Close",
            "RSI",
            "MACD",
            "MACD_signal",
            "MACD_hist",
            "SMA50",
            "SMA200",
            "BBP",
            "VolZ20",
            "pct_to_HH52",
            "ADX",
            "MR",
        ])

    data = df.copy()
    data = data.sort_index()

    close = pd.to_numeric(data.get("Close"), errors="coerce")
    high = pd.to_numeric(data.get("High"), errors="coerce")
    low = pd.to_numeric(data.get("Low"), errors="coerce")
    volume = pd.to_numeric(data.get("Volume"), errors="coerce") if "Volume" in data else pd.Series(index=data.index, dtype="float64")

    rsi = _rsi_ema(close)
    macd_line, macd_signal, macd_hist = _macd(close)
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    bbp = _bollinger_bbp(close)
    volz = _volume_z(volume)
    pct_to_hh52 = _pct_to_hh52(close)
    adx = _adx(high, low, close)
    mr = _mean_reversion_score(close)

    out = pd.DataFrame({
        "Close": close,
        "RSI": rsi,
        "MACD": macd_line,
        "MACD_signal": macd_signal,
        "MACD_hist": macd_hist,
        "SMA50": sma50,
        "SMA200": sma200,
        "BBP": bbp,
        "VolZ20": volz,
        "pct_to_HH52": pct_to_hh52,
        "ADX": adx,
        "MR": mr,
    })

    return out


def _dynamic_weights(rv: float | pd.Series | None) -> Tuple[Dict[str, float], float, float]:
    """Allocate dynamic weights between momentum and mean-reversion blocks."""

    if isinstance(rv, pd.Series):
        rv_value = float(rv.iloc[-1]) if not rv.empty else float("nan")
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


def compute_score(df: pd.DataFrame) -> Tuple[float, str, Dict[str, object]]:
    """Compute the composite score, signal, and diagnostics for a history."""

    kpis = compute_kpis(df)
    if kpis.empty:
        neutral_details = {
            "RSI": np.nan,
            "MACD": np.nan,
            "ADX": np.nan,
            "SMA_CT": np.nan,
            "SMA_LT": np.nan,
            "HH52": np.nan,
            "VOLZ": np.nan,
            "BBP": np.nan,
            "MR": np.nan,
            "weights": {},
            "rv": np.nan,
            "p_up": 0.5,
            "score_norm": 0.0,
            "score_smooth": 0.0,
        }
        return 0.0, "HOLD", neutral_details

    if "Close" in kpis.columns:
        close_series = pd.to_numeric(kpis["Close"], errors="coerce")
    elif df is not None and "Close" in df.columns:
        close_series = pd.to_numeric(df["Close"], errors="coerce")
        kpis["Close"] = close_series
    else:
        close_series = pd.Series(index=kpis.index, dtype="float64")

    if "SMA50" not in kpis.columns:
        kpis["SMA50"] = close_series.rolling(50, min_periods=25).mean()

    if "close_above_sma50" not in kpis.columns:
        kpis["close_above_sma50"] = (close_series > kpis["SMA50"]).astype(int)

    ret = close_series.pct_change()
    rv_series = ret.rolling(20, min_periods=5).std(ddof=0) * np.sqrt(252.0)

    score_history: list[float] = []
    for idx, row in kpis.iterrows():
        rv_val = rv_series.loc[idx] if idx in rv_series.index else np.nan
        weights, _, _ = _dynamic_weights(rv_val)
        subs = _compute_subscores(row)
        score_norm_row, _ = _aggregate_score(subs, weights)
        score_history.append(score_norm_row)

    score_series = pd.Series(score_history, index=kpis.index)
    score_series = score_series.replace([np.inf, -np.inf], np.nan)
    score_series = score_series.dropna()

    if score_series.empty:
        score_norm = 0.0
        score_smooth = 0.0
    else:
        score_norm = float(score_series.iloc[-1])
        if score_series.size >= 3:
            score_smooth = float(score_series.ewm(span=3, adjust=False).mean().iloc[-1])
        else:
            score_smooth = score_norm

    score_final = float(np.clip(score_smooth * 5.0, -5.0, 5.0))
    score_final = round(score_final, 2)

    rv_latest = float(rv_series.iloc[-1]) if len(rv_series) else np.nan
    weights_latest, wmom, wmr = _dynamic_weights(rv_latest)
    subscores_latest = _compute_subscores(kpis.iloc[-1])
    score_norm_latest, normalized_weights = _aggregate_score(subscores_latest, weights_latest)

    if np.isnan(score_norm_latest):
        score_norm_latest = 0.0

    if np.isnan(score_smooth):
        score_smooth = score_norm_latest

    try:
        p_up = float(1.0 / (1.0 + np.exp(-score_smooth)))
    except OverflowError:
        p_up = float(score_smooth > 0.0)

    action: str
    if score_final >= 3.0:
        action = "BUY"
    elif score_final >= 1.5:
        action = "WATCH"
    elif score_final > -1.5:
        action = "HOLD"
    elif score_final > -3.0:
        action = "REDUCE"
    else:
        action = "SELL"

    weights_details = dict(normalized_weights)
    weights_details["wmom"] = wmom
    weights_details["wmr"] = wmr

    close_above_latest = (
        kpis["close_above_sma50"].iloc[-1]
        if "close_above_sma50" in kpis.columns and not kpis.empty
        else np.nan
    )
    if pd.notna(close_above_latest):
        close_above_latest = int(close_above_latest)

    details = {
        "RSI": subscores_latest.get("RSI"),
        "MACD": subscores_latest.get("MACD"),
        "ADX": subscores_latest.get("ADX"),
        "SMA_CT": subscores_latest.get("SMA_CT"),
        "SMA_LT": subscores_latest.get("SMA_LT"),
        "HH52": subscores_latest.get("HH52"),
        "VOLZ": subscores_latest.get("VOLZ"),
        "BBP": subscores_latest.get("BBP"),
        "MR": subscores_latest.get("MR"),
        "close_above_sma50": close_above_latest,
        "weights": weights_details,
        "rv": rv_latest,
        "p_up": p_up,
        "score_norm": score_norm_latest,
        "score_smooth": score_smooth,
    }

    return score_final, action, details


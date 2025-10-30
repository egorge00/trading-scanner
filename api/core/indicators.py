import pandas as pd
import numpy as np

# --- Helpers sûrs sans TA-Lib ---

def rsi_ewm(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=period, adjust=False).mean()
    roll_down = down.ewm(span=period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    ma = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    upper = ma + k * std
    lower = ma - k * std
    width = (upper - lower) / ma
    return upper, ma, lower, width

def pct_to_hhll(close: pd.Series, lookback: int = 252):
    hh = close.rolling(lookback).max()
    ll = close.rolling(lookback).min()
    pct_to_hh = close / hh - 1.0
    pct_from_ll = close / ll - 1.0
    return hh, ll, pct_to_hh, pct_from_ll

def volume_z(vol: pd.Series, n: int = 20) -> pd.Series:
    mu = vol.rolling(n).mean()
    sd = vol.rolling(n).std(ddof=0)
    return (vol - mu) / sd.replace(0, np.nan)

def breakout_flag(high: pd.Series, low: pd.Series, close: pd.Series, lb: int = 20) -> pd.Series:
    maxh = high.rolling(lb).max().shift(1)
    minh = low.rolling(lb).min().shift(1)
    bo_up = (close > maxh).astype(int)
    bo_dn = (close < minh).astype(int)
    return bo_up, bo_dn

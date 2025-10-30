from dataclasses import dataclass
import pandas as pd
from .indicators import rsi_ewm, macd, sma, atr, bollinger, pct_to_hhll, volume_z, breakout_flag

@dataclass
class KPIResult:
    rsi: float
    macd_hist: float
    macd_pos: int
    close_above_sma50: int
    sma50_above_sma200: int
    pct_to_hh52: float
    pct_from_ll52: float
    bbw: float
    vol_z20: float
    atr_pct: float
    breakout20: int

@dataclass
class ScoreResult:
    score: float
    action: str  # "BUY"/"WATCH"/"HOLD"/"REDUCE"/"SELL"

def compute_kpis(df: pd.DataFrame) -> KPIResult:
    # df columns expected: Open, High, Low, Close, Volume
    rsi = rsi_ewm(df["Close"]).iloc[-1]
    macd_line, signal_line, hist = macd(df["Close"])
    macd_hist = float(hist.iloc[-1])
    macd_pos = int((macd_line.iloc[-1] > signal_line.iloc[-1]))

    sma50 = sma(df["Close"], 50)
    sma200 = sma(df["Close"], 200)
    close_above_sma50 = int(df["Close"].iloc[-1] > sma50.iloc[-1])
    sma50_above_sma200 = int(sma50.iloc[-1] > sma200.iloc[-1])

    hh52, ll52, pct_to_hh52, pct_from_ll52 = pct_to_hhll(df["Close"], 252)
    bbw_up, bb_mid, bbw_low, bbw = bollinger(df["Close"], 20, 2.0)
    volz = volume_z(df["Volume"], 20)
    atr14 = atr(df["High"], df["Low"], df["Close"], 14)
    atr_pct = float((atr14.iloc[-1] / df["Close"].iloc[-1]) if df["Close"].iloc[-1] else 0)
    bo_up20, _ = breakout_flag(df["High"], df["Low"], df["Close"], 20)

    return KPIResult(
        rsi=float(rsi),
        macd_hist=float(macd_hist),
        macd_pos=int(macd_pos),
        close_above_sma50=int(close_above_sma50),
        sma50_above_sma200=int(sma50_above_sma200),
        pct_to_hh52=float(pct_to_hh52.iloc[-1]),
        pct_from_ll52=float(pct_from_ll52.iloc[-1]),
        bbw=float(bbw.iloc[-1]),
        vol_z20=float(volz.iloc[-1]),
        atr_pct=float(atr_pct),
        breakout20=int(bo_up20.iloc[-1]),
    )

def compute_score(k: KPIResult) -> ScoreResult:
    # Poids simples (MVP) — on affinera ensuite
    s = 0.0
    # Trend
    s += 0.6 * k.close_above_sma50
    s += 0.6 * k.sma50_above_sma200
    # Momentum
    s += 0.5 if k.rsi >= 55 else (-0.5 if k.rsi <= 45 else 0.0)
    s += 0.5 if (k.macd_pos and k.macd_hist > 0) else (-0.5 if (not k.macd_pos and k.macd_hist < 0) else 0.0)
    # Breakout / 52w
    s += 0.7 if k.breakout20 else 0.0
    s += 0.4 if k.pct_to_hh52 > -0.02 else 0.0   # proche HH52
    # Volume
    s += 0.3 if k.vol_z20 > 0.5 else 0.0
    # Volatilité (pénalité si trop élevée)
    if k.atr_pct > 0.05:  # >5% ATR
        s -= 0.4
    # Bornes
    s = max(min(s, 5.0), -5.0)

    if s >= 3.0:
        action = "BUY"       # 🟢
    elif s >= 1.5:
        action = "WATCH"     # 🟢 clair
    elif s > -1.0:
        action = "HOLD"      # ⚪
    elif s > -2.0:
        action = "REDUCE"    # 🟠
    else:
        action = "SELL"      # 🔴

    return ScoreResult(score=round(s, 2), action=action)

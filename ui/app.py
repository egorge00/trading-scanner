import os
import json
import sys
import io
import difflib
import datetime as dt
import base64
import contextlib
import hashlib
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import bcrypt
import numpy as np
import pandas as pd
import streamlit as st
import requests
import traceback
import yfinance as yf


# --- Rerun robuste ---
def safe_rerun():
    import streamlit as st

    # pose un flag si le contexte ne permet pas le rerun immédiat
    st.session_state["_needs_rerun"] = True
    try:
        st.rerun()
    except Exception:
        # certains contextes (threads/callbacks) peuvent empêcher le rerun immédiat
        pass


USER_WL_PATH = Path("data/my_watchlist.csv")

AVAILABLE_MARKETS = ["US", "FR", "UK", "DE", "JP", "ETF"]


def load_user_watchlist() -> list[str]:
    """Charge la watchlist utilisateur depuis data/my_watchlist.csv (si présent)."""

    try:
        if USER_WL_PATH.exists():
            df = pd.read_csv(USER_WL_PATH)
            col = None
            for c in ["ticker", "Ticker", "symbol", "Symbol"]:
                if c in df.columns:
                    col = c
                    break
            if col is None:
                s = pd.read_csv(USER_WL_PATH, header=None).iloc[:, 0]
                return sorted(
                    {str(x).upper().strip() for x in s if str(x).strip()}
                )
            return sorted(
                {
                    str(x).upper().strip()
                    for x in df[col]
                    if str(x).strip()
                }
            )
    except Exception:
        pass
    return []


def save_user_watchlist(tickers: list[str]) -> None:
    """Sauvegarde la watchlist utilisateur dans data/my_watchlist.csv."""

    USER_WL_PATH.parent.mkdir(parents=True, exist_ok=True)
    uniq = sorted({str(x).upper().strip() for x in tickers if str(x).strip()})
    pd.DataFrame({"ticker": uniq}).to_csv(USER_WL_PATH, index=False)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _as_str_date(d):
    try:
        return pd.to_datetime(d).date().isoformat()
    except Exception:
        return None


def _last_notna(series: pd.Series):
    try:
        return series.dropna().iloc[-1]
    except Exception:
        return None


def _compute_daily_kpis_for_audit(df: pd.DataFrame) -> dict:
    """
    Minimal: recalcul local des indicateurs techniques utilisés par le score (mêmes définitions que compute_kpis).
    Entrée: df = OHLCV daily (yfinance) avec colonnes ['Open','High','Low','Close','Adj Close','Volume'].
    Retour: dict de valeurs scalaires (dernière observation).
    """

    out = {}

    if df is None or df.empty or "Close" not in df.columns:
        return out

    # RSI(14) (EWMA pour robustesse)
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=14, adjust=False).mean()
    roll_down = down.ewm(span=14, adjust=False).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    out["RSI14"] = _safe_float(_last_notna(rsi))

    # MACD (12,26,9) histogramme
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    out["MACD_hist"] = _safe_float(_last_notna(hist))

    # SMA50 / SMA200 + relation
    sma50 = df["Close"].rolling(50).mean()
    sma200 = df["Close"].rolling(200).mean()
    out["SMA50"] = _safe_float(_last_notna(sma50))
    out["SMA200"] = _safe_float(_last_notna(sma200))
    if out["SMA50"] is not None and out["SMA200"] is not None:
        out["SMA50_gt_SMA200"] = bool(out["SMA50"] > out["SMA200"])
    else:
        out["SMA50_gt_SMA200"] = None

    # Plus haut 52 semaines
    # 252 séances ≈ 52 semaines boursières
    hh52 = df["Close"].rolling(252).max()
    out["High_52w"] = _safe_float(_last_notna(hh52))
    if out["High_52w"] and "Close" in df.columns:
        last_close = _safe_float(_last_notna(df["Close"]))
        out["pct_to_High52w"] = _safe_float((last_close / out["High_52w"] - 1.0) if (last_close and out["High_52w"]) else None)
    else:
        out["pct_to_High52w"] = None

    # Z-Score Volume(20)
    if "Volume" in df.columns:
        vol20 = df["Volume"].rolling(20)
        mu = vol20.mean()
        sd = vol20.std(ddof=0)
        z = (df["Volume"] - mu) / sd
        out["Vol_Z20"] = _safe_float(_last_notna(z))
        out["Volume_last"] = _safe_float(_last_notna(df["Volume"]))
    else:
        out["Vol_Z20"] = None
        out["Volume_last"] = None

    # Clôture courante
    out["Close_last"] = _safe_float(_last_notna(df["Close"]))

    # Date de la dernière barre
    try:
        out["Last_bar_date"] = _as_str_date(df.index[-1])
    except Exception:
        out["Last_bar_date"] = None

    return out

# ============================
# 📦 Chargement pré-calculé par profil (Investisseur / Swing)
# ============================
PRECOMP_FILES = {
    "Investisseur": (
        "data/daily_scan_investisseur.parquet",
        "data/daily_scan_investisseur.json",
    ),
    "Swing": (
        "data/daily_scan_swing.parquet",
        "data/daily_scan_swing.json",
    ),
}


def load_precomputed_for_profile(profile: str):
    """Charge le parquet/json du profil demandé. Ne télécharge rien."""

    pq_path, js_path = PRECOMP_FILES.get(profile, (None, None))
    if not pq_path:
        return pd.DataFrame(), None, {"error": f"profile inconnu: {profile}"}

    pq, js = Path(pq_path), Path(js_path)
    if not pq.exists():
        return pd.DataFrame(), None, {"missing": str(pq)}

    df = pd.read_parquet(pq)

    # Arrondis légers pour l’UI
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    ts = None
    if js.exists():
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
            ts = meta.get("generated_at_utc")
        except Exception:
            ts = None

    return df, ts, {"ok": True, "file": str(pq)}


def cache_full_scan_in_session(profile: str, df: pd.DataFrame, ts: str):
    key = f"{_today_paris_str()}|{profile}"
    st.session_state.setdefault("daily_full_scan", {})
    st.session_state["daily_full_scan"][key] = {"df": df, "ts": ts}

# --- Colonnes standard (identiques pour scan & watchlist) ---
def main_table_columns(profile: str, df_cols: list[str]) -> list[str]:
    base = [
        "Ticker",
        "Name",
        "Market",
        "Signal",
        "ScoreFinal",
        "ScoreTech",
        "FscoreFund",
        "%toHH52",
        "Earnings",
    ]
    if profile != "Investisseur" and "RSI" in df_cols:
        base.insert(base.index("%toHH52"), "RSI")
    return [c for c in base if c in df_cols]

# --- Helpers d'arrondis numériques ---
def _round_numeric_cols(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].round(n)
    return df

# Styles optionnels (réutilisés partout)
def _style_earnings(series):
    out = []
    import re

    for v in series:
        s = str(v)
        m = re.search(r"J-(\d+)|J0", s)
        if m:
            out.append("background-color:#FFF4CC;")
            continue
        out.append("")
    return out


def _style_scorefinal(series):
    out = []
    for v in series:
        try:
            x = float(v)
        except Exception:
            out.append("")
            continue
        if x >= 80:
            out.append("background-color:#E8FAE6;")
        elif x < 30:
            out.append("background-color:#FDE8E8;")
        else:
            out.append("")
    return out


def render_results_table(
    df: pd.DataFrame,
    profile: str,
    title: str,
    hide_before_n: int = 0,
    allow_delete: bool = False,
):
    """Affiche le même tableau pour Scan & Watchlist."""

    if df is None or df.empty:
        st.info("Aucune donnée à afficher.")
        return

    view = df.copy()
    if "EarningsD" in view.columns and hide_before_n and hide_before_n > 0:
        view = view[(view["EarningsD"].isna()) | (view["EarningsD"] >= hide_before_n)]

    if "%toHH52" in view.columns:
        view["%toHH52"] = pd.to_numeric(view["%toHH52"], errors="coerce")

    view = _round_numeric_cols(view, 2)

    if title:
        st.subheader(title)

    cols = main_table_columns(profile, view.columns.tolist())

    if "ScoreFinal" in view.columns:
        view = view.sort_values(by=["ScoreFinal", "Ticker"], ascending=[False, True]).reset_index(drop=True)

    view_display = view.copy()

    def _fmt_blank(x):
        import pandas as pd

        if x is None:
            return "—"
        try:
            if pd.isna(x):
                return "—"
        except Exception:
            pass
        return x

    for c in ["FscoreFund", "ScoreFinal", "ScoreTech"]:
        if c in view_display.columns:
            view_display[c] = view_display[c].apply(_fmt_blank)

    hide_no_fund = st.checkbox(
        "Masquer les valeurs sans fondamentaux disponibles",
        value=False,
        help="N’affiche pas les lignes où FscoreFund est indisponible (None).",
        key=f"hide_no_fund_{hashlib.md5((title or '').encode('utf-8')).hexdigest()}",
    )
    if hide_no_fund and "FscoreFund" in view_display.columns:
        mask = view_display["FscoreFund"] != "—"
        view_display = view_display[mask].copy()
        view = view[mask].copy()

    styled = view_display[cols].style if len(cols) else view_display.style
    if "Earnings" in cols:
        styled = styled.apply(_style_earnings, subset=["Earnings"])
    if "ScoreFinal" in cols:
        styled = styled.apply(_style_scorefinal, subset=["ScoreFinal"])

    st.dataframe(styled, use_container_width=True, height=520)

    export_key = f"export_{hashlib.md5((title or '').encode('utf-8')).hexdigest()}"
    extras = [c for c in ["Ffund_valid"] if c in view.columns]
    export_cols = cols + [c for c in extras if c not in cols]
    st.download_button(
        "💾 Export CSV (vue affichée)",
        data=view[export_cols].to_csv(index=False).encode("utf-8"),
        file_name="scanner_view.csv",
        mime="text/csv",
        key=export_key,
    )

    if allow_delete and "Ticker" in view.columns:
        st.caption("🗑️ Retirer de la watchlist :")
        cols_rm = st.columns(min(5, max(1, len(view))))
        for i, (_, row) in enumerate(view.iterrows()):
            if i < len(cols_rm):
                with cols_rm[i]:
                    if st.button(f"🗑️ {row['Ticker']}", key=f"rmwl_{row['Ticker']}"):
                        wl = [
                            t
                            for t in st.session_state.get("my_watchlist", [])
                            if t != row["Ticker"]
                        ]
                        st.session_state["my_watchlist"] = wl
                        save_user_watchlist(wl)
                        safe_rerun()


def _daily_cache_key(profile: str) -> str:
    return f"{_today_paris_str()}|{profile}"


def _get_full_scan_df_for_today(profile: str) -> pd.DataFrame:
    key = _daily_cache_key(profile)
    meta = st.session_state.get("daily_full_scan", {}).get(key)
    if meta and isinstance(meta.get("df"), pd.DataFrame):
        return meta["df"]
    return pd.DataFrame()


@contextlib.contextmanager
def card(title: str = "", subtitle: str = ""):
    """Contexte visuel de 'carte' pour structurer les blocs."""

    box = st.container()
    with box:
        if title:
            st.markdown(f"### {title}")
        if subtitle:
            st.caption(subtitle)
        yield box


def signal_from_scorefinal(sf: float | None) -> str:
    if sf is None:
        return "⚪ HOLD"
    s = float(sf)
    if s >= 80:
        return "🟢 BUY"
    if s >= 65:
        return "🟡 WATCH"
    if s >= 45:
        return "⚪ HOLD"
    if s >= 30:
        return "🔵 REDUCE"
    return "🔴 SELL"


def _today_paris_str() -> str:
    return datetime.now(tz=ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d")


def _now_paris_str() -> str:
    return datetime.now(tz=ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d %H:%M:%S %Z")


def _now_paris_iso():
    return datetime.now(tz=ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d %H:%M:%S %Z")


def compute_full_scan_df(profile: str, max_workers: int = 8) -> pd.DataFrame:
    """Calcule le scan complet (sans Streamlit UI)."""

    uni = get_universe_normalized()
    tickers = (
        uni["ticker"].astype(str).str.upper().dropna().unique().tolist()
    )
    if not tickers:
        return pd.DataFrame()

    rows, failures = [], []
    max_workers = min(max_workers, max(2, (os.cpu_count() or 4) * 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(score_ticker_cached, t, profile): t for t in tickers}
        for fut in as_completed(futs):
            res = fut.result()
            if isinstance(res, dict) and res.get("error"):
                failures.append(res)
            else:
                rows.append(res)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    if "%toHH52" in out.columns:
        out["%toHH52"] = pd.to_numeric(out["%toHH52"], errors="coerce") * 100
    if "Market" in out.columns:
        out["Market"] = out["Market"].astype(str).str.strip().str.upper()

    out = _round_numeric_cols(out, 2)
    cols = [
        "Ticker",
        "Name",
        "Market",
        "Signal",
        "ScoreFinal",
        "ScoreTech",
        "FscoreFund",
        "RSI",
        "%toHH52",
        "Earnings",
        "EarningsDate",
        "EarningsD",
    ]
    out = out[[c for c in cols if c in out.columns]]
    if "ScoreFinal" in out.columns:
        out = out.sort_values(
            by=["ScoreFinal", "Ticker"], ascending=[False, True]
        ).reset_index(drop=True)
    return out


def run_full_scan_all_and_cache_ui(profile: str, max_workers: int = 8) -> pd.DataFrame:
    """Exécute le scan complet avec barre de progression et renvoie le DataFrame."""

    uni = get_universe_normalized()
    tickers = (
        uni["ticker"].astype(str).str.upper().dropna().unique().tolist()
    )
    if not tickers:
        st.warning("Univers vide : aucun ticker à scanner.")
        return pd.DataFrame()

    rows, failures = [], []
    progress = st.progress(0, text=f"Scan en cours… (0/{len(tickers)})")
    done = 0
    max_workers = min(max_workers, max(2, (os.cpu_count() or 4) * 2))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(score_ticker_cached, t, profile): t for t in tickers}
        total = len(futs)
        for fut in as_completed(futs):
            res = fut.result()
            done += 1
            progress.progress(done / total, text=f"Scan en cours… ({done}/{total})")
            if isinstance(res, dict) and res.get("error"):
                failures.append(res)
            else:
                rows.append(res)

    progress.empty()
    st.success(f"✅ Scan terminé ({done}/{len(tickers)})")

    if failures:
        with st.expander("Diagnostics (échecs)"):
            st.dataframe(pd.DataFrame(failures), use_container_width=True, height=240)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    if "%toHH52" in out.columns:
        out["%toHH52"] = pd.to_numeric(out["%toHH52"], errors="coerce") * 100
    if "Market" in out.columns:
        out["Market"] = out["Market"].astype(str).str.strip().str.upper()
    out = _round_numeric_cols(out, 2)

    cols = [
        "Ticker",
        "Name",
        "Market",
        "Signal",
        "ScoreFinal",
        "ScoreTech",
        "FscoreFund",
        "RSI",
        "%toHH52",
        "Earnings",
        "EarningsDate",
        "EarningsD",
    ]
    out = out[[c for c in cols if c in out.columns]]
    if "ScoreFinal" in out.columns:
        out = out.sort_values(
            by=["ScoreFinal", "Ticker"], ascending=[False, True]
        ).reset_index(drop=True)

    return out


@st.cache_data(ttl=86400, show_spinner=False)
def get_next_earnings(tkr: str):
    """Retourne (date_iso, days_to) pour la prochaine date d’earnings; ou dernière passée si aucune future.
       None, None si introuvable."""
    try:
        t = yf.Ticker(tkr)
        # API récente
        try:
            ed = t.get_earnings_dates(limit=12)
            if isinstance(ed, pd.DataFrame) and not ed.empty:
                now = pd.Timestamp(datetime.now(tz=ZoneInfo("Europe/Paris")).date())
                ed = ed.copy()
                ed.index = pd.to_datetime(ed.index).tz_localize(None)
                future = ed.index[ed.index >= now]
                d = future[0] if len(future) > 0 else ed.index.max()
                dt_local = d.to_pydatetime().replace(tzinfo=ZoneInfo("Europe/Paris"))
                days_to = (dt_local.date() - datetime.now(tz=ZoneInfo("Europe/Paris")).date()).days
                return dt_local.strftime("%Y-%m-%d"), int(days_to)
        except Exception:
            pass
        # Fallback: calendar
        try:
            cal = t.calendar
            if cal is not None and not cal.empty:
                for v in cal.iloc[0].values:
                    try:
                        candidate = pd.to_datetime(v).tz_localize(None)
                        dt_local = candidate.to_pydatetime().replace(tzinfo=ZoneInfo("Europe/Paris"))
                        days_to = (dt_local.date() - datetime.now(tz=ZoneInfo("Europe/Paris")).date()).days
                        return dt_local.strftime("%Y-%m-%d"), int(days_to)
                    except Exception:
                        continue
        except Exception:
            pass
    except Exception:
        pass
    return None, None


def fmt_earnings(date_iso: str | None, days_to: int | None) -> str:
    if not date_iso or days_to is None:
        return "—"
    if days_to < 0:
        return f"{date_iso} (J{days_to})"
    if days_to == 0:
        return f"{date_iso} (J0) 📣"
    if days_to <= 7:
        return f"{date_iso} (J-{days_to}) 📣"
    return f"{date_iso} (J-{days_to})"


# --- rendre importable le package "api" depuis /ui ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

DEBUG = bool(os.getenv("DEBUG", "0") == "1")
if DEBUG:
    from importlib import reload
    import api.core.scoring as scoring  # noqa: E402

    reload(scoring)

from api.core.scoring import (  # noqa: E402
    compute_kpis,
    compute_kpis_investor,
    compute_score,
    compute_score_investor,
)
# --- fondamentaux (safe) ---
try:  # noqa: E402
    from api.core.fundamentals import (  # type: ignore[attr-defined]
        compute_fscore_continuous_safe,
        get_fundamentals,
    )
except Exception as e:  # pragma: no cover - import fallback for Streamlit runtime
    import streamlit as st

    st.error(f"Import fondamentaux impossible: {e}")

    def get_fundamentals(_):
        return {}

    def compute_fscore_continuous_safe(_):
        return (None, {"reason": "import_failed"})

# --- Normalisation marchés ---
def _norm_market(m: str) -> str:
    if m is None:
        return ""
    s = str(m).strip().lower()
    MAP = {
        "us": "US",
        "usa": "US",
        "united states": "US",
        "uk": "UK",
        "gb": "UK",
        "great britain": "UK",
        "united kingdom": "UK",
        "fr": "FR",
        "france": "FR",
        "de": "DE",
        "germany": "DE",
        "nl": "NL",
        "netherlands": "NL",
        "ch": "CH",
        "switzerland": "CH",
        "it": "IT",
        "italy": "IT",
        "es": "ES",
        "spain": "ES",
        "ie": "IE",
        "ireland": "IE",
        "be": "BE",
        "belgium": "BE",
        "se": "SE",
        "sweden": "SE",
        "dk": "DK",
        "denmark": "DK",
        "no": "NO",
        "norway": "NO",
        "fi": "FI",
        "finland": "FI",
        "pt": "PT",
        "portugal": "PT",
        "etf": "ETF",
        "exchange traded fund": "ETF",
        "fund": "ETF",
    }
    return MAP.get(s, s.upper())


@st.cache_data(ttl=3600)
def get_universe_normalized():
    df = load_universe_df().copy()
    if "market" not in df.columns:
        df["market"] = ""
    df["market_norm"] = df["market"].apply(_norm_market)
    return df


DEBUG_ENV_DEFAULT = os.getenv("APP_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
DEBUG_TOGGLE_KEY = "debug_mode_checkbox"
IMPORT_ONLY = os.getenv("SCANNER_IMPORT_ONLY", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEBUG_MODE = DEBUG_ENV_DEFAULT

PROFILE_KEY = "analysis_profile"


def get_analysis_profile() -> str:
    return st.session_state.get(PROFILE_KEY, "Investisseur")


def get_score_label() -> str:
    return "Score (LT)" if get_analysis_profile() == "Investisseur" else "Score"


def rename_score_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return df
    score_label = get_score_label()
    if score_label != "Score" and score_label not in df.columns and "Score" in df.columns:
        return df.rename(columns={"Score": score_label})
    return df


def map_score_column(columns: list[str]) -> list[str]:
    score_label = get_score_label()
    return [score_label if c == "Score" else c for c in columns]


if not IMPORT_ONLY:
    # ---------- CONFIG ----------
    st.set_page_config(page_title="Trading Scanner", layout="wide")

    if st.session_state.get("_needs_rerun"):
        st.session_state["_needs_rerun"] = False
        st.rerun()

    # ---------- AUTH (simple & robuste) ----------
    USERNAME = "egorge"
    PASSWORD_HASH = os.getenv(
        "PASSWORD_HASH",
        "$2y$12$4LAav5U4KJwaT2YgzYTnf.qaTGo6VjxdkB6oueE//XreoI0D21RKe",
    )

    def login_form():
        st.title("Login")
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            remember = st.checkbox("Se souvenir de moi", value=True)
            submit = st.form_submit_button("Login")
        return submit, u, p, remember

    def check_password(raw_password: str, bcrypted: str) -> bool:
        try:
            return bcrypt.checkpw(raw_password.encode(), bcrypted.encode())
        except Exception:
            return False

    if "auth" not in st.session_state:
        st.session_state.auth = False

    if not st.session_state.auth:
        submitted, u, p, remember = login_form()
        if submitted:
            if u == USERNAME and check_password(p, PASSWORD_HASH):
                st.session_state.auth = True
                if remember:
                    st.session_state.remember_until = str(dt.datetime.utcnow() + dt.timedelta(days=30))
                st.rerun()
            else:
                st.error("Identifiants invalides")
        st.stop()

    # ---------- UI une fois connecté ----------
    st.sidebar.success(f"Connecté comme {USERNAME}")
    if st.sidebar.button("Se déconnecter"):
        st.session_state.clear()
        st.rerun()

    if PROFILE_KEY not in st.session_state:
        st.session_state[PROFILE_KEY] = "Investisseur"

    st.sidebar.selectbox(
        "Profil d’analyse",
        options=["Investisseur", "Swing"],
        index=0,
        key=PROFILE_KEY,
    )

    debug_default = bool(st.session_state.get(DEBUG_TOGGLE_KEY, DEBUG_ENV_DEFAULT))
    DEBUG_MODE = st.sidebar.checkbox(
        "Mode debug",
        value=debug_default,
        key=DEBUG_TOGGLE_KEY,
        help="Affiche davantage de diagnostics en cas d'erreur (colonnes, shapes).",
    )

# ========= FICHIERS =========
UNIVERSE_PATH = "data/watchlist.csv"
MY_WATCHLIST_KEY = "my_watchlist_df"

# ========= Helpers CSV =========
def normalize_cols(df: pd.DataFrame, expected=("isin","ticker","name","market")) -> pd.DataFrame:
    lower = [c.lower() for c in df.columns]
    mapping = {lc: c for lc, c in zip(lower, df.columns)}
    out = df.rename(columns={mapping.get("isin","isin"):"isin",
                             mapping.get("ticker","ticker"):"ticker",
                             mapping.get("name","name"):"name",
                             mapping.get("market","market"):"market"}).copy()
    for c in expected:
        if c not in out.columns:
            out[c] = ""
    out["isin"]   = out["isin"].astype(str).str.strip().str.upper()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["name"]   = out["name"].astype(str).str.strip()
    out["market"] = out["market"].astype(str).str.strip()
    return out[list(expected)]

@st.cache_data(show_spinner=False, ttl=3600)
def load_universe() -> pd.DataFrame:
    try:
        df = pd.read_csv(UNIVERSE_PATH)
        return normalize_cols(df)
    except Exception:
        return normalize_cols(pd.DataFrame([
            {"isin":"US0378331005","ticker":"AAPL","name":"Apple","market":"US"},
            {"isin":"FR0000120321","ticker":"OR.PA","name":"L'Oreal","market":"FR"},
        ]))

def load_my_watchlist() -> pd.DataFrame:
    if MY_WATCHLIST_KEY in st.session_state:
        return st.session_state[MY_WATCHLIST_KEY].copy()
    tickers = load_user_watchlist()
    if tickers:
        df = normalize_cols(pd.DataFrame({"ticker": tickers}))
    else:
        df = normalize_cols(pd.DataFrame(columns=["isin","ticker","name","market"]))
    st.session_state[MY_WATCHLIST_KEY] = df.copy()
    return df

def save_my_watchlist(df: pd.DataFrame):
    df_norm = normalize_cols(df)
    st.session_state[MY_WATCHLIST_KEY] = df_norm.copy()
    if "ticker" in df_norm.columns:
        save_user_watchlist(df_norm["ticker"].astype(str).tolist())
    else:
        save_user_watchlist([])

def export_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

# ========= GitHub persistence (my watchlist) =========
GITHUB_API = "https://api.github.com"

def _gh_headers():
    token = st.secrets.get("GITHUB_TOKEN", None)
    if not token:
        raise RuntimeError("Secret GITHUB_TOKEN manquant dans Streamlit.")
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }

def _gh_repo_branch_path():
    repo = st.secrets.get("GITHUB_REPO", "").strip() or "egorge00/trading-scanner"
    branch = st.secrets.get("GITHUB_BRANCH", "").strip() or "main"
    path = "data/my_watchlist.csv"
    return repo, branch, path

def gh_get_file(repo: str, path: str, ref: str = "main"):
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
    r = requests.get(url, headers=_gh_headers(), params={"ref": ref}, timeout=15)
    if r.status_code == 200:
        return r.json()
    if r.status_code == 404:
        return None
    raise RuntimeError(f"GitHub GET {path} a échoué: {r.status_code} {r.text}")

def gh_put_file(repo: str, path: str, message: str, content_text: str, sha: str | None, branch: str = "main"):
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
    b64 = base64.b64encode(content_text.encode()).decode()
    payload = {"message": message, "content": b64, "branch": branch}
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=_gh_headers(), json=payload, timeout=20)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"GitHub PUT {path} a échoué: {r.status_code} {r.text}")
    return r.json()

def load_my_watchlist_from_github() -> pd.DataFrame | None:
    try:
        repo, branch, path = _gh_repo_branch_path()
        meta = gh_get_file(repo, path, ref=branch)
        if not meta:
            return None
        content_b64 = meta.get("content", "")
        csv_text = base64.b64decode(content_b64).decode()
        df = pd.read_csv(io.StringIO(csv_text))
        return normalize_cols(df)
    except Exception as e:
        st.warning(f"Import GitHub impossible: {e}")
        return None

def save_my_watchlist_to_github(df: pd.DataFrame) -> bool:
    try:
        repo, branch, path = _gh_repo_branch_path()
        df_norm = normalize_cols(df)
        csv_text = df_norm.to_csv(index=False)
        sha = None
        existing = gh_get_file(repo, path, ref=branch)
        if existing:
            sha = existing.get("sha")
        gh_put_file(repo, path, "chore(watchlist): update my_watchlist.csv via app", csv_text, sha, branch)
        return True
    except Exception as e:
        st.warning(f"Sauvegarde GitHub impossible: {e}")
        return False

@st.cache_data(show_spinner=False, ttl=3600)
def get_name_map_from_universe() -> dict:
    uni = load_universe()
    mp = {}
    for _, r in uni.iterrows():
        t = str(r.get("ticker","")).strip().upper()
        n = str(r.get("name","")).strip()
        if t and n:
            mp[t] = n
    return mp

NAME_MAP = get_name_map_from_universe()
def get_name_for_ticker(tkr: str) -> str:
    return NAME_MAP.get(str(tkr).strip().upper(), "")

# ========= Données marché + scoring =========

@st.cache_data(ttl=900, show_spinner=False)
def safe_yf_download(tkr: str, period="9mo", interval="1d") -> pd.DataFrame:
    """
    Télécharge des quotes pour tkr de manière robuste.
    1) download(..., threads=False) sans group_by pour éviter MultiIndex/cache bugs
    2) fallback Ticker(tkr).history(...)
    3) fallback avec auto_adjust=True
    Retourne un DF plat avec colonnes OHLCV si dispo, sinon DF vide.
    """
    # Tentative 1 : download threads=False
    try:
        df = yf.download(
            tickers=tkr,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                lvl0 = df.columns.get_level_values(0)
                keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in set(lvl0)]
                if keep:
                    df = df[keep]
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            if cols:
                return df[cols].copy()
    except Exception:
        pass

    # Tentative 2 : Ticker().history auto_adjust=False
    try:
        h = yf.Ticker(tkr).history(period=period, interval=interval, auto_adjust=False)
        if isinstance(h, pd.DataFrame) and not h.empty:
            cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in h.columns]
            if cols:
                return h[cols].copy()
    except Exception:
        pass

    # Tentative 3 : Ticker().history auto_adjust=True (au cas où)
    try:
        h = yf.Ticker(tkr).history(period=period, interval=interval, auto_adjust=True)
        if isinstance(h, pd.DataFrame) and not h.empty:
            cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in h.columns]
            if cols:
                return h[cols].copy()
    except Exception:
        pass

    return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def score_ticker_cached(tkr: str, profile: str) -> dict:
    period = "36mo" if profile == "Investisseur" else "9mo"
    df = safe_yf_download(tkr, period=period, interval="1d")
    if df is None or df.empty or "Close" not in df.columns:
        return {"Ticker": tkr, "error": "no_data"}
    df = df.dropna(subset=["Close"]).copy()
    if df.empty:
        return {"Ticker": tkr, "error": "no_close"}

    if profile == "Investisseur":
        score, action = compute_score_investor(df)
        kpi = compute_kpis_investor(df)
        rsi_val = macd_h = volz20 = None
        pct_hh = (
            float(kpi["%toHH52w"].dropna().iloc[-1])
            if isinstance(kpi, pd.DataFrame) and "%toHH52w" in kpi.columns and not kpi["%toHH52w"].dropna().empty
            else None
        )
    else:
        score, action = compute_score(df)
        kpi = compute_kpis(df)
        rsi_val = (
            float(kpi["RSI"].dropna().iloc[-1])
            if isinstance(kpi, pd.DataFrame) and "RSI" in kpi.columns and not kpi["RSI"].dropna().empty
            else None
        )
        macd_h = (
            float(kpi["MACD_hist"].dropna().iloc[-1])
            if isinstance(kpi, pd.DataFrame) and "MACD_hist" in kpi.columns and not kpi["MACD_hist"].dropna().empty
            else None
        )
        pct_hh = (
            float(kpi["%toHH52"].dropna().iloc[-1])
            if isinstance(kpi, pd.DataFrame) and "%toHH52" in kpi.columns and not kpi["%toHH52"].dropna().empty
            else None
        )
        volz20 = (
            float(kpi["VolZ20"].dropna().iloc[-1])
            if isinstance(kpi, pd.DataFrame) and "VolZ20" in kpi.columns and not kpi["VolZ20"].dropna().empty
            else None
        )

    uni = get_universe_normalized()
    name = market = ""
    try:
        sel = uni.loc[
            uni["ticker"].astype(str).str.upper() == tkr, ["name", "market_norm"]
        ]
        if not sel.empty:
            name = str(sel["name"].iloc[0])
            market = str(sel["market_norm"].iloc[0])
    except Exception:
        pass

    # --- FONDAMENTAUX (cache lru dans le module) ---
    fund = get_fundamentals(tkr)
    fscore, fmeta = compute_fscore_continuous_safe(fund)

    raw_score = None
    if score is not None:
        try:
            raw_score = float(score)
        except Exception:
            raw_score = None

    tech100 = None
    if raw_score is not None:
        try:
            if not math.isnan(raw_score):
                tech100 = max(0.0, min(100.0, (raw_score + 5.0) * 10.0))
        except Exception:
            tech100 = None

    WEIGHT_TECH = 0.7
    WEIGHT_FUND = 0.3
    if fscore is None and tech100 is None:
        combo = None
    elif fscore is None:
        combo = float(tech100)
    elif tech100 is None:
        combo = float(fscore)
    else:
        combo = WEIGHT_TECH * float(tech100) + WEIGHT_FUND * float(fscore)

    sig = signal_from_scorefinal(combo)

    # Earnings
    edate, edays = get_next_earnings(tkr)
    earn_str = fmt_earnings(edate, edays)

    # -- ARRONDIS (2 décimales)
    def _r2(x):
        try:
            val = float(x)
        except Exception:
            return None
        if math.isnan(val):
            return None
        return round(val, 2)

    score_final_2 = _r2(combo) if "combo" in locals() else None
    score_tech_2 = _r2(tech100) if "tech100" in locals() else None
    fscore_fund_2 = _r2(fscore) if "fscore" in locals() else None

    return {
        "Ticker": tkr,
        "Name": name,
        "Market": market,
        "Signal": sig,
        "Score": raw_score,
        "ScoreTech": score_tech_2,
        "FscoreFund": fscore_fund_2,
        "Ffund_valid": (fmeta or {}).get("valid_metrics") if isinstance(fmeta, dict) else None,
        "ScoreFinal": score_final_2,
        "RSI": rsi_val,
        "MACD_hist": macd_h,
        "%toHH52": pct_hh,
        "VolZ20": volz20,
        "Earnings": earn_str,
        "EarningsDate": edate,
        "EarningsD": edays,
    }


def score_one(ticker: str, profile: str | None = None, *, debug: bool = False):
    """Thread-safe: ne modifie pas st.session_state. Retourne un dict score ou {'error','trace'}."""

    tkr = _norm_ticker(ticker)
    ok, why = validate_ticker(tkr)
    if not ok:
        return {"Ticker": tkr, "error": why}

    profile = (profile or "Investisseur").strip().title()
    if profile not in {"Investisseur", "Swing"}:
        profile = "Investisseur"

    df = pd.DataFrame()

    try:
        period = "36mo" if profile == "Investisseur" else "9mo"
        df = safe_yf_download(tkr, period=period, interval="1d")
        if df is None or df.empty or "Close" not in df.columns:
            return {"Ticker": tkr, "error": "no_data"}

        df = df.dropna(subset=["Close"]).copy()
        if df.empty:
            return {"Ticker": tkr, "error": "no_close"}

        rsi_val = None
        macd_h = None
        pct_hh = None
        volz20 = None
        close_gt_sma50 = None
        sma50_gt_sma200 = None

        action = None

        if profile == "Investisseur":
            score, action = compute_score_investor(df)
            kpis_lt = compute_kpis_investor(df)
            if isinstance(kpis_lt, pd.DataFrame) and "%toHH52w" in kpis_lt.columns:
                pct_series = kpis_lt["%toHH52w"].dropna()
                if not pct_series.empty:
                    pct_hh = float(pct_series.iloc[-1])
        else:
            kpis = compute_kpis(df)
            cs = compute_score(df)
            score = None
            action = None
            if isinstance(cs, (list, tuple)):
                if len(cs) > 0:
                    score = cs[0]
                if len(cs) > 1:
                    action = cs[1]
            else:
                score = cs

            if isinstance(kpis, pd.DataFrame):
                if "RSI" in kpis.columns:
                    rsi_series = kpis["RSI"].dropna()
                    if not rsi_series.empty:
                        rsi_val = float(rsi_series.iloc[-1])
                if "MACD_hist" in kpis.columns:
                    macd_series = kpis["MACD_hist"].dropna()
                    if not macd_series.empty:
                        macd_h = float(macd_series.iloc[-1])
                if "%toHH52" in kpis.columns:
                    pct_series = kpis["%toHH52"].dropna()
                    if not pct_series.empty:
                        pct_hh = float(pct_series.iloc[-1])
                if "VolZ20" in kpis.columns:
                    vol_series = kpis["VolZ20"].dropna()
                    if not vol_series.empty:
                        volz20 = float(vol_series.iloc[-1])
                close_series = df["Close"].dropna()
                close_last = float(close_series.iloc[-1]) if not close_series.empty else np.nan
                sma50_last = (
                    float(kpis["SMA50"].dropna().iloc[-1])
                    if "SMA50" in kpis.columns and not kpis["SMA50"].dropna().empty
                    else np.nan
                )
                sma200_last = (
                    float(kpis["SMA200"].dropna().iloc[-1])
                    if "SMA200" in kpis.columns and not kpis["SMA200"].dropna().empty
                    else np.nan
                )
                if not np.isnan(close_last) and not np.isnan(sma50_last):
                    close_gt_sma50 = close_last > sma50_last
                if not np.isnan(sma50_last) and not np.isnan(sma200_last):
                    sma50_gt_sma200 = sma50_last > sma200_last

        signal_code = action

        uni = get_universe_normalized()
        name, market = "", ""
        try:
            sel = uni.loc[
                uni["ticker"].astype(str).str.upper() == tkr, ["name", "market_norm"]
            ]
            if not sel.empty:
                name = str(sel["name"].iloc[0])
                market = str(sel["market_norm"].iloc[0])
        except Exception:
            pass

        return {
            "Ticker": tkr,
            "Name": name,
            "Market": market,
            "Signal": signal_code,
            "Score": float(score) if score is not None else None,
            "RSI": rsi_val,
            "MACD_hist": macd_h,
            "%toHH52": pct_hh,
            "VolZ20": volz20,
            "Close>SMA50": close_gt_sma50,
            "SMA50>SMA200": sma50_gt_sma200,
        }

    except Exception as e:
        debug_payload = None
        if debug:
            debug_payload = {
                "df_shape": getattr(df, "shape", None),
                "df_columns": list(df.columns) if isinstance(df, pd.DataFrame) else None,
            }
        return {
            "Ticker": tkr,
            "error": f"exception:{type(e).__name__}:{e}",
            "trace": traceback.format_exc(),
            "debug": debug_payload,
        }

# ========= Recherche dans l'univers (nom / isin / ticker) =========
def search_universe(query: str, topk: int = 50) -> pd.DataFrame:
    uni = load_universe()
    q = (query or "").strip()
    if not q:
        return uni.head(0)
    q_upper = q.upper()
    q_lower = q.lower()
    exact = uni[(uni["ticker"] == q_upper) | (uni["isin"] == q_upper) | (uni["name"].str.lower() == q_lower)]
    contains = uni[
        uni["ticker"].str.contains(q_upper, na=False) |
        uni["isin"].str.contains(q_upper, na=False) |
        uni["name"].str.lower().str.contains(q_lower, na=False)
    ]
    names = uni["name"].dropna().astype(str).tolist()
    fuzz = difflib.get_close_matches(q, names, n=topk, cutoff=0.6)
    fuzzy = uni[uni["name"].isin(fuzz)]
    out = pd.concat([exact, contains, fuzzy], ignore_index=True).drop_duplicates(subset=["isin","ticker","name"])
    return out.head(topk)

# ========= Positions (CSV persistant en session) =========
POSITIONS_KEY = "positions_df"

def load_positions() -> pd.DataFrame:
    if POSITIONS_KEY in st.session_state:
        return st.session_state[POSITIONS_KEY].copy()
    # structure: isin,ticker,opened_at,qty,entry_price,note,status
    df = pd.DataFrame(columns=["isin","ticker","opened_at","qty","entry_price","note","status"])
    # types
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    st.session_state[POSITIONS_KEY] = df.copy()
    return df

def save_positions(df: pd.DataFrame):
    # normalisation légère
    df = df.copy()
    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["opened_at"] = df["opened_at"].astype(str).str[:10]
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    df["note"] = df["note"].astype(str)
    df["status"] = df["status"].astype(str).str.lower().replace({"ouvert":"open","fermé":"closed","close":"closed"})
    st.session_state[POSITIONS_KEY] = df.copy()

def export_positions_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

@st.cache_data(show_spinner=False, ttl=30)
def last_close(ticker: str) -> float | None:
    try:
        df = safe_yf_download(ticker, period="5d", interval="1d")
        if df is None or df.empty or "Close" not in df.columns:
            return None
        return float(df["Close"].iloc[-1])
    except Exception:
        return None

def compute_pnl_row(row: pd.Series) -> dict:
    tkr = str(row.get("ticker","")).strip().upper()
    qty = float(row.get("qty") or 0)
    entry = float(row.get("entry_price") or 0)
    lc = last_close(tkr) if tkr else None
    if lc is None or qty == 0 or entry == 0:
        return {"last": None, "pnl_abs": None, "pnl_pct": None}
    pnl_abs = (lc - entry) * qty
    pnl_pct = (lc/entry - 1.0) * 100.0
    return {"last": lc, "pnl_abs": pnl_abs, "pnl_pct": pnl_pct}

def signal_for_ticker(ticker: str) -> tuple[str, float] | None:
    res = score_one(ticker, profile=get_analysis_profile(), debug=DEBUG_MODE)
    if not isinstance(res, dict) or res.get("error"):
        return None
    return res.get("Signal", ""), float(res.get("Score", 0))


# ===== Normalisation / Validation des tickers =====
def _norm_ticker(x: str) -> str:
    return str(x or "").strip().upper()


@st.cache_data(ttl=3600)
def load_universe_df() -> pd.DataFrame:
    return load_universe()


def validate_ticker(tkr: str) -> tuple[bool, str]:
    """Vérifie si le ticker est présent dans l'univers et ressemble à un symbole Yahoo valide."""

    t = _norm_ticker(tkr)
    uni = load_universe_df()
    if t in uni["ticker"].astype(str).str.upper().values:
        return True, ""

    import re

    if not re.match(r"^[A-Z0-9\.-]{1,12}$", t):
        return False, "format_ticker_invalide"
    return False, "ticker_hors_univers"
# ========= Onglets =========
tab_full, tab_single, tab_pos = st.tabs(
    ["🚀 Scanner complet", "📄 Fiche valeur", "💼 Positions"]
)

# --------- Onglet FICHE ---------
with tab_single:
    st.title("Fiche valeur (analyse individuelle)")
    profile = get_analysis_profile()
    score_label = get_score_label()
    ticker_default = st.session_state.get("ticker_fiche_valeur", "AAPL")
    tkr = st.text_input(
        "Ticker (Yahoo Finance)", value=ticker_default, key="ticker_fiche_valeur"
    )
    ticker_input = _norm_ticker(tkr)
    nm = get_name_for_ticker(ticker_input) if ticker_input else None
    if nm:
        st.caption(f"**{nm}**")

    if ticker_input:
        try:
            period = "36mo" if profile == "Investisseur" else "9mo"
            df_price = safe_yf_download(ticker_input, period=period, interval="1d")
            if isinstance(df_price.columns, pd.MultiIndex):
                df_price.columns = df_price.columns.get_level_values(0)
            if df_price.empty or "Close" not in df_price.columns:
                st.warning("Pas de données utilisables.")
            else:
                if profile == "Investisseur":
                    score, signal = compute_score_investor(df_price)
                    kpis_lt = compute_kpis_investor(df_price)
                    has_kpis_lt = isinstance(kpis_lt, pd.DataFrame) and not kpis_lt.empty
                    st.subheader(
                        f"{ticker_input} — {score_label}: {score:.2f} | Signal: {signal}"
                    )
                    last = (
                        kpis_lt.iloc[-1]
                        if has_kpis_lt
                        else pd.Series(dtype="float64")
                    )
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "SMA26w",
                            f"{float(last.get('SMA26w', np.nan)):.2f}"
                            if has_kpis_lt
                            else "–",
                        )
                        st.metric(
                            "SMA52w",
                            f"{float(last.get('SMA52w', np.nan)):.2f}"
                            if has_kpis_lt
                            else "–",
                        )
                    with col2:
                        pct_hh = (
                            float(last.get("%toHH52w", np.nan))
                            if has_kpis_lt
                            else np.nan
                        )
                        st.metric(
                            "% to 52w High",
                            f"{pct_hh*100:.2f}%" if not np.isnan(pct_hh) else "–",
                        )
                        mom = (
                            float(last.get("MOM_12m_minus_1m", np.nan))
                            if has_kpis_lt
                            else np.nan
                        )
                        st.metric(
                            "Momentum 12-1",
                            f"{mom*100:.2f}%" if not np.isnan(mom) else "–",
                        )
                    with col3:
                        rv = (
                            float(last.get("RV20w", np.nan))
                            if has_kpis_lt
                            else np.nan
                        )
                        st.metric(
                            "Volatilité 20w",
                            f"{rv:.2%}" if not np.isnan(rv) else "–",
                        )
                        dd = (
                            float(last.get("DD26w", np.nan))
                            if has_kpis_lt
                            else np.nan
                        )
                        st.metric(
                            "Drawdown 26w",
                            f"{dd*100:.2f}%" if not np.isnan(dd) else "–",
                        )
                else:
                    kpis = compute_kpis(df_price)
                    cs = compute_score(df_price)
                    score = np.nan
                    signal = None
                    if isinstance(cs, (list, tuple)):
                        score = cs[0] if len(cs) >= 1 else np.nan
                        signal = cs[1] if len(cs) >= 2 else None
                    else:
                        score = cs
                    st.subheader(
                        f"{ticker_input} — {score_label}: {score:.2f} | Signal: {signal}"
                    )
                    last = (
                        kpis.iloc[-1]
                        if isinstance(kpis, pd.DataFrame) and not kpis.empty
                        else pd.Series(dtype="float64")
                    )
                    has_kpis = isinstance(kpis, pd.DataFrame) and not kpis.empty
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "RSI(14)",
                            f"{float(last.get('RSI', np.nan)):.1f}"
                            if has_kpis
                            else "–",
                        )
                        st.metric(
                            "MACD hist",
                            f"{float(last.get('MACD_hist', np.nan)):.3f}"
                            if has_kpis
                            else "–",
                        )
                    with col2:
                        st.metric(
                            "Close > SMA50",
                            "✅"
                            if has_kpis
                            and float(last.get("Close", np.nan))
                            > float(last.get("SMA50", np.nan))
                            else ("❌" if has_kpis else "–"),
                        )
                        st.metric(
                            "SMA50 > SMA200",
                            "✅"
                            if has_kpis
                            and float(last.get("SMA50", np.nan))
                            > float(last.get("SMA200", np.nan))
                            else ("❌" if has_kpis else "–"),
                        )
                    with col3:
                        pct_hh = (
                            float(last.get("pct_to_HH52", np.nan)) * 100
                            if has_kpis
                            else np.nan
                        )
                        st.metric(
                            "% to 52w High",
                            f"{pct_hh:.2f}%" if not np.isnan(pct_hh) else "–",
                        )
                        volz = (
                            float(last.get("VolZ20", np.nan))
                            if has_kpis
                            else np.nan
                        )
                        st.metric(
                            "Vol Z20",
                            f"{volz:.2f}" if not np.isnan(volz) else "–",
                        )
                st.line_chart(df_price[["Close"]])
        except Exception as e:
            st.error(f"Erreur : {e}")

        st.markdown("### 🧾 Données utilisées pour le score (audit)")
        try:
            df = yf.download(
                ticker_input,
                period="18mo",
                interval="1d",
                group_by="column",
                auto_adjust=False,
                progress=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
                except Exception:
                    df = df.droplevel(level=1, axis=1)
            needed = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            present = [c for c in needed if c in df.columns]
            if not present:
                st.error("Pas de colonnes OHLCV disponibles pour ce ticker.")
                st.stop()

            tech = _compute_daily_kpis_for_audit(df)

            fund_raw = get_fundamentals(ticker_input)
            fscore, _ = compute_fscore_continuous_safe(fund_raw)

            edate, edays = None, None
            try:
                cal = yf.Ticker(ticker_input).calendar
                if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                    rawd = cal.loc["Earnings Date"].values[0]
                    edate = _as_str_date(rawd)
                    if edate:
                        d0 = pd.Timestamp.today(tz="Europe/Paris").normalize()
                        d1 = pd.to_datetime(edate)
                        edays = int((d1 - d0).days)
            except Exception:
                pass

            tech_rows = [
                {"Champ": "Close_last", "Valeur": tech.get("Close_last"), "Source": "yf.download"},
                {"Champ": "RSI14", "Valeur": tech.get("RSI14"), "Source": "yf.download"},
                {"Champ": "MACD_hist", "Valeur": tech.get("MACD_hist"), "Source": "yf.download"},
                {"Champ": "SMA50", "Valeur": tech.get("SMA50"), "Source": "yf.download"},
                {"Champ": "SMA200", "Valeur": tech.get("SMA200"), "Source": "yf.download"},
                {
                    "Champ": "SMA50_gt_SMA200",
                    "Valeur": tech.get("SMA50_gt_SMA200"),
                    "Source": "dérivé (Close)",
                },
                {"Champ": "High_52w", "Valeur": tech.get("High_52w"), "Source": "yf.download"},
                {
                    "Champ": "pct_to_High52w",
                    "Valeur": tech.get("pct_to_High52w"),
                    "Source": "dérivé (Close, High_52w)",
                },
                {"Champ": "Vol_Z20", "Valeur": tech.get("Vol_Z20"), "Source": "yf.download"},
                {"Champ": "Volume_last", "Valeur": tech.get("Volume_last"), "Source": "yf.download"},
                {"Champ": "Last_bar_date", "Valeur": tech.get("Last_bar_date"), "Source": "yf.download"},
            ]
            df_tech = pd.DataFrame(tech_rows)

            def _fmt_pct(x):
                if x is None:
                    return None
                try:
                    return round(100 * float(x), 2)
                except Exception:
                    return None

            fund_rows = [
                {
                    "Champ": "eps_growth (YoY)",
                    "Valeur": fund_raw.get("eps_growth"),
                    "Valeur_%": _fmt_pct(fund_raw.get("eps_growth")),
                    "Source": "Ticker.info['earningsGrowth'|'earningsQuarterlyGrowth']",
                },
                {
                    "Champ": "revenue_growth (YoY)",
                    "Valeur": fund_raw.get("revenue_growth"),
                    "Valeur_%": _fmt_pct(fund_raw.get("revenue_growth")),
                    "Source": "Ticker.info['revenueGrowth']",
                },
                {
                    "Champ": "profit_margin (TTM)",
                    "Valeur": fund_raw.get("profit_margin"),
                    "Valeur_%": _fmt_pct(fund_raw.get("profit_margin")),
                    "Source": "Ticker.info['profitMargins']",
                },
                {
                    "Champ": "ROE (TTM)",
                    "Valeur": fund_raw.get("roe"),
                    "Valeur_%": _fmt_pct(fund_raw.get("roe")),
                    "Source": "Ticker.info['returnOnEquity']",
                },
                {
                    "Champ": "Debt/Equity",
                    "Valeur": fund_raw.get("debt_to_equity"),
                    "Valeur_%": None,
                    "Source": "Ticker.info['debtToEquity']",
                },
                {
                    "Champ": "FscoreFund (0-100)",
                    "Valeur": fscore,
                    "Valeur_%": None,
                    "Source": "compute_fscore_continuous_safe(...)",
                },
            ]
            df_fund = pd.DataFrame(fund_rows)

            earn_rows = [
                {
                    "Champ": "Prochains earnings (date)",
                    "Valeur": edate,
                    "Source": "Ticker.calendar['Earnings Date']",
                },
                {
                    "Champ": "Jours jusqu’aux earnings",
                    "Valeur": edays,
                    "Source": "calcul (date Paris vs earnings)",
                },
            ]
            df_earn = pd.DataFrame(earn_rows)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🔧 Technique (yf.download)")
                st.dataframe(df_tech, use_container_width=True, height=380)
            with c2:
                st.subheader("🏦 Fondamental (Ticker.info)")
                st.dataframe(df_fund, use_container_width=True, height=380)

            st.subheader("📣 Earnings (contexte)")
            st.dataframe(df_earn, use_container_width=True, height=120)

            with st.expander("ℹ️ Comprendre les indicateurs (résumé)", expanded=False):
                st.markdown(
                    """
**Objectif** : lecture rapide des KPIs utilisés par le score (profil Investisseur).

### 📘 Indicateurs principaux
- **SMA26w** — moyenne mobile **26 semaines** (~6 mois).  
  • Au-dessus du cours ⇒ biais haussier moyen terme ; en dessous ⇒ prudence.  
- **SMA52w** — moyenne mobile **52 semaines** (1 an).  
  • Repère long terme : au-dessus ⇒ tendance LT haussière.  
- **% to 52w High** — distance au **plus haut 52 semaines**.  
  • Proche de 0% = près des sommets (momentum fort).  
- **Momentum 12-1** — performance **12 mois** en excluant le **dernier mois**.  
  • Évite l’effet de retournement court terme; >0% = dynamique positive.  
- **Volatilité 20w** — variabilité des **rendements hebdo**, **annualisée**.  
  • ~15–30% normal ; >50% très volatil.  
- **Drawdown 26w** — pire repli depuis le **plus haut 26 semaines**.  
  • -2% faible respiration ; -20% correction ; <-30% gros creux.

---

### 🧮 Formules (simplifiées)
- **SMA26w / SMA52w** : moyenne des clôtures sur 26 / 52 semaines.
- **% to 52w High** : \\( (Close - High_{52w}) / High_{52w} \\times 100 \\).
- **Momentum 12-1** : \\( (Close_{t-1m} - Close_{t-12m}) / Close_{t-12m} \\times 100 \\).
- **Volatilité 20w** :  
  1) rendements hebdo \\( r_t = \\ln(\\frac{P_t}{P_{t-1}}) \\)  
  2) \\( \\sigma_{20w} = std(r_t) \\) sur 20 semaines  
  3) **annualisation** : \\( \\sigma_{ann} = \\sigma_{20w} \\times \\sqrt{52} \\).
- **Drawdown 26w** : \\( (\\min_{après\\;max} P - P_{max})/P_{max} \\times 100 \\).

---

### 🧠 Rappels utiles
- **TTM** = *Trailing Twelve Months* (12 mois glissants).  
- Les **ETF** n’ont pas d’earnings (colonne vide = normal).  
- Les fondamentaux manquants ➜ affichés **“—”** (pas de faux 0).
                    """
                )

            audit_payload = {
                "ticker": ticker_input,
                "profile": profile,
                "tech": {r["Champ"]: r["Valeur"] for _, r in df_tech.iterrows()},
                "fund": {r["Champ"]: r["Valeur"] for _, r in df_fund.iterrows()},
                "earnings": {r["Champ"]: r["Valeur"] for _, r in df_earn.iterrows()},
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            st.download_button(
                "💾 Export JSON (données utilisées pour le score)",
                data=json.dumps(audit_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"audit_score_inputs_{ticker_input}.json",
                mime="application/json",
            )

            st.caption(
                "ℹ️ Les valeurs '—' ou vides signifient que YFinance ne fournit pas la donnée pour ce titre."
            )

        except Exception as e:
            st.error(f"Erreur dans la fiche valeur : {e}")

# --------- Onglet 🚀 SCANNER COMPLET (univers) ---------


with tab_full:
    st.title("Scanner complet — Univers entier")
    profile = get_analysis_profile()
    cache_key = _daily_cache_key(profile)

    st.session_state.setdefault("daily_full_scan", {})

    cache_entry = st.session_state["daily_full_scan"].get(cache_key, {})
    df_full_for_today = cache_entry.get("df", pd.DataFrame())
    ts_scan = cache_entry.get("ts")
    missing_precomp = False

    if df_full_for_today.empty:
        df_full_for_today, ts_scan, _ = load_precomputed_for_profile(profile)
        if not df_full_for_today.empty:
            cache_full_scan_in_session(profile, df_full_for_today, ts_scan or "—")
            cache_entry = st.session_state["daily_full_scan"].get(cache_key, {})
        else:
            missing_precomp = True

    status_placeholder = st.empty()

    # --- Univers normalisé ---
    uni = get_universe_normalized().copy()
    uni["market_norm"] = uni["market"].apply(_norm_market)

    # --- Marchés disponibles (force ETF dans l'UI) ---
    MARKETS_MAIN = [m for m in AVAILABLE_MARKETS if m != "ETF"] + ["ETF"]
    present = set(uni["market_norm"].dropna().unique().tolist())
    markets_all = sorted(set([m for m in MARKETS_MAIN if m in present] + ["ETF"]))

    # --- État de sélection persistant (multiselect) ---
    # --- Synchronise les sélections de marché avec les options disponibles ---
    st.session_state.setdefault("markets_selected", markets_all[:])
    existing_selected = st.session_state.get("markets_selected", [])
    filtered_selected = [m for m in existing_selected if m in markets_all]
    if not filtered_selected and existing_selected:
        # Les marchés choisis ne sont plus disponibles -> revenir au défaut
        filtered_selected = markets_all[:]
    if filtered_selected != existing_selected:
        st.session_state["markets_selected"] = filtered_selected[:]

    hide_before_n = 0
    # --- Bloc 1 : Panneau de contrôle ---
    with card("🔧 Panneau de contrôle"):
        st.subheader("🌍 Marchés & Filtres")

        selected_markets = st.multiselect(
            "Marchés à inclure",
            options=markets_all,
            default=st.session_state["markets_selected"],
            key="markets_selected",
            help="Sélectionne un ou plusieurs marchés à inclure dans le scan.",
        )
        selected_markets = [str(x).upper() for x in selected_markets]

        col_limit, col_hide, col_refresh = st.columns([2, 1, 1])
        with col_limit:
            limit_view = st.slider(
                "Limite d’affichage", min_value=50, max_value=1500, value=1000, step=50
            )
        with col_hide:
            hide_before_n = st.selectbox(
                "Masquer si earnings < N jours",
                [0, 1, 2, 3, 5, 7, 14],
                index=0,
                help="0 = ne rien masquer",
            )
        with col_refresh:
            refresh = st.button("🔄 Rafraîchir (scan manuel)", use_container_width=True)
        st.caption(
            "Astuce : le scan complet est mis en cache pour la journée. Utilise 🔄 Rafraîchir pour lancer un nouveau calcul."
        )

    if refresh:
        out_manual = run_full_scan_all_and_cache_ui(profile)
        if isinstance(out_manual, pd.DataFrame) and not out_manual.empty:
            cache_full_scan_in_session(profile, out_manual, _now_paris_str())
            missing_precomp = False

    meta = st.session_state.get("daily_full_scan", {}).get(cache_key, {})
    df = meta.get("df", pd.DataFrame())
    ts = meta.get("ts", "—")

    if isinstance(df, pd.DataFrame) and not df.empty:
        status_placeholder.caption(
            f"📦 Profil **{profile}** · Fichier chargé : `daily_scan_{profile.lower()}.parquet` · Dernier scan : {ts or 'n/a'} (UTC)"
        )
    else:
        if missing_precomp:
            status_placeholder.warning(
                f"⚠️ Pas de fichier pré-calculé trouvé pour le profil **{profile}** aujourd’hui.\n"
                "→ Lance le workflow GitHub correspondant ou utilise ton bouton 🔄 Rafraîchir manuel."
            )
        else:
            status_placeholder.warning(
                "⚠️ Aucun scan en cache pour aujourd’hui. Utilise 🔄 Rafraîchir pour lancer un calcul manuel."
            )

    # ---- Init 'my_watchlist' depuis fichier si absent en session ----
    if "my_watchlist" not in st.session_state:
        st.session_state["my_watchlist"] = load_user_watchlist()

    st.markdown(
        f"""
<div style="background:#F0F4F8;padding:10px;border-radius:10px;margin:8px 0;">
  🔒 <b>Règles</b> — Scan quotidien à <b>08:30 Europe/Paris</b> (Investisseur & Swing). Aucun scan au login.
  <br/>🕒 <b>Dernier scan chargé ({profile})</b> : <code>{ts}</code>
</div>
""",
        unsafe_allow_html=True,
    )

    if isinstance(df, pd.DataFrame) and not df.empty:
        view = df.copy()
        if selected_markets:
            view = view[view["Market"].isin(selected_markets)]
        else:
            view = view.iloc[0:0]

        if "EarningsD" in view.columns and hide_before_n and hide_before_n > 0:
            view = view[(view["EarningsD"].isna()) | (view["EarningsD"] >= hide_before_n)]

        view = _round_numeric_cols(view, 2)
        view = view.head(int(limit_view)).reset_index(drop=True)

        with card("📈 Statistiques du scan (cache du jour)"):
            sig_series = view.get("Signal", pd.Series(dtype="object")).fillna("")
            buy = sig_series.str.contains("BUY").mean() * 100 if not view.empty else 0.0
            hold = sig_series.str.contains("HOLD").mean() * 100 if not view.empty else 0.0
            sell = sig_series.str.contains("SELL").mean() * 100 if not view.empty else 0.0
            avg_score = (
                view["ScoreFinal"].mean()
                if "ScoreFinal" in view.columns and not view["ScoreFinal"].isna().all()
                else None
            )
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Tickers", f"{len(view):,}".replace(",", " "))
            with col2:
                st.metric("Signals BUY", f"{buy:.1f}%")
            with col3:
                st.metric("Signals HOLD", f"{hold:.1f}%")
            with col4:
                st.metric("Signals SELL", f"{sell:.1f}%")
            if avg_score is not None:
                st.caption(f"ScoreFinal moyen : {avg_score:.2f}")

        view_scan = view.copy()

        st.caption(f"🕒 Basé sur le scan du jour : {ts}")

        render_results_table(
            df=view_scan,
            profile=profile,
            title="📊 Résultats du scan",
            hide_before_n=hide_before_n,
            allow_delete=False,
        )

        try:
            st.session_state["__current_scan_df__"] = view_scan.copy()
        except Exception:
            st.session_state["__current_scan_df__"] = pd.DataFrame()

        st.markdown("---")
        st.subheader("⭐ Ma Watchlist (extrait du résultat du scan)")

        wl = [str(t).upper().strip() for t in st.session_state.get("my_watchlist", [])]
        base_df = st.session_state.get("__current_scan_df__", pd.DataFrame())

        if not len(wl):
            st.info("Watchlist vide. Ajoute des valeurs depuis le tableau ci-dessus.")
        elif base_df is None or base_df.empty:
            st.warning("Le scan du jour n'est pas chargé. Lance/Recharge le scan pour voir ta watchlist.")
        else:
            wl_df = base_df[base_df["Ticker"].astype(str).str.upper().isin(wl)].copy()

            if wl_df.empty:
                st.info("Aucune des valeurs de ta watchlist n'apparaît dans le tableau actuel (vérifie les filtres).")
            else:
                try:
                    render_results_table(
                        df=wl_df,
                        profile=profile,
                        title="",
                        hide_before_n=hide_before_n,
                        allow_delete=True,
                    )
                except Exception:
                    st.dataframe(wl_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🛠️ Gérer ma watchlist")

        if not view_scan.empty:
            opts = (
                view_scan.assign(
                    __label__=view_scan["Ticker"].astype(str).str.upper().str.strip()
                    + " — "
                    + view_scan["Name"].astype(str).str.strip()
                    + " ["
                    + view_scan["Market"].astype(str).str.upper().str.strip()
                    + "]",
                    __value__=view_scan["Ticker"].astype(str).str.upper().str.strip(),
                )[["__label__", "__value__"]]
                .drop_duplicates("__value__")
                .sort_values("__label__")
                .to_dict(orient="records")
            )
            labels = [o["__label__"] for o in opts]
            values = {o["__label__"]: o["__value__"] for o in opts}

            col_pick, col_add = st.columns([3, 1])
            with col_pick:
                choice_label = st.selectbox(
                    "Rechercher une valeur à ajouter",
                    options=["— Sélectionner —"] + labels,
                    index=0,
                )
            with col_add:
                add_clicked = st.button("➕ Ajouter", use_container_width=True)

            if add_clicked:
                if choice_label and choice_label != "— Sélectionner —":
                    tkr_to_add = values[choice_label]
                    wl = st.session_state.get("my_watchlist", [])
                    if tkr_to_add not in wl:
                        wl.append(tkr_to_add)
                        st.session_state["my_watchlist"] = sorted(set(wl))
                        save_user_watchlist(st.session_state["my_watchlist"])
                        st.success(f"{tkr_to_add} ajouté à la watchlist.")
                        safe_rerun()
                    else:
                        st.info(f"{tkr_to_add} est déjà dans la watchlist.")
        else:
            st.info(
                "Aucune donnée affichée : lance un scan ou ajuste les filtres de marché."
            )

        # ---------- Watchlist: affichage compact + corbeille sans collisions de keys ----------
        from hashlib import blake2b

        def _unique_key(prefix: str, tkr: str, idx: int = 0) -> str:
            # génère une clé courte unique déterministe à partir du ticker et d'un index
            h = blake2b(f"{prefix}|{tkr}|{idx}".encode("utf-8"), digest_size=6).hexdigest()
            return f"{prefix}_{h}"

        # Normalise & dédoublonne la watchlist une bonne fois
        wl_raw = st.session_state.get("my_watchlist", [])
        wl_norm = [str(x).upper().strip() for x in wl_raw if str(x).strip()]
        wl_uniq = sorted(set(wl_norm))
        if wl_uniq != wl_raw:
            st.session_state["my_watchlist"] = wl_uniq
            # persistance si tu l’as
            try:
                from pathlib import Path

                USER_WL_PATH = Path("data/my_watchlist.csv")
                pd.DataFrame({"ticker": wl_uniq}).to_csv(USER_WL_PATH, index=False)
            except Exception:
                pass

        if wl_uniq:
            st.caption(f"💾 Watchlist persistée ({len(wl_uniq)} tickers)")

            # grilles de 6 colonnes
            def _chunk(seq, n):
                for i in range(0, len(seq), n):
                    yield seq[i : i + n]

            section_prefix = "rmwl_grid"  # <- namespace unique à CETTE section
            for row in _chunk(wl_uniq, 6):
                cols = st.columns(len(row))
                for i, tkr in enumerate(row):
                    with cols[i]:
                        st.markdown(
                            "<div style='padding:6px 8px;border:1px solid #e5e7eb;border-radius:8px;margin-bottom:6px;'>"
                            f"<b>{tkr}</b></div>",
                            unsafe_allow_html=True,
                        )
                        btn_key = _unique_key(section_prefix, tkr, i)
                        if st.button("🗑️ Retirer", key=btn_key, use_container_width=True):
                            new_wl = [x for x in wl_uniq if x != tkr]
                            st.session_state["my_watchlist"] = new_wl
                            try:
                                pd.DataFrame({"ticker": new_wl}).to_csv(USER_WL_PATH, index=False)
                            except Exception:
                                pass
                            st.rerun()
        else:
            st.caption("Watchlist vide.")

        with st.expander("🔎 Détails techniques (colonnes supplémentaires)"):
            extra_cols = ["MACD_hist", "VolZ20"]
            extra_present = [c for c in extra_cols if c in view_scan.columns]
            if extra_present:
                extra_df = view_scan[["Ticker", "Name", "Market"] + extra_present]
                st.dataframe(extra_df, use_container_width=True)
            else:
                st.caption("Aucune colonne technique additionnelle disponible.")
    else:
        st.info(
            "Aucun cache disponible pour aujourd’hui. Clique sur **🚀 Lancer le scan** ou **🔄 Rafraîchir**."
        )
# --------- Onglet 💼 POSITIONS ---------
with tab_pos:
    st.title("💼 Positions en cours")

    pos = load_positions()

    with st.expander("➕ Ajouter une position", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            add_query = st.text_input("Nom / ISIN / Ticker (pour auto-remplir)", "")
            if st.button("🔎 Chercher dans la base"):
                candidates = search_universe(add_query, topk=20)
                if not candidates.empty:
                    st.session_state["pos_candidates"] = candidates
                else:
                    st.warning("Aucune correspondance dans la base.")

        candidates = st.session_state.get("pos_candidates")
        if isinstance(candidates, pd.DataFrame) and not candidates.empty:
            st.write("Sélectionne une ligne à préremplir :")
            st.dataframe(candidates[["ticker","name","isin","market"]], use_container_width=True, height=220)
            opt = candidates.apply(lambda r: f"{r['ticker']} — {r['name']} ({r['isin']})", axis=1).tolist()
            choice = st.selectbox("Choix", [""] + opt)
        else:
            choice = ""

        col4, col5, col6, col7 = st.columns(4)
        with col4:
            isin = st.text_input("ISIN", "")
        with col5:
            ticker_in = st.text_input("Ticker Yahoo", "")
        with col6:
            opened_at = st.date_input("Date d'entrée", value=dt.date.today())
        with col7:
            qty = st.number_input("Quantité", min_value=0.0, step=1.0, value=0.0)

        col8, col9 = st.columns(2)
        with col8:
            entry = st.number_input("Prix d'entrée", min_value=0.0, step=0.01, value=0.00)
        with col9:
            note = st.text_input("Note (optionnel)", "")

        if choice and st.button("📋 Préremplir depuis le choix"):
            r = candidates.iloc[opt.index(choice)]
            isin = r["isin"] or isin
            ticker_in = r["ticker"] or ticker_in
            st.session_state["prefill_isin"] = isin
            st.session_state["prefill_ticker"] = ticker_in
            st.rerun()

        # recharger valeurs préremplies si présentes
        isin = st.session_state.get("prefill_isin", isin)
        ticker_in = st.session_state.get("prefill_ticker", ticker_in)

        if st.button("➕ Ajouter la position"):
            if not (isin or ticker_in):
                st.warning("Renseigne au moins ISIN ou Ticker.")
            else:
                add = {
                    "isin": str(isin).strip().upper(),
                    "ticker": str(ticker_in).strip().upper(),
                    "opened_at": str(opened_at),
                    "qty": qty,
                    "entry_price": entry,
                    "note": note,
                    "status": "open",
                }
                pos = pd.concat([pos, pd.DataFrame([add])], ignore_index=True)
                pos = pos.drop_duplicates(subset=["isin","ticker","opened_at"], keep="last").reset_index(drop=True)
                save_positions(pos)
                st.success("Position ajoutée.")

    st.divider()
    st.subheader("Positions ouvertes — P&L & signaux")

    if pos.empty:
        st.info("Aucune position pour l’instant.")
    else:
        # Calcul P&L + signal
        rows = []
        for _, r in pos[pos["status"].str.lower().eq("open")].iterrows():
            pnl = compute_pnl_row(r)
            sig = signal_for_ticker(str(r.get("ticker","")).strip().upper()) if str(r.get("ticker","")).strip() else None
            rows.append({
                "Ticker": str(r.get("ticker","")).strip().upper(),
                "Name": get_name_for_ticker(str(r.get("ticker","")).strip().upper()),
                "ISIN": str(r.get("isin","")).strip().upper(),
                "Date entrée": r.get("opened_at",""),
                "Qté": r.get("qty", None),
                "Prix entrée": r.get("entry_price", None),
                "Dernier": pnl["last"],
                "PnL €": pnl["pnl_abs"],
                "PnL %": pnl["pnl_pct"],
                "Signal": (sig[0] if sig else ""),
                "Score": (sig[1] if sig else None),
                "Note": r.get("note",""),
            })
        if rows:
            dfp = pd.DataFrame(rows)
            # jolies colonnes
            disp = dfp[["Ticker","Name","ISIN","Date entrée","Qté","Prix entrée","Dernier","PnL €","PnL %","Signal","Score","Note"]]
            st.dataframe(disp, use_container_width=True)

            # Actions rapides
            st.markdown("### Actions rapides")
            c1, c2, c3 = st.columns(3)
            with c1:
                to_close = st.selectbox("Clôturer une position (par ticker)", [""] + dfp["Ticker"].dropna().unique().tolist())
                if st.button("✅ Marquer comme clôturée"):
                    if to_close:
                        pos.loc[pos["ticker"] == to_close, "status"] = "closed"
                        save_positions(pos)
                        st.success(f"{to_close} clôturée.")
                        st.rerun()
            with c2:
                to_delete = st.selectbox("🗑️ Supprimer une ligne (par ticker)", [""] + dfp["Ticker"].dropna().unique().tolist())
                if st.button("Supprimer définitivement"):
                    if to_delete:
                        before = len(pos)
                        pos = pos[pos["ticker"] != to_delete].reset_index(drop=True)
                        save_positions(pos)
                        st.success(f"{to_delete} supprimé ({before - len(pos)} ligne).")
                        st.rerun()
            with c3:
                st.download_button(
                    "⬇️ Exporter positions (CSV)",
                    data=export_positions_bytes(pos),
                    file_name="positions.csv",
                    mime="text/csv"
                )


# ===========================
# 📈 Onglet Backtest (basique)
# ===========================


def _load_universe(markets=None, limit=None) -> pd.DataFrame:
    """Lit data/watchlist.csv (format: ,ticker,name,market) → DataFrame normalisé."""

    df = pd.read_csv(
        "data/watchlist.csv", header=None, names=["_isin", "ticker", "name", "market"]
    )
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["market"] = df["market"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"]).reset_index(
        drop=True
    )
    if markets:
        df = df[df["market"].isin(markets)].copy()
    if limit:
        df = df.head(int(limit)).copy()
    return df[["ticker", "name", "market"]]


@st.cache_data(show_spinner=False, ttl=3600)
def _prefetch_history_batch(
    tickers: list[str], months_back: int
) -> dict[str, pd.DataFrame]:
    """Télécharge l'historique pour un lot de tickers en UNE requête yfinance, puis split par ticker."""

    if not tickers:
        return {}

    months = max(months_back + 6, 18)
    try:
        raw = yf.download(
            " ".join(tickers),
            period=f"{months}mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
    except Exception:
        return {}

    out: dict[str, pd.DataFrame] = {}
    need = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    if isinstance(raw.columns, pd.MultiIndex):
        tickers_found = sorted(set(raw.columns.get_level_values(0)))
        for tkr in tickers_found:
            sub = raw[tkr].copy()
            if not all(c in sub.columns for c in need):
                continue
            sub.index = pd.to_datetime(sub.index).tz_localize(None)
            for c in need:
                sub[c] = pd.to_numeric(sub[c], errors="coerce")
            sub = sub.dropna(subset=["Close"])
            if sub.empty:
                continue
            out[tkr.upper()] = sub
    else:
        sub = raw.copy()
        if all(c in sub.columns for c in need):
            sub.index = pd.to_datetime(sub.index).tz_localize(None)
            for c in need:
                sub[c] = pd.to_numeric(sub[c], errors="coerce")
            sub = sub.dropna(subset=["Close"])
            if not sub.empty:
                out[tickers[0].upper()] = sub

    return out


# --- helpers diagnostics ---
def _diag_panel(title: str, content):
    with st.expander(title, expanded=False):
        st.write(content)


def _check_universe(df_uni: pd.DataFrame):
    info = {
        "rows": len(df_uni),
        "markets":
            sorted(df_uni["market"].unique().tolist())
            if "market" in df_uni.columns and len(df_uni)
            else [],
        "head": df_uni.head(10).to_dict(orient="records") if len(df_uni) else [],
    }
    _diag_panel("🔎 Diagnostics univers", info)


# --- remplace _pick_dates par une version basée sur POSITIONS (jours de bourse) ---
def _pick_dates_by_pos(df: pd.DataFrame, months_back: int, horizon_days: int):
    """
    Choisit date_ref ~ 'il y a X mois' en prenant la date de marché la plus proche 'par le bas',
    puis prend la date à +horizon_days *en positions* (jours de bourse), sans exiger de date calendaire exacte.
    """

    if df.empty:
        return None, None

    idx = df.index.sort_values()
    approx_ref = idx.max() - pd.DateOffset(months=months_back)
    pos_ref = idx.searchsorted(approx_ref, side="right") - 1
    if pos_ref < 0:
        return None, None
    pos_h = min(pos_ref + int(horizon_days), len(idx) - 1)
    return idx[pos_ref], idx[pos_h]


def _normalize_signal(sig: str | None) -> str:
    """Mappe tout signal en {BUY, HOLD, WATCH, SELL, REDUCE} (ou 'NA')."""

    if not sig:
        return "NA"
    s = str(sig).strip().upper()
    if s in {"BUY", "LONG"}:
        return "BUY"
    if s in {"HOLD", "WATCH", "NEUTRAL"}:
        return "HOLD" if s == "HOLD" else "WATCH"
    if s in {"SELL", "SHORT"}:
        return "SELL"
    if s in {"REDUCE", "TRIM"}:
        return "REDUCE"
    return s


def _run_backtest(
    df_uni: pd.DataFrame, months_back: int, horizon_days: int, max_workers: int = 8
):
    """Boucle multi-threads avec barre de progression."""

    rows, errs = [], []
    progress = st.progress(0.0, text="Téléchargement & calcul…")
    N = len(df_uni)

    tickers_list = df_uni["ticker"].astype(str).str.upper().tolist()

    def _chunk(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i : i + n]

    hist_cache: dict[str, pd.DataFrame] = {}
    for batch in _chunk(tickers_list, 50):
        hist_cache.update(_prefetch_history_batch(batch, int(months_back)))

    def worker(tkr, name, mkt):
        row = {
            "Ticker": tkr,
            "Name": str(name),
            "Market": mkt,
            "DateRef": None,
            "DateH": None,
            "CloseRef": None,
            "CloseH": None,
            "Perf_%": None,
            "ScoreRef": None,
            "SignalRef": "NA",
            "error": None,
        }

        df = hist_cache.get(str(tkr).upper())
        if df is None or df.empty:
            row["error"] = "no_data"
            return row

        dref, dh = _pick_dates_by_pos(df, months_back, horizon_days)
        if dref is None or dh is None:
            row["error"] = "date_pick_failed"
            return row

        hist = df[df.index <= dref].copy()
        if len(hist) < 60:
            row["error"] = "not_enough_history"
            return row

        try:
            score_ref, signal_ref = compute_score_investor(hist)
        except Exception as e:
            row["error"] = f"score_failed:{e}"
            return row

        try:
            p_then = float(df.loc[dref, "Close"])
            p_h = float(df.loc[dh, "Close"])
            perf = (p_h / p_then - 1.0) * 100.0
        except Exception as e:
            row["error"] = f"perf_failed:{e}"
            return row

        row.update(
            {
                "DateRef": dref.date().isoformat(),
                "DateH": dh.date().isoformat(),
                "CloseRef": round(p_then, 4),
                "CloseH": round(p_h, 4),
                "Perf_%": round(perf, 2),
                "ScoreRef": round(float(score_ref), 2)
                if score_ref is not None
                else None,
                "SignalRef": _normalize_signal(signal_ref),
            }
        )
        return row

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(worker, r["ticker"], r["name"], r["market"]): r["ticker"]
            for _, r in df_uni.iterrows()
        }
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                res = fut.result()
                rows.append(res)
            except Exception as e:
                errs.append(str(e))
            progress.progress(k / N, text=f"Calcul {k}/{N}")

    progress.empty()
    df_res = pd.DataFrame(rows)
    return df_res, errs


# ============ UI Backtest ============
st.markdown("---")
st.header("📈 Backtest (basique)")

# Choix simples
AVAILABLE_MARKETS_BT = ["US", "FR", "UK", "DE", "JP", "ETF"]
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    months_back = st.number_input(
        "Mois en arrière (date de ref.)", min_value=3, max_value=36, value=12, step=1
    )
with c2:
    horizon_days = st.number_input(
        "Horizon (jours ouvrés)", min_value=10, max_value=180, value=60, step=5
    )
with c3:
    markets = st.multiselect(
        "Marchés", options=AVAILABLE_MARKETS_BT, default=["US", "FR"]
    )
with c4:
    limit = st.number_input(
        "Limite tickers", min_value=10, max_value=1000, value=100, step=10
    )

run_bt = st.button("🚀 Lancer le backtest", type="primary", use_container_width=True)

if run_bt:
    try:
        uni = _load_universe(markets=markets, limit=int(limit))
        _check_universe(uni)
        if uni.empty:
            st.warning(
                "Univers vide selon ces filtres (markets/limit). Essaie d’augmenter 'Limite' ou d’inclure plus de marchés."
            )
        else:
            df_bt, errors = _run_backtest(
                uni, int(months_back), int(horizon_days), max_workers=8
            )

            # ---------- Garanties de schéma (évite les KeyError) ----------
            for col, default in [
                ("Perf_%", np.nan),
                ("ScoreRef", np.nan),
                ("SignalRef", "NA"),
                ("error", None),
                ("DateRef", None),
                ("DateH", None),
                ("CloseRef", np.nan),
                ("CloseH", np.nan),
            ]:
                if col not in df_bt.columns:
                    df_bt[col] = default

            # =============================
            # ⚖️ Comparaison Originale vs Calibrée (quantiles)
            # =============================

            # 1) Helpers calibration & qualité
            def _calibrate_thresholds_from_df(df_scores: pd.DataFrame, buy_q=70, sell_q=30):
                """Retourne (buy_threshold, sell_threshold) sur ScoreRef, en ignorant NaN."""

                x = pd.to_numeric(df_scores.get("ScoreRef"), errors="coerce").dropna()
                if len(x) < 10:  # fallback si trop peu de points
                    return 1.0, -1.0
                return (
                    float(np.nanpercentile(x, buy_q)),
                    float(np.nanpercentile(x, sell_q)),
                )

            def _signal_calibrated(score, th_buy, th_sell):
                if pd.isna(score):
                    return "NA"
                s = float(score)
                if s >= th_buy:
                    return "BUY"
                if s <= th_sell:
                    return "SELL"
                return "HOLD"

            # qualité directionnelle (mêmes règles que précédemment)
            HOLD_BAND = 2.0  # neutre HOLD ±2%

            def _is_good(signal, perf):
                if pd.isna(perf):
                    return None
                s = str(signal).upper()
                if s == "BUY":
                    return perf > 0
                if s in ("SELL", "REDUCE"):
                    return perf < 0
                if s in ("HOLD", "WATCH"):
                    return abs(perf) <= HOLD_BAND
                return None

            def _label_quality(good):
                if good is True:
                    return "✅ Bon"
                if good is False:
                    return "❌ Mauvais"
                return "—"

            # Normalisations numériques
            df_bt["Perf_%"] = pd.to_numeric(df_bt["Perf_%"], errors="coerce")
            df_bt["ScoreRef"] = pd.to_numeric(df_bt["ScoreRef"], errors="coerce")

            # 2) Prépare les deux vues (Originale vs Calibrée)
            df_view_orig = df_bt.copy()
            df_view_orig["SignalUse"] = (
                df_view_orig["SignalRef"].astype(str).str.upper().fillna("NA")
            )
            df_view_orig["GoodAdvice2"] = df_view_orig.apply(
                lambda r: _is_good(r["SignalUse"], r["Perf_%"]), axis=1
            )
            df_view_orig["Quality2"] = df_view_orig["GoodAdvice2"].apply(_label_quality)

            # Vue calibrée (sur ScoreRef du backtest courant)
            th_buy, th_sell = _calibrate_thresholds_from_df(df_bt, buy_q=70, sell_q=30)
            df_view_cal = df_bt.copy()
            df_view_cal["SignalCal"] = df_view_cal["ScoreRef"].apply(
                lambda s: _signal_calibrated(s, th_buy, th_sell)
            )
            df_view_cal["GoodAdvice2"] = df_view_cal.apply(
                lambda r: _is_good(r["SignalCal"], r["Perf_%"]), axis=1
            )
            df_view_cal["Quality2"] = df_view_cal["GoodAdvice2"].apply(_label_quality)

            # 3) Sélecteur de vue (affichage unique)
            st.markdown("---")
            cvu1, cvu2, cvu3 = st.columns([1, 1, 2])
            with cvu1:
                view_mode = st.radio(
                    "Vue", ["Originale", "Calibrée (Q70/Q30)"], horizontal=False, index=0
                )
            with cvu2:
                sort_by = st.selectbox(
                    "Trier par", ["ScoreRef", "Perf_%", "Ticker"], index=0
                )
            with cvu3:
                sort_desc = st.checkbox("Tri décroissant", value=True)

            df_curr = df_view_orig if view_mode.startswith("Originale") else df_view_cal

            # 4) Indicateur global % de “Bons”
            if df_curr["GoodAdvice2"].notna().any():
                total = int(df_curr["GoodAdvice2"].notna().sum())
                bons = int((df_curr["GoodAdvice2"] == True).sum())
                pct_bon = round(bons / total * 100, 1) if total > 0 else 0.0
                st.markdown(
                    f"### {('🟦' if view_mode.startswith('Originale') else '🟪')} "
                    f"**{view_mode}** — ✅ {pct_bon}% de bons conseils ({bons}/{total})"
                    f" &nbsp;&nbsp;•&nbsp; seuils calibrés: BUY ≥ **{th_buy:.2f}**, SELL ≤ **{th_sell:.2f}**"
                    if view_mode.startswith("Calibrée")
                    else f"### 🟦 **{view_mode}** — ✅ {pct_bon}% de bons conseils ({bons}/{total})"
                )
            else:
                st.info(
                    "Aucune ligne exploitable pour calculer le % de bons conseils (Perf_% ou Signaux manquants)."
                )

            # 5) Options “liste seule”
            copt1, copt2, copt3 = st.columns([1, 1, 2])
            with copt1:
                only_with_perf = st.checkbox(
                    "Uniquement lignes avec performance",
                    value=True,
                    key=f"bt_onlyperf_{view_mode}",
                )
            with copt2:
                show_errors = st.checkbox(
                    "Afficher lignes en erreur",
                    value=False,
                    key=f"bt_err_{view_mode}",
                )
            with copt3:
                pass

            df_show = df_curr.copy()
            if only_with_perf:
                df_show = df_show[df_show["Perf_%"].notna()].copy()
            if not show_errors and "error" in df_show.columns:
                df_show = df_show[df_show["error"].isna()].copy()

            # 6) Colonnes & tri
            cols_order = [
                c
                for c in [
                    "Ticker",
                    "Name",
                    "Market",
                    "DateRef",
                    "DateH",
                    "CloseRef",
                    "CloseH",
                    "Perf_%",
                    "ScoreRef",
                    ("SignalUse" if view_mode.startswith("Originale") else "SignalCal"),
                    "Quality2",
                    "error",
                ]
                if c in df_show.columns
            ]

            if sort_by in df_show.columns:
                df_show = df_show.sort_values(
                    sort_by, ascending=not sort_desc, na_position="last"
                )

            # 7) Tableau + export (vue active)
            st.dataframe(df_show[cols_order], use_container_width=True, hide_index=True)

            import io

            csv_buf = io.StringIO()
            df_show[cols_order].to_csv(csv_buf, index=False)
            st.download_button(
                f"⬇️ Exporter la liste ({view_mode})",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name=f"backtest_{'cal' if view_mode.startswith('Calibrée') else 'orig'}_m{months_back}_h{horizon_days}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # ---------- Logs d’erreur (facultatif, sous expander) ----------
            if errors or ("error" in df_bt.columns and df_bt["error"].notna().any()):
                with st.expander("⚠️ Logs / Lignes en erreur", expanded=False):
                    if errors:
                        st.write(errors[:50])
                    if "error" in df_bt.columns:
                        st.dataframe(
                            df_bt[df_bt["error"].notna()][cols_order],
                            use_container_width=True,
                            hide_index=True,
                        )
    except Exception as e:
        st.error(f"Backtest impossible : {e}")

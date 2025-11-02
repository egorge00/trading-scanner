import os
import sys
import io
import difflib
import datetime as dt
import base64
import contextlib
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


def _color_score(s: pd.Series) -> list[str]:
    styles: list[str] = []
    for v in s:
        try:
            x = float(v)
        except Exception:
            styles.append("")
            continue
        if x >= 3:
            styles.append("background-color:#E8FAE6;")
        elif x <= -3:
            styles.append("background-color:#FDE8E8;")
        else:
            styles.append("")
    return styles


def _style_scorefinal(series: pd.Series) -> list[str]:
    out: list[str] = []
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


def _daily_cache_key(profile: str) -> str:
    return f"{_today_paris_str()}|{profile}"


def _now_paris_iso():
    return datetime.now(tz=ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d %H:%M:%S %Z")


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
from api.core.fundamentals import (  # noqa: E402
    compute_fscore_basic,
    get_fundamentals,
)

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
FULL_SCAN_WATCHLIST_KEY = "full_scan_watchlist_df"
FULL_SCAN_WATCHLIST_PATH = "data/full_scan_watchlist.csv"

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
    df = normalize_cols(pd.DataFrame(columns=["isin","ticker","name","market"]))
    st.session_state[MY_WATCHLIST_KEY] = df.copy()
    return df

def save_my_watchlist(df: pd.DataFrame):
    st.session_state[MY_WATCHLIST_KEY] = normalize_cols(df)

# ========= Persistance locale : watchlist du Scanner complet =========
def ensure_data_dir():
    os.makedirs("data", exist_ok=True)

def _normalize_full_wl(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["isin","ticker","name","market"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols].copy()
    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["name"] = df["name"].astype(str).str.strip()
    df["market"] = df["market"].astype(str).str.strip()
    return df

def load_full_scan_watchlist() -> pd.DataFrame:
    # 1) session
    if FULL_SCAN_WATCHLIST_KEY in st.session_state:
        return st.session_state[FULL_SCAN_WATCHLIST_KEY].copy()
    # 2) disque (CSV)
    try:
        if os.path.exists(FULL_SCAN_WATCHLIST_PATH):
            df = pd.read_csv(FULL_SCAN_WATCHLIST_PATH)
            df = _normalize_full_wl(df)
            st.session_state[FULL_SCAN_WATCHLIST_KEY] = df.copy()
            return df
    except Exception:
        pass
    # 3) vide
    df = _normalize_full_wl(pd.DataFrame(columns=["isin","ticker","name","market"]))
    st.session_state[FULL_SCAN_WATCHLIST_KEY] = df.copy()
    return df

def save_full_scan_watchlist(df: pd.DataFrame):
    df = _normalize_full_wl(df)
    st.session_state[FULL_SCAN_WATCHLIST_KEY] = df.copy()
    try:
        ensure_data_dir()
        df.to_csv(FULL_SCAN_WATCHLIST_PATH, index=False)
    except Exception:
        # on ne casse pas l'UI si l'écriture échoue (ex: FS en read-only)
        pass

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
    fscore100, _ = compute_fscore_basic(fund)

    raw_score = float(score) if score is not None else None
    tech100 = None if raw_score is None or np.isnan(raw_score) else max(0.0, min(100.0, (raw_score + 5.0) * 10.0))
    WEIGHT_TECH = 0.7
    WEIGHT_FUND = 0.3
    if tech100 is None or np.isnan(tech100):
        combo = fscore100
    else:
        combo = WEIGHT_TECH * tech100 + WEIGHT_FUND * fscore100

    sig = signal_from_scorefinal(combo)

    # Earnings
    edate, edays = get_next_earnings(tkr)
    earn_str = fmt_earnings(edate, edays)

    return {
        "Ticker": tkr,
        "Name": name,
        "Market": market,
        "Signal": sig,
        "Score": raw_score,
        "ScoreTech": round(tech100, 1) if tech100 is not None and not np.isnan(tech100) else None,
        "FscoreFund": round(fscore100, 1) if fscore100 is not None and not np.isnan(fscore100) else None,
        "ScoreFinal": round(combo, 1) if combo is not None and not np.isnan(combo) else None,
        "RSI": rsi_val,
        "MACD_hist": macd_h,
        "%toHH52": pct_hh,
        "VolZ20": volz20,
        "Earnings": earn_str,
        "EarningsDate": edate,
        "EarningsD": edays,
    }


def run_full_scan_all_and_cache(profile: str, max_workers: int = 8) -> pd.DataFrame:
    uni = get_universe_normalized()
    tickers = uni["ticker"].astype(str).str.upper().dropna().unique().tolist()

    if not tickers:
        st.warning("Univers vide : aucun ticker à scanner.")
        return pd.DataFrame()

    st.info(f"🚀 Scan complet de l’univers ({len(tickers)} tickers)…")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows, failures = [], []
    progress = st.progress(0, text=f"Scan en cours… (0/{len(tickers)})")
    done = 0
    max_workers = min(max_workers, max(2, (os.cpu_count() or 4) * 2))

    ts_start = _now_paris_str()
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
    st.success(f"✅ Scan terminé ({done}/{len(tickers)}) — démarré à {ts_start}")

    if failures:
        with st.expander("Diagnostics (échecs)"):
            df_fail = pd.DataFrame(failures)
            st.dataframe(df_fail, use_container_width=True, height=240)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    if "ScoreFinal" in out.columns:
        by = ["ScoreFinal"]
        asc = [False]
        if "Ticker" in out.columns:
            by.append("Ticker")
            asc.append(True)
        out = out.sort_values(by=by, ascending=asc)
    elif "Score" in out.columns:
        by = ["Score"]
        asc = [False]
        if "Ticker" in out.columns:
            by.append("Ticker")
            asc.append(True)
        out = out.sort_values(by=by, ascending=asc)
    out = out.reset_index(drop=True)
    cols = [
        "Ticker",
        "Name",
        "Market",
        "Signal",
        "Score",
        "ScoreTech",
        "FscoreFund",
        "ScoreFinal",
        "RSI",
        "MACD_hist",
        "%toHH52",
        "VolZ20",
        "Earnings",
        "EarningsDate",
        "EarningsD",
    ]
    out = out[[c for c in cols if c in out.columns]]

    if "Market" in out.columns:
        out["Market"] = out["Market"].astype(str).str.strip().str.upper()

    key = _daily_cache_key(profile)
    st.session_state.setdefault("daily_full_scan", {})
    st.session_state["daily_full_scan"][key] = {"df": out, "ts": _now_paris_str()}
    return out


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
tab_full, tab_scan, tab_single, tab_pos = st.tabs(
    ["🚀 Scanner complet", "🔎 Scanner (watchlist)", "📄 Fiche valeur", "💼 Positions"]
)
# --------- Onglet SCANNER (ma watchlist perso) ---------
with tab_scan:
    st.title("Scanner — Ma watchlist (perso)")
    my_wl = load_my_watchlist()
    uni = load_universe()
    score_label = get_score_label()
    profile_current = get_analysis_profile()

    with st.expander("📥 Ajouter depuis la base (nom / ISIN / ticker)", expanded=True):
        q = st.text_input("Rechercher dans la base", "")
        results = search_universe(q, topk=50) if q.strip() else uni.head(0)
        if not results.empty:
            st.dataframe(results[["ticker","name","isin","market"]], use_container_width=True, height=260)
            options = results.apply(lambda r: f"{r['ticker']} — {r['name']} ({r['isin']})", axis=1).tolist()
            pick = st.multiselect("Sélectionne ce que tu veux ajouter", options)
            if st.button("➕ Ajouter à ma watchlist"):
                to_add = []
                lookup = {(f"{r['ticker']} — {r['name']} ({r['isin']})"): r for _, r in results.iterrows()}
                for p in pick:
                    r = lookup.get(p)
                    if r is not None:
                        to_add.append({"isin":r["isin"],"ticker":r["ticker"],"name":r["name"],"market":r["market"]})
                if to_add:
                    add_df = normalize_cols(pd.DataFrame(to_add))
                    if "ticker" in add_df.columns:
                        add_df["ticker"] = add_df["ticker"].map(_norm_ticker)
                    my_wl = pd.concat([my_wl, add_df], ignore_index=True).drop_duplicates(subset=["isin","ticker"])
                    save_my_watchlist(my_wl)
                    st.success(f"{len(to_add)} valeur(s) ajoutée(s).")
        else:
            st.info("Tape un nom, un ISIN ou un ticker pour rechercher dans la base.")

    st.divider()
    st.subheader("📈 Scanner ma watchlist")
    run = st.button("🚀 Scanner maintenant (ma watchlist)")
    if run:
        rows = []
        failures = []
        prog = st.progress(0)
        tickers = [
            _norm_ticker(t)
            for t in my_wl.loc[my_wl["ticker"].astype(str).str.len() > 0, "ticker"].tolist()
        ]
        tickers = [t for t in tickers if t]
        n = len(tickers)
        for i, tkr in enumerate(tickers, start=1):
            prog.progress(i / max(n, 1))
            res = score_one(tkr, profile=profile_current, debug=DEBUG_MODE)
            if not isinstance(res, dict):
                failures.append({"Ticker": _norm_ticker(tkr), "error": "invalid_return"})
                continue
            if res.get("error"):
                failures.append(
                    {
                        "Ticker": res.get("Ticker", _norm_ticker(tkr)),
                        "error": res["error"],
                        "trace": res.get("trace"),
                    }
                )
            else:
                rows.append(res)
        if rows:
            df_out = (pd.DataFrame(rows)
                      .sort_values(by=["Score","Ticker"], ascending=[False, True])
                      .reset_index(drop=True))
            display_df = rename_score_for_display(df_out)
            cols = [
                "Ticker",
                "Name",
                "Market",
                "Score",
                "Signal",
                "RSI",
                "MACD_hist",
                "%toHH52",
                "VolZ20",
                "Close>SMA50",
                "SMA50>SMA200",
            ]
            display_cols = map_score_column(cols)
            display_cols = [c for c in display_cols if c in display_df.columns]
            st.dataframe(display_df[display_cols], use_container_width=True)

            st.markdown("**Top 10 opportunités 🟢**")
            top_cols = map_score_column([
                "Ticker",
                "Name",
                "Market",
                "Score",
                "Signal",
                "RSI",
                "MACD_hist",
                "%toHH52",
                "VolZ20",
            ])
            top_cols = [c for c in top_cols if c in display_df.columns]
            st.dataframe(display_df.head(10)[top_cols], use_container_width=True)

            st.markdown("### 🗑️ Supprimer une valeur depuis les résultats")
            for i, r in df_out.iterrows():
                c1, c2, c3, c4 = st.columns([2, 6, 2, 2])
                with c1:
                    st.write(f"**{r['Ticker']}**")
                with c2:
                    st.write(r.get("Name", ""))
                with c3:
                    st.write(f"{score_label}: {r['Score']}")
                with c4:
                    if st.button("🗑️ Retirer", key=f"del_{i}_{r['Ticker']}"):
                        before = len(my_wl)
                        my_wl = my_wl[my_wl["ticker"] != _norm_ticker(r["Ticker"])].reset_index(drop=True)
                        save_my_watchlist(my_wl)
                        st.success(f"{r['Ticker']} supprimé ({before - len(my_wl)} ligne).")
                        st.rerun()
        else:
            st.info("Aucun résultat exploitable sur la watchlist (tous en erreur).")

        if failures:
            df_fail = pd.DataFrame(failures)

            MAP = {
                "format_ticker_invalide": "Ticker invalide (format Yahoo).",
                "ticker_hors_univers": "Ticker hors de l'univers suivi.",
                "no_data": "Aucune donnée renvoyée par Yahoo.",
                "no_close": "Pas de clôtures exploitables.",
            }

            def _reason(e):
                if isinstance(e, str) and e.startswith("exception:"):
                    return "Erreur interne: " + e.split(":", 2)[1]
                return MAP.get(e, str(e))

            df_fail["raison"] = df_fail["error"].apply(_reason)
            with st.expander("Diagnostics (échecs)"):
                st.dataframe(df_fail[["Ticker", "raison"]], use_container_width=True)
                # Stacktrace éventuelle du dernier échec
                last_trace = None
                for f in failures:
                    if "trace" in f and f["trace"]:
                        last_trace = f["trace"]
                if last_trace:
                    st.code(last_trace, language="python")
                if DEBUG_MODE:
                    for f in failures:
                        if f.get("debug"):
                            st.caption(f"Debug {f.get('Ticker', '')}:")
                            st.json(f["debug"])

    st.markdown("---")
    st.subheader("💾 Persistance GitHub — Ma watchlist")

    colA, colB = st.columns(2)

    with colA:
        if st.button("⬇️ Importer depuis GitHub (data/my_watchlist.csv)"):
            gh_df = load_my_watchlist_from_github()
            if gh_df is None or gh_df.empty:
                st.info("Aucun fichier trouvé ou CSV vide sur GitHub.")
            else:
                save_my_watchlist(gh_df)
                st.success(f"Watchlist importée depuis GitHub ({len(gh_df)} lignes).")
                st.rerun()

    with colB:
        if st.button("⬆️ Sauvegarder sur GitHub (data/my_watchlist.csv)"):
            ok = save_my_watchlist_to_github(load_my_watchlist())
            if ok:
                st.success("Watchlist sauvegardée sur GitHub ✅")
            else:
                st.error("Échec de la sauvegarde GitHub.")

# --------- Onglet FICHE ---------
with tab_single:
    st.title("Fiche valeur (analyse individuelle)")
    ticker_input = st.text_input("Ticker Yahoo Finance", "AAPL")
    nm = get_name_for_ticker(ticker_input)
    if nm:
        st.caption(f"**{nm}**")
    profile = get_analysis_profile()
    score_label = get_score_label()
    if ticker_input:
        try:
            period = "36mo" if profile == "Investisseur" else "9mo"
            df = safe_yf_download(_norm_ticker(ticker_input), period=period, interval="1d")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or "Close" not in df.columns:
                st.warning("Pas de données utilisables.")
            else:
                if profile == "Investisseur":
                    score, signal = compute_score_investor(df)
                    kpis_lt = compute_kpis_investor(df)
                    st.subheader(f"{ticker_input} — {score_label}: {score:.2f} | Signal: {signal}")
                    last = kpis_lt.iloc[-1] if not kpis_lt.empty else pd.Series(dtype="float64")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("SMA26w", f"{float(last.get('SMA26w', np.nan)):.2f}" if not kpis_lt.empty else "–")
                        st.metric("SMA52w", f"{float(last.get('SMA52w', np.nan)):.2f}" if not kpis_lt.empty else "–")
                    with col2:
                        pct_hh = float(last.get("%toHH52w", np.nan)) if not kpis_lt.empty else np.nan
                        st.metric("% to 52w High", f"{pct_hh*100:.2f}%" if not np.isnan(pct_hh) else "–")
                        mom = float(last.get("MOM_12m_minus_1m", np.nan)) if not kpis_lt.empty else np.nan
                        st.metric("Momentum 12-1", f"{mom*100:.2f}%" if not np.isnan(mom) else "–")
                    with col3:
                        rv = float(last.get("RV20w", np.nan)) if not kpis_lt.empty else np.nan
                        st.metric("Volatilité 20w", f"{rv:.2%}" if not np.isnan(rv) else "–")
                        dd = float(last.get("DD26w", np.nan)) if not kpis_lt.empty else np.nan
                        st.metric("Drawdown 26w", f"{dd*100:.2f}%" if not np.isnan(dd) else "–")
                else:
                    kpis = compute_kpis(df)
                    cs = compute_score(df)
                    score = np.nan
                    signal = None
                    if isinstance(cs, (list, tuple)):
                        score = cs[0] if len(cs) >= 1 else np.nan
                        signal = cs[1] if len(cs) >= 2 else None
                    else:
                        score = cs
                    st.subheader(f"{ticker_input} — {score_label}: {score:.2f} | Signal: {signal}")
                    last = kpis.iloc[-1] if not kpis.empty else pd.Series(dtype="float64")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RSI(14)", f"{float(last.get('RSI', np.nan)):.1f}" if not kpis.empty else "–")
                        st.metric("MACD hist", f"{float(last.get('MACD_hist', np.nan)):.3f}" if not kpis.empty else "–")
                    with col2:
                        st.metric(
                            "Close > SMA50",
                            "✅" if float(last.get("Close", np.nan)) > float(last.get("SMA50", np.nan)) else "❌",
                        )
                        st.metric(
                            "SMA50 > SMA200",
                            "✅" if float(last.get("SMA50", np.nan)) > float(last.get("SMA200", np.nan)) else "❌",
                        )
                    with col3:
                        pct_hh = float(last.get("pct_to_HH52", np.nan)) * 100 if not kpis.empty else np.nan
                        st.metric("% to 52w High", f"{pct_hh:.2f}%" if not np.isnan(pct_hh) else "–")
                        volz = float(last.get("VolZ20", np.nan)) if not kpis.empty else np.nan
                        st.metric("Vol Z20", f"{volz:.2f}" if not np.isnan(volz) else "–")
                st.line_chart(df[["Close"]])
        except Exception as e:
            st.error(f"Erreur : {e}")

# --------- Onglet 🚀 SCANNER COMPLET (univers) ---------


with tab_full:
    st.title("Scanner complet — Univers entier")
    profile = get_analysis_profile()
    score_label = get_score_label()

    # --- Univers normalisé ---
    uni = get_universe_normalized().copy()
    uni["market_norm"] = uni["market"].apply(_norm_market)

    # --- Marchés disponibles (force ETF dans l'UI) ---
    MARKETS_MAIN = ["US", "FR", "DE", "UK", "ETF"]
    present = set(uni["market_norm"].dropna().unique().tolist())
    markets_all = sorted(set([m for m in MARKETS_MAIN if m in present] + ["ETF"]))

    # --- État de sélection persistant (multiselect) ---
    if "markets_selected" not in st.session_state:
        st.session_state["markets_selected"] = markets_all[:]

    # --- Bandeau d'état (toujours visible) ---
    key_today = _daily_cache_key(profile)
    meta_today = st.session_state.get("daily_full_scan", {}).get(key_today, {})
    ts_last = meta_today.get("ts", "—")
    cache_ok = (
        "daily_full_scan" in st.session_state
        and key_today in st.session_state["daily_full_scan"]
    )

    st.markdown(
        f"""
<div style="background:#F0F4F8;padding:10px 12px;border-radius:10px;margin-bottom:12px;">
  <b>📅 Dernier scan</b> : {ts_last} · <b>👤 Profil</b> : {profile} · <b>🧠 Cache</b> : {"OK" if cache_ok else "Absent"}
</div>
""",
        unsafe_allow_html=True,
    )

    hide_before_n = 0
    # --- Bloc 1 : Panneau de contrôle ---
    with card("🔧 Panneau de contrôle"):
        c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 1, 1])
        with c1:
            selected_markets = st.multiselect(
                "Marchés",
                options=markets_all,
                default=st.session_state["markets_selected"],
            )
            st.session_state["markets_selected"] = selected_markets or markets_all[:]
        with c2:
            limit_view = st.slider(
                "Limite d’affichage", min_value=50, max_value=1500, value=1000, step=50
            )
        with c3:
            hide_before_n = st.selectbox(
                "Masquer si earnings < N jours",
                [0, 1, 2, 3, 5, 7, 14],
                index=0,
                help="0 = ne rien masquer",
            )
        with c4:
            do_scan = st.button("🚀 Lancer le scan complet", use_container_width=True)
        with c5:
            refresh = st.button(
                "🔄 Rafraîchir (remplacer le cache)", use_container_width=True
            )
        st.caption("Astuce : le scan complet se lance automatiquement 1×/jour au premier login.")

    # --- Logique auto-scan / refresh ---
    if "daily_full_scan" not in st.session_state:
        st.session_state["daily_full_scan"] = {}

    if refresh:
        _ = run_full_scan_all_and_cache(profile)
    elif key_today not in st.session_state["daily_full_scan"]:
        _ = run_full_scan_all_and_cache(profile)
    elif do_scan:
        _ = run_full_scan_all_and_cache(profile)

    # --- Récupération du cache du jour ---
    meta = st.session_state["daily_full_scan"].get(key_today, {})
    out = meta.get("df")
    if isinstance(out, pd.DataFrame) and not out.empty:
        out = out.copy()
        need_cols = [
            "Ticker",
            "Name",
            "Market",
            "Signal",
            "Score",
            "ScoreTech",
            "FscoreFund",
            "ScoreFinal",
            "RSI",
            "MACD_hist",
            "%toHH52",
            "VolZ20",
            "Earnings",
            "EarningsDate",
            "EarningsD",
        ]
        out = out[[c for c in need_cols if c in out.columns]]
        if "Market" in out.columns:
            out["Market"] = out["Market"].astype(str).str.strip().str.upper()

        # --- Bloc 2 : Statistiques rapides ---
        with card("📈 Statistiques du scan (cache du jour)"):
            sig_series = out.get("Signal", pd.Series(dtype="object")).fillna("")
            buy = sig_series.str.contains("BUY").mean() * 100 if not out.empty else 0.0
            hold = sig_series.str.contains("HOLD").mean() * 100 if not out.empty else 0.0
            sell = sig_series.str.contains("SELL").mean() * 100 if not out.empty else 0.0
            avg_score = (
                out["ScoreFinal"].mean()
                if "ScoreFinal" in out.columns and not out["ScoreFinal"].isna().all()
                else float("nan")
            )
            st.info(
                f"**{buy:.0f}% BUY** · **{hold:.0f}% HOLD** · **{sell:.0f}% SELL** · **Score moyen : {avg_score:.1f}**"
            )

        # --- Bloc 3 : Résultats (vue filtrée dynamique sur le cache) ---
        with card("📊 Résultats du scan"):
            selected = st.session_state.get("markets_selected", markets_all)
            view = (
                out[out["Market"].isin([m.upper() for m in selected])]
                if selected
                else out.copy()
            )
            if "EarningsD" in meta.get("df", pd.DataFrame()).columns and hide_before_n and hide_before_n > 0:
                base = view.copy()
                view = base[(base["EarningsD"].isna()) | (base["EarningsD"] >= hide_before_n)]
            if "ScoreFinal" in view.columns:
                by = ["ScoreFinal"]
                asc = [False]
                if "Ticker" in view.columns:
                    by.append("Ticker")
                    asc.append(True)
                view = view.sort_values(by=by, ascending=asc)
            elif "Score" in view.columns:
                by = ["Score"]
                asc = [False]
                if "Ticker" in view.columns:
                    by.append("Ticker")
                    asc.append(True)
                view = view.sort_values(by=by, ascending=asc)
            view = view.head(int(limit_view)).reset_index(drop=True)

            cols_main = [
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
            if profile != "Investisseur":
                insert_at = cols_main.index("%toHH52")
                cols_main.insert(insert_at, "RSI")
            present = [c for c in cols_main if c in view.columns]
            display_view = rename_score_for_display(view)
            if "%toHH52" in display_view.columns:
                display_view["%toHH52"] = (
                    pd.to_numeric(display_view["%toHH52"], errors="coerce") * 100
                ).round(1)
            display_cols = map_score_column(present)
            display_cols = [c for c in display_cols if c in display_view.columns]

            score_col = map_score_column(["Score"])[0]
            main_df = display_view[display_cols]
            if not main_df.empty:
                styled = main_df.style
                if "ScoreFinal" in main_df.columns:
                    styled = styled.apply(_style_scorefinal, subset=["ScoreFinal"])
                if score_col in main_df.columns:
                    styled = styled.apply(_color_score, subset=[score_col])
                st.dataframe(styled, use_container_width=True, height=520)
            else:
                st.dataframe(main_df, use_container_width=True, height=520)

            with st.expander("🔎 Détails techniques (colonnes supplémentaires)"):
                extra_cols = ["MACD_hist", "VolZ20"]
                extra_present = [c for c in extra_cols if c in view.columns]
                if extra_present:
                    extra_df = view[["Ticker", "Name", "Market"] + extra_present]
                    st.dataframe(extra_df, use_container_width=True)
                else:
                    st.caption("Aucune colonne technique additionnelle disponible.")

            st.download_button(
                "💾 Export CSV (vue filtrée)",
                data=view.to_csv(index=False).encode("utf-8"),
                file_name="full_scan_view.csv",
                mime="text/csv",
            )

        # --- Bloc 4 : Watchlist (gestion) ---
        with card("⭐ Ma Watchlist"):
            st.caption(
                "Ajoute/supprime des valeurs ; la watchlist se scanne dans son onglet dédié."
            )

            filtered_uni = uni.copy()
            sel_markets = st.session_state.get("markets_selected", markets_all)
            if sel_markets:
                filtered_uni = filtered_uni[
                    filtered_uni["market_norm"].isin([m.upper() for m in sel_markets])
                ]

            needed = ["isin", "ticker", "name", "market_norm"]
            for col in needed:
                if col not in filtered_uni.columns:
                    filtered_uni[col] = ""
            filtered_uni = (
                filtered_uni[[c for c in needed if c in filtered_uni.columns]]
                .dropna(how="all")
                .copy()
            )
            filtered_uni = filtered_uni.rename(columns={"market_norm": "Market"})

            picked = []
            if not filtered_uni.empty:
                filtered_uni["label"] = filtered_uni.apply(
                    lambda r: f"{r['ticker']} — {r['name']} ({r['isin']})", axis=1
                )
                picked = st.multiselect(
                    "Choisis des valeurs depuis l'univers filtré",
                    options=filtered_uni["label"].tolist(),
                    default=[],
                    key="full_scan_pick",
                )
            else:
                st.info("Aucune valeur disponible dans la sélection filtrée.")

            if st.button("Ajouter à la watchlist du Scanner complet"):
                if picked:
                    lookup = {row["label"]: row for _, row in filtered_uni.iterrows()}
                    to_add = []
                    for p in picked:
                        row = lookup.get(p)
                        if row is not None:
                            to_add.append(
                                {
                                    "isin": row.get("isin", ""),
                                    "ticker": row.get("ticker", ""),
                                    "name": row.get("name", ""),
                                    "market": row.get("Market", ""),
                                }
                            )
                    if to_add:
                        full_wl = load_full_scan_watchlist()
                        add_df = pd.DataFrame(to_add)
                        if "ticker" in add_df.columns:
                            add_df["ticker"] = add_df["ticker"].map(_norm_ticker)
                        full_wl = pd.concat([full_wl, add_df], ignore_index=True)
                        full_wl = (
                            full_wl.drop_duplicates(subset=["ticker", "isin"]).reset_index(drop=True)
                        )
                        save_full_scan_watchlist(full_wl)
                        st.success(f"{len(to_add)} valeur(s) ajoutée(s).")
                    else:
                        st.info("Sélection invalide : aucune valeur à ajouter.")
                else:
                    st.info("Sélectionne au moins une valeur.")

            st.markdown("#### Watchlist du Scanner complet (sélection utilisateur)")
            full_wl = load_full_scan_watchlist()

            if full_wl.empty:
                st.info(
                    "Ta watchlist du Scanner complet est vide. Ajoute des valeurs depuis la sélection ci-dessus."
                )
            else:
                rows_wl = []
                failures = []
                for tkr in (
                    full_wl["ticker"].dropna().astype(str).str.strip().unique().tolist()
                ):
                    norm_tkr = _norm_ticker(tkr)
                    if not norm_tkr:
                        failures.append({"Ticker": "", "error": "format_ticker_invalide"})
                        continue
                    res = score_one(norm_tkr, profile=profile, debug=DEBUG_MODE)
                    if not isinstance(res, dict):
                        failures.append({"Ticker": norm_tkr, "error": "invalid_return"})
                        continue
                    if res.get("error"):
                        failures.append(
                            {
                                "Ticker": res.get("Ticker", norm_tkr),
                                "error": res["error"],
                                "trace": res.get("trace"),
                            }
                        )
                    else:
                        rows_wl.append(res)

                if rows_wl:
                    df_wl = (
                        pd.DataFrame(rows_wl)
                        .sort_values(by=["Score", "Ticker"], ascending=[False, True])
                        .reset_index(drop=True)
                    )
                    wl_cols = [
                        "Ticker",
                        "Name",
                        "Market",
                        "Score",
                        "Signal",
                        "RSI",
                        "MACD_hist",
                        "%toHH52",
                        "VolZ20",
                        "Close>SMA50",
                        "SMA50>SMA200",
                    ]
                    df_wl = df_wl[[c for c in wl_cols if c in df_wl.columns]]
                    display_wl = rename_score_for_display(df_wl)
                    display_cols = map_score_column(wl_cols)
                    display_cols = [c for c in display_cols if c in display_wl.columns]
                    st.dataframe(display_wl[display_cols], use_container_width=True)

                    st.markdown("#### Retirer une valeur de cette watchlist")
                    for i, r in df_wl.iterrows():
                        c1, c2, c3 = st.columns([5, 4, 1])
                        with c1:
                            st.write(f"**{r['Ticker']}** — {r.get('Name', '')}")
                        with c2:
                            st.write(
                                f"{score_label}: {r['Score']} | Signal: {r.get('Signal', '')}"
                            )
                        with c3:
                            if st.button("🗑️", key=f"full_wl_del_{i}_{r['Ticker']}"):
                                wl = load_full_scan_watchlist()
                                wl = wl[wl["ticker"] != _norm_ticker(r["Ticker"])].reset_index(
                                    drop=True
                                )
                                save_full_scan_watchlist(wl)
                                st.rerun()

                    buf = io.StringIO()
                    full_wl.to_csv(buf, index=False)
                    st.download_button(
                        "Exporter la watchlist du Scanner complet (CSV)",
                        data=buf.getvalue().encode(),
                        file_name="full_scan_watchlist.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("Impossible de scorer la sélection (tickers invalides ou indisponibles).")

                if failures:
                    df_fail = pd.DataFrame(failures)

                    MAP = {
                        "format_ticker_invalide": "Ticker invalide (format Yahoo).",
                        "ticker_hors_univers": "Ticker hors de l'univers suivi.",
                        "no_data": "Aucune donnée renvoyée par Yahoo.",
                        "no_close": "Pas de clôtures exploitables.",
                    }

                    def _reason(e):
                        if isinstance(e, str) and e.startswith("exception:"):
                            return "Erreur interne: " + e.split(":", 2)[1]
                        return MAP.get(e, str(e))

                    df_fail["raison"] = df_fail["error"].apply(_reason)
                    with st.expander("Diagnostics (échecs)"):
                        st.dataframe(df_fail[["Ticker", "raison"]], use_container_width=True)
                        last_trace = None
                        for f in failures:
                            if "trace" in f and f["trace"]:
                                last_trace = f["trace"]
                        if last_trace:
                            st.code(last_trace, language="python")
                        if DEBUG_MODE:
                            for f in failures:
                                if f.get("debug"):
                                    st.caption(f"Debug {f.get('Ticker', '')}:")
                                    st.json(f["debug"])
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
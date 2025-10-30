import os
import sys
import io
import time
import difflib
import datetime as dt
import bcrypt
import pandas as pd
import streamlit as st
import yfinance as yf
import concurrent.futures as cf

# --- rendre importable le package "api" depuis /ui ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.core.scoring import compute_kpis, compute_score
from api.core.isin_resolver import resolve_isin_to_ticker

# ---------- CONFIG ----------
st.set_page_config(page_title="Scanner", layout="wide")

# ---------- AUTH (simple & robuste) ----------
USERNAME = "egorge"
PASSWORD_HASH = os.getenv(
    "PASSWORD_HASH",
    "$2y$12$4LAav5U4KJwaT2YgzYTnf.qaTGo6VjxdkB6oueE//XreoI0D21RKe"
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
    df = normalize_cols(pd.DataFrame(columns=["isin","ticker","name","market"]))
    st.session_state[MY_WATCHLIST_KEY] = df.copy()
    return df

def save_my_watchlist(df: pd.DataFrame):
    st.session_state[MY_WATCHLIST_KEY] = normalize_cols(df)

def export_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

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
@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_df_cached(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker, period="6mo", interval="1d",
            group_by="column", auto_adjust=False, progress=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                df = df.droplevel(1, axis=1)
        needed = {"Open","High","Low","Close","Volume"}
        if df.empty or not needed.issubset(df.columns):
            return None
        return df
    except Exception:
        return None

def score_one(ticker: str):
    df = fetch_df_cached(ticker)
    if df is None:
        return None
    try:
        k = compute_kpis(df)
        s = compute_score(k)
        return {
            "Ticker": ticker,
            "Name": get_name_for_ticker(ticker),
            "Score": s.score,
            "Action": s.action,
            "RSI": round(k.rsi, 1),
            "MACD_hist": round(k.macd_hist, 3),
            "Close>SMA50": "✅" if k.close_above_sma50 else "❌",
            "SMA50>SMA200": "✅" if k.sma50_above_sma200 else "❌",
            "%toHH52": round(k.pct_to_hh52 * 100, 2),
            "VolZ20": round(k.vol_z20, 2),
        }
    except Exception:
        return None

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

# ========= Onglets =========
tab_scan, tab_single, tab_full = st.tabs(["🔎 Scanner (watchlist)", "📄 Fiche valeur", "🚀 Scanner complet"])

# --------- Onglet SCANNER (ma watchlist perso) ---------
with tab_scan:
    st.title("Scanner — Ma watchlist (perso)")
    my_wl = load_my_watchlist()
    uni = load_universe()

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
                    my_wl = pd.concat([my_wl, add_df], ignore_index=True).drop_duplicates(subset=["isin","ticker"])
                    save_my_watchlist(my_wl)
                    st.success(f"{len(to_add)} valeur(s) ajoutée(s).")
        else:
            st.info("Tape un nom, un ISIN ou un ticker pour rechercher dans la base.")

    st.markdown("---")
    st.subheader("📈 Scanner ma watchlist")
    run = st.button("🚀 Scanner maintenant (ma watchlist)")
    if run:
        rows = []
        prog = st.progress(0)
        tickers = my_wl.loc[my_wl["ticker"].astype(str).str.len() > 0, "ticker"].tolist()
        n = len(tickers)
        for i, tkr in enumerate(tickers, start=1):
            prog.progress(i / max(n, 1))
            res = score_one(tkr)
            if res is not None:
                rows.append(res)
        if rows:
            df_out = (pd.DataFrame(rows)
                      .sort_values(by=["Score","Ticker"], ascending=[False, True])
                      .reset_index(drop=True))
            cols = ["Ticker","Name","Score","Action","RSI","MACD_hist","%toHH52","VolZ20","Close>SMA50","SMA50>SMA200"]
            df_out = df_out[[c for c in cols if c in df_out.columns]]
            st.dataframe(df_out, use_container_width=True)

            st.markdown("**Top 10 opportunités 🟢**")
            st.dataframe(df_out.head(10)[["Ticker","Name","Score","Action","RSI","MACD_hist","%toHH52","VolZ20"]],
                         use_container_width=True)

            st.markdown("### 🗑️ Supprimer une valeur depuis les résultats")
            for i, r in df_out.iterrows():
                c1, c2, c3, c4 = st.columns([2, 6, 2, 2])
                with c1:
                    st.write(f"**{r['Ticker']}**")
                with c2:
                    st.write(r.get("Name", ""))
                with c3:
                    st.write(f"Score: {r['Score']}")
                with c4:
                    if st.button("🗑️ Retirer", key=f"del_{i}_{r['Ticker']}"):
                        before = len(my_wl)
                        my_wl = my_wl[my_wl["ticker"] != str(r["Ticker"]).strip().upper()].reset_index(drop=True)
                        save_my_watchlist(my_wl)
                        st.success(f"{r['Ticker']} supprimé ({before - len(my_wl)} ligne).")
                        st.rerun()
        else:
            st.info("Aucun résultat (tickers invalides).")

# --------- Onglet FICHE ---------
with tab_single:
    st.title("Fiche valeur (analyse individuelle)")
    ticker_input = st.text_input("Ticker Yahoo Finance", "AAPL")
    nm = get_name_for_ticker(ticker_input)
    if nm:
        st.caption(f"**{nm}**")
    if ticker_input:
        try:
            df = yf.download(ticker_input, period="6mo", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or "Close" not in df.columns:
                st.warning("Pas de données utilisables.")
            else:
                kpis = compute_kpis(df)
                score = compute_score(kpis)
                st.subheader(f"{ticker_input} — Score: {score.score} | Action: {score.action}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RSI(14)", f"{kpis.rsi:.1f}")
                    st.metric("MACD hist", f"{kpis.macd_hist:.3f}")
                with col2:
                    st.metric("Close > SMA50", "✅" if kpis.close_above_sma50 else "❌")
                    st.metric("SMA50 > SMA200", "✅" if kpis.sma50_above_sma200 else "❌")
                with col3:
                    st.metric("% to 52w High", f"{kpis.pct_to_hh52*100:.2f}%")
                    st.metric("Vol Z20", f"{kpis.vol_z20:.2f}")
                st.line_chart(df[["Close"]])
        except Exception as e:
            st.error(f"Erreur : {e}")

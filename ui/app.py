import os
import sys
import io
import datetime as dt
import bcrypt
import pandas as pd
import streamlit as st
import yfinance as yf

# --- rendre importable le package "api" depuis /ui ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.core.scoring import compute_kpis, compute_score

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

# ========= Helpers Watchlist (ISIN clé primaire) =========
WATCHLIST_PATH = "data/watchlist.csv"
ISIN_MAP_PATH = "data/isin_map.csv"

DEFAULT_WATCHLIST = pd.DataFrame([
    {"isin":"US0378331005","ticker":"AAPL","name":"Apple","market":"US"},
    {"isin":"FR0000120321","ticker":"OR.PA","name":"L'Oreal","market":"FR"},
    {"isin":"FR0000121014","ticker":"MC.PA","name":"LVMH","market":"FR"},
    {"isin":"NL0000235190","ticker":"AIR.PA","name":"Airbus","market":"FR"},
])

def load_csv_safe(path: str, expected_cols: list[str]) -> pd.DataFrame|None:
    try:
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        lower = [c.lower() for c in df.columns]
        mapping = {lc: c for lc, c in zip(lower, df.columns)}
        if not all(c in lower for c in [c.lower() for c in expected_cols]):
            return None
        return df.rename(columns={mapping[c.lower()]: c for c in expected_cols})
    except Exception:
        return None

def load_watchlist() -> pd.DataFrame:
    if "watchlist_df" in st.session_state:
        return st.session_state.watchlist_df.copy()
    df = load_csv_safe(WATCHLIST_PATH, ["isin","ticker","name","market"])
    if df is None:
        df = DEFAULT_WATCHLIST.copy()
    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    st.session_state.watchlist_df = df
    return df.copy()

def save_watchlist_to_session(df: pd.DataFrame):
    st.session_state.watchlist_df = df.copy()

def export_watchlist_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

def load_isin_map() -> dict:
    m = {}
    df = load_csv_safe(ISIN_MAP_PATH, ["isin","ticker"])
    if df is not None:
        for _, r in df.iterrows():
            m[str(r["isin"]).strip().upper()] = str(r["ticker"]).strip().upper()
    return m

ISIN_MAP = load_isin_map()

# ========= Données marché =========
def fetch_df(ticker: str) -> pd.DataFrame | None:
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
        needed = {"Open", "High", "Low", "Close", "Volume"}
        if df.empty or not needed.issubset(df.columns):
            return None
        return df
    except Exception:
        return None

def score_one(ticker: str):
    df = fetch_df(ticker)
    if df is None:
        return None
    try:
        k = compute_kpis(df)
        s = compute_score(k)
        return {
            "Ticker": ticker,
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

# ========== ONGLETS ==========
tab_scan, tab_single = st.tabs(["🔎 Scanner (watchlist)", "📄 Fiche valeur"])

# --------- Onglet SCANNER (watchlist ISIN) ---------
with tab_scan:
    st.title("Scanner — Watchlist (ISIN)")

    wl = load_watchlist()

    with st.expander("➕ Ajouter / ➖ Supprimer (clé ISIN)", expanded=False):
        colA, colB = st.columns(2)

        with colA:
            st.markdown("**Ajouter par ISIN**")
            new_isin = st.text_input("ISIN (obligatoire)", "")
            new_ticker = st.text_input("Ticker Yahoo (si connu)", "")
            new_name = st.text_input("Nom (optionnel)", "")
            new_mkt = st.text_input("Marché (optionnel)", "")
            if st.button("Ajouter à la watchlist"):
                if new_isin.strip():
                    isin = new_isin.strip().upper()
                    # tentative de résolution via mapping, si pas de ticker fourni
                    ticker = new_ticker.strip().upper() if new_ticker.strip() else ISIN_MAP.get(isin, "")
                    add = {"isin": isin, "ticker": ticker, "name": new_name.strip(), "market": new_mkt.strip()}
                    wl = pd.concat([wl, pd.DataFrame([add])], ignore_index=True)
                    wl = wl.drop_duplicates(subset=["isin"]).reset_index(drop=True)
                    save_watchlist_to_session(wl)
                    msg = f"{isin} ajouté."
                    if not ticker:
                        msg += " (⚠️ à résoudre : pas de ticker)"
                    st.success(msg)
                else:
                    st.warning("ISIN requis.")

        with colB:
            st.markdown("**Supprimer par ISIN**")
            del_isin = st.selectbox("Choisir un ISIN à supprimer", [""] + wl["isin"].tolist())
            if st.button("Supprimer de la watchlist"):
                if del_isin:
                    wl = wl[wl["isin"] != del_isin].reset_index(drop=True)
                    save_watchlist_to_session(wl)
                    st.success(f"{del_isin} supprimé.")

        st.download_button(
            "⬇️ Télécharger watchlist.csv (sauvegarde)",
            data=export_watchlist_csv_bytes(wl),
            file_name="watchlist.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.subheader("Résultats du scan (scores calculés aujourd’hui)")
    unresolved = wl[wl["ticker"].isin(["", "NA", "NAN"]) | wl["ticker"].isna()]
    if len(unresolved) > 0:
        st.warning(f"{len(unresolved)} ligne(s) à résoudre (ISIN sans ticker). Elles ne seront pas scannées.")
        st.dataframe(unresolved, use_container_width=True)

    run = st.button("🚀 Lancer le scan maintenant")
    if run:
        rows = []
        progress = st.progress(0)
        tickers = wl.loc[wl["ticker"].astype(str).str.len() > 0, "ticker"].tolist()
        n = len(tickers)
        for i, tkr in enumerate(tickers, start=1):
            progress.progress(i / max(n, 1))
            res = score_one(tkr)
            if res is not None:
                rows.append(res)
        if rows:
            df_out = pd.DataFrame(rows)
            df_out = df_out.sort_values(by=["Score", "Ticker"], ascending=[False, True]).reset_index(drop=True)
            st.dataframe(df_out, use_container_width=True)
            st.markdown("**Top 10 opportunités 🟢**")
            st.dataframe(df_out.head(10)[["Ticker", "Score", "Action", "RSI", "MACD_hist", "%toHH52", "VolZ20"]], use_container_width=True)
        else:
            st.info("Aucun résultat (tickers manquants ou invalides).")

# --------- Onglet FICHE ---------
with tab_single:
    st.title("Fiche valeur (analyse individuelle)")
    # Ici on te laisse taper le ticker, l’ISIN n’est pas compris par Yahoo
    ticker_input = st.text_input("Ticker Yahoo Finance (ex: AAPL, OR.PA, MC.PA)", "AAPL")

    if ticker_input:
        try:
            df = yf.download(
                ticker_input, period="6mo", interval="1d",
                group_by="column", auto_adjust=False, progress=False
            )
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df.columns = df.columns.get_level_values(0)
                except Exception:
                    df = df.droplevel(1, axis=1)

            needed = {"Open", "High", "Low", "Close", "Volume"}
            if df.empty or not needed.issubset(df.columns):
                st.warning("Pas de données utilisables pour ce ticker.")
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
                st.write("Dernières valeurs :")
                st.dataframe(df.tail(5))
        except Exception as e:
            st.error(f"Erreur de récupération des données : {e}")

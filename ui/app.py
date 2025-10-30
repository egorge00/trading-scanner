import os
import datetime as dt
import bcrypt
import pandas as pd
import streamlit as st
import yfinance as yf
import sys, os
# permettre d'importer depuis la racine du repo (parent de /ui)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ----- IMPORT MOTEUR KPI/SCORE -----
from api.core.scoring import compute_kpis, compute_score

# ---------- CONFIG ----------
st.set_page_config(page_title="Scanner", layout="wide")

# ---------- AUTH (simple & robuste) ----------
USERNAME = "egorge"
# Utilise la variable d'env si définie, sinon le hash déjà fourni :
PASSWORD_HASH = os.getenv(
    "PASSWORD_HASH",
    "$2y$12$4LAav5U4KJwaT2YgzYTnf.qaTGo6VjxdkB6oueE//XreoI0D21RKe"  # <-- ton hash bcrypt
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

st.title("Scanner d’opportunités – Daily")

ticker = st.text_input("Ticker Yahoo Finance (ex: AAPL, OR.PA, MC.PA)", "AAPL")

# ---- Données + KPI/Score ----
if ticker:
    try:
        # Téléchargement (colonnes plates)
        df = yf.download(
            ticker, period="6mo", interval="1d",
            group_by="column", auto_adjust=False, progress=False
        )

        # Aplanir si MultiIndex résiduel
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                df = df.droplevel(1, axis=1)

        if df.empty or not set(["Open", "High", "Low", "Close", "Volume"]).issubset(df.columns):
            st.warning("Pas de données utilisables pour ce ticker.")
        else:
            # Calcul KPIs + Score (moteur)
            kpis = compute_kpis(df)
            score = compute_score(kpis)

            st.subheader(f"{ticker} — Score: {score.score} | Action: {score.action}")

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

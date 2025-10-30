import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
import bcrypt
import os

# ---------- CONFIG ----------
st.set_page_config(page_title="Scanner", layout="wide")

# ---------- AUTH (maison, simple & robuste) ----------
USERNAME = "egorge"
PASSWORD_HASH = os.getenv("PASSWORD_HASH", "$2y$12$4LAav5U4KJwaT2YgzYTnf.qaTGo6VjxdkB6oueE//XreoI0D21RKe")  # <= ton hash bcrypt déjà mis

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

# ---- Données ----
if ticker:
    try:
        # 1) Forcer des colonnes plates (pas de MultiIndex)
        df = yf.download(
            ticker, period="6mo", interval="1d",
            group_by="column", auto_adjust=False, progress=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            # Si jamais MultiIndex malgré tout (rare)
            df = df.xs("AAPL", level=1, axis=1) if "AAPL" in df.columns.get_level_values(1) else df.droplevel(1, axis=1)

        if df.empty or "Close" not in df.columns:
            st.warning("Pas de données utilisables pour ce ticker.")
        else:
            # 2) RSI (EWMA, plus robuste)
            delta = df["Close"].diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            roll_up = up.ewm(span=14, adjust=False).mean()
            roll_down = down.ewm(span=14, adjust=False).mean()
            rs = roll_up / roll_down
            df["RSI"] = 100 - (100 / (1 + rs))

            st.subheader(f"{ticker} – Clôtures & RSI")
            st.line_chart(df[["Close", "RSI"]].dropna())

            st.write("Dernières valeurs :")
            st.dataframe(df.tail(5))
    except Exception as e:
        st.error(f"Erreur de récupération des données : {e}")

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
            st.experimental_rerun()
        else:
            st.error("Identifiants invalides")
    st.stop()

# ---------- UI une fois connecté ----------
st.sidebar.success(f"Connecté comme {USERNAME}")
if st.sidebar.button("Se déconnecter"):
    st.session_state.clear()
    st.experimental_rerun()

st.title("Scanner d’opportunités – Daily")

ticker = st.text_input("Ticker Yahoo Finance (ex: AAPL, OR.PA, MC.PA)", "AAPL")

if ticker:
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty:
            st.warning("Pas de données pour ce ticker.")
        else:
            # RSI simple (placeholder)
            delta = df["Close"].diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            rs = up.rolling(14).mean() / down.rolling(14).mean()
            df["RSI"] = 100 - (100 / (1 + rs))

            st.subheader(f"{ticker} – Clôtures & RSI")
            st.line_chart(df[["Close", "RSI"]])

            st.write("Dernières valeurs :")
            st.dataframe(df.tail(5))
    except Exception as e:
        st.error(f"Erreur de récupération des données : {e}")

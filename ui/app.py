import streamlit as st
import streamlit_authenticator as stauth
import yfinance as yf
import pandas as pd
import datetime as dt

# ---------- CONFIG ----------
st.set_page_config(page_title="Scanner", layout="wide")

# ---- AUTHENTIFICATION ----
USERNAME = "egorge"
PASSWORD_HASH = "$2y$12$8BCPvSkXXJeIRu1zn3DqDuNZUD5T1uxbAImQ7dj79joZ.kLwXI.2."  # On mettra le vrai hash après

credentials = {
    "usernames": {
        USERNAME: {"name": USERNAME, "password": PASSWORD_HASH}
    }
}
authenticator = stauth.Authenticate(
    credentials, "scanner_cookie", "scanner_key", cookie_expiry_days=30
)
name, auth_status, username = authenticator.login(location="main")

# ---- SI CONNECTÉ ----
if auth_status:
    authenticator.logout("Se déconnecter", "sidebar")
    st.success(f"Bienvenue {name} 👋")

    st.title("Scanner d’opportunités – Daily")

    # ---- Entrée utilisateur ----
    st.write("Sélectionne un ticker pour afficher les indicateurs :")
    ticker = st.text_input("Ticker Yahoo Finance (ex: AAPL, OR.PA, MC.PA)", "AAPL")

    # ---- Données ----
    if ticker:
        try:
            df = yf.download(ticker, period="6mo", interval="1d")
            df["RSI"] = (
                pd.Series(df["Close"]).diff().clip(lower=0).rolling(14).mean()
                / pd.Series(df["Close"]).diff().abs().rolling(14).mean()
            ) * 100

            # ---- Graph ----
            st.subheader(f"{ticker} – Clôtures & RSI")
            st.line_chart(df[["Close", "RSI"]])

            # ---- Résumé ----
            st.write("Dernières valeurs :")
            st.dataframe(df.tail(5))
        except Exception as e:
            st.error(f"Erreur de récupération des données : {e}")

elif auth_status is False:
    st.error("Identifiants invalides")
else:
    st.info("Veuillez vous connecter pour accéder au scanner.")

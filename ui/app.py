import os
import sys
import io
import time
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
    "$2y$12$4LAav5U4KJwaT2YgzYTnf.qaTGo6VjxdkB6oueE//XreoI0D21RKe"  # <-- remplace si besoin
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
    df["isin"]   = df["isin"].astype(str).str.strip().str.upper()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["name"]   = df["name"].astype(str).str.strip()
    df["market"] = df["market"].astype(str).str.strip()
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

# ---- Name map (Ticker -> Name) pour affichage ----
def get_name_map() -> dict:
    wl = st.session_state.get("watchlist_df")
    if wl is None:
        wl = load_watchlist()
    mp = {}
    for _, r in wl.iterrows():
        t = str(r.get("ticker","")).strip().upper()
        n = str(r.get("name","")).strip()
        if t:
            mp[t] = n
    return mp

NAME_MAP = get_name_map()

def get_name_for_ticker(tkr: str) -> str:
    return NAME_MAP.get(str(tkr).strip().upper(), "")

# ========= Données marché =========
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
        needed = {"Open", "High", "Low", "Close", "Volume"}
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
            "Name": get_name_for_ticker(ticker),  # <--- colonne Name
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

def resolve_and_update_row(wl: pd.DataFrame, isin: str, name: str) -> tuple[pd.DataFrame, str|None]:
    tkr = resolve_isin_to_ticker(isin, name_hint=name or None)
    if tkr:
        wl.loc[wl["isin"] == isin, "ticker"] = tkr
        save_watchlist_to_session(wl)
        # refresh du name map
        global NAME_MAP
        NAME_MAP = get_name_map()
        return wl, tkr
    return wl, None

# ========== ONGLETS ==========
tab_scan, tab_single, tab_full = st.tabs(["🔎 Scanner (watchlist)", "📄 Fiche valeur", "🚀 Scanner complet"])

# --------- Onglet SCANNER (watchlist ISIN) ---------
with tab_scan:
    st.title("Scanner — Watchlist (ISIN)")
    wl = load_watchlist()

    with st.expander("➕ Ajouter / ➖ Supprimer / 🔍 Résoudre (clé ISIN)", expanded=False):
        colA, colB, colC = st.columns(3)

        # --- Ajouter ---
        with colA:
            st.markdown("**Ajouter par ISIN**")
            new_isin = st.text_input("ISIN (obligatoire)", key="add_isin")
            new_name = st.text_input("Nom (optionnel)", key="add_name")
            new_mkt  = st.text_input("Marché (optionnel)", key="add_mkt")
            new_ticker = st.text_input("Ticker Yahoo (si connu)", key="add_ticker")

            if st.button("Ajouter à la watchlist"):
                if new_isin.strip():
                    isin = new_isin.strip().upper()
                    ticker = new_ticker.strip().upper()
                    if not ticker:
                        t = resolve_isin_to_ticker(isin, name_hint=new_name or None)
                        if t:
                            ticker = t
                    add = {"isin": isin, "ticker": ticker, "name": (new_name or "").strip(), "market": (new_mkt or "").strip()}
                    wl = pd.concat([wl, pd.DataFrame([add])], ignore_index=True)
                    wl = wl.drop_duplicates(subset=["isin"]).reset_index(drop=True)
                    save_watchlist_to_session(wl)
                    # refresh du name map
                    NAME_MAP = get_name_map()
                    st.success(f"{isin} ajouté. " + (f"Résolu → {ticker}" if ticker else "⚠️ ticker à résoudre"))
                else:
                    st.warning("ISIN requis.")

        # --- Supprimer ---
        with colB:
            st.markdown("**Supprimer par ISIN**")
            del_isin = st.selectbox("Choisir un ISIN à supprimer", [""] + wl["isin"].tolist(), key="del_isin")
            if st.button("Supprimer de la watchlist"):
                if del_isin:
                    wl = wl[wl["isin"] != del_isin].reset_index(drop=True)
                    save_watchlist_to_session(wl)
                    # refresh du name map
                    NAME_MAP = get_name_map()
                    st.success(f"{del_isin} supprimé.")

        # --- Résoudre ---
        with colC:
            st.markdown("**🔍 Résoudre l’ISIN → ticker**")
            unresolved = wl[wl["ticker"].isin(["", "NA", "NAN"]) | wl["ticker"].isna()]
            if len(unresolved) > 0:
                pick = st.selectbox("ISIN à résoudre", unresolved["isin"].tolist(), key="res_isin")
                name_hint = wl.loc[wl["isin"] == pick, "name"].iloc[0] if not wl[wl["isin"] == pick].empty else ""
                if st.button("🔍 Résoudre cet ISIN"):
                    wl, t = resolve_and_update_row(wl, pick, name_hint)
                    if t:
                        st.success(f"{pick} → {t}")
                    else:
                        st.warning("Aucun ticker trouvé automatiquement.")

            if st.button("🔎 Résoudre tous les ISIN sans ticker"):
                count_ok = 0
                unresolved = wl[wl["ticker"].isin(["", "NA", "NAN"]) | wl["ticker"].isna()]
                for _, r in unresolved.iterrows():
                    wl, t = resolve_and_update_row(wl, r["isin"], r.get("name",""))
                    if t:
                        count_ok += 1
                st.success(f"Résolution terminée : {count_ok} trouvé(s).")

        # Export CSV mis à jour
        st.download_button(
            "⬇️ Télécharger watchlist.csv (sauvegarde)",
            data=export_watchlist_csv_bytes(wl),
            file_name="watchlist.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.subheader("Résultats du scan (rapide)")
    run = st.button("🚀 Lancer le scan maintenant (watchlist affichée)")
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
            df_out = (pd.DataFrame(rows)
                      .sort_values(by=["Score","Ticker"], ascending=[False, True])
                      .reset_index(drop=True))
            # ordre avec Name juste après Ticker
            cols = ["Ticker","Name","Score","Action","RSI","MACD_hist","%toHH52","VolZ20","Close>SMA50","SMA50>SMA200"]
            df_out = df_out[[c for c in cols if c in df_out.columns]]
            st.dataframe(df_out, use_container_width=True)

            st.markdown("**Top 10 opportunités 🟢**")
            st.dataframe(
                df_out.head(10)[["Ticker","Name","Score","Action","RSI","MACD_hist","%toHH52","VolZ20"]],
                use_container_width=True
            )
        else:
            st.info("Aucun résultat (tickers manquants ou invalides).")

# --------- Onglet FICHE ---------
with tab_single:
    st.title("Fiche valeur (analyse individuelle)")
    ticker_input = st.text_input("Ticker Yahoo Finance (ex: AAPL, OR.PA, MC.PA)", "AAPL")

    # Affiche le nom s'il existe dans ta watchlist
    nm = get_name_for_ticker(ticker_input)
    if nm:
        st.caption(f"**{nm}**")

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

# --------- Onglet 🚀 SCANNER COMPLET ---------
with tab_full:
    st.title("Scanner complet — Univers entier")

    wl = load_watchlist()
    base = wl.loc[wl["ticker"].astype(str).str.len() > 0].copy()

    # Filtres
    colf1, colf2, colf3 = st.columns([1,2,1])
    markets = sorted([m for m in base["market"].unique() if isinstance(m, str) and m])
    sel_markets = colf1.multiselect("Marchés", options=["(Tous)"] + markets, default=["(Tous)"])
    query = colf2.text_input("Recherche (ticker ou nom contient…)", "")
    limit = colf3.number_input("Limite (nb tickers à scanner)", min_value=10, max_value=2000, value=300, step=10)

    dfv = base.copy()
    if sel_markets and "(Tous)" not in sel_markets:
        dfv = dfv[dfv["market"].isin(sel_markets)]
    if query.strip():
        q = query.strip().lower()
        dfv = dfv[dfv["ticker"].str.lower().str.contains(q) | dfv["name"].str.lower().str.contains(q)]

    tickers = dfv["ticker"].dropna().astype(str).str.strip().tolist()[: int(limit)]
    st.caption(f"{len(tickers)} tickers sélectionnés pour le scan.")

    # Exécution
    if st.button("🚀 Lancer le scan complet (parallèle)"):
        start = time.time()
        rows = []
        prog = st.progress(0)
        done = 0

        def worker(tkr):
            return score_one(tkr)

        max_workers = min(8, max(2, os.cpu_count() or 4))
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(worker, t): t for t in tickers}
            for fut in cf.as_completed(futures):
                res = fut.result()
                if res is not None:
                    rows.append(res)
                done += 1
                prog.progress(done / max(len(tickers), 1))

        elapsed = time.time() - start

        if rows:
            out = (pd.DataFrame(rows)
                   .sort_values(by=["Score","Ticker"], ascending=[False, True])
                   .reset_index(drop=True))

            # Signal visuel + colonnes ordonnées avec Name juste après Ticker
            def to_signal(a:str)->str:
                return {"BUY":"🟢 BUY","WATCH":"🟢 WATCH","HOLD":"⚪ HOLD","REDUCE":"🟠 REDUCE","SELL":"🔴 SELL"}.get(a, a)
            out["Signal"] = out["Action"].map(to_signal)

            cols = ["Ticker","Name","Signal","Score","RSI","MACD_hist","%toHH52","VolZ20","Action"]
            out = out[[c for c in cols if c in out.columns]]

            st.success(f"Scan terminé en {elapsed:.1f}s — {len(out)} lignes")
            st.dataframe(out, use_container_width=True)

            # Export CSV des résultats
            buffer = io.StringIO()
            out.to_csv(buffer, index=False)
            st.download_button(
                "⬇️ Exporter résultats (CSV)",
                data=buffer.getvalue().encode(),
                file_name="scan_results.csv",
                mime="text/csv"
            )

            st.markdown("**Top 25 opportunités 🟢**")
            st.dataframe(out.head(25)[["Ticker","Name","Signal","Score","RSI","MACD_hist","%toHH52","VolZ20"]],
                         use_container_width=True)
        else:
            st.info("Aucun résultat (tickers invalides ou indisponibles).")

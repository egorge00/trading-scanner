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
UNIVERSE_PATH = "data/watchlist.csv"   # tout l'univers (indices)
MY_WATCHLIST_KEY = "my_watchlist_df"   # watchlist perso en session

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
        # fallback minimal
        return normalize_cols(pd.DataFrame([
            {"isin":"US0378331005","ticker":"AAPL","name":"Apple","market":"US"},
            {"isin":"FR0000120321","ticker":"OR.PA","name":"L'Oreal","market":"FR"},
        ]))

def load_my_watchlist() -> pd.DataFrame:
    if MY_WATCHLIST_KEY in st.session_state:
        return st.session_state[MY_WATCHLIST_KEY].copy()
    # par défaut vide (tu ajoutes depuis l'univers)
    df = normalize_cols(pd.DataFrame(columns=["isin","ticker","name","market"]))
    st.session_state[MY_WATCHLIST_KEY] = df.copy()
    return df

def save_my_watchlist(df: pd.DataFrame):
    st.session_state[MY_WATCHLIST_KEY] = normalize_cols(df)

def export_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

# ---- Name map basé sur l'univers (pour avoir les noms partout) ----
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

    # 1) matches exacts très prioritaires
    exact = uni[(uni["ticker"] == q_upper) | (uni["isin"] == q_upper) | (uni["name"].str.lower() == q_lower)]

    # 2) contient
    contains = uni[
        uni["ticker"].str.contains(q_upper, na=False) |
        uni["isin"].str.contains(q_upper, na=False) |
        uni["name"].str.lower().str.contains(q_lower, na=False)
    ]

    # 3) fuzzy (sur le nom) avec difflib
    names = uni["name"].dropna().astype(str).tolist()
    fuzz = difflib.get_close_matches(q, names, n=topk, cutoff=0.6)
    fuzzy = uni[uni["name"].isin(fuzz)]

    out = pd.concat([exact, contains, fuzzy], ignore_index=True).drop_duplicates(subset=["isin","ticker","name"])
    # limiter
    return out.head(topk)

# ========= Onglets =========
tab_scan, tab_single, tab_full = st.tabs(["🔎 Scanner (watchlist)", "📄 Fiche valeur", "🚀 Scanner complet"])

# --------- Onglet SCANNER (ma watchlist perso) ---------
with tab_scan:
    st.title("Scanner — Ma watchlist (perso)")
    st.caption("Ici tu gères **ta** liste perso. Ajoute depuis la base (univers complet) ou manuellement, puis scanne uniquement ces valeurs.")

    my_wl = load_my_watchlist()
    uni = load_universe()

    with st.expander("📥 Ajouter depuis la base (nom / ISIN / ticker)", expanded=True):
        q = st.text_input("Rechercher dans la base (nom, ISIN, ticker)", "")
        results = search_universe(q, topk=50) if q.strip() else uni.head(0)
        if not results.empty:
            st.dataframe(results[["ticker","name","isin","market"]], use_container_width=True, height=260)
            # sélection des lignes à ajouter
            options = results.apply(lambda r: f"{r['ticker']} — {r['name']} ({r['isin']})", axis=1).tolist()
            pick = st.multiselect("Sélectionne ce que tu veux ajouter à ta watchlist", options)
            if st.button("➕ Ajouter la sélection à ma watchlist"):
                to_add = []
                # reconstruire depuis la string
                lookup = {(f"{r['ticker']} — {r['name']} ({r['isin']})"): r for _, r in results.iterrows()}
                for p in pick:
                    r = lookup.get(p)
                    if r is not None:
                        to_add.append({"isin":r["isin"],"ticker":r["ticker"],"name":r["name"],"market":r["market"]})
                if to_add:
                    add_df = normalize_cols(pd.DataFrame(to_add))
                    my_wl = pd.concat([my_wl, add_df], ignore_index=True)
                    my_wl = my_wl.drop_duplicates(subset=["isin","ticker"]).reset_index(drop=True)
                    save_my_watchlist(my_wl)
                    st.success(f"{len(to_add)} valeur(s) ajoutée(s) à ta watchlist.")
        else:
            st.info("Tape un nom, un ISIN ou un ticker pour rechercher dans la base.")

    with st.expander("➕ Ajouter / ➖ Supprimer (manuel) / 🔍 Résoudre (clé ISIN)", expanded=False):
        colA, colB, colC = st.columns(3)
        # --- Ajouter manuel ---
        with colA:
            st.markdown("**Ajouter (manuel)**")
            new_isin = st.text_input("ISIN", key="add_isin_manual")
            new_name = st.text_input("Nom (optionnel)", key="add_name_manual")
            new_mkt  = st.text_input("Marché (optionnel)", key="add_mkt_manual")
            new_ticker = st.text_input("Ticker Yahoo (si connu)", key="add_ticker_manual")
            if st.button("Ajouter à ma watchlist (manuel)"):
                if new_isin.strip() or new_ticker.strip():
                    isin = new_isin.strip().upper()
                    ticker = new_ticker.strip().upper()
                    if not ticker and isin:
                        t = resolve_isin_to_ticker(isin, name_hint=new_name or None)
                        if t:
                            ticker = t
                    add = {"isin": isin, "ticker": ticker, "name": (new_name or "").strip(), "market": (new_mkt or "").strip()}
                    my_wl = pd.concat([my_wl, pd.DataFrame([add])], ignore_index=True)
                    my_wl = my_wl.drop_duplicates(subset=["isin","ticker"]).reset_index(drop=True)
                    save_my_watchlist(my_wl)
                    st.success(f"Ajouté. " + (f"Résolu → {ticker}" if ticker else "⚠️ ticker à résoudre"))
                else:
                    st.warning("Renseigne au moins un ISIN ou un ticker.")

        # --- Supprimer ---
        with colB:
            st.markdown("**Supprimer (dans ma watchlist)**")
            del_choice = st.selectbox("Choisir une ligne à supprimer", [""] + (my_wl.apply(lambda r: f"{r['ticker']} — {r['name']} ({r['isin']})", axis=1).tolist()))
            if st.button("Supprimer de ma watchlist"):
                if del_choice:
                    # retrouver l'ISIN/ticker à partir du libellé
                    parts = del_choice.split(" (")
                    left = parts[0] if parts else ""
                    isin = del_choice.split("(")[-1].rstrip(")") if "(" in del_choice else ""
                    ticker = left.split(" — ")[0] if " — " in left else left
                    before = len(my_wl)
                    my_wl = my_wl[~((my_wl["ticker"]==ticker) | (my_wl["isin"]==isin))].reset_index(drop=True)
                    save_my_watchlist(my_wl)
                    st.success(f"Supprimé ({before-len(my_wl)} ligne).")

        # --- Résoudre les lignes sans ticker dans ma watchlist ---
        with colC:
            st.markdown("**🔍 Résoudre (ISIN → ticker) dans ma watchlist**")
            unresolved = my_wl[(my_wl["ticker"]=="") | (my_wl["ticker"].isna())]
            if len(unresolved) > 0:
                pick = st.selectbox("ISIN à résoudre", unresolved["isin"].tolist(), key="res_isin_mywl")
                name_hint = my_wl.loc[my_wl["isin"] == pick, "name"].iloc[0] if not my_wl[my_wl["isin"] == pick].empty else ""
                if st.button("🔍 Résoudre cet ISIN (ma watchlist)"):
                    t = resolve_isin_to_ticker(pick, name_hint=name_hint)
                    if t:
                        my_wl.loc[my_wl["isin"] == pick, "ticker"] = t
                        save_my_watchlist(my_wl)
                        st.success(f"{pick} → {t}")
                    else:
                        st.warning("Aucun ticker trouvé automatiquement.")

        # Export de TA watchlist perso
        st.download_button(
            "⬇️ Télécharger MA watchlist (CSV)",
            data=export_csv_bytes(my_wl),
            file_name="my_watchlist.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.subheader("📈 Scanner ma watchlist (résultats du jour)")
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
    if rows:
    df_out = (pd.DataFrame(rows)
              .sort_values(by=["Score","Ticker"], ascending=[False, True])
              .reset_index(drop=True))
    cols = ["Ticker","Name","Score","Action","RSI","MACD_hist","%toHH52","VolZ20","Close>SMA50","SMA50>SMA200"]
    df_out = df_out[[c for c in cols if c in df_out.columns]]
    st.dataframe(df_out, use_container_width=True)

    st.markdown("**Top 10 opportunités 🟢 (ma watchlist)**")
    st.dataframe(
        df_out.head(10)[["Ticker","Name","Score","Action","RSI","MACD_hist","%toHH52","VolZ20"]],
        use_container_width=True
    )

    # --- Corbeille par ligne : suppression rapide depuis les résultats ---
    st.markdown("### 🗑️ Supprimer une valeur directement depuis les résultats")
    st.caption("Clique sur la corbeille pour retirer une valeur de **ta** watchlist, puis la page se relance.")

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
                my_wl = load_my_watchlist()
                before = len(my_wl)
                my_wl = my_wl[my_wl["ticker"] != str(r["Ticker"]).strip().upper()].reset_index(drop=True)
                save_my_watchlist(my_wl)
                st.success(f"{r['Ticker']} supprimé de ta watchlist ({before - len(my_wl)} ligne).")
                st.rerun()
else:
    st.info("Aucun résultat (aucun ticker dans ta watchlist ou tickers invalides).")

# --------- Onglet FICHE ---------
with tab_single:
    st.title("Fiche valeur (analyse individuelle)")
    ticker_input = st.text_input("Ticker Yahoo Finance (ex: AAPL, OR.PA, MC.PA)", "AAPL")

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
            needed = {"Open","High","Low","Close","Volume"}
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

# --------- Onglet 🚀 SCANNER COMPLET (univers) ---------
with tab_full:
    st.title("Scanner complet — Univers entier")
    uni = load_universe()
    base = uni.loc[uni["ticker"].astype(str).str.len() > 0].copy()

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
            def to_signal(a:str)->str:
                return {"BUY":"🟢 BUY","WATCH":"🟢 WATCH","HOLD":"⚪ HOLD","REDUCE":"🟠 REDUCE","SELL":"🔴 SELL"}.get(a, a)
            out["Signal"] = out["Action"].map(to_signal)
            cols = ["Ticker","Name","Signal","Score","RSI","MACD_hist","%toHH52","VolZ20","Action"]
            out = out[[c for c in cols if c in out.columns]]

            st.success(f"Scan terminé en {elapsed:.1f}s — {len(out)} lignes")
            st.dataframe(out, use_container_width=True)

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

import os
import sys
import io
import time
import difflib
import datetime as dt
import base64
import json
import bcrypt
import pandas as pd
import streamlit as st
import yfinance as yf
import concurrent.futures as cf
import requests

# --- rendre importable le package "api" depuis /ui ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.core.scoring import compute_kpis, compute_score
from api.core.isin_resolver import resolve_isin_to_ticker

# ---------- CONFIG ----------
st.set_page_config(page_title="Trading Scanner", layout="wide")

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
FULL_SCAN_WATCHLIST_KEY = "full_scan_watchlist_df"

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

def load_full_scan_watchlist() -> pd.DataFrame:
    if FULL_SCAN_WATCHLIST_KEY in st.session_state:
        return st.session_state[FULL_SCAN_WATCHLIST_KEY].copy()
    df = pd.DataFrame(columns=["isin", "ticker", "name", "market"])
    st.session_state[FULL_SCAN_WATCHLIST_KEY] = df.copy()
    return df

def save_full_scan_watchlist(df: pd.DataFrame):
    cols = ["isin", "ticker", "name", "market"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols].copy()
    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["name"] = df["name"].astype(str).str.strip()
    df["market"] = df["market"].astype(str).str.strip()
    st.session_state[FULL_SCAN_WATCHLIST_KEY] = df.copy()

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
        df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
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
    res = score_one(ticker)
    if not res: 
        return None
    return res.get("Action",""), float(res.get("Score", 0))

# ========= Onglets =========
tab_full, tab_scan, tab_single, tab_pos = st.tabs(
    ["🚀 Scanner complet", "🔎 Scanner (watchlist)", "📄 Fiche valeur", "💼 Positions"]
)
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

    st.divider()
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

# --------- Onglet 🚀 SCANNER COMPLET (univers) ---------

with tab_full:
    st.title("Scanner complet — Univers entier")

    with st.form("full_scan_form", clear_on_submit=False):
        colf1, colf2, colf3, colf4 = st.columns([1, 2, 1, 1])

        uni = load_universe()
        base = uni.loc[uni["ticker"].astype(str).str.len() > 0].copy()

        markets = sorted([m for m in base["market"].unique() if isinstance(m, str) and m])
        sel_markets = colf1.multiselect("Marchés", options=["(Tous)"] + markets, default=["(Tous)"])
        query = colf2.text_input("Rechercher (ticker ou nom)", "")
        limit = colf3.number_input("Limite", min_value=10, max_value=2000, value=300, step=10)

        do_scan = colf4.form_submit_button("🚀 Lancer le scan complet")

    dfv = base.copy()
    if sel_markets and "(Tous)" not in sel_markets:
        dfv = dfv[dfv["market"].isin(sel_markets)]
    if query.strip():
        q = query.strip().lower()
        dfv = dfv[
            dfv["ticker"].str.lower().str.contains(q)
            | dfv["name"].str.lower().str.contains(q)
        ]
    tickers = dfv["ticker"].dropna().astype(str).str.strip().tolist()[: int(limit)]
    st.caption(f"{len(tickers)} tickers sélectionnés pour le scan.")

    # ---------- 2) RÉSULTATS DU SCAN ----------
    if do_scan:
        start = time.time()
        rows = []
        done = 0
        prog = st.progress(0)

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
            out = (
                pd.DataFrame(rows)
                .sort_values(by=["Score", "Ticker"], ascending=[False, True])
                .reset_index(drop=True)
            )

            def to_signal(a: str) -> str:
                return {
                    "BUY": "🟢 BUY",
                    "WATCH": "🟢 WATCH",
                    "HOLD": "⚪ HOLD",
                    "REDUCE": "🟠 REDUCE",
                    "SELL": "🔴 SELL",
                }.get(a, a)

            out["Signal"] = out["Action"].map(to_signal)

            # --- Sauvegarde journalière ---
            today = dt.date.today().strftime("%Y-%m-%d")
            os.makedirs("data/scans", exist_ok=True)
            daily_path = f"data/scans/{today}.csv"
            try:
                out.to_csv(daily_path, index=False)
                st.caption(f"Résultats sauvegardés : {daily_path}")
            except Exception as e:
                st.warning(f"Impossible de sauvegarder le scan du jour: {e}")

            # --- Comparaison classement vs veille ---
            yesterday = (dt.date.today() - dt.timedelta(days=1)).strftime("%Y-%m-%d")
            prev_path = f"data/scans/{yesterday}.csv"

            try:
                if os.path.exists(prev_path):
                    prev = pd.read_csv(prev_path)

                    out["Rank_today"] = out["Score"].rank(ascending=False, method="min")
                    if "Score" in prev.columns and "Ticker" in prev.columns:
                        prev = prev.copy()
                        prev["Rank_yesterday"] = prev["Score"].rank(ascending=False, method="min")
                        merged = out.merge(prev[["Ticker", "Rank_yesterday"]], on="Ticker", how="left")

                        def trend_from_ranks(row):
                            ry = row.get("Rank_yesterday", float("nan"))
                            rt = row.get("Rank_today", float("nan"))
                            if pd.isna(ry):
                                return "🆕"
                            if rt < ry:
                                return "↗️"
                            if rt > ry:
                                return "↘️"
                            return "🟰"

                        merged["Trend"] = merged.apply(trend_from_ranks, axis=1)
                        out = merged
                    else:
                        st.info("Fichier de veille trouvé mais colonnes manquantes (Ticker/Score).")
                else:
                    st.info("Pas de fichier de veille — aucune comparaison de classement affichée.")
            except Exception as e:
                st.warning(f"Comparaison classement vs veille impossible: {e}")

            cols = [
                "Ticker",
                "Name",
                "Signal",
                "Score",
                "Trend",
                "RSI",
                "MACD_hist",
                "%toHH52",
                "VolZ20",
                "Action",
                "Close>SMA50",
                "SMA50>SMA200",
            ]
            out = out[[c for c in cols if c in out.columns]]

            st.success(f"Scan terminé en {elapsed:.1f}s — {len(out)} lignes")
            st.dataframe(out, use_container_width=True)

            buffer = io.StringIO()
            out.to_csv(buffer, index=False)
            st.download_button(
                "⬇️ Exporter résultats (CSV)",
                data=buffer.getvalue().encode(),
                file_name="scan_results.csv",
                mime="text/csv",
            )
        else:
            st.info("Aucun résultat (tickers invalides ou indisponibles).")

    st.divider()

    # ---------- 3) SÉLECTION ----------
    st.subheader("Sélectionner des valeurs à suivre")
    full_wl = load_full_scan_watchlist()

    cand = dfv[["isin", "ticker", "name", "market"]].dropna().copy()
    if not cand.empty:
        cand["label"] = cand.apply(
            lambda r: f"{r['ticker']} — {r['name']} ({r['isin']})", axis=1
        )
        picked = st.multiselect(
            "Choisis des valeurs dans la liste filtrée ci-dessus",
            options=cand["label"].tolist(),
            default=[],
            key="full_scan_pick",
        )
        if st.button("Ajouter à la watchlist du Scanner complet"):
            if picked:
                lookup = {row["label"]: row for _, row in cand.iterrows()}
                to_add = []
                for p in picked:
                    r = lookup.get(p)
                    if r is not None:
                        to_add.append(
                            {
                                "isin": r["isin"],
                                "ticker": r["ticker"],
                                "name": r["name"],
                                "market": r["market"],
                            }
                        )
                if to_add:
                    add_df = pd.DataFrame(to_add)
                    full_wl = pd.concat([full_wl, add_df], ignore_index=True)
                    full_wl = full_wl.drop_duplicates(subset=["ticker", "isin"]).reset_index(drop=True)
                    save_full_scan_watchlist(full_wl)
                    st.success(f"{len(to_add)} valeur(s) ajoutée(s).")
            else:
                st.info("Sélectionne au moins une valeur.")
    else:
        st.info("Aucune valeur dans la liste filtrée actuelle.")

    st.divider()

    # ---------- 4) WATCHLIST SCORÉE ----------
    st.subheader("Watchlist du Scanner complet (sélection utilisateur)")
    full_wl = load_full_scan_watchlist()

    if full_wl.empty:
        st.info(
            "Ta watchlist du Scanner complet est vide. Ajoute des valeurs depuis la sélection ci-dessus."
        )
    else:
        rows_wl = []
        for tkr in full_wl["ticker"].dropna().astype(str).str.strip().unique().tolist():
            res = score_one(tkr)
            if res:
                rows_wl.append(res)

        if rows_wl:
            df_wl = (
                pd.DataFrame(rows_wl)
                .sort_values(by=["Score", "Ticker"], ascending=[False, True])
                .reset_index(drop=True)
            )
            cols = [
                "Ticker",
                "Name",
                "Score",
                "Action",
                "RSI",
                "MACD_hist",
                "%toHH52",
                "VolZ20",
                "Close>SMA50",
                "SMA50>SMA200",
            ]
            df_wl = df_wl[[c for c in cols if c in df_wl.columns]]
            st.dataframe(df_wl, use_container_width=True)

            st.markdown("#### Retirer une valeur de cette watchlist")
            for i, r in df_wl.iterrows():
                c1, c2, c3 = st.columns([5, 4, 1])
                with c1:
                    st.write(f"**{r['Ticker']}** — {r.get('Name', '')}")
                with c2:
                    st.write(f"Score: {r['Score']} | Action: {r['Action']}")
                with c3:
                    if st.button("🗑️", key=f"full_wl_del_{i}_{r['Ticker']}"):
                        wl = load_full_scan_watchlist()
                        wl = wl[wl["ticker"] != str(r["Ticker"]).strip().upper()].reset_index(
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

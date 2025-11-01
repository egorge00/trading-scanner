import os
import sys
import time
import typing as t
import pandas as pd
import requests

EXCLUDE_TICKERS = {"BRO", "BRK.B"}  # valeurs à force-exclure

W_SNP = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
W_CAC = "https://en.wikipedia.org/wiki/CAC_40"
W_ESTOXX = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
W_FTSE100 = "https://en.wikipedia.org/wiki/FTSE_100_Index"
W_DAX = "https://en.wikipedia.org/wiki/DAX"

UA = {"User-Agent": "Mozilla/5.0 (compatible; WatchlistBot/1.0; +https://example.com)"}

def fetch_tables(url: str, retries: int = 3, sleep_s: float = 1.5) -> list[pd.DataFrame]:
    """Télécharge une page et renvoie toutes les tables HTML en DataFrames.
    Robuste aux erreurs réseau & aux 429 via retries."""
    last_exc = None
    for i in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=30, headers=UA)
            r.raise_for_status()
            tables = pd.read_html(r.text)
            print(f"[OK] {url} -> {len(tables)} table(s)")
            return tables
        except Exception as e:
            last_exc = e
            print(f"[WARN] fetch {url} (try {i}/{retries}) failed: {e}")
            time.sleep(sleep_s * i)
    print(f"[ERROR] giving up on {url}: {last_exc}")
    return []

def build_sp500() -> pd.DataFrame:
    tabs = fetch_tables(W_SNP)
    if not tabs:
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    # La 1ère table est la liste des constituants
    t = tabs[0].copy()
    # colonnes qui bougent rarement
    col_map = {"Symbol":"ticker","Security":"name"}
    for k in list(col_map.keys()):
        if k not in t.columns:
            # fallback: ex. 'Symbol' parfois nommé 'Ticker symbol'
            if k == "Symbol" and "Ticker symbol" in t.columns:
                col_map["Ticker symbol"] = "ticker"
                col_map.pop("Symbol")
    t = t.rename(columns=col_map)
    if "ticker" not in t.columns or "name" not in t.columns:
        print("[WARN] S&P500: colonnes inattendues, on abandonne cette source.")
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    t["ticker"] = t["ticker"].astype(str).str.upper().str.strip()
    t["name"] = t["name"].astype(str).str.strip()
    t["isin"] = ""
    t["market"] = "US"
    out = t[["isin","ticker","name","market"]].dropna()
    print(f"[INFO] S&P500 rows: {len(out)}")
    return out

def build_cac40() -> pd.DataFrame:
    tabs = fetch_tables(W_CAC)
    if not tabs:
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    tb = next((x for x in tabs if "Company" in x.columns and ("Ticker" in x.columns or "Ticker symbol" in x.columns)), None)
    if tb is None:
        print("[WARN] CAC40: table non trouvée")
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    col_tick = "Ticker" if "Ticker" in tb.columns else "Ticker symbol"
    t = tb.rename(columns={"Company":"name", col_tick:"ticker"}).copy()
    t["ticker"] = (t["ticker"].astype(str)
                   .str.replace(r"^EPA:", "", regex=True)
                   .str.strip())
    t["name"] = t["name"].astype(str).str.strip()
    t["isin"] = ""
    t["market"] = "FR"
    out = t[["isin","ticker","name","market"]].dropna()
    print(f"[INFO] CAC40 rows: {len(out)}")
    return out

def build_eurostoxx50() -> pd.DataFrame:
    tabs = fetch_tables(W_ESTOXX)
    if not tabs:
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    tb = next((x for x in tabs if "Company" in x.columns), None)
    if tb is None:
        print("[WARN] ESTOXX50: table non trouvée")
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    t = tb.rename(columns={"Company":"name", "Ticker":"ticker"}).copy()
    def to_yahoo(sym: t.Any) -> str:
        s = str(sym).strip().upper()
        if any(s.endswith(suf) for suf in (".PA",".AS",".BR",".MI",".MC",".DE",".F",".BE",".SW",".VI",".LS",".IR",".OL",".HE",".VX",".DK",".FI",".NO",".SE",".L")):
            return s
        # pas de suffixe → on laisse vide (Option A)
        return ""
    if "ticker" in t.columns:
        t["ticker"] = t["ticker"].apply(to_yahoo)
    else:
        t["ticker"] = ""
    t["name"] = t["name"].astype(str).str.strip()
    t["isin"] = ""
    def guess_market(tk: str) -> str:
        mapping = {
            ".PA":"FR",".DE":"DE",".F":"DE",".AS":"NL",".MI":"IT",".MC":"ES",".BE":"BE",
            ".SW":"CH",".VX":"CH",".LS":"PT",".DK":"DK",".FI":"FI",".NO":"NO",".SE":"SE",".L":"UK"
        }
        for suf, mkt in mapping.items():
            if tk.endswith(suf): return mkt
        return "EU"
    t["market"] = t["ticker"].apply(guess_market)
    out = t[["isin","ticker","name","market"]].drop_duplicates(subset=["name"]).reset_index(drop=True)
    print(f"[INFO] EURO STOXX 50 rows: {len(out)}")
    return out

def build_ftse100() -> pd.DataFrame:
    tabs = fetch_tables(W_FTSE100)
    if not tabs:
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    tb = None
    for x in tabs:
        cols = set(map(str, x.columns))
        if "Company" in cols and ({"EPIC", "Ticker", "Ticker symbol"} & cols):
            tb = x
            break
    if tb is None:
        print("[WARN] FTSE100: table non trouvée")
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    col_tick = "EPIC" if "EPIC" in tb.columns else ("Ticker" if "Ticker" in tb.columns else "Ticker symbol")
    t = tb.rename(columns={"Company":"name", col_tick:"ticker"}).copy()
    t["ticker"] = (t["ticker"].astype(str).str.upper().str.strip()
                   .str.replace(r"\s+", "", regex=True))
    # Yahoo Londres → suffixe .L
    t["ticker"] = t["ticker"].apply(lambda s: f"{s}.L" if s and not s.endswith(".L") else s)
    t["name"] = t["name"].astype(str).str.strip()
    t["isin"] = ""
    t["market"] = "UK"
    out = t[["isin","ticker","name","market"]].dropna()
    print(f"[INFO] FTSE100 rows: {len(out)}")
    return out

def build_dax40() -> pd.DataFrame:
    tabs = fetch_tables(W_DAX)
    if not tabs:
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    tb = None
    for x in tabs:
        cols = set(map(str, x.columns))
        if "Company" in cols and ({"Ticker", "Ticker symbol", "Symbol"} & cols):
            tb = x
            break
    if tb is None:
        # fallback: noms seuls
        tb2 = next((x for x in tabs if "Company" in set(map(str, x.columns))), None)
        if tb2 is None:
            print("[WARN] DAX40: table non trouvée")
            return pd.DataFrame(columns=["isin","ticker","name","market"])
        t2 = tb2.rename(columns={"Company":"name"}).copy()
        t2["ticker"] = ""
        t2["isin"] = ""
        t2["market"] = "DE"
        out2 = t2[["isin","ticker","name","market"]]
        print(f"[INFO] DAX40 rows (fallback, sans tickers): {len(out2)}")
        return out2
    col_tick = "Ticker" if "Ticker" in tb.columns else ("Ticker symbol" if "Ticker symbol" in tb.columns else "Symbol")
    t = tb.rename(columns={"Company":"name", col_tick:"ticker"}).copy()
    t["ticker"] = t["ticker"].astype(str).str.upper().str.strip()
    t["ticker"] = t["ticker"].apply(lambda s: f"{s}.DE" if s and "." not in s else s)
    t["name"] = t["name"].astype(str).str.strip()
    t["isin"] = ""
    t["market"] = "DE"
    out = t[["isin","ticker","name","market"]].dropna()
    print(f"[INFO] DAX40 rows: {len(out)}")
    return out

def main():
    os.makedirs("data", exist_ok=True)

    frames = []
    for builder, label in [
        (build_sp500, "S&P500"),
        (build_cac40, "CAC40"),
        (build_eurostoxx50, "EUROSTOXX50"),
        (build_ftse100, "FTSE100"),
        (build_dax40, "DAX40"),
    ]:
        try:
            df = builder()
            print(f"[SUM] {label}: {len(df)} lignes")
            frames.append(df)
        except Exception as e:
            print(f"[ERROR] {label} failed: {e}")

    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["isin","ticker","name","market"])
    # Nettoyage & dédup
    for c in ("isin","ticker","name","market"):
        if c in all_df.columns:
            all_df[c] = all_df[c].astype(str).str.strip()
    all_df = all_df.drop_duplicates(subset=["ticker","name"]).reset_index(drop=True)
    all_df["isin"] = all_df.get("isin", "").fillna("")

    if "ticker" in all_df.columns:
        all_df["ticker"] = all_df["ticker"].astype(str).str.strip().str.upper()
        all_df = all_df[~all_df["ticker"].isin(EXCLUDE_TICKERS)].reset_index(drop=True)

    all_df.to_csv("data/watchlist.csv", index=False)
    print(f"[DONE] data/watchlist.csv écrit ({len(all_df)} lignes)")
    # On sort avec code 0 même si des sources manquent, pour éviter l'échec du job
    sys.exit(0)

if __name__ == "__main__":
    main()

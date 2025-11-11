import os
import re
import sys
import time
import typing as t
import pandas as pd
import requests

ETF_STATIC = [
    # --- US Broad & Style ---
    {"isin": "", "ticker": "SPY", "name": "SPDR S&P 500", "market": "ETF"},
    {"isin": "", "ticker": "IVV", "name": "iShares Core S&P 500", "market": "ETF"},
    {"isin": "", "ticker": "VOO", "name": "Vanguard S&P 500", "market": "ETF"},
    {"isin": "", "ticker": "VTI", "name": "Vanguard Total Stock Market", "market": "ETF"},
    {"isin": "", "ticker": "QQQ", "name": "Invesco QQQ Trust", "market": "ETF"},
    {"isin": "", "ticker": "IWM", "name": "iShares Russell 2000", "market": "ETF"},
    {"isin": "", "ticker": "DIA", "name": "SPDR Dow Jones Industrial Average", "market": "ETF"},
    {"isin": "", "ticker": "MTUM", "name": "iShares MSCI USA Momentum", "market": "ETF"},
    {"isin": "", "ticker": "QUAL", "name": "iShares MSCI USA Quality", "market": "ETF"},
    {"isin": "", "ticker": "VLUE", "name": "iShares MSCI USA Value Factor", "market": "ETF"},
    {"isin": "", "ticker": "SIZE", "name": "iShares MSCI USA Size Factor", "market": "ETF"},

    # --- US Sectors (SPDR) ---
    {"isin": "", "ticker": "XLK", "name": "Technology Select Sector SPDR", "market": "ETF"},
    {"isin": "", "ticker": "XLY", "name": "Consumer Discretionary Select SPDR", "market": "ETF"},
    {"isin": "", "ticker": "XLP", "name": "Consumer Staples Select SPDR", "market": "ETF"},
    {"isin": "", "ticker": "XLE", "name": "Energy Select Sector SPDR", "market": "ETF"},
    {"isin": "", "ticker": "XLF", "name": "Financial Select Sector SPDR", "market": "ETF"},
    {"isin": "", "ticker": "XLV", "name": "Health Care Select Sector SPDR", "market": "ETF"},
    {"isin": "", "ticker": "XLI", "name": "Industrial Select Sector SPDR", "market": "ETF"},
    {"isin": "", "ticker": "XLU", "name": "Utilities Select Sector SPDR", "market": "ETF"},
    {"isin": "", "ticker": "XLB", "name": "Materials Select Sector SPDR", "market": "ETF"},
    {"isin": "", "ticker": "XLRE", "name": "Real Estate Select Sector SPDR", "market": "ETF"},

    # --- US Bonds / Gold / Others ---
    {"isin": "", "ticker": "LQD", "name": "iShares iBoxx $ Inv Grade Corporate Bd", "market": "ETF"},
    {"isin": "", "ticker": "HYG", "name": "iShares iBoxx $ High Yield Corporate", "market": "ETF"},
    {"isin": "", "ticker": "TLT", "name": "iShares 20+ Year Treasury", "market": "ETF"},
    {"isin": "", "ticker": "IEF", "name": "iShares 7-10 Year Treasury", "market": "ETF"},
    {"isin": "", "ticker": "SHY", "name": "iShares 1-3 Year Treasury", "market": "ETF"},
    {"isin": "", "ticker": "GLD", "name": "SPDR Gold Shares", "market": "ETF"},
    {"isin": "", "ticker": "IAU", "name": "iShares Gold Trust", "market": "ETF"},
    {"isin": "", "ticker": "SLV", "name": "iShares Silver Trust", "market": "ETF"},
    {"isin": "", "ticker": "USO", "name": "United States Oil", "market": "ETF"},

    # --- Global / International (US listings) ---
    {"isin": "", "ticker": "VEA", "name": "Vanguard FTSE Developed Markets", "market": "ETF"},
    {"isin": "", "ticker": "VWO", "name": "Vanguard FTSE Emerging Markets", "market": "ETF"},
    {"isin": "", "ticker": "IEFA", "name": "iShares Core MSCI EAFE", "market": "ETF"},
    {"isin": "", "ticker": "IEMG", "name": "iShares Core MSCI Emerging Markets", "market": "ETF"},
    {"isin": "", "ticker": "VT", "name": "Vanguard Total World Stock", "market": "ETF"},

    # --- Europe-domiciled UCITS (Yahoo suffix .L/.DE/.AS...) ---
    {"isin": "", "ticker": "CSPX.L", "name": "iShares Core S&P 500 UCITS (GBP) Acc", "market": "ETF"},
    {"isin": "", "ticker": "SXR8.DE", "name": "iShares Core S&P 500 UCITS (EUR) Acc", "market": "ETF"},
    {"isin": "", "ticker": "VUAA.AS", "name": "Vanguard S&P 500 UCITS (EUR) Acc", "market": "ETF"},
    {"isin": "", "ticker": "VWCE.DE", "name": "Vanguard FTSE All-World UCITS (EUR) Acc", "market": "ETF"},
    {"isin": "", "ticker": "EQQQ.L", "name": "Invesco EQQQ NASDAQ-100 UCITS (GBP)", "market": "ETF"},
    {"isin": "", "ticker": "IUSA.L", "name": "iShares S&P 500 UCITS (GBP) Dist", "market": "ETF"},
    {"isin": "", "ticker": "XD9U.DE", "name": "Xtrackers MSCI USA UCITS (EUR) Acc", "market": "ETF"},
    {"isin": "", "ticker": "VUKE.L", "name": "Vanguard FTSE 100 UCITS (GBP) Dist", "market": "ETF"},
    {"isin": "", "ticker": "IMEU.L", "name": "iShares Core MSCI Europe UCITS (GBP)", "market": "ETF"},
    {"isin": "", "ticker": "EIMI.L", "name": "iShares Core MSCI EM IMI UCITS (GBP)", "market": "ETF"},
]

W_SNP = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
W_CAC = "https://en.wikipedia.org/wiki/CAC_40"
W_ESTOXX = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
W_FTSE100 = "https://en.wikipedia.org/wiki/FTSE_100_Index"
W_DAX = "https://en.wikipedia.org/wiki/DAX"

UA = {"User-Agent": "Mozilla/5.0 (compatible; WatchlistBot/1.0; +https://example.com)"}


def _clean_name(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

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
    table = tabs[0].copy()
    # colonnes qui bougent rarement
    col_map = {"Symbol":"ticker","Security":"name"}
    for k in list(col_map.keys()):
        if k not in table.columns:
            # fallback: ex. 'Symbol' parfois nommé 'Ticker symbol'
            if k == "Symbol" and "Ticker symbol" in table.columns:
                col_map["Ticker symbol"] = "ticker"
                col_map.pop("Symbol")
    table = table.rename(columns=col_map)
    if "ticker" not in table.columns or "name" not in table.columns:
        print("[WARN] S&P500: colonnes inattendues, on abandonne cette source.")
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    table["ticker"] = table["ticker"].astype(str).str.upper().str.strip()
    table["name"] = table["name"].astype(str).str.strip()
    table["isin"] = ""
    table["market"] = "US"
    out = table[["isin","ticker","name","market"]].dropna()
    print(f"[INFO] S&P500 rows: {len(out)}")
    return out

def build_cac40() -> pd.DataFrame:
    tabs = fetch_tables(W_CAC)
    if not tabs:
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    table = next((x for x in tabs if "Company" in x.columns and ("Ticker" in x.columns or "Ticker symbol" in x.columns)), None)
    if table is None:
        print("[WARN] CAC40: table non trouvée")
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    col_tick = "Ticker" if "Ticker" in table.columns else "Ticker symbol"
    table = table.rename(columns={"Company":"name", col_tick:"ticker"}).copy()
    table["ticker"] = (
        table["ticker"].astype(str)
        .str.replace(r"^EPA:", "", regex=True)
        .str.strip()
    )
    table["name"] = table["name"].astype(str).str.strip()
    table["isin"] = ""
    table["market"] = "FR"
    out = table[["isin","ticker","name","market"]].dropna()
    print(f"[INFO] CAC40 rows: {len(out)}")
    return out

def build_eurostoxx50() -> pd.DataFrame:
    tabs = fetch_tables(W_ESTOXX)
    if not tabs:
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    table = next((x for x in tabs if "Company" in x.columns), None)
    if table is None:
        print("[WARN] ESTOXX50: table non trouvée")
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    table = table.rename(columns={"Company":"name", "Ticker":"ticker"}).copy()
    def to_yahoo(sym: t.Any) -> str:
        s = str(sym).strip().upper()
        if any(s.endswith(suf) for suf in (".PA",".AS",".BR",".MI",".MC",".DE",".F",".BE",".SW",".VI",".LS",".IR",".OL",".HE",".VX",".DK",".FI",".NO",".SE",".L")):
            return s
        # pas de suffixe → on laisse vide (Option A)
        return ""
    if "ticker" in table.columns:
        table["ticker"] = table["ticker"].apply(to_yahoo)
    else:
        table["ticker"] = ""
    table["name"] = table["name"].astype(str).str.strip()
    table["isin"] = ""
    def guess_market(tk: str) -> str:
        mapping = {
            ".PA":"FR",".DE":"DE",".F":"DE",".AS":"NL",".MI":"IT",".MC":"ES",".BE":"BE",
            ".SW":"CH",".VX":"CH",".LS":"PT",".DK":"DK",".FI":"FI",".NO":"NO",".SE":"SE",".L":"UK"
        }
        for suf, mkt in mapping.items():
            if tk.endswith(suf):
                return mkt
        return "EU"
    table["market"] = table["ticker"].apply(guess_market)
    out = table[["isin","ticker","name","market"]].drop_duplicates(subset=["name"]).reset_index(drop=True)
    print(f"[INFO] EURO STOXX 50 rows: {len(out)}")
    return out

def build_ftse100() -> pd.DataFrame:
    tabs = fetch_tables(W_FTSE100)
    if not tabs:
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    table = None
    for x in tabs:
        cols = set(map(str, x.columns))
        if "Company" in cols and ({"EPIC", "Ticker", "Ticker symbol"} & cols):
            table = x
            break
    if table is None:
        print("[WARN] FTSE100: table non trouvée")
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    col_tick = "EPIC" if "EPIC" in table.columns else ("Ticker" if "Ticker" in table.columns else "Ticker symbol")
    table = table.rename(columns={"Company":"name", col_tick:"ticker"}).copy()
    table["ticker"] = (
        table["ticker"].astype(str).str.upper().str.strip().str.replace(r"\s+", "", regex=True)
    )
    # Yahoo Londres → suffixe .L
    table["ticker"] = table["ticker"].apply(lambda s: f"{s}.L" if s and not s.endswith(".L") else s)
    table["name"] = table["name"].astype(str).str.strip()
    table["isin"] = ""
    table["market"] = "UK"
    out = table[["isin","ticker","name","market"]].dropna()
    print(f"[INFO] FTSE100 rows: {len(out)}")
    return out

def build_dax40() -> pd.DataFrame:
    tabs = fetch_tables(W_DAX)
    if not tabs:
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    table = None
    for x in tabs:
        cols = set(map(str, x.columns))
        if "Company" in cols and ({"Ticker", "Ticker symbol", "Symbol"} & cols):
            table = x
            break
    if table is None:
        # fallback: noms seuls
        tb2 = next((x for x in tabs if "Company" in set(map(str, x.columns))), None)
        if tb2 is None:
            print("[WARN] DAX40: table non trouvée")
            return pd.DataFrame(columns=["isin","ticker","name","market"])
        table_fallback = tb2.rename(columns={"Company":"name"}).copy()
        table_fallback["ticker"] = ""
        table_fallback["isin"] = ""
        table_fallback["market"] = "DE"
        out2 = table_fallback[["isin","ticker","name","market"]]
        print(f"[INFO] DAX40 rows (fallback, sans tickers): {len(out2)}")
        return out2
    col_tick = "Ticker" if "Ticker" in table.columns else ("Ticker symbol" if "Ticker symbol" in table.columns else "Symbol")
    table = table.rename(columns={"Company":"name", col_tick:"ticker"}).copy()
    table["ticker"] = table["ticker"].astype(str).str.upper().str.strip()
    table["ticker"] = table["ticker"].apply(lambda s: f"{s}.DE" if s and "." not in s else s)
    table["name"] = table["name"].astype(str).str.strip()
    table["isin"] = ""
    table["market"] = "DE"
    out = table[["isin","ticker","name","market"]].dropna()
    print(f"[INFO] DAX40 rows: {len(out)}")
    return out


def fetch_nikkei225() -> pd.DataFrame:
    """Récupère les constituants du Nikkei 225 et normalise les colonnes."""

    url = "https://en.wikipedia.org/wiki/Nikkei_225"
    try:
        tables = pd.read_html(url, flavor="bs4")
    except Exception as exc:
        print(f"[WARN] Nikkei225 read_html failed: {exc}")
        tables = []

    candidate = None
    for table in tables:
        cols = {str(col).lower(): col for col in table.columns}
        has_code = any(key in cols for key in ["code", "ticker"])
        has_name = any(key in cols for key in ["company", "company name", "name"])
        if has_code and has_name:
            candidate = table.rename(
                columns={
                    cols.get("code", cols.get("ticker")): "Code",
                    cols.get("company", cols.get("company name", cols.get("name"))): "Company",
                }
            )
            break

    if candidate is None or candidate.empty:
        print("[WARN] Nikkei225 table not found")
        return pd.DataFrame(columns=["isin", "ticker", "name", "market"])

    candidate["ticker"] = (
        candidate["Code"].astype(str).str.replace(".T", "", regex=False).str.strip() + ".T"
    )
    candidate["name"] = candidate["Company"].astype(str).map(_clean_name)
    candidate["market"] = "JP"
    candidate["isin"] = ""

    out = (
        candidate[["isin", "ticker", "name", "market"]]
        .drop_duplicates(subset=["ticker"])
        .reset_index(drop=True)
    )
    print(f"[INFO] Nikkei225 rows: {len(out)}")
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
        (fetch_nikkei225, "NIKKEI225"),
    ]:
        try:
            df = builder()
            print(f"[SUM] {label}: {len(df)} lignes")
            frames.append(df)
        except Exception as e:
            print(f"[ERROR] {label} failed: {e}")

    df_final = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["isin","ticker","name","market"])

    # Nettoyage & dédup
    for c in ("isin", "ticker", "name", "market"):
        if c in df_final.columns:
            df_final[c] = df_final[c].astype(str).str.strip()
    if {"ticker", "name"} <= set(df_final.columns):
        df_final = df_final.drop_duplicates(subset=["ticker", "name"]).reset_index(drop=True)
    for col in ("isin", "ticker", "name", "market"):
        if col not in df_final.columns:
            df_final[col] = ""
    df_final["isin"] = df_final["isin"].fillna("")
    df_final["ticker"] = df_final["ticker"].fillna("").astype(str).str.strip().str.upper()

    etf_df = pd.DataFrame(ETF_STATIC)
    for c in ["isin", "ticker", "name", "market"]:
        if c not in etf_df.columns:
            etf_df[c] = ""
    etf_df["isin"] = etf_df["isin"].fillna("")
    etf_df["ticker"] = etf_df["ticker"].astype(str).str.strip().str.upper()
    etf_df["name"] = etf_df["name"].astype(str)
    etf_df["market"] = "ETF"

    # --- Exclusion manuelle de tickers problématiques ---
    EXCLUDE_TICKERS = {"BRK.B", "BRO", "BF.B"}
    df_final = df_final[~df_final["ticker"].isin(EXCLUDE_TICKERS)].reset_index(drop=True)
    df_final = df_final[~df_final["isin"].isin(EXCLUDE_TICKERS)].reset_index(drop=True)
    etf_df = etf_df[~etf_df["ticker"].isin(EXCLUDE_TICKERS)].reset_index(drop=True)
    etf_df = etf_df[~etf_df["isin"].isin(EXCLUDE_TICKERS)].reset_index(drop=True)

    keep_cols = ["isin", "ticker", "name", "market"]
    df_final = pd.concat([df_final[keep_cols], etf_df[keep_cols]], ignore_index=True)
    df_final = df_final.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    df_final.loc[df_final["ticker"].isin(etf_df["ticker"]), "market"] = "ETF"

    csv_path = "data/watchlist.csv"
    df_final.to_csv(csv_path, index=False)
    print(f"✅ watchlist.csv mis à jour : {len(df_final)} lignes (incl. ETF)")
    # On sort avec code 0 même si des sources manquent, pour éviter l'échec du job
    sys.exit(0)

if __name__ == "__main__":
    main()

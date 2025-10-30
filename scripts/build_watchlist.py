import pandas as pd
import requests

W_SNP = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
W_CAC = "https://en.wikipedia.org/wiki/CAC_40"
W_ESTOXX = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
W_FTSE100 = "https://en.wikipedia.org/wiki/FTSE_100_Index"
W_DAX = "https://en.wikipedia.org/wiki/DAX"

def fetch_tables(url: str) -> list[pd.DataFrame]:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_html(r.text)

def build_sp500() -> pd.DataFrame:
    t = fetch_tables(W_SNP)[0].copy()  # constituents table
    t = t.rename(columns={"Symbol":"ticker","Security":"name"})
    t["ticker"] = t["ticker"].astype(str).str.upper().str.strip()
    t["name"] = t["name"].astype(str).str.strip()
    t["isin"] = ""
    t["market"] = "US"
    return t[["isin","ticker","name","market"]]

def build_cac40() -> pd.DataFrame:
    tables = fetch_tables(W_CAC)
    tb = next((x for x in tables if "Company" in x.columns and "Ticker" in x.columns), None)
    if tb is None:
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    t = tb.rename(columns={"Company":"name","Ticker":"ticker"}).copy()
    t["ticker"] = (t["ticker"].astype(str)
                   .str.replace(r"^EPA:", "", regex=True)
                   .str.strip())
    t["name"] = t["name"].astype(str).str.strip()
    t["isin"] = ""
    t["market"] = "FR"
    return t[["isin","ticker","name","market"]]

def build_eurostoxx50() -> pd.DataFrame:
    tables = fetch_tables(W_ESTOXX)
    tb = next((x for x in tables if "Company" in x.columns), None)
    if tb is None:
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    t = tb.rename(columns={"Company":"name", "Ticker":"ticker"}).copy()
    # Heuristique Yahoo: garder ticker s’il ressemble à un suffixe de place connu, sinon vide
    def to_yahoo(sym: str) -> str:
        s = str(sym).strip().upper()
        if any(s.endswith(suf) for suf in (".PA",".AS",".BR",".MI",".MC",".DE",".F",".BE",".SW",".VI",".LS",".IR",".OL",".HE",".VX",".DK",".FI",".NO",".SE")):
            return s
        return ""
    t["ticker"] = t["ticker"].apply(to_yahoo) if "ticker" in t.columns else ""
    t["name"] = t["name"].astype(str).str.strip()
    t["isin"] = ""
    def guess_market(tk: str) -> str:
        if tk.endswith(".PA"): return "FR"
        if tk.endswith(".DE") or tk.endswith(".F"): return "DE"
        if tk.endswith(".AS"): return "NL"
        if tk.endswith(".MI"): return "IT"
        if tk.endswith(".MC"): return "ES"
        if tk.endswith(".BE"): return "BE"
        if tk.endswith(".SW") or tk.endswith(".VX"): return "CH"
        if tk.endswith(".LS"): return "PT"
        if tk.endswith(".DK"): return "DK"
        if tk.endswith(".FI"): return "FI"
        if tk.endswith(".NO"): return "NO"
        if tk.endswith(".SE"): return "SE"
        return "EU"
    t["market"] = t["ticker"].apply(guess_market)
    out = t[["isin","ticker","name","market"]].drop_duplicates(subset=["name"]).reset_index(drop=True)
    return out

def build_ftse100() -> pd.DataFrame:
    tables = fetch_tables(W_FTSE100)
    # Rechercher une table avec 'Company' et un champ de ticker (souvent 'EPIC', 'Ticker' ou 'Ticker symbol')
    tb = None
    for x in tables:
        cols = set(map(str, x.columns))
        if "Company" in cols and ({"EPIC", "Ticker", "Ticker symbol"} & cols):
            tb = x
            break
    if tb is None:
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    col_tick = "EPIC" if "EPIC" in tb.columns else ("Ticker" if "Ticker" in tb.columns else "Ticker symbol")
    t = tb.rename(columns={"Company":"name", col_tick:"ticker"}).copy()
    t["ticker"] = (t["ticker"].astype(str).str.upper().str.strip()
                   .str.replace(r"\s+", "", regex=True))
    # Yahoo Londres = suffixe .L
    t["ticker"] = t["ticker"].apply(lambda s: f"{s}.L" if s and not s.endswith(".L") else s)
    t["name"] = t["name"].astype(str).str.strip()
    t["isin"] = ""
    t["market"] = "UK"
    return t[["isin","ticker","name","market"]]

def build_dax40() -> pd.DataFrame:
    tables = fetch_tables(W_DAX)
    # Rechercher table avec 'Company' et un champ de ticker
    tb = None
    for x in tables:
        cols = set(map(str, x.columns))
        if "Company" in cols and ({"Ticker", "Ticker symbol", "Symbol"} & cols):
            tb = x
            break
    if tb is None:
        # fallback: si pas de ticker dispo, on sort seulement les noms (à résoudre plus tard)
        tb2 = next((x for x in tables if "Company" in set(map(str, x.columns))), None)
        if tb2 is None:
            return pd.DataFrame(columns=["isin","ticker","name","market"])
        t2 = tb2.rename(columns={"Company":"name"}).copy()
        t2["ticker"] = ""
        t2["isin"] = ""
        t2["market"] = "DE"
        return t2[["isin","ticker","name","market"]]
    col_tick = "Ticker" if "Ticker" in tb.columns else ("Ticker symbol" if "Ticker symbol" in tb.columns else "Symbol")
    t = tb.rename(columns={"Company":"name", col_tick:"ticker"}).copy()
    t["ticker"] = t["ticker"].astype(str).str.upper().str.strip()
    # Heuristique Yahoo Francfort: .DE (Xetra) si pas de suffixe
    t["ticker"] = t["ticker"].apply(lambda s: f"{s}.DE" if s and "." not in s else s)
    t["name"] = t["name"].astype(str).str.strip()
    t["isin"] = ""
    t["market"] = "DE"
    return t[["isin","ticker","name","market"]]

def main():
    sp = build_sp500()
    cac = build_cac40()
    es = build_eurostoxx50()
    ft = build_ftse100()
    dax = build_dax40()
    all_df = pd.concat([sp, cac, es, ft, dax], ignore_index=True)

    # Nettoyage & dédupe
    for c in ("isin","ticker","name","market"):
        all_df[c] = all_df[c].astype(str).str.strip()
    # On garde au moins un identifiant (ticker ou nom) pour dédupe
    all_df = all_df.drop_duplicates(subset=["ticker","name"]).reset_index(drop=True)

    # Option A: ISIN vide
    all_df["isin"] = all_df["isin"].fillna("")

    all_df.to_csv("data/watchlist.csv", index=False)
    print(f"Written data/watchlist.csv with {len(all_df)} rows.")

if __name__ == "__main__":
    main()

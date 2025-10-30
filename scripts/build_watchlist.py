import pandas as pd
import requests

W_SNP = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
W_CAC = "https://en.wikipedia.org/wiki/CAC_40"
W_ESTOXX = "https://en.wikipedia.org/wiki/EURO_STOXX_50"

def fetch_tables(url: str) -> list[pd.DataFrame]:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_html(r.text)

def build_sp500() -> pd.DataFrame:
    # Wikipedia S&P500 table: columns include Symbol, Security
    t = fetch_tables(W_SNP)[0].copy()
    t = t.rename(columns={"Symbol":"ticker","Security":"name"})
    t["ticker"] = t["ticker"].astype(str).str.upper().str.strip()
    t["name"] = t["name"].astype(str).str.strip()
    t["isin"] = ""
    t["market"] = "US"
    return t[["isin","ticker","name","market"]]

def build_cac40() -> pd.DataFrame:
    # EN wiki has a tidy table with Company + Ticker
    tables = fetch_tables(W_CAC)
    # pick a table that contains Company and Ticker
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
    # take first table that has 'Company' column
    tb = next((x for x in tables if "Company" in x.columns), None)
    if tb is None:
        return pd.DataFrame(columns=["isin","ticker","name","market"])
    t = tb.rename(columns={"Company":"name", "Ticker":"ticker"}).copy()
    # ticker not always Yahoo-compatible → keep if looks like Yahoo with suffix; else blank
    def to_yahoo(sym: str) -> str:
        s = str(sym).strip().upper()
        if any(s.endswith(suf) for suf in (".PA",".AS",".BR",".MI",".MC",".DE",".F",".BE",".SW",".VI",".LS",".IR",".OL",".HE",".VX")):
            return s
        return ""
    if "ticker" in t.columns:
        t["ticker"] = t["ticker"].apply(to_yahoo)
    else:
        t["ticker"] = ""
    t["name"] = t["name"].astype(str).str.strip()
    t["isin"] = ""
    # approximate market from suffix
    def guess_market(tk: str) -> str:
        if tk.endswith(".PA"): return "FR"
        if tk.endswith(".DE") or tk.endswith(".F"): return "DE"
        if tk.endswith(".AS"): return "NL"
        if tk.endswith(".MI"): return "IT"
        if tk.endswith(".MC"): return "ES"
        if tk.endswith(".BE"): return "BE"
        if tk.endswith(".SW") or tk.endswith(".VX"): return "CH"
        if tk.endswith(".LS"): return "PT"
        return "EU"
    t["market"] = t["ticker"].apply(guess_market)
    out = t[["isin","ticker","name","market"]].drop_duplicates(subset=["name"]).reset_index(drop=True)
    return out

def main():
    sp = build_sp500()
    cac = build_cac40()
    es = build_eurostoxx50()
    # concat + dedupe
    all_df = pd.concat([sp, cac, es], ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["ticker","name"]).reset_index(drop=True)
    all_df.to_csv("data/watchlist.csv", index=False)
    print(f"Written data/watchlist.csv with {len(all_df)} rows.")

if __name__ == "__main__":
    main()

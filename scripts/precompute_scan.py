import os
import sys
import json
import time
from datetime import datetime, timezone

import pandas as pd

# Accès aux modules locaux
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ui.app import score_ticker_cached, get_universe_normalized  # réutilise tes fonctions

PROFILE = os.getenv("PROFILE", "Investisseur")  # "Investisseur" ou "Swing"

OUT_PARQUET = f"data/daily_scan_{PROFILE.lower()}.parquet"
OUT_JSON = f"data/daily_scan_{PROFILE.lower()}.json"


def compute_full_scan_df(profile: str, max_workers: int = 8) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    uni = get_universe_normalized()
    tickers = uni["ticker"].astype(str).str.upper().dropna().unique().tolist()
    if not tickers:
        return pd.DataFrame()

    rows = []
    max_workers = min(max_workers, max(2, (os.cpu_count() or 4) * 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(score_ticker_cached, t, profile): t for t in tickers}
        for fut in as_completed(futs):
            res = fut.result()
            if isinstance(res, dict) and not res.get("error"):
                rows.append(res)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Normalisations affichage
    if "%toHH52" in df.columns:
        df["%toHH52"] = pd.to_numeric(df["%toHH52"], errors="coerce") * 100
    if "Market" in df.columns:
        df["Market"] = df["Market"].astype(str).str.strip().str.upper()

    # Arrondi universel 2 décimales
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].round(2)

    keep = [
        "Ticker",
        "Name",
        "Market",
        "Signal",
        "ScoreFinal",
        "ScoreTech",
        "FscoreFund",
        "RSI",
        "%toHH52",
        "Earnings",
        "EarningsDate",
        "EarningsD",
    ]
    df = df[[c for c in keep if c in df.columns]]
    if "ScoreFinal" in df.columns:
        df = df.sort_values(by=["ScoreFinal", "Ticker"], ascending=[False, True]).reset_index(
            drop=True
        )
    return df


def main():
    t0 = time.time()
    df = compute_full_scan_df(PROFILE)
    if df.empty:
        print("No data produced")
        sys.exit(1)

    os.makedirs("data", exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "profile": PROFILE,
        "rows": len(df),
        "columns": list(df.columns),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ {PROFILE}: {OUT_PARQUET} written in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

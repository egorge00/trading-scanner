import os, sys, json, time
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

# accès modules du repo
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ⚠️ on évite d'importer la UI Streamlit; on importe seulement la logique
from ui.app import get_universe_normalized, score_ticker_cached  # si indispo, mettre ces fonctions dans api/core/scan.py

PROFILE = os.getenv("PROFILE", "Investisseur")
OUT_PARQUET = f"data/daily_scan_{PROFILE.lower()}.parquet"
OUT_JSON    = f"data/daily_scan_{PROFILE.lower()}.json"

def compute_full_scan_df(profile: str, max_workers: int = 8) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    uni = get_universe_normalized()
    tickers = (
        uni["ticker"].astype(str).str.upper().dropna().unique().tolist()
        if "ticker" in uni.columns else []
    )
    if not tickers:
        print("[precompute] universe empty or no 'ticker' column", file=sys.stderr)
        return pd.DataFrame()

    rows, failures = [], []
    max_workers = min(max_workers, max(2, (os.cpu_count() or 4) * 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(score_ticker_cached, t, profile): t for t in tickers}
        total = len(futs); done = 0
        for fut in as_completed(futs):
            res = fut.result()
            done += 1
            if isinstance(res, dict) and not res.get("error"):
                rows.append(res)
            else:
                failures.append(res)
            if done % 50 == 0:
                print(f"[precompute] progress {done}/{total}")

    if not rows:
        print("[precompute] no rows produced", file=sys.stderr)
        if failures:
            print(f"[precompute] failures examples: {failures[:3]}", file=sys.stderr)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # normalisations
    if "%toHH52" in df.columns:
        df["%toHH52"] = pd.to_numeric(df["%toHH52"], errors="coerce") * 100
    if "Market" in df.columns:
        df["Market"] = df["Market"].astype(str).str.strip().str.upper()

    # arrondis
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].round(2)

    keep = ["Ticker","Name","Market","Signal","ScoreFinal","ScoreTech","FscoreFund","RSI",
            "%toHH52","Earnings","EarningsDate","EarningsD"]
    df = df[[c for c in keep if c in df.columns]]
    if "ScoreFinal" in df.columns:
        df = df.sort_values(by=["ScoreFinal","Ticker"], ascending=[False, True]).reset_index(drop=True)
    print(f"[precompute] built dataframe: rows={len(df)} cols={list(df.columns)}")
    return df

def main():
    t0 = time.time()
    Path("data").mkdir(parents=True, exist_ok=True)
    df = compute_full_scan_df(PROFILE)
    if df.empty:
        print("[precompute] empty df, abort", file=sys.stderr)
        sys.exit(1)

    df.to_parquet(OUT_PARQUET, index=False)
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "profile": PROFILE,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "file": OUT_PARQUET,
        "size_bytes": int(Path(OUT_PARQUET).stat().st_size),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[precompute] ✅ saved {OUT_PARQUET} ({meta['rows']} rows, {meta['size_bytes']} bytes) in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()

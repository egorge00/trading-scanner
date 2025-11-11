import os, sys, json, time, subprocess
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

# accès modules du repo (sans Streamlit UI)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    # idéalement tu as ces helpers hors UI
    from api.core.scan import get_universe_normalized, score_ticker_cached
except Exception:
    # fallback (si pas encore factorisé) : importer depuis ui.app
    from ui.app import get_universe_normalized, score_ticker_cached  # type: ignore

PROFILE = os.getenv("PROFILE", "Investisseur").strip()
assert PROFILE in {"Investisseur", "Swing"}, f"PROFILE invalide: {PROFILE}"

OUT_PARQUET = f"data/daily_scan_{PROFILE.lower()}.parquet"
OUT_JSON    = f"data/daily_scan_{PROFILE.lower()}.json"

def compute_full_scan_df(profile: str, max_workers: int = 8) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    uni = get_universe_normalized()
    if "ticker" not in uni.columns:
        print("[precompute] universe missing 'ticker' col", file=sys.stderr)
        return pd.DataFrame()

    tickers = (
        uni["ticker"].astype(str).str.upper().dropna().unique().tolist()
    )
    if not tickers:
        print("[precompute] empty tickers", file=sys.stderr)
        return pd.DataFrame()

    rows, failures = [], []
    max_workers = min(max_workers, max(2, (os.cpu_count() or 4) * 2))
    print(f"[precompute] PROFILE={profile} total={len(tickers)}")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(score_ticker_cached, t, profile): t for t in tickers}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                res = fut.result()
                if isinstance(res, dict) and not res.get("error"):
                    rows.append(res)
                else:
                    failures.append(res)
            except Exception as e:
                failures.append({"ticker": futs[fut], "error": str(e)})
            if i % 100 == 0:
                print(f"[precompute] progress {i}/{len(tickers)}")

    if not rows:
        print("[precompute] no rows produced", file=sys.stderr)
        if failures:
            print(f"[precompute] failures examples: {failures[:3]}", file=sys.stderr)
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Normalisations/arrondis (colonnes optionnelles selon ton calcul)
    if "%toHH52" in df.columns:
        df["%toHH52"] = pd.to_numeric(df["%toHH52"], errors="coerce")
    if "Market" in df.columns:
        df["Market"] = df["Market"].astype(str).str.upper().str.strip()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].round(2)

    keep = ["Ticker","Name","Market","Signal","ScoreFinal","ScoreTech","FscoreFund",
            "RSI","%toHH52","Earnings","EarningsDate","EarningsD","Trend"]
    df = df[[c for c in keep if c in df.columns]]
    if "ScoreFinal" in df.columns:
        df = df.sort_values(by=["ScoreFinal","Ticker"], ascending=[False, True]).reset_index(drop=True)

    print(f"[precompute] built {profile}: rows={len(df)} cols={list(df.columns)}")
    return df

def main():
    Path("data").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = compute_full_scan_df(PROFILE)
    if df.empty:
        print(f"[precompute] empty df for {PROFILE}, abort", file=sys.stderr)
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

    print(f"[precompute] ✅ saved {OUT_PARQUET} ({meta['rows']} rows) in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()

"""Smoke test script for validating KPIs and scoring on a small ticker set."""

from __future__ import annotations

import os

os.environ.setdefault("SCANNER_IMPORT_ONLY", "1")

import pandas as pd

from api.core.scoring import (
    compute_kpis,
    compute_kpis_investor,
    compute_score,
    compute_score_investor,
)
from ui.app import safe_yf_download


TICKERS = ["AAPL", "MSFT", "MC.PA", "OR.PA", "BMW.DE", "SHEL.L"]


def run() -> None:
    ok: list[str] = []
    for ticker in TICKERS:
        df_daily = safe_yf_download(ticker, period="9mo", interval="1d")
        assert isinstance(df_daily, pd.DataFrame)
        if not df_daily.empty:
            compute_kpis(df_daily)
            compute_score(df_daily)

        df_weekly_source = safe_yf_download(ticker, period="36mo", interval="1d")
        if not df_weekly_source.empty:
            compute_kpis_investor(df_weekly_source)
            compute_score_investor(df_weekly_source)

        ok.append(ticker)

    print("SMOKE OK for:", ok)


if __name__ == "__main__":
    run()


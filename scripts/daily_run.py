#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Daily runner for the Trading Scanner
------------------------------------
Subcommands:
  - fetch  : télécharge les données de marché et calcule les scores
  - email  : envoie un rapport HTML simple par email (Gmail SMTP app password)
  - alerts : placeholder
  - run    : fetch + email

Conçu pour GitHub Actions (réseau autorisé). Résilient aux tickers sans data.
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
import yfinance as yf

# ----- Logging -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("daily_run")

# ----- Scoring (fallback si import échoue) -----
try:
    from api.core.scoring import compute_kpis, compute_score
except Exception as e:
    logger.warning("Could not import api.core.scoring (%s). Using fallback scoring.", e)

    def compute_kpis(_df):  # type: ignore
        return pd.DataFrame()

    def compute_score(_df):  # type: ignore
        return 0.0, "HOLD"

# =============================================================================
# Helpers marché
# =============================================================================

def _flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Aplati et normalise un DataFrame yfinance pour récupérer
    colonnes: ['Open','High','Low','Close','Volume'] si possible.
    Retourne None si inutilisable.
    """
    if df is None or df.empty:
        return None

    # MultiIndex → aplatir
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            try:
                df = df.droplevel(1, axis=1)
            except Exception:
                return None

    # Normaliser les noms
    def _norm(c: str) -> str:
        c = str(c).strip()
        if c.lower().replace(" ", "") in ("adjclose", "adjustedclose", "adjusted_close"):
            return "Adj Close"
        c = c.replace(".", " ").strip()
        return c.title()

    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]

    # Si 'Close' absent mais 'Adj Close' présent → utiliser Adj Close
    cols = set(df.columns)
    if "Close" not in cols and "Adj Close" in cols:
        df["Close"] = df["Adj Close"]

    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(set(df.columns)):
        # tolère Volume manquant
        missing = needed.difference(set(df.columns))
        if missing == {"Volume"}:
            df["Volume"] = pd.NA
        else:
            return None

    # Nettoyage
    try:
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
    except KeyError:
        return None

    if df.empty:
        return None
    return df


def fetch_history(ticker: str, period: str = "6mo") -> pd.DataFrame | None:
    """Télécharge un historique daily et renvoie un DF plat OHLCV, sinon None."""
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            group_by="column",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        logger.warning("yfinance download failed for %s: %s", ticker, e)
        return None

    out = _flatten_ohlcv(df)
    if out is None:
        logger.info("No usable OHLCV for %s (period=%s). Skipping.", ticker, period)
    return out

# =============================================================================
# Scoring
# =============================================================================

def score_universe(watchlist: pd.DataFrame, period: str = "6mo") -> tuple[list[dict], list[dict]]:
    """
    Parcourt les tickers de la watchlist, calcule KPIs + score.
    Retourne (scores, errors) :
      - scores: liste de dicts (Ticker, Name, Score, Action, KPIs…)
      - errors: liste de dicts {ticker, reason}
    """
    rows: list[dict] = []
    errors: list[dict] = []

    wl = watchlist.copy()
    # Normaliser colonnes attendues
    lower = {c.lower(): c for c in wl.columns}
    for want in ("isin", "ticker", "name", "market"):
        if want not in lower:
            wl[want] = ""
        else:
            real = lower[want]
            if real != want:
                wl = wl.rename(columns={real: want})

    wl["ticker"] = wl["ticker"].astype(str).str.strip().str.upper()
    wl["name"] = wl.get("name", "").astype(str)
    wl = wl[wl["ticker"].str.len() > 0]

    for _, r in wl.iterrows():
        tkr = r["ticker"]
        name = r.get("name", "")
        try:
            history = fetch_history(tkr, period=period)
            if history is None:
                errors.append({"ticker": tkr, "reason": "no_usable_history"})
                continue

            kpis = compute_kpis(history)
            cs = compute_score(history)
            score = cs[0] if isinstance(cs, (list, tuple)) and len(cs) >= 1 else cs
            action = cs[1] if isinstance(cs, (list, tuple)) and len(cs) >= 2 else None
            last = kpis.iloc[-1] if not kpis.empty else pd.Series(dtype="float64")
            rows.append({
                "Ticker": tkr,
                "Name": name,
                "Score": float(score),
                "Action": str(action) if action is not None else "",
                "RSI": float(last.get("RSI", np.nan)),
                "MACD_hist": float(last.get("MACD_hist", np.nan)),
                "Close>SMA50": bool(float(last.get("Close", np.nan)) > float(last.get("SMA50", np.nan))),
                "SMA50>SMA200": bool(float(last.get("SMA50", np.nan)) > float(last.get("SMA200", np.nan))),
                "%toHH52": float(last.get("pct_to_HH52", np.nan)),
                "VolZ20": float(last.get("VolZ20", np.nan)),
            })
        except Exception as e:
            errors.append({"ticker": tkr, "reason": f"exception:{type(e).__name__}"})

    return rows, errors

# =============================================================================
# Tasks
# =============================================================================

def run_fetch(watchlist_path: str, positions_path: str | None, output_dir: str, period: str = "6mo"):
    logger.info("Loading watchlist from %s", watchlist_path)
    wl = pd.read_csv(watchlist_path)

    logger.info("Scoring universe (period=%s)...", period)
    scores, errors = score_universe(wl, period=period)
    logger.info("Done. success=%d, errors=%d", len(scores), len(errors))

    os.makedirs(output_dir, exist_ok=True)
    out_scores = os.path.join(output_dir, "scores.csv")
    pd.DataFrame(scores).to_csv(out_scores, index=False)

    if errors:
        out_err = os.path.join(output_dir, "errors.csv")
        pd.DataFrame(errors).to_csv(out_err, index=False)
        logger.warning("Some tickers failed. See %s", out_err)
    else:
        logger.info("All tickers processed successfully.")

def run_email(
    watchlist_path: str,
    output_dir: str,
    period: str,
    sender: str,
    recipients: str,
    smtp_user: str,
    smtp_password: str,
):
    """Compose et envoie un rapport simple par email."""
    scores_file = os.path.join(output_dir, "scores.csv")
    if not os.path.exists(scores_file):
        logger.warning("scores.csv not found, running fetch first.")
        run_fetch(watchlist_path, None, output_dir, period=period)

    try:
        df = pd.read_csv(scores_file)
    except Exception as e:
        logger.error("Cannot read %s: %s", scores_file, e)
        df = pd.DataFrame()

    if df.empty:
        body = "<p>Aucune donnée disponible aujourd’hui.</p>"
    else:
        df_sorted = df.sort_values(by="Score", ascending=False).head(10)
        show_cols = [c for c in ["Ticker","Name","Score","Action","RSI","MACD_hist","%toHH52","VolZ20"] if c in df_sorted.columns]
        body = "<h3>Top 10 opportunités 🟢</h3>" + df_sorted[show_cols].to_html(index=False, justify="center", border=0)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Daily Market Scanner"
    msg["From"] = sender
    msg["To"] = recipients
    msg.attach(MIMEText(body, "html", "utf-8"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(sender, [x.strip() for x in recipients.split(",") if x.strip()], msg.as_string())
        logger.info("Email sent successfully to %s", recipients)
    except Exception as e:
        logger.error("SMTP send failed: %s", e)
        raise

# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Daily runner for trading scanner")
    parser.add_argument("--watchlist", required=False, default="data/watchlist.csv")
    parser.add_argument("--positions", required=False, default=None)
    parser.add_argument("--output", required=False, default="out")
    parser.add_argument("--period", required=False, default="6mo")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="Download market data & compute scores")
    sub.add_parser("alerts", help="Placeholder for alert logic")

    email_p = sub.add_parser("email", help="Send daily report by email")
    email_p.add_argument("--sender", required=True)
    email_p.add_argument("--recipients", required=True)
    email_p.add_argument("--smtp-user", required=True)
    email_p.add_argument("--smtp-password", required=True)

    sub.add_parser("run", help="Run fetch then email")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    if args.command == "fetch":
        logger.info("Executing command: fetch")
        run_fetch(args.watchlist, args.positions, args.output, period=args.period)
        return 0

    if args.command == "email":
        logger.info("Executing command: email")
        run_email(
            args.watchlist,
            args.output,
            args.period,
            args.sender,
            args.recipients,
            args.smtp_user,
            args.smtp_password,
        )
        return 0

    if args.command == "run":
        logger.info("Executing command: run (fetch + email)")
        run_fetch(args.watchlist, args.positions, args.output, period=args.period)
        run_email(
            args.watchlist,
            args.output,
            args.period,
            os.getenv("EMAIL_FROM", "scanner@example.com"),
            os.getenv("EMAIL_RECIPIENTS", ""),
            os.getenv("SMTP_LOGIN", ""),
            os.getenv("SMTP_PASSWORD", ""),
        )
        return 0

    parser.print_help()
    return 2

if __name__ == "__main__":
    sys.exit(main())

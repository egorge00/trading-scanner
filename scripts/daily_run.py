#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Daily runner for the Trading Scanner
------------------------------------
Handles:
- fetch: télécharge les données de marché et calcule les scores
- email: envoie le rapport par mail
- alerts: placeholder pour alertes futures
- run: wrapper combinant tout

Compatible avec GitHub Actions (pas d’exception fatale si un ticker échoue).
"""

import os
import sys
import io
import time
import argparse
import logging
import smtplib
import pandas as pd
import yfinance as yf
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Si besoin : importer ton module interne
try:
    from api.core.scoring import compute_kpis, compute_score
except Exception:
    def compute_kpis(df):  # fallback minimal pour standalone
        class KPIs:
            rsi = 50
            macd_hist = 0
            close_above_sma50 = False
            sma50_above_sma200 = False
            pct_to_hh52 = 0
            vol_z20 = 0
        return KPIs()
    def compute_score(k):
        class Score:
            score = 0
            action = "HOLD"
        return Score()

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# -------------------- Fetch helper --------------------
def fetch_history(ticker: str, period: str = "6mo") -> pd.DataFrame | None:
    """Télécharge un historique plat OHLCV, sinon None"""
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            group_by="column",
            auto_adjust=False,
            progress=False,
        )
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

        needed = {"Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(set(df.columns)):
            return None

        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
        if df.empty:
            return None
        return df
    except Exception:
        return None


# -------------------- Scoring --------------------
def score_universe(watchlist: pd.DataFrame, period: str = "6mo") -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    errors: list[dict] = []

    wl = watchlist.copy()
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

            k = compute_kpis(history)
            s = compute_score(k)
            rows.append({
                "Ticker": tkr,
                "Name": name,
                "Score": s.score,
                "Action": s.action,
                "RSI": round(k.rsi, 1),
                "MACD_hist": round(k.macd_hist, 3),
                "Close>SMA50": bool(k.close_above_sma50),
                "SMA50>SMA200": bool(k.sma50_above_sma200),
                "%toHH52": float(k.pct_to_hh52),
                "VolZ20": float(k.vol_z20),
            })
        except Exception as e:
            errors.append({"ticker": tkr, "reason": f"exception:{type(e).__name__}"})
            continue

    return rows, errors


# -------------------- Main tasks --------------------
def run_fetch(watchlist_path: str, positions_path: str | None, output_dir: str, period: str = "6mo"):
    logger.info("Loading watchlist from %s", watchlist_path)
    wl = pd.read_csv(watchlist_path)

    wl_cols = {c.lower(): c for c in wl.columns}
    rename_map = {}
    for want in ("isin", "ticker", "name", "market"):
        if want not in wl_cols:
            wl[want] = ""
        else:
            rename_map[wl_cols[want]] = want
    wl = wl.rename(columns=rename_map)

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

    df = pd.read_csv(scores_file)
    if df.empty:
        body = "<p>Aucune donnée disponible aujourd’hui.</p>"
    else:
        df_sorted = df.sort_values(by="Score", ascending=False).head(10)
        body = "<h3>Top 10 opportunités 🟢</h3>"
        body += df_sorted.to_html(index=False, justify="center", border=0)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Daily Market Scanner"
    msg["From"] = sender
    msg["To"] = recipients
    msg.attach(MIMEText(body, "html", "utf-8"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(sender, recipients.split(","), msg.as_string())
        logger.info("Email sent successfully to %s", recipients)
    except Exception as e:
        logger.error("SMTP send failed: %s", e)
        raise


# -------------------- CLI --------------------
def main():
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

    elif args.command == "email":
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

    elif args.command == "run":
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

    else:
        parser.print_help()


if __name__ == "__main__":
    sys.exit(main())

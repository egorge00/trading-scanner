"""Daily pipeline orchestrator for the trading scanner."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yfinance as yf

from api.core.scoring import KPIResult, ScoreResult, compute_kpis, compute_score
from api.notifications import DailyEmailContext, send_daily_email

LOGGER = logging.getLogger("daily_run")

DEFAULT_OUTPUT_DIR = Path("data/generated")
DEFAULT_ALERT_STATE = DEFAULT_OUTPUT_DIR / "alerts_state.json"


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot complete."""


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_watchlist(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df[df["ticker"] != ""]
    return df


def load_positions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.lower().str.strip()
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df[df.get("ticker", "") != ""]
    return df


def fetch_history(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False, threads=False)
    except Exception as exc:  # pragma: no cover - network errors
        LOGGER.warning("Failed to download %s: %s", ticker, exc)
        return None
    if df.empty:
        LOGGER.warning("No data for %s", ticker)
        return None
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if len(df) < 200:
        LOGGER.debug("Not enough history for %s (%d rows)", ticker, len(df))
        return None
    return df


def score_universe(watchlist: pd.DataFrame, period: str = "2y") -> tuple[pd.DataFrame, list[str]]:
    rows = []
    errors: list[str] = []
    generated_at = datetime.now(timezone.utc).isoformat()
    for entry in watchlist.itertuples():
        ticker = entry.ticker
        history = fetch_history(ticker, period=period)
        if history is None:
            errors.append(ticker)
            continue
        try:
            kpi: KPIResult = compute_kpis(history)
            score: ScoreResult = compute_score(kpi)
        except Exception as exc:  # pragma: no cover - scoring errors
            LOGGER.exception("Scoring failed for %s: %s", ticker, exc)
            errors.append(ticker)
            continue
        rows.append(
            {
                "ticker": ticker,
                "name": getattr(entry, "name", ""),
                "market": getattr(entry, "market", ""),
                "score": score.score,
                "action": score.action,
                "rsi": kpi.rsi,
                "macd_hist": kpi.macd_hist,
                "vol_z20": kpi.vol_z20,
                "pct_to_hh52": kpi.pct_to_hh52,
                "pct_from_ll52": kpi.pct_from_ll52,
                "generated_at": generated_at,
            }
        )
    scores_df = pd.DataFrame(rows)
    if not scores_df.empty:
        scores_df = scores_df.sort_values("score", ascending=False).reset_index(drop=True)
        scores_df["rank"] = scores_df.index + 1
    return scores_df, errors


def build_top(scores: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    columns = ["rank", "ticker", "name", "score", "action"]
    if scores.empty:
        return pd.DataFrame(columns=columns)
    subset = scores.nsmallest(limit, columns=["rank"]).sort_values("rank")
    for col in columns:
        if col not in subset.columns:
            subset[col] = ""
    return subset[columns]


def merge_positions(scores: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return positions.copy()
    if scores.empty:
        merged = positions.copy()
        merged["score"] = 0.0
        merged["action"] = "UNKNOWN"
        return merged
    merged = positions.merge(scores[["ticker", "score", "action"]], on="ticker", how="left")
    merged["score"] = merged["score"].fillna(0.0)
    merged["action"] = merged["action"].fillna("UNKNOWN")
    return merged


def _load_alert_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        LOGGER.warning("Alert state file is corrupted, ignoring.")
        return {}


def _save_alert_state(path: Path, state: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def check_alerts(
    positions: pd.DataFrame,
    cooldown_hours: int = 6,
    state_path: Path = DEFAULT_ALERT_STATE,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    state = _load_alert_state(state_path)
    triggered: list[dict[str, Any]] = []
    keep_keys = set()
    for entry in positions.itertuples():
        if getattr(entry, "status", "open") != "open":
            continue
        score_value = getattr(entry, "score", 0)
        if score_value is None:
            continue
        ticker = entry.ticker
        keep_keys.add(ticker)
        if float(score_value) > -2.0:
            continue
        last_ts = state.get(ticker)
        if last_ts:
            try:
                last_dt = datetime.fromisoformat(last_ts)
            except ValueError:
                last_dt = None
            if last_dt and now - last_dt < timedelta(hours=cooldown_hours):
                continue
        payload = {
            "ticker": ticker,
            "score": float(score_value),
            "action": getattr(entry, "action", "SELL"),
            "opened_at": getattr(entry, "opened_at", ""),
            "note": getattr(entry, "note", ""),
        }
        triggered.append(payload)
        state[ticker] = now.isoformat()
    # Clean-up: remove non-open positions from state
    for key in list(state.keys()):
        if key not in keep_keys:
            state.pop(key)
    _save_alert_state(state_path, state)
    return triggered


def dataframe_to_records(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return df[columns].to_dict(orient="records")


def persist_results(
    scores: pd.DataFrame,
    top: pd.DataFrame,
    positions: pd.DataFrame,
    errors: list[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    _ensure_output_dir(output_dir)
    scores_path = output_dir / "scores.csv"
    top_path = output_dir / "top10.csv"
    pos_path = output_dir / "positions.csv"
    meta_path = output_dir / "meta.json"
    scores.to_csv(scores_path, index=False)
    top.to_csv(top_path, index=False)
    positions.to_csv(pos_path, index=False)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
        "scores_path": str(scores_path),
        "positions_path": str(pos_path),
        "top_path": str(top_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))


def run_fetch(
    watchlist_path: Path,
    positions_path: Path,
    output_dir: Path,
    period: str = "2y",
) -> dict[str, Any]:
    LOGGER.info("Loading watchlist from %s", watchlist_path)
    watchlist = load_watchlist(watchlist_path)
    scores, errors = score_universe(watchlist, period=period)
    LOGGER.info("Computed scores for %d tickers (errors=%d)", len(scores), len(errors))
    LOGGER.info("Loading positions from %s", positions_path)
    positions = load_positions(positions_path)
    positions_with_scores = merge_positions(scores, positions)
    top = build_top(scores)
    persist_results(scores, top, positions_with_scores, errors, output_dir=output_dir)
    return {
        "scores": scores,
        "positions": positions_with_scores,
        "top": top,
        "errors": errors,
    }


def load_latest(output_dir: Path) -> dict[str, pd.DataFrame]:
    scores_path = output_dir / "scores.csv"
    top_path = output_dir / "top10.csv"
    pos_path = output_dir / "positions.csv"
    if not scores_path.exists():
        raise PipelineError(f"Scores file not found in {output_dir}")
    return {
        "scores": pd.read_csv(scores_path),
        "top": pd.read_csv(top_path) if top_path.exists() else pd.DataFrame(),
        "positions": pd.read_csv(pos_path) if pos_path.exists() else pd.DataFrame(),
    }


def _parse_recipients(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def run_email(
    output_dir: Path,
    subject: str,
    sender: str,
    recipients: list[str],
    smtp_user: str,
    smtp_password: str,
) -> None:
    datasets = load_latest(output_dir)
    scores = datasets["scores"]
    top = datasets["top"]
    positions = datasets["positions"]
    alerts: list[dict[str, Any]] = []
    for row in dataframe_to_records(positions, ["ticker", "score", "action", "opened_at", "note"]):
        try:
            score_val = float(row.get("score", 0))
        except (TypeError, ValueError):
            continue
        if score_val <= -2.0:
            row["score"] = score_val
            alerts.append(row)
    events: list[dict[str, Any]] = []
    context = DailyEmailContext(
        generated_at=datetime.now(timezone.utc),
        top_opportunities=dataframe_to_records(top, ["rank", "ticker", "name", "score", "action"]),
        positions=dataframe_to_records(positions, ["ticker", "opened_at", "score", "action", "note"]),
        alerts=alerts,
        events=events,
    )
    send_daily_email(subject, sender, recipients, smtp_user, smtp_password, context)


def run_alerts(
    output_dir: Path,
    cooldown_hours: int,
    sender: str,
    recipients: list[str],
    smtp_user: str,
    smtp_password: str,
    subject: str,
) -> list[dict[str, Any]]:
    datasets = load_latest(output_dir)
    positions = datasets["positions"]
    triggered = check_alerts(positions, cooldown_hours=cooldown_hours)
    if triggered and recipients:
        context = DailyEmailContext(
            generated_at=datetime.now(timezone.utc),
            top_opportunities=[],
            positions=dataframe_to_records(positions, ["ticker", "opened_at", "score", "action", "note"]),
            alerts=triggered,
            events=[],
        )
        send_daily_email(subject, sender, recipients, smtp_user, smtp_password, context)
    return triggered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily trading scanner pipeline")
    parser.add_argument("--watchlist", type=Path, default=Path("data/watchlist.csv"))
    parser.add_argument("--positions", type=Path, default=Path("data/positions.csv"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--period", default="2y")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("fetch", help="Fetch market data and compute scores")

    email_parser = sub.add_parser("email", help="Send the daily email report")
    email_parser.add_argument("--subject", default="Daily Market Scanner")
    email_parser.add_argument("--sender", default="scanner@example.com")
    email_parser.add_argument("--recipients")
    email_parser.add_argument("--smtp-user")
    email_parser.add_argument("--smtp-password")

    alert_parser = sub.add_parser("alerts", help="Trigger alert email for red positions")
    alert_parser.add_argument("--subject", default="Alertes positions - Trading Scanner")
    alert_parser.add_argument("--cooldown", type=int, default=6)
    alert_parser.add_argument("--sender", default="scanner@example.com")
    alert_parser.add_argument("--recipients")
    alert_parser.add_argument("--smtp-user")
    alert_parser.add_argument("--smtp-password")

    sub.add_parser("run", help="Fetch data then send email")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    watchlist_path: Path = args.watchlist
    positions_path: Path = args.positions
    output_dir: Path = args.output
    period: str = args.period

    command = args.command or "run"
    LOGGER.info("Executing command: %s", command)

    if command in {"fetch", "run"}:
        run_fetch(watchlist_path, positions_path, output_dir, period=period)
        if command == "fetch":
            return 0

    if command in {"email", "run"}:
        subject = getattr(args, "subject", "Daily Market Scanner")
        sender = getattr(args, "sender", "scanner@example.com")
        recipients = _parse_recipients(
            getattr(args, "recipients", None) or os.getenv("EMAIL_RECIPIENTS")
        )
        smtp_user = getattr(args, "smtp_user", None) or os.getenv("SMTP_LOGIN") or sender
        smtp_password = getattr(args, "smtp_password", None) or os.getenv("SMTP_PASSWORD")
        if not smtp_password:
            raise PipelineError("SMTP password is required to send emails")
        if not recipients:
            LOGGER.warning("No recipients configured, skipping email sending")
        else:
            run_email(output_dir, subject, sender, recipients, smtp_user, smtp_password)
        if command == "email":
            return 0

    if command == "alerts":
        subject = getattr(args, "subject", "Alertes positions - Trading Scanner")
        sender = getattr(args, "sender", "scanner@example.com")
        cooldown = getattr(args, "cooldown", 6)
        recipients = _parse_recipients(
            getattr(args, "recipients", None) or os.getenv("EMAIL_RECIPIENTS")
        )
        smtp_user = getattr(args, "smtp_user", None) or os.getenv("SMTP_LOGIN") or sender
        smtp_password = getattr(args, "smtp_password", None) or os.getenv("SMTP_PASSWORD")
        if not smtp_password:
            raise PipelineError("SMTP password is required for alerts")
        triggered = run_alerts(output_dir, cooldown, sender, recipients, smtp_user, smtp_password, subject)
        LOGGER.info("Triggered alerts: %d", len(triggered))
        return 0

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())

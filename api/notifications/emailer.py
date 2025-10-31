"""SMTP email helper for the daily trading report."""
from __future__ import annotations

import dataclasses
from datetime import datetime
import html
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Iterable, Sequence


@dataclasses.dataclass
class DailyEmailContext:
    """Container for the HTML email sections."""

    generated_at: datetime
    top_opportunities: list[dict]
    positions: list[dict]
    alerts: list[dict]
    events: list[dict]


def _render_table(rows: Iterable[dict], columns: Sequence[tuple[str, str]]) -> str:
    """Render a HTML table for the provided rows.

    Args:
        rows: Iterable of dictionaries.
        columns: Sequence of tuples (key, label).
    """

    rows = list(rows)
    if not rows:
        return "<p><em>Aucune donnée disponible.</em></p>"

    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body_parts: list[str] = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.2f}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body_parts.append(f"<tr>{''.join(cells)}</tr>")
    body = "".join(body_parts)
    return f"<table border='1' cellpadding='6' cellspacing='0'>" \
           f"<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def build_daily_email(context: DailyEmailContext) -> tuple[str, str]:
    """Build the plain text and HTML version of the daily email."""

    generated_str = context.generated_at.strftime("%Y-%m-%d %H:%M UTC")

    sections = []
    sections.append(
        """
        <h2>Top 10 opportunités haussières 🟢</h2>
        {table}
        """.format(
            table=_render_table(
                context.top_opportunities,
                [
                    ("rank", "#"),
                    ("ticker", "Ticker"),
                    ("name", "Nom"),
                    ("score", "Score"),
                    ("action", "Action"),
                ],
            )
        )
    )
    sections.append(
        """
        <h2>Positions en cours</h2>
        {table}
        """.format(
            table=_render_table(
                context.positions,
                [
                    ("ticker", "Ticker"),
                    ("opened_at", "Entrée"),
                    ("score", "Score"),
                    ("action", "Action"),
                    ("note", "Note"),
                ],
            )
        )
    )
    sections.append(
        """
        <h2>Alertes ventes 🔴</h2>
        {table}
        """.format(
            table=_render_table(
                context.alerts,
                [
                    ("ticker", "Ticker"),
                    ("score", "Score"),
                    ("action", "Action"),
                    ("opened_at", "Entrée"),
                    ("note", "Note"),
                ],
            )
        )
    )
    sections.append(
        """
        <h2>Événements à venir</h2>
        {table}
        """.format(
            table=_render_table(
                context.events,
                [
                    ("ticker", "Ticker"),
                    ("event", "Événement"),
                    ("date", "Date"),
                    ("note", "Note"),
                ],
            )
        )
    )

    html_body = (
        """
        <html>
            <body>
                <p>Bonjour 👋,</p>
                <p>Voici le rapport quotidien généré le {generated}.</p>
                {sections}
                <p>Bonne journée et bons trades !</p>
            </body>
        </html>
        """
    ).format(generated=html.escape(generated_str), sections="\n".join(sections))

    def _as_text(rows: Iterable[dict], columns: Sequence[tuple[str, str]]) -> str:
        rows = list(rows)
        if not rows:
            return "  - (aucune donnée)"
        lines = []
        for row in rows:
            formatted = ", ".join(f"{label}: {row.get(key, '')}" for key, label in columns)
            lines.append(f"  - {formatted}")
        return "\n".join(lines)

    text_sections = [
        "Top 10 opportunités haussières:\n" + _as_text(
            context.top_opportunities,
            [
                ("rank", "#"),
                ("ticker", "Ticker"),
                ("name", "Nom"),
                ("score", "Score"),
                ("action", "Action"),
            ],
        ),
        "Positions en cours:\n" + _as_text(
            context.positions,
            [
                ("ticker", "Ticker"),
                ("opened_at", "Entrée"),
                ("score", "Score"),
                ("action", "Action"),
                ("note", "Note"),
            ],
        ),
        "Alertes ventes:\n" + _as_text(
            context.alerts,
            [
                ("ticker", "Ticker"),
                ("score", "Score"),
                ("action", "Action"),
                ("opened_at", "Entrée"),
                ("note", "Note"),
            ],
        ),
        "Événements à venir:\n" + _as_text(
            context.events,
            [
                ("ticker", "Ticker"),
                ("event", "Événement"),
                ("date", "Date"),
                ("note", "Note"),
            ],
        ),
    ]
    text_body = f"Rapport généré le {generated_str}\n\n" + "\n\n".join(text_sections)
    return text_body, html_body


def send_daily_email(
    subject: str,
    sender: str,
    recipients: Sequence[str],
    smtp_user: str,
    smtp_password: str,
    context: DailyEmailContext,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
) -> None:
    """Send the daily HTML email via Gmail SMTP."""

    if not recipients:
        raise ValueError("At least one recipient must be provided")

    text_body, html_body = build_daily_email(context)

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.attach(MIMEText(text_body, "plain", "utf-8"))
    message.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(sender, list(recipients), message.as_string())

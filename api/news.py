"""News and macro feed integration using Alpha Vantage NEWS_SENTIMENT.

This module exposes a FastAPI router and helpers to fetch and normalize
news items so they can be reused by the Streamlit UI and the API layer.
"""
from __future__ import annotations

import os
import time
from typing import List, Literal, Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl, ValidationError

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
NEWS_TOPICS = "economy_macro,economy_fiscal,economy_monetary,financial_markets"
NEWS_TTL_SECONDS = 60


class MissingApiKeyError(RuntimeError):
    """Raised when the Alpha Vantage API key is not configured."""


class NewsProviderError(RuntimeError):
    """Raised when the upstream news provider returns an error."""


class NewsItem(BaseModel):
    id: str
    headline: str
    summary: Optional[str]
    source: Optional[str]
    url: HttpUrl
    published_at: str
    sentiment_score: Optional[float]
    sentiment_label: Literal["bullish", "bearish", "neutral"]
    tickers: List[str]


router = APIRouter(prefix="/api/news", tags=["news"])

_cache_data: list[NewsItem] | None = None
_cache_timestamp: float | None = None


def _compute_sentiment_label(score: Optional[float]) -> Literal["bullish", "bearish", "neutral"]:
    if score is None:
        return "neutral"
    if score > 0.15:
        return "bullish"
    if score < -0.15:
        return "bearish"
    return "neutral"


def _normalize_feed_item(item: dict) -> NewsItem | None:
    score_raw = item.get("overall_sentiment_score")
    try:
        score_val = float(score_raw)
    except Exception:
        score_val = None

    tickers = item.get("ticker_sentiment") or []
    tickers_list = sorted(
        {
            str(t.get("ticker", ""))
            .strip()
            .upper()
            for t in tickers
            if str(t.get("ticker", "")).strip()
        }
    )

    summary = item.get("summary")
    if not summary:
        summary = str(item.get("overall_sentiment_label") or "").strip() or None

    normalized = {
        "id": str(item.get("uuid") or item.get("title") or ""),
        "headline": str(item.get("title") or "").strip(),
        "summary": summary,
        "source": str(item.get("source") or "").strip() or None,
        "url": item.get("url"),
        "published_at": str(item.get("time_published") or "").strip(),
        "sentiment_score": score_val,
        "sentiment_label": _compute_sentiment_label(score_val),
        "tickers": tickers_list,
    }

    if not normalized["headline"] or not normalized["url"]:
        return None

    try:
        return NewsItem(**normalized)
    except ValidationError:
        return None


def _fetch_provider_payload() -> dict:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise MissingApiKeyError("ALPHA_VANTAGE_API_KEY not set")

    params = {
        "function": "NEWS_SENTIMENT",
        "topics": NEWS_TOPICS,
        "sort": "LATEST",
        "limit": 50,
        "apikey": api_key,
    }
    try:
        resp = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=10)
    except Exception as exc:  # pragma: no cover - network layer
        raise NewsProviderError("News provider error") from exc

    if resp.status_code != 200:
        raise NewsProviderError("News provider error")

    try:
        return resp.json()
    except Exception as exc:  # pragma: no cover - invalid JSON
        raise NewsProviderError("News provider error") from exc


def _load_news_from_provider() -> list[NewsItem]:
    payload = _fetch_provider_payload()
    feed = payload.get("feed") if isinstance(payload, dict) else None
    if not isinstance(feed, list):
        raise NewsProviderError("News provider error")

    out: list[NewsItem] = []
    for item in feed:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_feed_item(item)
        if normalized:
            out.append(normalized)
    return out


def get_news_items(force_refresh: bool = False) -> list[NewsItem]:
    global _cache_data, _cache_timestamp

    now = time.time()
    if (
        not force_refresh
        and _cache_data is not None
        and _cache_timestamp is not None
        and now - _cache_timestamp < NEWS_TTL_SECONDS
    ):
        return _cache_data

    news_items = _load_news_from_provider()
    _cache_data = news_items
    _cache_timestamp = now
    return news_items


@router.get("/econ", response_model=list[NewsItem])
def read_econ_news() -> list[NewsItem]:
    try:
        return get_news_items()
    except MissingApiKeyError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except NewsProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

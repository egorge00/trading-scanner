"""Minimal FastAPI application exposing the news endpoints."""
from fastapi import FastAPI

from api.news import router as news_router

app = FastAPI(title="Trading Scanner API")
app.include_router(news_router)

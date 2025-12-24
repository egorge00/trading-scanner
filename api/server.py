"""Minimal FastAPI application exposing the news endpoints."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.news import router as news_router

app = FastAPI(title="Trading Scanner API")
app.include_router(news_router)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def read_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

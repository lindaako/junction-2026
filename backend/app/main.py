"""FastAPI backend: the single place your frontend talks to.

Run:  uvicorn app.main:app --reload --port 8000
Docs: http://localhost:8000/docs  (try endpoints right in the browser)
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import db
from .providers import ChatRequest, ProviderError, available_providers, call_model


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()
    yield


app = FastAPI(title="Junction starter", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a hackathon demo; tighten for real products
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatIn(BaseModel):
    message: str
    provider: str | None = None
    system: str | None = None
    image_b64: str | None = None
    image_mime: str = "image/png"
    history: list[dict] = []


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/providers")
def providers():
    return available_providers()


@app.post("/api/chat")
async def chat(body: ChatIn):
    req = ChatRequest(
        message=body.message,
        system=body.system,
        image_b64=body.image_b64,
        image_mime=body.image_mime,
        history=body.history,
    )
    started = time.perf_counter()
    try:
        reply = await call_model(req, body.provider)
    except ProviderError as e:
        raise HTTPException(status_code=502, detail=str(e))
    latency_ms = int((time.perf_counter() - started) * 1000)
    used = body.provider or "default"
    db.save(used, body.message, reply, latency_ms)
    return {"reply": reply, "provider": used, "latency_ms": latency_ms}


@app.get("/api/history")
def history(limit: int = 20):
    return db.recent(limit)

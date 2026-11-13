"""Tiny storage layer. SQLite by default (zero setup).
Set DATABASE_URL=postgresql://user:pass@host:5432/db to use Postgres instead
(local docker-compose, Aiven, Tiger Data, ...)."""
from __future__ import annotations

import os
import sqlite3
import time

SQLITE_PATH = os.getenv("SQLITE_PATH", "app.db")


def _url() -> str:
    return os.getenv("DATABASE_URL", "")


def _is_pg() -> bool:
    return _url().startswith("postgres")


def _connect():
    if _is_pg():
        import psycopg  # only needed when using Postgres
        return psycopg.connect(_url())
    return sqlite3.connect(SQLITE_PATH)


def _q(sql: str) -> str:
    return sql.replace("?", "%s") if _is_pg() else sql


def init() -> None:
    pk = "SERIAL PRIMARY KEY" if _is_pg() else "INTEGER PRIMARY KEY AUTOINCREMENT"
    with _connect() as c:
        c.execute(
            f"""CREATE TABLE IF NOT EXISTS messages (
                id {pk},
                created_at DOUBLE PRECISION,
                provider TEXT,
                prompt TEXT,
                reply TEXT,
                latency_ms INTEGER
            )"""
        )


def save(provider: str, prompt: str, reply: str, latency_ms: int) -> None:
    with _connect() as c:
        c.execute(
            _q("INSERT INTO messages (created_at, provider, prompt, reply, latency_ms) VALUES (?,?,?,?,?)"),
            (time.time(), provider, prompt, reply, latency_ms),
        )


def recent(limit: int = 20) -> list[dict]:
    with _connect() as c:
        cur = c.execute(
            _q("SELECT id, created_at, provider, prompt, reply, latency_ms FROM messages ORDER BY id DESC LIMIT ?"),
            (limit,),
        )
        cols = ["id", "created_at", "provider", "prompt", "reply", "latency_ms"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

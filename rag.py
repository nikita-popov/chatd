#!/usr/bin/env python3
"""rag.py — L1.5 external semantic recall store for chatd.

Implements the request-scoped RAG sidecar layer (L1.5) — a chatd-owned
SQLite store completely separate from mempalace. Conversation turns are
embedded via Ollama and retrieved by cosine similarity for the current
user message.

L1.5 is always injected into the system prompt alongside KG recall.
It does not replace or modify any mempalace layer.
"""
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import requests

from config import (
    OLLAMA_API,
    RAG_DB_PATH,
    RAG_EMBED_MODEL,
    RAG_MAX_CHARS_PER_CHUNK,
    RAG_MIN_SCORE,
    RAG_TOP_K,
)

log = logging.getLogger("chatd.rag")


def _db_path() -> Path:
    return Path(os.path.expanduser(RAG_DB_PATH))


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    """Initialise the L1.5 external RAG store (chatd-owned, separate from mempalace)."""
    conn = _connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                chunk TEXT NOT NULL,
                embedding TEXT NOT NULL,
                added_at TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_source ON rag_chunks(source)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_added_at ON rag_chunks(added_at)"
        )
        conn.commit()
        log.info("RAG: store ready at %s", path := _db_path())
    finally:
        conn.close()


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _chunk_turn(user: str, assistant: str) -> str:
    user = _truncate(user, RAG_MAX_CHARS_PER_CHUNK // 2)
    assistant = _truncate(assistant, RAG_MAX_CHARS_PER_CHUNK // 2)
    return f"User: {user}\nAssistant: {assistant}".strip()


def _embed(text: str) -> List[float]:
    payload = {"model": RAG_EMBED_MODEL, "input": text}
    r = requests.post(f"{OLLAMA_API}/api/embed", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    embeddings = data.get("embeddings") or []
    if not embeddings:
        raise RuntimeError("embed API returned no embeddings")
    return embeddings[0]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


def index_turn(source: str, user: str, assistant: str) -> None:
    """Index one conversation turn into the L1.5 RAG store."""
    chunk = _chunk_turn(user, assistant)
    if not chunk:
        return
    try:
        embedding = _embed(chunk)
    except Exception as e:
        log.warning("RAG: embedding failed during index_turn: %s", e)
        return

    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO rag_chunks(source, chunk, embedding, added_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                source,
                chunk,
                json.dumps(embedding),
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
        )
        conn.commit()
        log.debug("RAG: indexed chunk for %s", source)
    except Exception as e:
        log.warning("RAG: failed to index chunk for %s: %s", source, e)
    finally:
        conn.close()


def retrieve(query: str, top_k: int = RAG_TOP_K) -> Optional[str]:
    """Return L1.5 RAG context: top-k semantically similar past turns for *query*."""
    query = (query or "").strip()
    if not query:
        return None

    try:
        qvec = _embed(query)
    except Exception as e:
        log.warning("RAG: embedding failed during retrieve: %s", e)
        return None

    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT source, chunk, embedding, added_at FROM rag_chunks ORDER BY id DESC LIMIT 200"
        ).fetchall()
    except Exception as e:
        log.warning("RAG: retrieval query failed: %s", e)
        conn.close()
        return None
    finally:
        conn.close()

    scored = []
    for row in rows:
        try:
            emb = json.loads(row["embedding"])
        except Exception:
            continue
        score = _cosine_similarity(qvec, emb)
        if score >= RAG_MIN_SCORE:
            scored.append((score, row["source"], row["chunk"], row["added_at"]))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    blocks = []
    for score, source, chunk, added_at in scored[:top_k]:
        blocks.append(
            f"[source={source} score={score:.3f} at={added_at}]\n{chunk}"
        )
    log.debug("RAG: retrieved %d chunks", len(blocks))
    return "\n\n---\n\n".join(blocks)

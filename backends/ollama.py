"""backends/ollama.py — Ollama local inference backend.

Wire format: Ollama NDJSON over HTTP.
Embed: POST /api/embed.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List

import requests

from config import OLLAMA_API

log = logging.getLogger("chatd.backends.ollama")


class OllamaBackend:
    """Ollama local runtime. Default backend for all unprefixed models."""

    def chat_stream(
        self, payload: Dict[str, Any]
    ) -> Generator[bytes, None, None]:
        r = requests.post(
            f"{OLLAMA_API}/api/chat",
            json=payload, timeout=3600, stream=True,
        )
        r.raise_for_status()
        with r:
            for line in r.iter_lines():
                if line:
                    yield line + b"\n"

    def chat_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(
            f"{OLLAMA_API}/api/chat",
            json=payload, timeout=3600,
        )
        r.raise_for_status()
        return r.json()

    def embed(self, text: str) -> List[float]:
        from config import RAG_EMBED_MODEL
        r = requests.post(
            f"{OLLAMA_API}/api/embed",
            json={"model": RAG_EMBED_MODEL, "input": text},
            timeout=120,
        )
        r.raise_for_status()
        embeddings = r.json().get("embeddings") or []
        if not embeddings:
            raise RuntimeError("Ollama embed returned no embeddings")
        return embeddings[0]

"""backends/ollama.py — Ollama local inference backend.

Wire format: Ollama NDJSON over HTTP.
Embed: POST /api/embed.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Generator, List

import requests

OLLAMA_API: str = os.environ.get("OLLAMA_API", "http://127.0.0.1:11434")

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

    def embed(self, text: str, model: str) -> List[float]:
        r = requests.post(
            f"{OLLAMA_API}/api/embed",
            json={"model": model, "input": text},
            timeout=120,
        )
        r.raise_for_status()
        embeddings = r.json().get("embeddings") or []
        if not embeddings:
            raise RuntimeError("Ollama embed returned no embeddings")
        return embeddings[0]

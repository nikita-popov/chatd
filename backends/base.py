"""backends/base.py — Backend Protocol (structural interface).

Every backend must satisfy this Protocol. No inheritance required —
duck typing is sufficient, but explicit registration is encouraged.

Chat contract:
  chat_stream(payload) -> Iterator[bytes]
    Yields Ollama-format NDJSON lines (bytes).
    Each line: {"model":..., "message":{"role":"assistant","content":...}, "done":bool}

  chat_sync(payload) -> dict
    Returns a single Ollama-format response dict.

Embed contract:
  embed(text, model) -> list[float]
    Returns a normalised float vector of fixed dimensionality.
    model — the embedding model identifier to use.
    Raises RuntimeError if the backend does not support embeddings.

payload is always an Ollama-format dict:
  {"model": str, "messages": [...], "options": {...}, "tools": [...], "stream": bool}
"""
from __future__ import annotations

from typing import Any, Dict, Generator, List, Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """Structural interface for a chatd inference backend."""

    def chat_stream(
        self, payload: Dict[str, Any]
    ) -> Generator[bytes, None, None]:
        """Yield Ollama-format NDJSON chunks as bytes."""
        ...

    def chat_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a single Ollama-format response dict."""
        ...

    def embed(self, text: str, model: str) -> List[float]:
        """Return an embedding vector for *text* using *model*.

        Raise RuntimeError('embed not supported') if unsupported.
        """
        ...

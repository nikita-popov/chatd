"""backends/openrouter.py — OpenRouter inference backend.

Wire format: OpenAI /chat/completions over HTTPS.
Embed: not supported (raises RuntimeError).
       Use OllamaBackend.embed() for RAG — it is always available locally.

Model name convention: strip "or/" prefix before sending to OpenRouter.
Example: "or/mistralai/mistral-7b-instruct:free" -> "mistralai/mistral-7b-instruct:free"
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Generator, List, Optional

import requests

OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_API_BASE: str = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_PREFIX: str = os.environ.get("OPENROUTER_PREFIX", "or/")

log = logging.getLogger("chatd.backends.openrouter")

_PREFIX: str = OPENROUTER_PREFIX

# In-process cache: model_id (without prefix) -> raw OR model object.
# Populated lazily on the first /api/tags request; lives until process restart.
_model_cache: Dict[str, Dict[str, Any]] = {}
_model_cache_loaded: bool = False


def _strip(model: str) -> str:
    return model[len(_PREFIX):] if model.startswith(_PREFIX) else model


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/nikita-popov/chatd",
    }


def _load_model_cache() -> None:
    """Fetch GET /models and populate _model_cache (id -> object).

    No-op if the cache is already loaded. Thread-safety is not critical here:
    a duplicate fetch on concurrent startup is harmless.
    """
    global _model_cache, _model_cache_loaded
    if _model_cache_loaded:
        return
    try:
        r = requests.get(
            f"{OPENROUTER_API_BASE}/models",
            headers=_headers(),
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        models = data.get("data") or []
        _model_cache = {m["id"]: m for m in models if m.get("id")}
        _model_cache_loaded = True
        log.info("[openrouter] model cache loaded: %d models", len(_model_cache))
    except Exception as exc:
        log.warning("[openrouter] failed to load model cache: %s", exc)


def fetch_model_info(model_with_prefix: str) -> Optional[Dict[str, Any]]:
    """Return OR model metadata for *model_with_prefix*, or None if not found.

    Uses the bulk GET /models endpoint (cached in-process) because OR does
    not expose a per-model GET /models/{id} endpoint.
    """
    _load_model_cache()
    model_id = _strip(model_with_prefix)
    result = _model_cache.get(model_id)
    if result is None:
        log.warning("[openrouter] model not found in catalog: %s", model_id)
    return result


def _to_openai(payload: Dict[str, Any], stream: bool) -> Dict[str, Any]:
    opts = payload.get("options", {})
    result: Dict[str, Any] = {
        "model":    _strip(payload["model"]),
        "messages": payload.get("messages", []),
        "stream":   stream,
    }
    if "num_predict" in opts:
        result["max_tokens"] = opts["num_predict"]
    if "temperature" in opts:
        result["temperature"] = opts["temperature"]
    if "top_p" in opts:
        result["top_p"] = opts["top_p"]
    if payload.get("tools"):
        result["tools"] = payload["tools"]
        result["tool_choice"] = "auto"
    return result


def _normalize_tool_calls(raw: List[Dict]) -> List[Dict]:
    result = []
    for tc in raw:
        fn = tc.get("function") or {}
        args = fn.get("arguments") or "{}"
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        result.append({"function": {"name": fn.get("name", ""), "arguments": args}})
    return result


def _sse_to_ndjson(
    response: requests.Response, model: str
) -> Generator[bytes, None, None]:
    """Convert OpenAI SSE stream to Ollama NDJSON bytes."""
    for raw in response.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8").removeprefix("data: ")
        if line.strip() == "[DONE]":
            yield (json.dumps({
                "model": model,
                "message": {"role": "assistant", "content": ""},
                "done": True, "done_reason": "stop",
            }) + "\n").encode()
            return
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        choice = (chunk.get("choices") or [{}])[0]
        delta  = choice.get("delta") or {}
        finish = choice.get("finish_reason")
        out: Dict[str, Any] = {
            "model":   model,
            "message": {"role": "assistant", "content": delta.get("content") or ""},
            "done":    finish is not None,
        }
        if delta.get("tool_calls"):
            out["message"]["tool_calls"] = _normalize_tool_calls(delta["tool_calls"])
        if finish:
            out["done_reason"] = finish
        yield (json.dumps(out, ensure_ascii=False) + "\n").encode()


class OpenRouterBackend:
    """OpenRouter cloud runtime. Selected for models prefixed with 'or/'."""

    def chat_stream(
        self, payload: Dict[str, Any]
    ) -> Generator[bytes, None, None]:
        r = requests.post(
            f"{OPENROUTER_API_BASE}/chat/completions",
            headers=_headers(),
            json=_to_openai(payload, stream=True),
            timeout=3600, stream=True,
        )
        r.raise_for_status()
        yield from _sse_to_ndjson(r, payload["model"])

    def chat_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(
            f"{OPENROUTER_API_BASE}/chat/completions",
            headers=_headers(),
            json=_to_openai(payload, stream=False),
            timeout=3600,
        )
        r.raise_for_status()
        data   = r.json()
        choice = (data.get("choices") or [{}])[0]
        msg    = choice.get("message") or {}
        result: Dict[str, Any] = {
            "model":   payload["model"],
            "message": {"role": "assistant", "content": msg.get("content") or ""},
            "done":    True, "done_reason": "stop",
        }
        if msg.get("tool_calls"):
            result["message"]["tool_calls"] = _normalize_tool_calls(msg["tool_calls"])
        return result

    def embed(self, text: str, model: str) -> List[float]:
        raise RuntimeError(
            "OpenRouterBackend does not support embed. "
            "Configure RAG_EMBED_MODEL on a local Ollama instance."
        )

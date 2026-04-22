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
from requests.exceptions import HTTPError

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


def supports_tools(model_with_prefix: str) -> bool:
    """Return True if the OR model declares 'tools' in supported_parameters.

    Falls back to True when the cache is unavailable (safe default: let OR
    reject the request if needed rather than silently stripping tools from a
    model that actually supports them).
    """
    _load_model_cache()
    model_id = _strip(model_with_prefix)
    info = _model_cache.get(model_id)
    if info is None:
        # Cache miss: model unknown, assume supported to avoid silent data loss.
        return True
    params: List[str] = info.get("supported_parameters") or []
    return "tools" in params


def _to_openai(payload: Dict[str, Any], stream: bool) -> Dict[str, Any]:
    model = payload["model"]
    opts = payload.get("options", {})
    result: Dict[str, Any] = {
        "model":    _strip(model),
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
        if supports_tools(model):
            result["tools"] = payload["tools"]
            result["tool_choice"] = "auto"
        else:
            log.debug(
                "[openrouter] stripping tools from payload: %s does not support function calling",
                _strip(model),
            )
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


def _extract_or_error(response: requests.Response) -> str:
    """Pull the human-readable message out of an OR error body, or fall back
    to the raw HTTP status line."""
    try:
        body = response.json()
        err = body.get("error") or {}
        msg = err.get("message") or ""
        code = err.get("code")
        return f"{msg} (code={code})" if code else msg or response.text[:200]
    except Exception:
        return response.text[:200]


def _log_http_error(exc: HTTPError) -> str:
    """Log an HTTPError at the appropriate level and return a short description.

    429/503 are transient — WARNING. Everything else is ERROR.
    Retry-After is extracted and included in the log when present.
    """
    r = exc.response
    status = r.status_code if r is not None else 0
    detail = _extract_or_error(r) if r is not None else str(exc)

    extra = ""
    if r is not None and status == 429:
        retry_after = r.headers.get("Retry-After")
        if retry_after:
            extra = f" | Retry-After: {retry_after}s"

    msg = f"[openrouter] HTTP {status}{extra}: {detail}"
    if status in (429, 503):
        log.warning(msg)
    else:
        log.error(msg)

    return detail


def _error_chunk(model: str, detail: str, status: int) -> bytes:
    """Produce a terminal Ollama-style NDJSON chunk carrying the error text."""
    return (json.dumps({
        "model": model,
        "message": {"role": "assistant", "content": f"[OpenRouter error {status}] {detail}"},
        "done": True,
        "done_reason": "error",
    }, ensure_ascii=False) + "\n").encode()


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
        """Stream chat completions from OR, converting SSE to Ollama NDJSON.

        HTTP errors are caught here rather than propagated: the caller receives
        a terminal error chunk so the session finaliser can run normally.
        """
        model = payload["model"]
        try:
            r = requests.post(
                f"{OPENROUTER_API_BASE}/chat/completions",
                headers=_headers(),
                json=_to_openai(payload, stream=True),
                timeout=3600, stream=True,
            )
            r.raise_for_status()
        except HTTPError as exc:
            detail = _log_http_error(exc)
            status = exc.response.status_code if exc.response is not None else 0
            yield _error_chunk(model, detail, status)
            return
        yield from _sse_to_ndjson(r, model)

    def chat_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Non-streaming chat completion (used by sidecar / tool-call loops).

        HTTP errors are re-raised as RuntimeError with a clean message so the
        caller does not have to handle requests internals.
        """
        try:
            r = requests.post(
                f"{OPENROUTER_API_BASE}/chat/completions",
                headers=_headers(),
                json=_to_openai(payload, stream=False),
                timeout=3600,
            )
            r.raise_for_status()
        except HTTPError as exc:
            detail = _log_http_error(exc)
            status = exc.response.status_code if exc.response is not None else 0
            raise RuntimeError(f"OpenRouter HTTP {status}: {detail}") from exc
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

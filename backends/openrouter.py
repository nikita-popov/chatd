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
    _load_model_cache()
    model_id = _strip(model_with_prefix)
    result = _model_cache.get(model_id)
    if result is None:
        log.warning("[openrouter] model not found in catalog: %s", model_id)
    return result


def supports_tools(model_with_prefix: str) -> bool:
    _load_model_cache()
    model_id = _strip(model_with_prefix)
    info = _model_cache.get(model_id)
    if info is None:
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


def _merge_tool_call_deltas(acc: List[Dict], deltas: List[Dict]) -> List[Dict]:
    """Merge a tool_calls delta into the accumulator.

    OpenAI streaming sends tool_calls in fragments:
      chunk 1: [{index:0, id:"call_x", function:{name:"foo", arguments:""}}]
      chunk 2: [{index:0, function:{arguments:'{"k":'}}]
      chunk 3: [{index:0, function:{arguments:'"v"}'}}]

    We accumulate by index, concatenating the arguments strings, and taking
    the first non-empty name/id we see.
    """
    for delta in deltas:
        idx = delta.get("index", 0)
        # Extend accumulator to cover this index.
        while len(acc) <= idx:
            acc.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        entry = acc[idx]
        if delta.get("id") and not entry["id"]:
            entry["id"] = delta["id"]
        fn = delta.get("function") or {}
        if fn.get("name") and not entry["function"]["name"]:
            entry["function"]["name"] = fn["name"]
        if fn.get("arguments"):
            entry["function"]["arguments"] += fn["arguments"]
    return acc


def _finalise_tool_calls(acc: List[Dict]) -> List[Dict]:
    """Parse accumulated argument strings and normalise to chatd internal format."""
    result = []
    for entry in acc:
        fn = entry.get("function") or {}
        args_str = fn.get("arguments") or "{}"
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            log.warning("[openrouter] failed to parse tool_call arguments: %r", args_str[:200])
            args = {}
        result.append({"function": {"name": fn.get("name", ""), "arguments": args}})
    return result


def _extract_or_error(response: requests.Response) -> str:
    try:
        body = response.json()
        err = body.get("error") or {}
        msg = err.get("message") or ""
        code = err.get("code")
        return f"{msg} (code={code})" if code else msg or response.text[:200]
    except Exception:
        return response.text[:200]


def _or_response_meta(r: requests.Response) -> str:
    parts = []
    provider = (
        r.headers.get("X-OR-Provider")
        or r.headers.get("x-or-provider")
    )
    gen_id = (
        r.headers.get("X-OR-Generation-ID")
        or r.headers.get("x-or-generation-id")
    )
    if provider:
        parts.append(f"provider={provider}")
    if gen_id:
        parts.append(f"generation={gen_id}")
    return " ".join(parts)


def _log_http_error(exc: HTTPError) -> str:
    r = exc.response
    status = r.status_code if r is not None else 0
    detail = _extract_or_error(r) if r is not None else str(exc)

    extras: List[str] = []
    if r is not None:
        if status == 429:
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                extras.append(f"Retry-After={retry_after}s")
        meta = _or_response_meta(r)
        if meta:
            extras.append(meta)

    extra_str = " | " + ", ".join(extras) if extras else ""
    msg = f"[openrouter] HTTP {status}{extra_str}: {detail}"
    if status in (429, 503):
        log.warning(msg)
    else:
        log.error(msg)

    return detail


def _error_chunk(model: str, detail: str, status: int) -> bytes:
    return (json.dumps({
        "model": model,
        "message": {"role": "assistant", "content": f"[OpenRouter error {status}] {detail}"},
        "done": True,
        "done_reason": "error",
    }, ensure_ascii=False) + "\n").encode()


def _sse_to_ndjson(
    response: requests.Response, model: str
) -> Generator[bytes, None, None]:
    """Convert OpenAI SSE stream to Ollama NDJSON bytes.

    Tool call handling:
      OpenAI streams tool_calls as delta fragments across multiple chunks
      (each chunk adds a piece of the function name or arguments string).
      We accumulate all fragments in `_tc_acc` and emit them only once in
      the final done=True chunk.  This prevents chatd.py from seeing dozens
      of intermediate chunks with empty names and from picking up an
      incomplete last-delta as the resolved tool call.
    """
    _tc_acc: List[Dict] = []   # accumulates raw delta fragments
    content_acc: str = ""

    for raw in response.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8").removeprefix("data: ")
        if line.strip() == "[DONE]":
            # Emit a single terminal chunk with the fully assembled tool_calls.
            out: Dict[str, Any] = {
                "model":   model,
                "message": {"role": "assistant", "content": ""},
                "done":    True,
                "done_reason": "stop",
            }
            if _tc_acc:
                finalised = _finalise_tool_calls(_tc_acc)
                out["message"]["tool_calls"] = finalised
                log.debug(
                    "[openrouter] assembled %d tool_call(s): %s",
                    len(finalised),
                    [tc["function"]["name"] for tc in finalised],
                )
            yield (json.dumps(out, ensure_ascii=False) + "\n").encode()
            return
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        choice = (chunk.get("choices") or [{}])[0]
        delta  = choice.get("delta") or {}
        finish = choice.get("finish_reason")

        # Accumulate tool_call fragments silently — do NOT yield intermediate chunks.
        if delta.get("tool_calls"):
            _merge_tool_call_deltas(_tc_acc, delta["tool_calls"])
            continue

        content = delta.get("content") or ""
        if content:
            content_acc += content

        out = {
            "model":   model,
            "message": {"role": "assistant", "content": content},
            "done":    finish is not None,
        }
        if finish:
            out["done_reason"] = finish
            # If finish arrived without [DONE] and we have accumulated tool calls,
            # attach them to this final chunk.
            if _tc_acc:
                finalised = _finalise_tool_calls(_tc_acc)
                out["message"]["tool_calls"] = finalised
                log.debug(
                    "[openrouter] assembled %d tool_call(s) at finish: %s",
                    len(finalised),
                    [tc["function"]["name"] for tc in finalised],
                )
        yield (json.dumps(out, ensure_ascii=False) + "\n").encode()


class OpenRouterBackend:
    """OpenRouter cloud runtime. Selected for models prefixed with 'or/'."""

    def chat_stream(
        self, payload: Dict[str, Any]
    ) -> Generator[bytes, None, None]:
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

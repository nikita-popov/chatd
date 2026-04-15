#!/usr/bin/env python3
import json
import logging
import os
import re
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

import requests
from requests.exceptions import ReadTimeout
from flask import Flask, request, Response, jsonify, stream_with_context

import memory
from config import (
    OLLAMA_API,
    THINKING,
    DEFAULT_OPTIONS,
    MEMPALACE_ALLOWED_TOOLS,
    MEMPALACE_WRITE_TOOLS,
    TOOL_DESCRIPTION_OVERRIDES,
)
from mcp_client import MCPClient, discover_mcp_servers
from think_remapper import ThinkingRemapper

# ── logging ────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chatd")

# ── app ─────────────────────────────────────────────────────────────────────────────

TOOLS: List[Dict] = []
TOOL_REGISTRY: Dict[str, MCPClient] = {}

app = Flask(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"


def make_chunk(model: str, content: str, done: bool = False) -> bytes:
    obj: Dict[str, Any] = {
        "model": model,
        "created_at": now_iso(),
        "message": {"role": "assistant", "content": content},
        "done": done,
    }
    if done:
        obj["done_reason"] = "stop"
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def make_keepalive(model: str) -> bytes:
    """Empty content chunk — keeps the HTTP connection alive while tools run."""
    return make_chunk(model, "")


def proxy_get(path: str) -> Response:
    r = requests.get(f"{OLLAMA_API}{path}", timeout=600)
    return Response(
        r.content,
        status=r.status_code,
        content_type=r.headers.get("Content-Type", "application/json"),
    )


def merge_options(client_options: Optional[Dict]) -> Dict[str, Any]:
    opts = dict(DEFAULT_OPTIONS)
    if client_options:
        opts.update(client_options)
    return opts


def ensure_system_prompt(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    system_prompt = memory.wake_up()
    if not isinstance(messages, list) or not messages:
        return [{"role": "system", "content": system_prompt}]
    if messages[0].get("role") != "system":
        return [{"role": "system", "content": system_prompt}] + messages
    return messages


def build_model_messages(raw_messages: List[Dict]) -> List[Dict]:
    result = []
    for m in raw_messages:
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        content = m.get("content", "") or ""
        if role == "assistant":
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        entry: Dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and m.get("tool_calls"):
            entry["tool_calls"] = m["tool_calls"]
        result.append(entry)
    return result


def make_ollama_payload(
        model: str,
        messages: List[Dict],
        options: Dict[str, Any],
        stream: bool,
) -> Dict[str, Any]:
    """Build an Ollama /api/chat payload, respecting the global THINKING flag."""
    payload: Dict[str, Any] = {
        "model":    model,
        "messages": messages,
        "stream":   stream,
        "options":  options,
    }
    if TOOLS:
        payload["tools"] = TOOLS
    if not THINKING:
        payload["think"] = False
    return payload


# ── CORS ──────────────────────────────────────────────────────────────────────────

@app.after_request
def add_cors(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/chat", methods=["OPTIONS"])
@app.route("/chat",     methods=["OPTIONS"])
def options_chat():
    return Response(status=204)


# ── tools ───────────────────────────────────────────────────────────────────────

def load_tools():
    tools    = []
    registry = {}

    for source, cmd in discover_mcp_servers().items():
        client = MCPClient(cmd)
        try:
            server_tools = client.list_tools()
        except Exception as e:
            log.error("[tools] %s: failed to list tools: %s", source, e)
            continue
        log.info("[tools] %s: %d tools loaded", source, len(server_tools))

        for t in server_tools:
            name         = getattr(t, "name", None)
            description  = getattr(t, "description", "") or ""
            input_schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)

            if not name or not input_schema:
                log.warning("[tools] %s: skipping tool without name/schema: %r", source, t)
                continue

            description = TOOL_DESCRIPTION_OVERRIDES.get(name, description)
            registry[name] = client

            # Hide non-whitelisted mempalace tools from model context.
            if source == "mempalace" and name not in MEMPALACE_ALLOWED_TOOLS:
                log.debug("[tools] mempalace: hiding from model context: %s", name)
                continue

            tools.append({
                "type": "function",
                "function": {
                    "name":        name,
                    "description": description,
                    "parameters":  input_schema,
                },
            })
            log.debug("[tools] registered: %s", name)

    return tools, registry


def init_tools():
    global TOOLS, TOOL_REGISTRY
    log.info("[tools] initializing MCP tools...")
    TOOLS, TOOL_REGISTRY = load_tools()
    log.info("[tools] total: %d tools in registry", len(TOOLS))


def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    client = TOOL_REGISTRY.get(name)
    if client is None:
        raise RuntimeError(f"Unknown tool: {name}")
    log.debug("[tool] calling %s with args: %s", name, arguments)
    result = client.call_tool(name, arguments)
    log.debug("[tool] %s result: %s", name, str(result)[:200])
    if name in MEMPALACE_WRITE_TOOLS:
        # Keep in-process KG dict consistent after a structured fact write.
        if name == "mempalace_kg_add":
            memory.kg_add(
                arguments.get("subject", ""),
                arguments.get("predicate", ""),
                arguments.get("object", ""),
            )
        memory.invalidate()
    return result


# ── stream yield helper ──────────────────────────────────────────────────────────────────

def _log_yield(req_id: str, label: str, data: bytes) -> bytes:
    log.debug("[%s] yield %-12s %d bytes", req_id, label, len(data))
    return data


# ── routes ───────────────────────────────────────────────────────────────────────

@app.get("/health")
@app.get("/api/health")
def health():
    return jsonify({"ok": True})


@app.get("/tags")
@app.get("/api/tags")
def tags():
    return proxy_get("/api/tags")


def chat_stream_generator(
    model: str,
    messages: List[Dict],
    options: Dict[str, Any],
    req_id: str = "-",
) -> Generator[bytes, None, None]:
    MAX_TOOL_ROUNDS = 5
    prev_tool_names: Optional[List[str]] = None
    repeat_count = 0

    data = make_keepalive(model)
    log.debug("[%s] yield %-12s %d bytes", req_id, "keepalive/0", len(data))
    yield data

    try:
        for round_num in range(MAX_TOOL_ROUNDS):
            log.info("[%s] stream round %d, messages=%d", req_id, round_num, len(messages))

            payload = make_ollama_payload(
                model, build_model_messages(messages), options, stream=True,
            )
            log.debug(
                "[%s] -> Ollama payload (messages=%d, tools=%d, options=%s, think=%s)",
                req_id, len(payload["messages"]), len(TOOLS), options, THINKING,
            )

            try:
                r = requests.post(
                    f"{OLLAMA_API}/api/chat",
                    json=payload, timeout=3600, stream=True,
                )
                r.raise_for_status()
                log.debug("[%s] Ollama HTTP %d, headers: %s", req_id, r.status_code, dict(r.headers))
            except Exception as e:
                log.error("[%s] Ollama request failed: %s", req_id, e)
                yield make_chunk(model, f"\n[error: {e}]\n", done=True)
                return

            last_tool_calls: Optional[List[Dict]] = None
            chunk_count = 0
            remapper = ThinkingRemapper(model)

            with r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError as e:
                        log.warning("[%s] bad JSON line: %s | %r", req_id, e, line[:120])
                        continue

                    chunk_count += 1
                    msg  = chunk.get("message") or {}
                    done = chunk.get("done", False)
                    tc   = msg.get("tool_calls")

                    if tc:
                        log.info("[%s] tool_calls: %s", req_id,
                                 [c.get("function", {}).get("name") for c in tc])
                        last_tool_calls = tc

                    if done:
                        log.info(
                            "[%s] round %d done, chunks=%d, tool_calls=%s, content_len=%d",
                            req_id, round_num, chunk_count,
                            bool(last_tool_calls), len(remapper.content_acc),
                        )
                        closing = remapper.close()
                        if closing:
                            yield _log_yield(req_id, "think/close", closing)
                        if last_tool_calls:
                            yield _log_yield(req_id, "keepalive/tc", make_keepalive(model))
                        else:
                            yield _log_yield(req_id, "done", remapper.feed(line))
                        break

                    out = remapper.feed(line)
                    if not tc:
                        log.debug(
                            "[%s] yield %-12s %d bytes: ...%s",
                            req_id, f"chunk/{chunk_count}", len(out),
                            out[-60:].decode("utf-8", errors="replace").strip(),
                        )
                        yield out

            log.debug("[%s] exited iter_lines, last_tool_calls=%s", req_id, bool(last_tool_calls))

            if not last_tool_calls:
                log.info("[%s] stream complete, no more tool rounds", req_id)
                break

            # ── loop detection ───────────────────────────────────────────────────────────────
            current_tool_names = [
                (tc.get("function") or {}).get("name") for tc in last_tool_calls
            ]
            if current_tool_names == prev_tool_names:
                repeat_count += 1
                if repeat_count >= 2:
                    log.warning("[%s] tool loop detected (%s x%d), aborting",
                                req_id, current_tool_names, repeat_count)
                    yield make_chunk(model, "\n[ошибка: цикл инструментов, остановлено]\n", done=True)
                    return
            else:
                repeat_count = 0
            prev_tool_names = current_tool_names

            # ── tool execution ─────────────────────────────────────────────────────────────────
            messages.append({
                "role":       "assistant",
                "content":    remapper.content_acc,
                "tool_calls": last_tool_calls,
            })

            for tc in last_tool_calls:
                fn        = tc.get("function") or {}
                tool_name = fn.get("name") or "unknown"
                tool_args = fn.get("arguments") or {}
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        log.warning("[%s] failed to parse tool_args for %s", req_id, tool_name)
                        tool_args = {}

                log.info("[%s] executing tool: %s(%s)", req_id, tool_name, tool_args)
                yield _log_yield(req_id, "tool/ann", make_chunk(model, f"\n[→ {tool_name}]\n"))

                try:
                    tool_result = call_tool(tool_name, tool_args)
                except Exception as e:
                    log.error("[%s] tool %s failed: %s", req_id, tool_name, e)
                    tool_result = {"error": str(e)}

                messages.append({
                    "role":    "tool",
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

            yield _log_yield(req_id, "keepalive/tr", make_keepalive(model))
            last_tool_calls = None

        log.debug("[%s] generator exiting normally", req_id)

    except GeneratorExit:
        log.warning("[%s] GeneratorExit: client disconnected", req_id)
    except Exception as e:
        log.error("[%s] unhandled exception: %s", req_id, traceback.format_exc())
        yield make_chunk(model, f"\n[internal error: {e}]\n", done=True)


@app.post("/chat")
@app.post("/api/chat")
def chat():
    req_id = uuid.uuid4().hex[:8]

    try:
        original_payload = request.get_json(force=True, silent=False)
    except Exception as e:
        log.error("[%s] failed to parse request JSON: %s", req_id, e)
        return jsonify({"error": "invalid JSON"}), 400

    model       = original_payload.get("model") or "qwen3:8b"
    messages    = original_payload.get("messages") or []
    options     = merge_options(original_payload.get("options"))
    want_stream = original_payload.get("stream", True)

    messages = ensure_system_prompt(messages)

    log.info("[%s] POST /api/chat model=%s stream=%s messages=%d options=%s",
             req_id, model, want_stream, len(messages), options)

    if not want_stream:
        try:
            MAX_TOOL_ROUNDS = 5
            resp: Dict = {}
            prev_tc_names: Optional[List[str]] = None
            rep = 0

            for i in range(MAX_TOOL_ROUNDS):
                payload = make_ollama_payload(
                    model, build_model_messages(messages), options, stream=False,
                )
                log.debug("[%s] non-stream round %d, think=%s", req_id, i, THINKING)
                r = requests.post(f"{OLLAMA_API}/api/chat", json=payload, timeout=3600)
                r.raise_for_status()
                resp       = r.json()
                msg        = resp.get("message") or {}
                tool_calls = msg.get("tool_calls") or []

                if not tool_calls:
                    break

                tc_names = [(tc.get("function") or {}).get("name") for tc in tool_calls]
                log.info("[%s] non-stream tool_calls: %s", req_id, tc_names)

                if tc_names == prev_tc_names:
                    rep += 1
                    if rep >= 2:
                        log.warning("[%s] non-stream tool loop, aborting", req_id)
                        return jsonify({"error": "tool loop detected"}), 500
                else:
                    rep = 0
                prev_tc_names = tc_names

                messages.append({
                    "role":       "assistant",
                    "content":    msg.get("content") or "",
                    "tool_calls": tool_calls,
                })
                for tc in tool_calls:
                    fn        = tc.get("function") or {}
                    tool_name = fn.get("name")
                    tool_args = fn.get("arguments") or {}
                    if isinstance(tool_args, str):
                        tool_args = json.loads(tool_args)
                    try:
                        tool_result = call_tool(tool_name, tool_args)
                    except Exception as e:
                        tool_result = {"error": str(e)}
                    messages.append({
                        "role":    "tool",
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    })

            msg    = resp.get("message") or {}
            answer = (msg.get("content") or "").strip() or "[пустой ответ]"
            log.info("[%s] non-stream done, answer_len=%d", req_id, len(answer))
            return jsonify({"model": model, "answer": answer, "raw": resp})

        except ReadTimeout:
            log.error("[%s] ollama timeout", req_id)
            return jsonify({"error": "ollama timeout"}), 504
        except Exception as e:
            log.error("[%s] non-stream error: %s", req_id, traceback.format_exc())
            return jsonify({"error": str(e)}), 502

    log.info("[%s] starting stream generator", req_id)
    return Response(
        stream_with_context(chat_stream_generator(model, messages, options, req_id)),
        content_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control":     "no-cache",
            "Transfer-Encoding": "chunked",
        },
    )


# ── entry point ──────────────────────────────────────────────────────────────────────

init_tools()
memory.init()

if __name__ == "__main__":
    log.info("started")
    app.run(host="0.0.0.0", port=5001, debug=False)

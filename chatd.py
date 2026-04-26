#!/usr/bin/env python3
import atexit
import json
import logging
import os
import re
import threading
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
from requests.exceptions import ReadTimeout, RequestException
from flask import Flask, request, Response, jsonify, stream_with_context

import backends
import memory
import rag
import session as sess
from config import (
    THINKING,
    TOOLS_FILTER,
    TOOLS_ALLOWED,
    DEFAULT_OPTIONS,
    MAX_TOOL_ROUNDS,
    MEMPALACE_WRITE_TOOLS,
    TOOL_OVERRIDE,
    TOOL_DESCRIPTION_OVERRIDES,
    CHATD_SUMMARY_MODEL,
    CHATD_COMPRESS_EVERY,
    OPENROUTER_API_MODELS,
    CHATD_EVENT_TOKEN,
    CHATD_EVENT_MODEL,
)
from backends.ollama import OLLAMA_API
from backends.openrouter import fetch_model_info, OPENROUTER_PREFIX
from mcp_client import MCPClient, discover_mcp_servers
from think_remapper import ThinkingRemapper

# ── logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chatd")

# ── app ────────────────────────────────────────────────────────────────────────

VERSION = "1.0.0"

TOOLS: List[Dict] = []
TOOL_REGISTRY: Dict[str, MCPClient] = {}

# All started MCP clients — used for clean shutdown via atexit.
_MCP_CLIENTS: List[MCPClient] = []

app = Flask(__name__)
app.json.ensure_ascii = False

_GLOBAL_SUMMARY_LOCK = threading.Lock()

# How much to boost num_predict for tool follow-up rounds.
TOOL_ROUND_NUM_PREDICT_BOOST = 1024

# Maximum seconds to wait for run_tool_loop inside /api/event.
EVENT_TOOL_LOOP_TIMEOUT = 120

_EVENT_DIRECTIVE = (
    "\n\n"
    "This is an automated system event — there is no human in the conversation.\n"
    "You MUST respond by calling the appropriate tools to handle this event.\n"
    "Do NOT just describe what you would do — actually call the tools now.\n"
    "After completing all actions, write a brief summary of what was done."
)


# ── helpers ────────────────────────────────────────────────────────────────────

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
    return make_chunk(model, "")


def proxy_get(path: str) -> Response:
    r = requests.get(f"{OLLAMA_API}{path}", timeout=600)
    return Response(
        r.content,
        status=r.status_code,
        content_type=r.headers.get("Content-Type", "application/json"),
    )


def proxy_post(path: str, json_body: Dict[str, Any]) -> Response:
    r = requests.post(f"{OLLAMA_API}{path}", json=json_body, timeout=600, stream=True)
    return Response(
        r.iter_content(chunk_size=None),
        status=r.status_code,
        content_type=r.headers.get("Content-Type", "application/x-ndjson"),
    )


def merge_options(client_options: Optional[Dict]) -> Dict[str, Any]:
    opts = dict(DEFAULT_OPTIONS)
    if client_options:
        opts.update(client_options)
    return opts


def extract_chat_id(flask_request) -> Optional[str]:
    """Extract chat session ID from the Referer header.

    Hollama sends requests from /sessions/<id> — the browser automatically
    includes this URL as the Referer header, so we parse it from there.
    Falls back to scanning messages for a legacy chatId field.

    Example Referer: http://host/sessions/e8unjq  →  "e8unjq"
    """
    referer = flask_request.headers.get("Referer", "")
    if referer:
        m = re.search(r"/sessions/([^/?#]+)", referer)
        if m:
            return m.group(1)

    # Legacy fallback: chatId field in messages (pre-hollama WebUI)
    messages = flask_request.get_json(silent=True) or {}
    for msg in (messages.get("messages") or []):
        chat_id = msg.get("chatId")
        if chat_id is not None:
            return str(chat_id)

    return None


def _last_user_text(messages: List[Dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return (m.get("content") or "")
    return ""


def _check_event_token() -> Optional[Tuple[Response, int]]:
    """Return (Response, status) tuple if CHATD_EVENT_TOKEN is set and Bearer doesn't match.

    Returns None when the request is allowed through.
    """
    if not CHATD_EVENT_TOKEN:
        return None
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return jsonify({"error": "missing Authorization header"}), 401
    token = auth.removeprefix("Bearer ").strip()
    if token != CHATD_EVENT_TOKEN:
        return jsonify({"error": "invalid token"}), 401
    return None


# ── system prompt assembly ─────────────────────────────────────────────────────

def ensure_system_prompt(
    messages: List[Dict[str, Any]],
    per_chat_summary: str = "",
    global_summary: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Assemble the always-loaded memory block and prepend it as system message.

    Returns (messages, layer_sizes) where layer_sizes maps layer name to char count:
      L0        mempalace identity + wake-up base
      L0.5g     global compressed summary
      L0.5p     per-chat compressed summary
      L1        mempalace Essential Story (included in wake_up base)
      L1.5_kg   KG recall sidecar
      L1.5_rag  RAG recall sidecar
    """
    base_prompt = memory.wake_up(
        per_chat_summary=per_chat_summary,
        global_summary=global_summary,
    )
    layer_sizes: Dict[str, int] = {
        "L0+L1":   len(base_prompt),
        "L0.5g":   len(global_summary),
        "L0.5p":   len(per_chat_summary),
        "L1.5_kg": 0,
        "L1.5_rag": 0,
    }

    last_user = _last_user_text(messages)
    if last_user:
        extra = memory.kg_recall_from_text(last_user)
        if extra:
            layer_sizes["L1.5_kg"] = len(extra)
            log.debug("[sidecar] injecting %d-char KG recall into system prompt", len(extra))
            base_prompt = f"{base_prompt}\n\n# Recalled facts\n{extra}"

        rag_ctx = rag.retrieve(last_user)
        if rag_ctx:
            layer_sizes["L1.5_rag"] = len(rag_ctx)
            log.debug("[sidecar] injecting %d-char RAG recall into system prompt", len(rag_ctx))
            base_prompt = f"{base_prompt}\n\n# Relevant past context\n{rag_ctx}"

    system_prompt = base_prompt

    if not isinstance(messages, list) or not messages:
        return [{"role": "system", "content": system_prompt}], layer_sizes
    if messages[0].get("role") != "system":
        result = [{"role": "system", "content": system_prompt}] + messages
    else:
        result = [{"role": "system", "content": system_prompt}] + messages[1:]
    return result, layer_sizes


def build_model_messages(
    raw_messages: List[Dict],
    max_history_turns: int = 20,
) -> List[Dict]:
    """Sanitise messages for the backend.

    - Strip fields unknown to Ollama (chatId, id, createdAt, updatedAt …).
    - Remove <think>…</think> blocks from previous assistant turns.
    - Preserve tool_calls on assistant messages.
    - Keep system message as-is.
    - Limit history to *max_history_turns* user+assistant pairs.
    """
    system: Optional[Dict] = None
    turns: List[Dict] = []

    for m in raw_messages:
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        content = m.get("content", "") or ""
        if role == "assistant":
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        if role == "system":
            system = {"role": "system", "content": content}
            continue

        entry: Dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and m.get("tool_calls"):
            entry["tool_calls"] = m["tool_calls"]
        turns.append(entry)

    if len(turns) > max_history_turns * 2:
        turns = turns[-(max_history_turns * 2):]

    result = []
    if system:
        result.append(system)
    result.extend(turns)
    return result


def _log_payload_sizes(
    req_id: str,
    messages: List[Dict],
    tools_count: int,
    layer_sizes: Optional[Dict[str, int]] = None,
) -> None:
    sizes: Dict[str, int] = {"system": 0, "user": 0, "assistant": 0, "tool": 0}
    for m in messages:
        role = m.get("role", "other")
        content = m.get("content") or ""
        sizes[role] = sizes.get(role, 0) + len(content)
    total = sum(sizes.values())

    if layer_sizes:
        log.debug(
            "[%s] system layers: L0+L1=%d L0.5g=%d L0.5p=%d L1.5_kg=%d L1.5_rag=%d  total=%d",
            req_id,
            layer_sizes.get("L0+L1", 0),
            layer_sizes.get("L0.5g", 0),
            layer_sizes.get("L0.5p", 0),
            layer_sizes.get("L1.5_kg", 0),
            layer_sizes.get("L1.5_rag", 0),
            sizes.get("system", 0),
        )
    log.debug(
        "[%s] payload sizes: system=%d user=%d assistant=%d tool=%d  total_chars=%d  tools=%d",
        req_id,
        sizes.get("system", 0),
        sizes.get("user", 0),
        sizes.get("assistant", 0),
        sizes.get("tool", 0),
        total,
        tools_count,
    )


def make_ollama_payload(
        model: str,
        messages: List[Dict],
        options: Dict[str, Any],
        stream: bool,
) -> Dict[str, Any]:
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


# ── tool loop helper ──────────────────────────────────────────────────────────────────

def run_tool_loop(
    model: str,
    messages: List[Dict],
    options: Dict[str, Any],
    layer_sizes: Dict[str, int],
    req_id: str,
) -> str:
    """Run the non-streaming tool loop and return the final assistant text.

    Shared by /api/chat (non-stream), /api/generate (non-stream), and /api/event.
    Mutates *messages* in-place (appends assistant + tool turns).
    Returns the final assistant content string.
    """
    prev_tc_names: Optional[List[str]] = None
    rep = 0
    resp: Dict = {}
    backend = backends.get_backend(model)

    for i in range(MAX_TOOL_ROUNDS):
        round_options = _options_for_round(options, i)
        built = build_model_messages(messages)
        payload = make_ollama_payload(model, built, round_options, stream=False)
        _log_payload_sizes(req_id, built, len(TOOLS), layer_sizes if i == 0 else None)
        log.debug("[%s] tool_loop round %d", req_id, i)

        resp       = backend.chat_sync(payload)
        msg        = resp.get("message") or {}
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            break

        tc_names = [(tc.get("function") or {}).get("name") for tc in tool_calls]
        log.info("[%s] tool_calls: %s", req_id, tc_names)

        if tc_names == prev_tc_names:
            rep += 1
            if rep >= 2:
                log.warning("[%s] tool loop detected, aborting", req_id)
                break
        else:
            rep = 0
        prev_tc_names = tc_names

        messages.append({
            "role":       "assistant",
            "content":    msg.get("content") or "",
            "tool_calls": tool_calls,
        })
        for tc in tool_calls:
            fn   = tc.get("function") or {}
            name = fn.get("name") or "unknown"
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            result = call_tool(name, args)
            if isinstance(result, dict) and "error" in result:
                log.warning("[%s] tool %s error: %s", req_id, name, result["error"])
            messages.append({
                "role":    "tool",
                "content": json.dumps(result, ensure_ascii=False),
            })

    return ((resp.get("message") or {}).get("content") or "").strip()


# ── CORS ───────────────────────────────────────────────────────────────────────

@app.after_request
def add_cors(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/chat",  methods=["OPTIONS"])
@app.route("/chat",      methods=["OPTIONS"])
@app.route("/api/event", methods=["OPTIONS"])
def options_chat():
    return Response(status=204)


# ── tools ──────────────────────────────────────────────────────────────────────

def load_tools():
    tools    = []
    registry = {}
    clients  = []

    for source, cmd in discover_mcp_servers().items():
        client = MCPClient(cmd)
        try:
            client.start()
        except Exception as e:
            log.error("[tools] %s: failed to start MCP server: %s", source, e)
            continue

        clients.append(client)

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

            if TOOL_OVERRIDE and name in TOOL_DESCRIPTION_OVERRIDES:
                description = TOOL_DESCRIPTION_OVERRIDES.get(name, description)

            registry[name] = client

            if TOOLS_FILTER:
                if name not in TOOLS_ALLOWED:
                    log.debug("[tools] hiding from model context: %s", name)
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

    return tools, registry, clients


def _stop_all_mcp_clients() -> None:
    for client in _MCP_CLIENTS:
        try:
            client.stop()
        except Exception as e:
            log.debug("[mcp] atexit stop error: %s", e)


def init_tools():
    global TOOLS, TOOL_REGISTRY, _MCP_CLIENTS
    log.info("[tools] initializing MCP tools...")
    TOOLS, TOOL_REGISTRY, _MCP_CLIENTS = load_tools()
    atexit.register(_stop_all_mcp_clients)
    log.info("[tools] total: %d tools in registry", len(TOOLS))


def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """Invoke a tool by name.

    Returns the tool result dict on success.  On failure (unknown tool,
    MCP error, exception) returns an error dict so the model can see what
    went wrong and potentially correct its call.
    """
    if name == "mempalacekgquery":
        entity = arguments.get("entity", "")
        if entity:
            fast = memory.kg_recall(entity)
            if fast:
                log.info(
                    "[tool] mempalacekgquery intercepted by FastMemory (entity=%s)",
                    entity,
                )
                return {"facts": fast, "source": "fast_memory"}

    client = TOOL_REGISTRY.get(name)
    if client is None:
        log.error("[tool] unknown tool requested: %s", name)
        return {"error": f"Unknown tool: {name}", "tool": name, "args": arguments}

    log.debug("[tool] calling %s with args: %s", name, arguments)
    try:
        result = client.call_tool(name, arguments)
    except Exception as exc:
        log.error("[tool] %s raised exception: %s", name, exc)
        return {"error": str(exc), "tool": name, "args": arguments}

    log.debug("[tool] %s result: %s", name, str(result)[:300])

    if isinstance(result, dict) and result.get("success") is False:
        err_msg = result.get("error") or "unknown MCP error"
        log.warning("[tool] %s returned MCP error: %s", name, err_msg)
        return {
            "error":       err_msg,
            "tool":        name,
            "args":        arguments,
            "mcp_result":  result,
        }

    if name in MEMPALACE_WRITE_TOOLS:
        memory.invalidate()
    return result


# ── compressed summary + external RAG indexing ────────────────────────────────

_SUMMARIZE_SYSTEM = (
    "You are a memory compressor for a personal AI assistant.\n"
    "Task: given an existing summary and new conversation turns, "
    "produce an UPDATED summary that merges both.\n"
    "Rules:\n"
    "- Do NOT repeat facts already in the existing summary verbatim.\n"
    "- Add only NEW facts, decisions, preferences, open questions from the new turns.\n"
    "- If a new turn contradicts the summary, update the fact (do not keep both).\n"
    "- Drop: greetings, tool call details, apologies, filler phrases.\n"
    "- Output: plain text, max 1250 words, no markdown, no bullet points.\n"
    "- Language: match the language of the conversation (Russian if Russian)."
)

_GLOBAL_SUMMARIZE_SYSTEM = (
    "You are a global activity compressor for a personal AI assistant.\n"
    "Task: merge the previous global summary with new conversation turns and "
    "keep only stable medium-term context about recent work, ongoing themes, "
    "and repeated user interests.\n"
    "Rules:\n"
    "- Keep it compact and factual.\n"
    "- Prefer persistent patterns over one-off details.\n"
    "- Remove stale or superseded facts.\n"
    "- Output: plain text, max 900 words, no markdown, no bullet points.\n"
    "- Language: match the language of the conversation (Russian if Russian)."
)


def _summarize_text(system_prompt: str, previous_summary: str, turns_text: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"Previous summary:\n{previous_summary or '(empty)'}\n\n"
            f"New exchanges:\n{turns_text}"
        )},
    ]
    payload = {
        "model":    CHATD_SUMMARY_MODEL,
        "messages": messages,
        "stream":   False,
        "think":    False,
        "options":  {"num_predict": 1500, "temperature": 0.2},
    }
    backend = backends.get_backend(CHATD_SUMMARY_MODEL)
    resp = backend.chat_sync(payload)
    return ((resp.get("message") or {}).get("content") or "").strip()


def _update_global_summary(turns_text: str) -> None:
    with _GLOBAL_SUMMARY_LOCK:
        previous = memory.read_global_summary()
        try:
            new_summary = _summarize_text(_GLOBAL_SUMMARIZE_SYSTEM, previous, turns_text)
            if new_summary and len(new_summary) >= 20:
                memory.write_global_summary(new_summary)
                log.info("[global-summary] updated (%d chars)", len(new_summary))
        except Exception as e:
            log.warning("[global-summary] update failed: %s", e)


def _compress_summary(session: sess.Session) -> None:
    """Compress ALL accumulated raw_turns into session.summary."""
    if not session.raw_turns:
        return

    turns = list(session.raw_turns)
    turns_text = "\n".join(
        f"User: {t['user']}\nAssistant: {t['assistant']}"
        for t in turns
    )
    log.debug(
        "[session:%s] compressing %d turns (%d chars) with model %s",
        session.session_id, len(turns), len(turns_text), CHATD_SUMMARY_MODEL,
    )
    try:
        new_summary = _summarize_text(_SUMMARIZE_SYSTEM, session.summary, turns_text)
        if new_summary and len(new_summary) >= 20:
            session.summary = new_summary
            for turn in turns:
                rag.index_turn(
                    source=f"session:{session.session_id}",
                    user=turn["user"],
                    assistant=turn["assistant"],
                )
            _update_global_summary(turns_text)
            session.raw_turns.clear()
            session.save()
            log.info(
                "[session:%s] summary updated (%d turns → %d chars)",
                session.session_id, len(turns), len(new_summary),
            )
        else:
            log.warning(
                "[session:%s] summariser returned empty content, clearing raw_turns anyway",
                session.session_id,
            )
            session.raw_turns.clear()
            session.save()
    except Exception as e:
        log.warning("[session:%s] compress failed: %s", session.session_id, e)


def maybe_compress_async(session: sess.Session) -> None:
    if len(session.raw_turns) < CHATD_COMPRESS_EVERY:
        return
    if not session._compress_lock.acquire(blocking=False):
        log.debug("[session:%s] compress already running, skipping", session.session_id)
        return

    def _run():
        try:
            _compress_summary(session)
        finally:
            session._compress_lock.release()

    t = threading.Thread(
        target=_run,
        daemon=True,
        name=f"compress-{session.session_id}",
    )
    t.start()
    log.debug("[session:%s] compression thread started", session.session_id)


def _backfill_session(session: sess.Session, messages: List[Dict]) -> None:
    if session.raw_turns or session.summary:
        return
    pairs: List[Dict] = []
    last_user: Optional[str] = None
    for m in messages:
        role = m.get("role")
        if role == "user":
            last_user = (m.get("content") or "")
        elif role == "assistant" and last_user is not None:
            pairs.append({
                "user":      last_user,
                "assistant": (m.get("content") or ""),
            })
            last_user = None
    if not pairs:
        return
    session.raw_turns = pairs[-10:]
    log.info(
        "[session:%s] backfilled %d turns from incoming history",
        session.session_id, len(session.raw_turns),
    )
    maybe_compress_async(session)


def _log_yield(req_id: str, label: str, data: bytes) -> bytes:
    log.debug("[%s] yield %-12s %d bytes", req_id, label, len(data))
    return data


# ── augmented /api/tags ────────────────────────────────────────────────────────

def _or_model_to_tag(model_name: str) -> Optional[Dict[str, Any]]:
    info = fetch_model_info(model_name)
    if info is None:
        return None

    context_length: int = info.get("context_length") or 0
    architecture: str   = (info.get("architecture") or {}).get("modality") or "text"
    description: str    = info.get("description") or ""

    return {
        "name":        model_name,
        "model":       model_name,
        "modified_at": now_iso(),
        "size":        context_length,
        "digest":      "",
        "details": {
            "family":         "openrouter",
            "families":       ["openrouter"],
            "format":         "api",
            "parameter_size": description[:64] if description else "",
            "quantization_level": architecture,
        },
    }


def _build_or_tags() -> List[Dict[str, Any]]:
    result = []
    for model_name in OPENROUTER_API_MODELS:
        entry = _or_model_to_tag(model_name)
        if entry:
            result.append(entry)
            log.debug("[tags] OR model added: %s", model_name)
        else:
            log.warning("[tags] OR model skipped (fetch failed): %s", model_name)
    return result


# ── routes ──────────────────────────────────────────────────────────────────────

@app.get("/")
@app.get("/api")
def root():
    return Response("Ollama is running", status=200, mimetype="text/plain")


@app.get("/version")
@app.get("/api/version")
def version():
    r = requests.get(f"{OLLAMA_API}/api/version")
    ollama_version = r.json()['version']
    log.debug("ollama version: %s", ollama_version)
    return jsonify({"chatd": f"{VERSION}", "ollama": f"{ollama_version}"})


@app.get("/health")
@app.get("/api/health")
def health():
    return jsonify({"ok": True, "tools": len(TOOLS)})


@app.post("/pull")
@app.post("/api/pull")
def pull():
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception as e:
        log.error("pull: bad JSON: %s", e)
        return jsonify({"error": "invalid JSON"}), 400

    model_name = payload.get("name") or payload.get("model") or ""
    log.debug("[pull] fake success for model %s", model_name)

    def _stream() -> Generator[bytes, None, None]:
        chunk = {
            "status":    "success",
            "digest":    "",
            "total":     1,
            "completed": 1,
            "model":     model_name,
        }
        yield (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")

    return Response(
        _stream(),
        content_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control":     "no-cache",
            "Transfer-Encoding": "chunked",
        },
    )


@app.post("/show")
@app.post("/api/show")
def show_model():
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception as e:
        log.error("show: bad JSON: %s", e)
        return jsonify({"error": "invalid JSON"}), 400

    model_name = payload.get("name") or payload.get("model")
    if not model_name:
        return jsonify({"error": "missing model name"}), 400

    if model_name.startswith(OPENROUTER_PREFIX):
        info = fetch_model_info(model_name)
        if info is None:
            return jsonify({"error": f"model not found: {model_name}"}), 404

        description = info.get("description") or ""
        context_len = info.get("context_length") or 0
        arch = (info.get("architecture") or {}).get("modality") or "text"

        result = {
            "model": model_name,
            "modified_at": now_iso(),
            "size": context_len,
            "digest": "",
            "details": {
                "format": "api",
                "family": "openrouter",
                "families": ["openrouter"],
                "parameter_size": description[:64],
                "quantization_level": arch,
            },
            "capabilities": ["chat", "tools"],
        }
        return jsonify(result)

    try:
        r = requests.post(
            f"{OLLAMA_API}/api/show",
            json={"name": model_name},
            timeout=600,
        )
        return Response(
            r.content,
            status=r.status_code,
            content_type=r.headers.get("Content-Type", "application/json"),
        )
    except Exception as e:
        log.error("[show] proxy failed: %s", e)
        return jsonify({"error": str(e)}), 502


@app.get("/tags")
@app.get("/api/tags")
def tags():
    r = requests.get(f"{OLLAMA_API}/api/tags", timeout=600)
    if not OPENROUTER_API_MODELS:
        return Response(
            r.content,
            status=r.status_code,
            content_type=r.headers.get("Content-Type", "application/json"),
        )

    try:
        data = r.json()
    except Exception:
        return Response(
            r.content,
            status=r.status_code,
            content_type=r.headers.get("Content-Type", "application/json"),
        )

    or_entries = _build_or_tags()
    data.setdefault("models", [])
    data["models"].extend(or_entries)
    log.info("[tags] ollama=%d or=%d total=%d",
             len(data["models"]) - len(or_entries),
             len(or_entries),
             len(data["models"]))
    return jsonify(data)


def _options_for_round(options: Dict[str, Any], round_num: int) -> Dict[str, Any]:
    """Return options dict adjusted for the current tool round.

    Round 0: options as-is.
    Round 1+: num_predict boosted by TOOL_ROUND_NUM_PREDICT_BOOST.
    """
    if round_num == 0:
        return options
    boosted = dict(options)
    base = boosted.get("num_predict", 512)
    boosted["num_predict"] = base + TOOL_ROUND_NUM_PREDICT_BOOST
    log.debug("tool round %d: num_predict %d -> %d", round_num, base, boosted["num_predict"])
    return boosted


def chat_stream_generator(
    model: str,
    messages: List[Dict],
    options: Dict[str, Any],
    session: Optional[sess.Session],
    layer_sizes: Dict[str, int],
    req_id: str = "-",
) -> Generator[bytes, None, None]:
    prev_tool_names: Optional[List[str]] = None
    repeat_count = 0

    last_user_msg = _last_user_text(messages)

    data = make_keepalive(model)
    log.debug("[%s] yield %-12s %d bytes", req_id, "keepalive/0", len(data))
    yield data

    final_assistant_content = ""
    backend = backends.get_backend(model)

    try:
        for round_num in range(MAX_TOOL_ROUNDS):
            log.info("[%s] stream round %d, messages=%d", req_id, round_num, len(messages))

            round_options = _options_for_round(options, round_num)
            built_messages = build_model_messages(messages)
            payload = make_ollama_payload(model, built_messages, round_options, stream=True)
            _log_payload_sizes(req_id, built_messages, len(TOOLS),
                               layer_sizes if round_num == 0 else None)
            log.debug(
                "[%s] -> backend payload (messages=%d, tools=%d, options=%s, think=%s)",
                req_id, len(built_messages), len(TOOLS), round_options, THINKING,
            )

            last_tool_calls: Optional[List[Dict]] = None
            chunk_count = 0
            remapper = ThinkingRemapper(model)

            try:
                stream = backend.chat_stream(payload)
            except Exception as e:
                log.error("[%s] backend stream init failed: %s", req_id, e)
                yield make_chunk(model, f"\n[Ошибка соединения: {e}]\n", done=True)
                return

            try:
                for line in stream:
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8") if isinstance(line, bytes) else line)
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
                            final_assistant_content = remapper.content_acc
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

            except RequestException as e:
                log.error("[%s] network error during stream: %s", req_id, e)
                yield make_chunk(
                    model,
                    f"\n[Ошибка сети: {type(e).__name__}: {e}]\n",
                    done=True,
                )
                return

            log.debug("[%s] exited stream, last_tool_calls=%s", req_id, bool(last_tool_calls))

            if not last_tool_calls:
                log.info("[%s] stream complete, no more tool rounds", req_id)
                break

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
                yield _log_yield(req_id, "tool/ann", make_chunk(model, f"\n[→ {tool_name}]\n\n"))

                tool_result = call_tool(tool_name, tool_args)

                if isinstance(tool_result, dict) and "error" in tool_result:
                    log.warning(
                        "[%s] tool %s error (passed to model): %s",
                        req_id, tool_name, tool_result["error"],
                    )

                messages.append({
                    "role":    "tool",
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

            yield _log_yield(req_id, "keepalive/tr", make_keepalive(model))
            last_tool_calls = None

        log.debug("[%s] generator exiting normally", req_id)

    except Exception as e:
        log.error("[%s] unhandled exception: %s", req_id, traceback.format_exc())
        yield make_chunk(model, f"\n[internal error: {e}]\n", done=True)

    finally:
        if session and last_user_msg and final_assistant_content:
            log.debug(
                "[%s] [finally] saving turn: session=%s user=%d chars assistant=%d chars",
                req_id, session.session_id,
                len(last_user_msg), len(final_assistant_content),
            )
            sess.record_turn(session, last_user_msg, final_assistant_content)
            maybe_compress_async(session)
        else:
            log.debug(
                "[%s] [finally] skipping save: session=%s user_empty=%s answer_empty=%s",
                req_id,
                session.session_id if session else "None",
                not last_user_msg,
                not final_assistant_content,
            )


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

    chat_id = extract_chat_id(request)
    session = sess.get_session(chat_id) if chat_id else None

    if session:
        _backfill_session(session, messages)

    history_depth = sum(1 for m in messages if m.get("role") in ("user", "assistant"))
    if history_depth <= 1:
        summary = ""
        log.debug("[%s] fresh chat (depth=%d), suppressing summary", req_id, history_depth)
    else:
        summary = session.summary if session else ""

    global_summary = memory.read_global_summary()

    log.info("[%s] chat_id=%s summary_len=%d global_summary_len=%d",
             req_id, chat_id, len(summary), len(global_summary))

    messages, layer_sizes = ensure_system_prompt(
        messages,
        per_chat_summary=summary,
        global_summary=global_summary,
    )

    log.info("[%s] POST /api/chat model=%s stream=%s messages=%d options=%s",
             req_id, model, want_stream, len(messages), options)

    if not want_stream:
        try:
            answer = run_tool_loop(model, messages, options, layer_sizes, req_id)

            if session and answer:
                last_user = _last_user_text(messages)
                if last_user:
                    log.debug(
                        "[%s] non-stream: saving turn session=%s user=%d chars answer=%d chars",
                        req_id, session.session_id, len(last_user), len(answer),
                    )
                    sess.record_turn(session, last_user, answer)
                    maybe_compress_async(session)

            log.info("[%s] non-stream done, answer_len=%d", req_id, len(answer))
            return jsonify({"model": model, "answer": answer})

        except ReadTimeout:
            log.error("[%s] backend timeout", req_id)
            return jsonify({"error": "backend timeout"}), 504
        except Exception as e:
            log.error("[%s] non-stream error: %s", req_id, traceback.format_exc())
            return jsonify({"error": str(e)}), 502

    log.info("[%s] starting stream generator", req_id)
    return Response(
        stream_with_context(
            chat_stream_generator(model, messages, options, session, layer_sizes, req_id)
        ),
        content_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control":     "no-cache",
            "Transfer-Encoding": "chunked",
        },
    )


@app.post("/generate")
@app.post("/api/generate")
def generate():
    req_id = uuid.uuid4().hex[:8]

    try:
        original_payload = request.get_json(force=True, silent=False)
    except Exception as e:
        log.error("[%s] generate: failed to parse request JSON: %s", req_id, e)
        return jsonify({"error": "invalid JSON"}), 400

    model       = original_payload.get("model") or "qwen3:8b"
    prompt      = original_payload.get("prompt") or ""
    system_text = original_payload.get("system") or ""
    want_stream = original_payload.get("stream", True)
    options     = merge_options(original_payload.get("options"))

    session = None

    messages: List[Dict[str, Any]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": prompt})

    global_summary = memory.read_global_summary()

    log.info(
        "[%s] POST /api/generate model=%s stream=%s prompt_len=%d",
        req_id, model, want_stream, len(prompt),
    )

    messages, layer_sizes = ensure_system_prompt(
        messages,
        per_chat_summary="",
        global_summary=global_summary,
    )

    if not want_stream:
        try:
            answer = run_tool_loop(model, messages, options, layer_sizes, req_id)
            log.info("[%s] generate non-stream done, answer_len=%d", req_id, len(answer))
            return jsonify({
                "model":      model,
                "created_at": now_iso(),
                "response":   answer,
                "done":       True,
            })
        except ReadTimeout:
            log.error("[%s] generate backend timeout", req_id)
            return jsonify({"error": "backend timeout"}), 504
        except Exception as e:
            log.error("[%s] generate non-stream error: %s", req_id, traceback.format_exc())
            return jsonify({"error": str(e)}), 502

    def generate_stream():
        inner = chat_stream_generator(model, messages, options, session, layer_sizes, req_id)
        for raw in inner:
            try:
                obj = json.loads(raw.decode("utf-8"))
            except Exception:
                yield raw
                continue

            msg     = obj.get("message") or {}
            content = msg.get("content") or ""

            out = {
                "model":      obj.get("model") or model,
                "created_at": obj.get("created_at") or now_iso(),
                "response":   content,
                "done":       obj.get("done", False),
            }
            if obj.get("done"):
                out["done_reason"] = obj.get("done_reason", "stop")

            yield (json.dumps(out, ensure_ascii=False) + "\n").encode("utf-8")

    log.info("[%s] starting generate stream", req_id)
    return Response(
        stream_with_context(generate_stream()),
        content_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control":     "no-cache",
            "Transfer-Encoding": "chunked",
        },
    )


# ── /api/event ──────────────────────────────────────────────────────────────────────

@app.post("/api/event")
def system_event():
    req_id = uuid.uuid4().hex[:8]

    denied = _check_event_token()
    if denied:
        return denied

    try:
        ev = request.get_json(force=True, silent=False) or {}
    except Exception as e:
        log.error("[%s] event: bad JSON: %s", req_id, e)
        return jsonify({"error": "invalid JSON"}), 400

    event_type   = ev.get("type", "event")
    event_source = ev.get("source", "unknown")
    # Use explicit "payload" key when present; otherwise collect all non-reserved fields.
    # Check key existence explicitly so that an empty dict payload is preserved.
    if "payload" in ev:
        event_payload = ev["payload"]
    else:
        event_payload = {k: v for k, v in ev.items()
                         if k not in ("type", "source", "model", "options")}
    model   = ev.get("model") or CHATD_EVENT_MODEL or "qwen3:8b"
    options = merge_options(ev.get("options"))

    log.info(
        "[%s] system event: type=%s source=%s model=%s",
        req_id, event_type, event_source, model,
    )

    event_text = (
        f"[SYSTEM EVENT] type={event_type} source={event_source}\n"
        f"{json.dumps(event_payload, ensure_ascii=False, indent=2)}"
        f"{_EVENT_DIRECTIVE}"
    )
    messages: List[Dict[str, Any]] = [{"role": "user", "content": event_text}]

    global_summary = memory.read_global_summary()
    messages, layer_sizes = ensure_system_prompt(
        messages,
        per_chat_summary="",
        global_summary=global_summary,
    )

    # Run tool loop in a daemon thread with a hard timeout so a hanging model
    # cannot block a gunicorn worker indefinitely.
    result_box: Dict[str, Any] = {}

    def _run():
        try:
            result_box["answer"] = run_tool_loop(model, messages, options, layer_sizes, req_id)
        except Exception as exc:
            result_box["exc"] = exc

    t = threading.Thread(target=_run, daemon=True, name=f"event-{req_id}")
    t.start()
    t.join(timeout=EVENT_TOOL_LOOP_TIMEOUT)

    if t.is_alive():
        log.error("[%s] event: tool loop timed out after %ds", req_id, EVENT_TOOL_LOOP_TIMEOUT)
        return jsonify({"error": "tool loop timeout"}), 504

    if "exc" in result_box:
        exc = result_box["exc"]
        if isinstance(exc, ReadTimeout):
            log.error("[%s] event: backend timeout", req_id)
            return jsonify({"error": "backend timeout"}), 504
        log.error("[%s] event error: %s", req_id, traceback.format_exc())
        return jsonify({"error": str(exc)}), 502

    answer = result_box.get("answer", "")
    log.info("[%s] event done, answer_len=%d", req_id, len(answer))
    return jsonify({
        "ok":       True,
        "req_id":   req_id,
        "response": answer,
    })


init_tools()
memory.init()
rag.init()

if __name__ == "__main__":
    log.info("started")
    app.run(host="0.0.0.0", port=5001, debug=False)

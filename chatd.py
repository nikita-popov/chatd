import functools
import json
import logging
import os
import re
import subprocess
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

import requests
from requests.exceptions import ReadTimeout
from flask import Flask, request, Response, jsonify, stream_with_context

from mcp_client import MCPClient, MCP_MONITOR_CMD, MCP_ALERTS_CMD, MCP_NOTES_CMD, MCP_MEMPALACE_CMD

# ─── logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chatd")

# ─── context ─────────────────────────────────────────────────────────────────

def _run_wakeup() -> str:
    """Execute mempalace wake-up and return raw stdout (L0 + L1 text)."""
    env = {**os.environ, "MEMPALACE_PALACE_PATH": "/var/lib/mempalace"}
    try:
        result = subprocess.run(
            ["/opt/chatd/venv/bin/python", "-m", "mempalace", "wake-up"],
            capture_output=True, text=True, timeout=10, env=env,
        )
        if result.returncode != 0:
            log.warning("mempalace wake-up exited %d: %s", result.returncode, result.stderr.strip())
        return result.stdout.strip()
    except Exception as e:
        log.warning("mempalace wake-up failed: %s", e)
        return ""


@functools.lru_cache(maxsize=1)
def _wakeup_cached() -> str:
    """Return cached wake-up text.

    Cache is a single slot — subsequent calls return the same string until
    cache_clear() is called (e.g. after mempalace_kg_add writes new facts).
    """
    text = _run_wakeup()
    log.info("wake-up cache populated (%d chars)", len(text))
    return text


def get_system_prompt() -> str:
    """Return current system prompt (L0 + L1).

    Uses lru_cache so the subprocess is only spawned once per cache epoch.
    Cache is invalidated by invalidate_wakeup_cache() after kg_add writes.
    Falls back to empty string on error — ensure_system_prompt handles that.
    """
    return _wakeup_cached()


def invalidate_wakeup_cache() -> None:
    """Evict the lru_cache so the next request re-runs wake-up."""
    _wakeup_cached.cache_clear()
    log.info("wake-up cache invalidated")


# ─── config ──────────────────────────────────────────────────────────────────

OLLAMA_API = "http://127.0.0.1:11434"

# Set to True to enable qwen3 extended thinking (<think>...</think>).
# When False, "think": false is sent with every Ollama request.
THINKING = False

TOOL_DESCRIPTION_OVERRIDES = {
    "alerts_list": (
        "List Alertmanager alerts. "
        "Call ONLY when the user asks about server alerts, incidents, or firing rules."
    ),
    "alerts_summary": (
        "Summarize active Alertmanager alerts by name/severity/state. "
        "Call ONLY when asked for a monitoring overview or alert statistics."
    ),
    "monitor_query": (
        "Query VictoriaMetrics with PromQL. "
        "Call ONLY when the user asks for specific metrics (CPU, RAM, uptime, etc.)."
    ),
    "notes_search": (
        "Search Memos notes by text query. "
        "Call ONLY when the user asks to find or list their notes."
    ),
    "mempalace_status": (
        "Returns palace overview (wings, rooms, drawer count), AAAK dialect spec, "
        "and memory protocol instructions. "
        "Call ONCE at the start of every session before using any other memory tool. "
        "The returned protocol tells you how to correctly use add_drawer and search."
    ),
    "mempalace_search": (
        "Semantic vector search over all stored memories (drawers). "
        "Call when the user asks about their preferences, habits, past decisions, "
        "or any personal fact not present in the current conversation. "
        "Use the 'wing' and 'room' filters to narrow results when the topic is clear "
        "(e.g. wing='Person', room='preferences'). "
        "Fall back to this if mempalace_kg_query returned no results."
    ),
    "mempalace_add_drawer": (
        "File verbatim text into the palace (writes to ChromaDB, persists long-term). "
        "Use for ANY content longer than a short phrase: paragraphs, user bios, "
        "conversation summaries, decisions, preferences in natural language. "
        "Required args: "
        "'wing' (domain, e.g. 'wing_general' or 'wing_person'), "
        "'room' (topic, e.g. 'hall_preferences' or 'hall_facts'), "
        "'content' (the verbatim text to store, in AAAK dialect if possible). "
        "Call after EVERY turn where the user shared personal facts worth keeping. "
        "Do NOT call for greetings or transient context."
    ),
    "mempalace_kg_query": (
        "Query the structured knowledge graph for facts about a named entity. "
        "Call FIRST (before mempalace_search) when the user asks about their own "
        "attributes: favorite tools, location, job, language, etc. "
        "Required arg: 'entity' (e.g. 'user'). "
        "Returns subject→predicate→object triples with temporal validity. "
        "If result is empty, follow up with mempalace_search."
    ),
    "mempalace_kg_add": (
        "Add ONE atomic subject→predicate→object triple to the knowledge graph. "
        "Use for short, structured facts ONLY: single values, not sentences or paragraphs. "
        "Args MUST be ASCII, no Cyrillic, no spaces in values. "
        "Good: subject='user', predicate='favorite_editor', object='emacs'. "
        "Good: subject='user', predicate='location', object='Omsk'. "
        "BAD: object='Enthusiast of computer networks, programs in C, Go...' — "
        "use mempalace_add_drawer for multi-sentence content instead. "
        "When given a block of text with multiple facts: decompose into individual "
        "triples (one call per fact) AND also file the full text via add_drawer. "
        "Call ONLY when the user EXPLICITLY asks you to remember a specific fact."
    ),
}

# Sampling options.
# Tuned for qwen3 non-thinking mode (THINKING=False).
# When THINKING=True you may want to raise temperature to 0.7-1.0
# and remove repeat_penalty to let the model explore freely.
DEFAULT_OPTIONS: Dict[str, Any] = {
    "num_predict":    768,
    "num_ctx":        8192,
    "temperature":    0.3,
    "top_p":          0.9,
    "top_k":          40,
    "repeat_penalty": 1.05,
}

MEMPALACE_ALLOWED_TOOLS = {
    "mempalace_status",
    "mempalace_search",
    "mempalace_add_drawer",
    "mempalace_kg_query",
    # second priority — add when needed:
    # "mempalace_list_wings",
    "mempalace_kg_add",
    # "mempalace_memories_filed_away",
}

# Tools that write to memory and require wake-up cache invalidation afterward.
MEMPALACE_WRITE_TOOLS = {
    "mempalace_kg_add",
    "mempalace_add_drawer",
}

TOOLS: List[Dict] = []
TOOL_REGISTRY: Dict[str, MCPClient] = {}

app = Flask(__name__)


# ─── helpers ─────────────────────────────────────────────────────────────────

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
    """Empty content chunk to keep the HTTP connection alive.

    Yielded at stream start so the browser sees an immediate response,
    preventing 499 while Ollama loads the model into VRAM.
    Also yielded when tool_calls arrive (done=True but no content yet)
    and after all tools complete before the next Ollama round-trip.
    """
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
    system_prompt = get_system_prompt()
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
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "stream": stream,
        "options": options,
    }
    if not THINKING:
        payload["think"] = False
    return payload


# ─── CORS ────────────────────────────────────────────────────────────────────

@app.after_request
def add_cors(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/chat", methods=["OPTIONS"])
@app.route("/chat", methods=["OPTIONS"])
def options_chat():
    return Response(status=204)


# ─── tools ───────────────────────────────────────────────────────────────────

def load_tools():
    tools = []
    registry = {}
    servers = {
        #"monitor":   MCPClient(MCP_MONITOR_CMD),
        #"notes":     MCPClient(MCP_NOTES_CMD),
        #"alerts":    MCPClient(MCP_ALERTS_CMD),
        "mempalace": MCPClient(MCP_MEMPALACE_CMD),
    }
    for source, client in servers.items():
        server_tools = client.list_tools()
        log.info("[tools] %s: %d tools loaded", source, len(server_tools))
        for t in server_tools:
            name = getattr(t, "name", None)
            description = getattr(t, "description", "") or ""
            # Override description if specified to improve model tool-selection.
            description = TOOL_DESCRIPTION_OVERRIDES.get(name, description)
            input_schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)
            if not name or not input_schema:
                log.warning("[tools] %s: skipping tool without name or schema: %r", source, t)
                continue
            registry[name] = client
            if source == "mempalace" and name not in MEMPALACE_ALLOWED_TOOLS:
                log.debug("[tools] mempalace: skipping from model context: %s", name)
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": input_schema,
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
    # Invalidate wake-up cache so the next request sees fresh L1 facts.
    if name in MEMPALACE_WRITE_TOOLS:
        invalidate_wakeup_cache()
    return result


# ─── ThinkingRemapper ────────────────────────────────────────────────────────

class ThinkingRemapper:
    """
    Converts Ollama's message.thinking stream into <think>...</think> in
    message.content so frontends that only read content still receive thinking.

    State machine:
      idle     - no thinking seen yet
      thinking - inside a thinking block, <think> already emitted
      closed   - </think> emitted, back to normal content

    thinking_acc: raw thinking text (no tags), for logging/debugging.
    content_acc:  only the real response text, used for conversation history.
    """

    def __init__(self, model: str):
        self.model = model
        self._state: str = "idle"
        self.thinking_acc: str = ""
        self.content_acc: str = ""

    def feed(self, line: bytes) -> bytes:
        """Process one NDJSON line. Returns the (possibly rewritten) line."""
        try:
            chunk = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return line

        msg      = chunk.get("message") or {}
        thinking = msg.get("thinking") or ""
        content  = msg.get("content")  or ""

        self.thinking_acc += thinking
        self.content_acc  += content

        if not thinking and self._state == "idle":
            return line

        if "thinking" in msg:
            del msg["thinking"]

        if thinking:
            if self._state == "idle":
                self._state = "thinking"
                msg["content"] = "<think>" + thinking
            else:
                msg["content"] = thinking
        else:
            if self._state == "thinking":
                self._state = "closed"
                msg["content"] = "</think>" + content

        chunk["message"] = msg
        return (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")

    def close(self) -> Optional[bytes]:
        """If stream ends while still in thinking state, emit closing tag."""
        if self._state == "thinking":
            self._state = "closed"
            return make_chunk(self.model, "</think>")
        return None


# ─── stream yield helper ─────────────────────────────────────────────────────

def _yield_chunk(req_id: str, label: str, data: bytes):
    """Log every outgoing chunk with its label and byte size, then yield it."""
    log.debug("[%s] yield %-12s %d bytes: %s", req_id, label, len(data), data[:120])
    return data


# ─── routes ──────────────────────────────────────────────────────────────────

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

    # Yield an empty chunk immediately so the browser sees a response and does
    # not abort the request while Ollama loads the model into VRAM (499 fix).
    data = make_keepalive(model)
    log.debug("[%s] yield %-12s %d bytes", req_id, "keepalive/0", len(data))
    yield data

    try:
        for round_num in range(MAX_TOOL_ROUNDS):
            log.info("[%s] stream round %d, messages=%d", req_id, round_num, len(messages))

            payload = make_ollama_payload(
                model,
                build_model_messages(messages),
                options,
                stream=True,
            )

            log.debug("[%s] -> Ollama payload (messages count=%d, tools=%d, options=%s, think=%s)",
                      req_id, len(payload["messages"]), len(TOOLS), options, THINKING)

            try:
                r = requests.post(
                    f"{OLLAMA_API}/api/chat",
                    json=payload,
                    timeout=3600,
                    stream=True,
                )
                r.raise_for_status()
                log.debug("[%s] Ollama HTTP %d, headers: %s",
                          req_id, r.status_code, dict(r.headers))
            except Exception as e:
                log.error("[%s] Ollama request failed: %s", req_id, e)
                data = make_chunk(model, f"\n[error: {e}]\n", done=True)
                log.debug("[%s] yield %-12s %d bytes", req_id, "error", len(data))
                yield data
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
                        log.info("[%s] tool_calls received: %s",
                                 req_id, [c.get("function", {}).get("name") for c in tc])
                        last_tool_calls = tc

                    if done:
                        log.info("[%s] round %d done, chunks=%d, tool_calls=%s, content_len=%d",
                                 req_id, round_num, chunk_count,
                                 bool(last_tool_calls), len(remapper.content_acc))
                        closing = remapper.close()
                        if closing:
                            log.debug("[%s] yield %-12s %d bytes", req_id, "think/close", len(closing))
                            yield closing
                        if last_tool_calls:
                            data = make_keepalive(model)
                            log.debug("[%s] yield %-12s %d bytes", req_id, "keepalive/tc", len(data))
                            yield data
                        else:
                            data = remapper.feed(line)
                            log.debug("[%s] yield %-12s %d bytes", req_id, "done", len(data))
                            yield data
                        break

                    out = remapper.feed(line)
                    if not tc:
                        log.debug("[%s] yield %-12s %d bytes: ...%s",
                                  req_id, f"chunk/{chunk_count}", len(out),
                                  out[-60:].decode("utf-8", errors="replace").strip())
                        yield out

            log.debug("[%s] exited Ollama iter_lines loop, last_tool_calls=%s",
                      req_id, bool(last_tool_calls))

            if not last_tool_calls:
                log.info("[%s] stream complete, no more tool rounds", req_id)
                break

            # ── tool loop detection ─────────────────────────────────────────
            current_tool_names = [
                (tc.get("function") or {}).get("name") for tc in last_tool_calls
            ]
            if current_tool_names == prev_tool_names:
                repeat_count += 1
                if repeat_count >= 2:
                    log.warning("[%s] tool loop detected (%s x%d), aborting",
                                req_id, current_tool_names, repeat_count)
                    data = make_chunk(model, "\n[ошибка: цикл инструментов, остановлено]\n", done=True)
                    log.debug("[%s] yield %-12s %d bytes", req_id, "loop/abort", len(data))
                    yield data
                    return
            else:
                repeat_count = 0
            prev_tool_names = current_tool_names

            # ── tool execution ────────────────────────────────────────────
            messages.append({
                "role": "assistant",
                "content": remapper.content_acc,
                "tool_calls": last_tool_calls,
            })

            for tc in last_tool_calls:
                fn = tc.get("function") or {}
                tool_name = fn.get("name") or "unknown"
                tool_args = fn.get("arguments") or {}
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        log.warning("[%s] failed to parse tool_args for %s", req_id, tool_name)
                        tool_args = {}

                log.info("[%s] executing tool: %s(%s)", req_id, tool_name, tool_args)
                data = make_chunk(model, f"\n[→ {tool_name}]\n")
                log.debug("[%s] yield %-12s %d bytes", req_id, "tool/ann", len(data))
                yield data

                try:
                    tool_result = call_tool(tool_name, tool_args)
                except Exception as e:
                    log.error("[%s] tool %s failed: %s", req_id, tool_name, e)
                    tool_result = {"error": str(e)}

                messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

            data = make_keepalive(model)
            log.debug("[%s] yield %-12s %d bytes", req_id, "keepalive/tr", len(data))
            yield data
            last_tool_calls = None

        log.debug("[%s] generator exiting normally", req_id)

    except GeneratorExit:
        log.warning("[%s] GeneratorExit: client disconnected mid-stream", req_id)
    except Exception as e:
        log.error("[%s] unhandled exception in generator: %s", req_id, traceback.format_exc())
        data = make_chunk(model, f"\n[internal error: {e}]\n", done=True)
        log.debug("[%s] yield %-12s %d bytes", req_id, "exception", len(data))
        yield data


@app.post("/chat")
@app.post("/api/chat")
def chat():
    import uuid
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
                ollama_payload = make_ollama_payload(
                    model,
                    build_model_messages(messages),
                    options,
                    stream=False,
                )

                log.debug("[%s] non-stream round %d, think=%s", req_id, i, THINKING)
                r = requests.post(
                    f"{OLLAMA_API}/api/chat",
                    json=ollama_payload,
                    timeout=3600,
                )
                r.raise_for_status()
                resp = r.json()
                msg = resp.get("message") or {}
                tool_calls = msg.get("tool_calls") or []

                if not tool_calls:
                    break

                tc_names = [(tc.get("function") or {}).get("name") for tc in tool_calls]
                log.info("[%s] non-stream tool_calls: %s", req_id, tc_names)

                if tc_names == prev_tc_names:
                    rep += 1
                    if rep >= 2:
                        log.warning("[%s] non-stream tool loop detected, aborting", req_id)
                        return jsonify({"error": "tool loop detected"}), 500
                else:
                    rep = 0
                prev_tc_names = tc_names

                messages.append({
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": tool_calls,
                })

                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    tool_name = fn.get("name")
                    tool_args = fn.get("arguments") or {}
                    if isinstance(tool_args, str):
                        tool_args = json.loads(tool_args)
                    try:
                        tool_result = call_tool(tool_name, tool_args)
                    except Exception as e:
                        tool_result = {"error": str(e)}
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    })

            msg = resp.get("message") or {}
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
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked",
        },
    )


# ─── entry point ─────────────────────────────────────────────────────────────

init_tools()

if __name__ == "__main__":
    log.info("started")
    app.run(host="0.0.0.0", port=5001, debug=False)

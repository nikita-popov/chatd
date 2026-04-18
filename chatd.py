#!/usr/bin/env python3
import json
import logging
import os
import re
import threading
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

import requests
from requests.exceptions import ReadTimeout
from flask import Flask, request, Response, jsonify, stream_with_context

import memory
import rag
import session as sess
from config import (
    OLLAMA_API,
    THINKING,
    TOOLS_FILTER,
    TOOLS_ALLOWED,
    DEFAULT_OPTIONS,
    MEMPALACE_WRITE_TOOLS,
    TOOL_OVERRIDE,
    TOOL_DESCRIPTION_OVERRIDES,
    CHATD_SUMMARY_MODEL,
    CHATD_COMPRESS_EVERY,
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

# ── app ───────────────────────────────────────────────────────────────────────────

TOOLS: List[Dict] = []
TOOL_REGISTRY: Dict[str, MCPClient] = {}

app = Flask(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────────

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


def extract_chat_id(messages: List[Dict]) -> Optional[str]:
    """Return chatId from the first message that carries it, or None."""
    for m in messages:
        chat_id = m.get("chatId")
        if chat_id is not None:
            return str(chat_id)
    return None


def _last_user_text(messages: List[Dict]) -> str:
    """Return the content of the most recent user message, or empty string."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return (m.get("content") or "")
    return ""


def ensure_system_prompt(
    messages: List[Dict[str, Any]],
    summary: str = "",
    global_summary: str = "",
) -> List[Dict[str, Any]]:
    """Replace or prepend the system message with the full L0+L0.5+L1+L1.5 block.

    Also performs two sidecar enrichments for the last user message:
    1. KG recall for entity-like facts.
    2. External RAG retrieval for semantically related past turns.
    """
    base_prompt = memory.wake_up(summary=summary, global_summary=global_summary)

    last_user = _last_user_text(messages)
    if last_user:
        extra = memory.kg_recall_from_text(last_user)
        if extra:
            log.debug("[sidecar] injecting %d-char KG recall into system prompt", len(extra))
            base_prompt = f"{base_prompt}\n\n# Recalled facts\n{extra}"

        rag_ctx = rag.retrieve(last_user)
        if rag_ctx:
            log.debug("[sidecar] injecting %d-char RAG recall into system prompt", len(rag_ctx))
            base_prompt = f"{base_prompt}\n\n# Relevant past context\n{rag_ctx}"

    system_prompt = base_prompt

    if not isinstance(messages, list) or not messages:
        return [{"role": "system", "content": system_prompt}]
    if messages[0].get("role") != "system":
        return [{"role": "system", "content": system_prompt}] + messages
    return [{"role": "system", "content": system_prompt}] + messages[1:]


def build_model_messages(
    raw_messages: List[Dict],
    max_history_turns: int = 20,
) -> List[Dict]:
    """Sanitise messages for the Ollama API.

    - Strip fields unknown to Ollama (chatId, id, createdAt, updatedAt …).
    - Remove <think>…</think> blocks from previous assistant turns.
    - Keep system message as-is (already assembled by ensure_system_prompt).
    - Limit history to *max_history_turns* user+assistant pairs to prevent
      context-window overflow.  The system message is never trimmed.
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


# ── CORS ─────────────────────────────────────────────────────────────────────────────

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


# ── tools ─────────────────────────────────────────────────────────────────────────

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

    return tools, registry


def init_tools():
    global TOOLS, TOOL_REGISTRY
    log.info("[tools] initializing MCP tools...")
    TOOLS, TOOL_REGISTRY = load_tools()
    log.info("[tools] total: %d tools in registry", len(TOOLS))


def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
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
        raise RuntimeError(f"Unknown tool: {name}")
    log.debug("[tool] calling %s with args: %s", name, arguments)
    result = client.call_tool(name, arguments)
    log.debug("[tool] %s result: %s", name, str(result)[:200])
    if name in MEMPALACE_WRITE_TOOLS:
        memory.invalidate()
    return result


# ── L1.5: rolling summary + external RAG indexing ─────────────────────────────

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
    r = requests.post(f"{OLLAMA_API}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    return ((r.json().get("message") or {}).get("content") or "").strip()


def _update_global_summary(turns_text: str) -> None:
    previous = memory.read_global_summary()
    try:
        new_summary = _summarize_text(_GLOBAL_SUMMARIZE_SYSTEM, previous, turns_text)
        if new_summary and len(new_summary) >= 20:
            memory.write_global_summary(new_summary)
            log.info("[global-summary] updated (%d chars)", len(new_summary))
    except Exception as e:
        log.warning("[global-summary] update failed: %s", e)


def _compress_summary(session: sess.Session) -> None:
    """Blocking compress call — meant to run in a background thread."""
    if not session.raw_turns:
        return

    turns = session.raw_turns[-CHATD_COMPRESS_EVERY:]
    turns_text = "\n".join(
        f"User: {t['user']}\nAssistant: {t['assistant']}"
        for t in turns
    )
    try:
        log.debug("[session:%s] compression model: %s", session.session_id, CHATD_SUMMARY_MODEL)
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
                "[session:%s] summary updated (%d chars)",
                session.session_id, len(new_summary),
            )
        else:
            log.warning("[session:%s] summariser returned empty content", session.session_id)
            session.raw_turns.clear()
            session.save()
    except Exception as e:
        log.warning("[session:%s] compress failed: %s", session.session_id, e)


def maybe_compress_async(session: sess.Session) -> None:
    if len(session.raw_turns) < CHATD_COMPRESS_EVERY:
        return
    t = threading.Thread(
        target=_compress_summary,
        args=(session,),
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
    session: Optional[sess.Session],
    req_id: str = "-",
) -> Generator[bytes, None, None]:
    MAX_TOOL_ROUNDS = 5
    prev_tool_names: Optional[List[str]] = None
    repeat_count = 0

    last_user_msg = _last_user_text(messages)

    data = make_keepalive(model)
    log.debug("[%s] yield %-12s %d bytes", req_id, "keepalive/0", len(data))
    yield data

    final_assistant_content = ""

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

            log.debug("[%s] exited iter_lines, last_tool_calls=%s", req_id, bool(last_tool_calls))

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

    chat_id = extract_chat_id(messages)
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

    log.info("[%s] chat_id=%s summary_len=%d global_summary_len=%d", req_id, chat_id, len(summary), len(global_summary))

    messages = ensure_system_prompt(messages, summary=summary, global_summary=global_summary)

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
            return jsonify({"model": model, "answer": answer, "raw": resp})

        except ReadTimeout:
            log.error("[%s] ollama timeout", req_id)
            return jsonify({"error": "ollama timeout"}), 504
        except Exception as e:
            log.error("[%s] non-stream error: %s", req_id, traceback.format_exc())
            return jsonify({"error": str(e)}), 502

    log.info("[%s] starting stream generator", req_id)
    return Response(
        stream_with_context(
            chat_stream_generator(model, messages, options, session, req_id)
        ),
        content_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control":     "no-cache",
            "Transfer-Encoding": "chunked",
        },
    )


init_tools()
memory.init()
rag.init()

if __name__ == "__main__":
    log.info("started")
    app.run(host="0.0.0.0", port=5001, debug=False)

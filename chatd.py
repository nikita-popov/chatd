import json
import logging
import re
import traceback
from typing import Any, Dict, Generator, List, Optional

import requests
from requests.exceptions import ReadTimeout
from flask import Flask, request, Response, jsonify, stream_with_context

from mcp_client import MCPClient, MCP_MONITOR_CMD, MCP_ALERTS_CMD, MCP_NOTES_CMD

# ─── logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chatd")

# ─── config ─────────────────────────────────────────────────────────────────

OLLAMA_API = "http://127.0.0.1:11434"

SYSTEM_PROMPT = (
    "Ты локальный ассистент домашней инфраструктуры. "
    "Отвечай кратко, по делу, по-русски, без рассуждений и воды."
)

TOOLS: List[Dict] = []
TOOL_REGISTRY: Dict[str, MCPClient] = {}

app = Flask(__name__)


# ─── helpers ────────────────────────────────────────────────────────────────

def proxy_get(path: str) -> Response:
    r = requests.get(f"{OLLAMA_API}{path}", timeout=60)
    return Response(
        r.content,
        status=r.status_code,
        content_type=r.headers.get("Content-Type", "application/json"),
    )


def ensure_system_prompt(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        return [{"role": "system", "content": SYSTEM_PROMPT}]
    if messages[0].get("role") != "system":
        return [{"role": "system", "content": SYSTEM_PROMPT}] + messages
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


# ─── CORS ───────────────────────────────────────────────────────────────────

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


# ─── tools ──────────────────────────────────────────────────────────────────

def load_tools():
    tools = []
    registry = {}
    servers = {
        "monitor": MCPClient(MCP_MONITOR_CMD),
        "notes":   MCPClient(MCP_NOTES_CMD),
        "alerts":  MCPClient(MCP_ALERTS_CMD),
    }
    for source, client in servers.items():
        server_tools = client.list_tools()
        log.info("[tools] %s: %d tools loaded", source, len(server_tools))
        for t in server_tools:
            name = getattr(t, "name", None)
            description = getattr(t, "description", "") or ""
            input_schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)
            if not name or not input_schema:
                log.warning("[tools] %s: skipping tool without name or schema: %r", source, t)
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": input_schema,
                },
            })
            registry[name] = client
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
    return result


# ─── routes ─────────────────────────────────────────────────────────────────

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
    options: Optional[Dict] = None,
    req_id: str = "-",
) -> Generator[bytes, None, None]:
    MAX_TOOL_ROUNDS = 5

    try:
        for round_num in range(MAX_TOOL_ROUNDS):
            log.info("[%s] stream round %d, messages=%d", req_id, round_num, len(messages))

            payload: Dict[str, Any] = {
                "model": model,
                "messages": build_model_messages(messages),
                "tools": TOOLS,
                "stream": True,
            }
            if options:
                payload["options"] = options

            log.debug("[%s] -> Ollama payload (messages count=%d, tools=%d)",
                      req_id, len(payload["messages"]), len(TOOLS))

            try:
                r = requests.post(
                    f"{OLLAMA_API}/api/chat",
                    json=payload,
                    timeout=300,
                    stream=True,
                )
                r.raise_for_status()
            except Exception as e:
                log.error("[%s] Ollama request failed: %s", req_id, e)
                err_chunk = json.dumps({
                    "model": model,
                    "message": {"role": "assistant", "content": f"\n[error: {e}]\n"},
                    "done": True,
                }, ensure_ascii=False)
                yield (err_chunk + "\n").encode("utf-8")
                return

            content_acc = ""
            last_tool_calls: Optional[List[Dict]] = None
            chunk_count = 0

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
                    msg = chunk.get("message") or {}
                    done = chunk.get("done", False)
                    delta = msg.get("content") or ""
                    content_acc += delta

                    tc = msg.get("tool_calls")
                    if tc:
                        log.info("[%s] tool_calls received: %s",
                                 req_id, [c.get("function", {}).get("name") for c in tc])
                        last_tool_calls = tc

                    # keepalive: при наличии tool_calls контент не стримим,
                    # но раз в 5 чанков шлём пустую строку чтобы nginx/браузер
                    # не посчитали соединение зависшим.
                    if last_tool_calls and chunk_count % 5 == 0:
                        yield b"\n"

                    if done:
                        log.info("[%s] round %d done, chunks=%d, tool_calls=%s, content_len=%d",
                                 req_id, round_num, chunk_count,
                                 bool(last_tool_calls), len(content_acc))
                        if last_tool_calls:
                            # flush перед переходом к следующему раунду
                            yield b"\n"
                            break
                        else:
                            yield line + b"\n"
                        break

                    if not tc:
                        yield line + b"\n"

            if not last_tool_calls:
                log.info("[%s] stream complete, no more tool rounds", req_id)
                break

            # ── tool execution ────────────────────────────────────────────
            messages.append({
                "role": "assistant",
                "content": content_acc,
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

                thinking_chunk = json.dumps({
                    "model": model,
                    "message": {
                        "role": "assistant",
                        "content": f"\n[→ {tool_name}]\n",
                    },
                    "done": False,
                }, ensure_ascii=False)
                yield (thinking_chunk + "\n").encode("utf-8")

                try:
                    tool_result = call_tool(tool_name, tool_args)
                except Exception as e:
                    log.error("[%s] tool %s failed: %s", req_id, tool_name, e)
                    tool_result = {"error": str(e)}

                messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

            last_tool_calls = None

    except Exception as e:
        log.error("[%s] unhandled exception in generator: %s", req_id, traceback.format_exc())
        err_chunk = json.dumps({
            "model": model,
            "message": {"role": "assistant", "content": f"\n[internal error: {e}]\n"},
            "done": True,
        }, ensure_ascii=False)
        yield (err_chunk + "\n").encode("utf-8")


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

    model    = original_payload.get("model") or "qwen3:4b"
    messages = original_payload.get("messages") or []
    options  = original_payload.get("options") or {}
    want_stream = original_payload.get("stream", True)

    messages = ensure_system_prompt(messages)

    log.info("[%s] POST /api/chat model=%s stream=%s messages=%d options=%s",
             req_id, model, want_stream, len(messages), options)

    if not want_stream:
        try:
            MAX_TOOL_ROUNDS = 5
            resp: Dict = {}
            for i in range(MAX_TOOL_ROUNDS):
                ollama_payload: Dict[str, Any] = {
                    "model": model,
                    "messages": build_model_messages(messages),
                    "tools": TOOLS,
                    "stream": False,
                }
                if options:
                    ollama_payload["options"] = options

                log.debug("[%s] non-stream round %d", req_id, i)
                r = requests.post(
                    f"{OLLAMA_API}/api/chat",
                    json=ollama_payload,
                    timeout=300,
                )
                r.raise_for_status()
                resp = r.json()
                msg = resp.get("message") or {}
                tool_calls = msg.get("tool_calls") or []

                if not tool_calls:
                    break

                log.info("[%s] non-stream tool_calls: %s", req_id,
                         [tc.get("function", {}).get("name") for tc in tool_calls])

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
        stream_with_context(chat_stream_generator(model, messages, options or None, req_id)),
        content_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked",
        },
    )


# ─── entry point ────────────────────────────────────────────────────────────

init_tools()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)

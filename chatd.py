import json
import re
import time
from typing import Any, Dict, Generator, List, Optional

import requests
from requests.exceptions import ReadTimeout
from flask import Flask, request, Response, jsonify, stream_with_context

from mcp_client import MCPClient, MCP_MONITOR_CMD, MCP_ALERTS_CMD, MCP_NOTES_CMD

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


# ─── tools ──────────────────────────────────────────────────────────────────

def load_tools():
    tools = []
    registry = {}
    servers = {
        "monitor": MCPClient(MCP_MONITOR_CMD),
        "notes":   MCPClient(MCP_NOTES_CMD),
        "alerts":  MCPClient(MCP_ALERTS_CMD),
    }
    for _source, client in servers.items():
        for t in client.list_tools():
            name = getattr(t, "name", None)
            description = getattr(t, "description", "") or ""
            input_schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)
            if not name or not input_schema:
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
    return tools, registry


def init_tools():
    global TOOLS, TOOL_REGISTRY
    TOOLS, TOOL_REGISTRY = load_tools()


def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    client = TOOL_REGISTRY.get(name)
    if client is None:
        raise RuntimeError(f"Unknown tool: {name}")
    return client.call_tool(name, arguments)


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
) -> Generator[bytes, None, None]:
    """
    Генератор, который:
    1. Стримит ответ Ollama напрямую клиенту
    2. При tool_calls — паузует стрим, исполняет инструменты, продолжает
    Клиент видит один непрерывный NDJSON-стрим.
    """
    MAX_TOOL_ROUNDS = 5

    for _round in range(MAX_TOOL_ROUNDS):
        payload: Dict[str, Any] = {
            "model": model,
            "messages": build_model_messages(messages),
            "tools": TOOLS,
            "stream": True,
        }
        if options:
            payload["options"] = options

        with requests.post(
            f"{OLLAMA_API}/api/chat",
            json=payload,
            timeout=300,
            stream=True,
        ) as r:
            r.raise_for_status()

            content_acc = ""
            last_tool_calls: Optional[List[Dict]] = None  # только из финального чанка

            for line in r.iter_lines():
                if not line:
                    continue

                chunk = json.loads(line.decode("utf-8"))
                msg = chunk.get("message") or {}
                done = chunk.get("done", False)
                delta = msg.get("content") or ""
                content_acc += delta

                tc = msg.get("tool_calls")
                if tc:
                    last_tool_calls = tc  # запомнили, но НЕ прерываем стрим

                if done:
                    if last_tool_calls:
                        # Будет следующий раунд — не отдаём финальный done
                        break
                    else:
                        # Финальный done без tool_calls — отдаём клиенту
                        yield line + b"\n"
                    break

                if not tc:
                    # Обычный контентный чанк — сразу клиенту
                    yield line + b"\n"

        if not last_tool_calls:
            break

        # ── tool execution ───────────────────────────────────────────────
        messages.append({
            "role": "assistant",
            "content": content_acc,
            "tool_calls": last_tool_calls,
        })

        for tc in last_tool_calls:
            fn = tc.get("function") or {}
            tool_name = fn.get("name")
            tool_args = fn.get("arguments") or {}
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    tool_args = {}

            thinking_chunk = {
                "model": model,
                "message": {
                    "role": "assistant",
                    "content": f"\n[tool: {tool_name}]\n",
                },
                "done": False,
            }
            yield (json.dumps(thinking_chunk, ensure_ascii=False) + "\n").encode("utf-8")

            try:
                tool_result = call_tool(tool_name, tool_args)
            except Exception as e:
                tool_result = {"error": str(e)}

            messages.append({
                "role": "tool",
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

        last_tool_calls = None


@app.post("/chat")
@app.post("/api/chat")
def chat():
    original_payload = request.get_json(force=True, silent=False)
    model    = original_payload.get("model") or "qwen3:4b"
    messages = original_payload.get("messages") or []
    options  = original_payload.get("options") or {}
    messages = ensure_system_prompt(messages)

    want_stream = original_payload.get("stream", True)

    if not want_stream:
        # Не-стриминговый путь — для тестирования через curl
        try:
            MAX_TOOL_ROUNDS = 5
            resp: Dict = {}
            for _ in range(MAX_TOOL_ROUNDS):
                ollama_payload: Dict[str, Any] = {
                    "model": model,
                    "messages": build_model_messages(messages),
                    "tools": TOOLS,
                    "stream": False,
                }
                if options:
                    ollama_payload["options"] = options

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
            return jsonify({"model": model, "answer": answer, "raw": resp})

        except ReadTimeout:
            return jsonify({"error": "ollama timeout"}), 504
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify({"error": str(e)}), 502

    # Стриминговый путь — основной
    return Response(
        stream_with_context(chat_stream_generator(model, messages, options or None)),
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
    app.run(host="0.0.0.0", port=5001, debug=False)

#!/usr/bin/env python3
import json
import time
from typing import Any, Dict, List

import requests
from requests.exceptions import ReadTimeout

from flask import Flask, request, Response, jsonify, stream_with_context

from mcp_client import MCPClient, MCP_MONITOR_CMD, MCP_ALERTS_CMD, MCP_NOTES_CMD

# Ollama native API
OLLAMA_API = "http://127.0.0.1:11434"

SYSTEM_PROMPT = (
    "Ты локальный ассистент домашней инфраструктуры. "
    "Отвечай кратко, по делу, по-русски, без рассуждений и воды."
)

app = Flask(__name__)

# MCP clients
mcp_monitor = MCPClient(MCP_MONITOR_CMD)
mcp_alerts = MCPClient(MCP_ALERTS_CMD)
mcp_notes = MCPClient(MCP_NOTES_CMD)

mcp_tools_cache: List[Dict[str, Any]] = []


def proxy_get(path: str) -> Response:
    r = requests.get(f"{OLLAMA_API}{path}", timeout=60)
    return Response(
        r.content,
        status=r.status_code,
        content_type=r.headers.get("Content-Type", "application/json"),
    )


def ensure_system_prompt(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(messages, list):
        return messages
    if not messages:
        return [{"role": "system", "content": SYSTEM_PROMPT}]
    if messages[0].get("role") != "system":
        return [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    return messages


def load_tools():
    tools = []
    registry = {}

    servers = {
        "monitor": MCPClient(MCP_MONITOR_CMD),
        "notes": MCPClient(MCP_NOTES_CMD),
        "alerts": MCPClient(MCP_ALERTS_CMD),
    }

    for source, client in servers.items():
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


def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    if name.startswith("monitor_"):
        client = mcp_monitor
    elif name.startswith("smarthome_"):
        client = mcp_smarthome
    elif name.startswith("notes_"):
        client = mcp_notes
    else:
        raise RuntimeError(f"Unknown tool prefix for {name}")

    return client.call_tool(name, arguments)


def build_model_messages(raw_messages):
    result = []
    for m in raw_messages:
        role = m.get("role")
        content = m.get("content", "")

        if role not in ("system", "user", "assistant"):
            continue

        # Вырежем старые "рассуждения", если они были — например, если они размечены как-то спецом.
        # Самый простой KISS-вариант: из прошлых assistant оставляем только последнюю или вообще убираем.
        if role == "assistant":
            continue  # сначала попробуй вообще не слать прошлые ответы модели

        result.append({"role": role, "content": content})

    return result


def call_ollama_with_tools(model, messages):
    tools, _ = load_tools()
    #model_messages = build_model_messages(messages)
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": True,
    }

    with requests.post(f"{OLLAMA_API}/api/chat",
                       json=payload,
                       timeout=300,
                       stream=True) as r:
        r.raise_for_status()

        full_message = ""
        final_resp = None

        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            final_resp = chunk
            msg = chunk.get("message") or {}
            content = msg.get("content")
            if content:
                full_message += content

        if final_resp is None:
            raise RuntimeError("empty response from ollama")

        if full_message and "message" in final_resp:
            final_resp["message"]["content"] = full_message

        return final_resp


@app.get("/health")
@app.get("/api/health")
def health():
    return jsonify({"ok": True})


@app.get("/tags")
@app.get("/api/tags")
def tags():
    return proxy_get("/api/tags")


@app.post("/chat")
@app.post("/api/chat")
def chat():
    payload = request.get_json(force=True, silent=False)

    model = payload.get("model") or "qwen3:4b"
    messages = payload.get("messages") or []

    messages = ensure_system_prompt(messages)

    started = time.time()
    try:
        resp = call_ollama_with_tools(model, messages)
    except ReadTimeout:
        return jsonify({"error": "ollama timeout"}), 504
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 502

    msg = resp.get("message") or {}
    answer = (msg.get("content") or "").strip()
    if not answer:
        answer = "[пустой ответ]"

    total_sec = resp.get("total_duration", 0) / 1e9 if resp.get("total_duration") else (time.time() - started)
    eval_count = resp.get("eval_count")

    return jsonify({
        "model": model,
        "answer": answer,
        "raw": resp,
        "meta": {
            "total_sec": total_sec,
            "eval_count": eval_count,
        }
    })


if __name__ == "__main__":
    mcp_alerts.start()
    mcp_monitor.start()
    mcp_notes.start()
    app.run(host="0.0.0.0", port=5001, debug=False)

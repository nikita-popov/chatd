#!/usr/bin/env python3

import json
import re
import time
from typing import Any, Dict, List

import requests
from requests.exceptions import ReadTimeout

from flask import Flask, request, Response, jsonify

from mcp_client import MCPClient, MCP_MONITOR_CMD, MCP_ALERTS_CMD, MCP_NOTES_CMD

OLLAMA_API = "http://127.0.0.1:11434"

SYSTEM_PROMPT = (
    "Ты локальный ассистент домашней инфраструктуры. "
    "Отвечай кратко, по делу, по-русски, без рассуждений и воды."
)

TOOLS: List[Dict] = []
TOOL_REGISTRY: Dict[str, MCPClient] = {}

app = Flask(__name__)


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


def init_tools():
    global TOOLS, TOOL_REGISTRY
    TOOLS, TOOL_REGISTRY = load_tools()


def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    client = TOOL_REGISTRY.get(name)
    if client is None:
        raise RuntimeError(f"Unknown tool: {name}")
    return client.call_tool(name, arguments)


def build_model_messages(raw_messages):
    result = []
    for m in raw_messages:
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        content = m.get("content", "") or ""
        if role == "assistant":
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        result.append({"role": role, "content": content})
    return result


def call_ollama_with_tools(model, messages):
    #model_messages = build_model_messages(messages)
    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
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
        MAX_TOOL_ROUNDS = 5
        for _ in range(MAX_TOOL_ROUNDS):
            resp = call_ollama_with_tools(model, messages)
            msg = resp.get("message") or {}
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                break

            # Добавляем ответ модели с tool_calls в историю
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})

            # Исполняем каждый инструмент
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

    return jsonify({
        "model": model,
        "answer": answer,
        "raw": resp,
        "meta": {
            "total_sec": total_sec,
            "eval_count": resp.get("eval_count"),
        }
    })


if __name__ == "__main__":
    init_tools()
    app.run(host="0.0.0.0", port=5001, debug=False)

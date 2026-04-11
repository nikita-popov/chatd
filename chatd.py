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
        # Сохраняем tool_calls если они есть (нужны для истории)
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


# ─── ollama stream ───────────────────────────────────────────────────────────

def _ollama_stream_request(messages: List[Dict]) -> requests.Response:
    """Открывает стриминговый запрос к Ollama. Возвращает Response объект."""
    payload = {
        "model": _current_model,   # см. ниже — пробрасываем через контекст
        "messages": build_model_messages(messages),
        "tools": TOOLS,
        "stream": True,
    }
    return requests.post(
        f"{OLLAMA_API}/api/chat",
        json=payload,
        timeout=300,
        stream=True,
    )


def _collect_stream(
    r: requests.Response,
) -> tuple[str, Optional[List[Dict]], Dict]:
    """
    Читает NDJSON-стрим от Ollama.
    Возвращает (accumulated_content, tool_calls_or_None, final_chunk).
    """
    content_acc = ""
    tool_calls = None
    final_chunk = {}

    for line in r.iter_lines():
        if not line:
            continue
        chunk = json.loads(line.decode("utf-8"))
        final_chunk = chunk
        msg = chunk.get("message") or {}
        delta = msg.get("content") or ""
        content_acc += delta
        if msg.get("tool_calls"):
            tool_calls = msg["tool_calls"]

    return content_acc, tool_calls, final_chunk


# ─── routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
@app.get("/api/health")
def health():
    return jsonify({"ok": True})


@app.get("/tags")
@app.get("/api/tags")
def tags():
    return proxy_get("/api/tags")


# Глобальная переменная модели — прокидывается из chat() в генератор
# (thread-local не нужен — у нас один worker, но можно заменить на threading.local)
_current_model = "qwen3:4b"


def chat_stream_generator(model: str, messages: List[Dict]) -> Generator[bytes, None, None]:
    """
    Генератор, который:
    1. Стримит ответ Ollama напрямую клиенту
    2. При tool_calls — паузует стрим, исполняет инструменты, продолжает
    Клиент видит один непрерывный NDJSON-стрим.
    """
    global _current_model
    _current_model = model

    MAX_TOOL_ROUNDS = 5

    for round_num in range(MAX_TOOL_ROUNDS):
        payload = {
            "model": model,
            "messages": build_model_messages(messages),
            "tools": TOOLS,
            "stream": True,
        }

        with requests.post(
            f"{OLLAMA_API}/api/chat",
            json=payload,
            timeout=300,
            stream=True,
        ) as r:
            r.raise_for_status()

            content_acc = ""
            tool_calls = None
            final_chunk = {}
            is_last_round = False

            for line in r.iter_lines():
                if not line:
                    continue

                chunk = json.loads(line.decode("utf-8"))
                final_chunk = chunk
                msg = chunk.get("message") or {}

                # Собираем tool_calls (не стримим их клиенту — нет смысла)
                if msg.get("tool_calls"):
                    tool_calls = msg["tool_calls"]

                delta = msg.get("content") or ""
                content_acc += delta
                done = chunk.get("done", False)

                if tool_calls and done:
                    # Не отдаём финальный done-чанк клиенту сейчас —
                    # будет ещё один раунд
                    break

                if not tool_calls:
                    # Обычный чанк — пробрасываем клиенту как есть
                    yield line + b"\n"

                if done:
                    is_last_round = True
                    break

            if is_last_round or not tool_calls:
                # Финальный раунд — больше нечего делать
                break

            # ── tool execution ───────────────────────────────────────────
            # Добавляем assistant-сообщение с tool_calls в историю
            messages.append({
                "role": "assistant",
                "content": content_acc,
                "tool_calls": tool_calls,
            })

            # Исполняем инструменты и добавляем результаты
            for tc in tool_calls:
                fn = tc.get("function") or {}
                tool_name = fn.get("name")
                tool_args = fn.get("arguments") or {}
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}

                # Отправляем клиенту "thinking" чанк — пусть видит что идёт работа
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

            tool_calls = None  # сброс для следующего раунда


@app.post("/chat")
@app.post("/api/chat")
def chat():
    payload = request.get_json(force=True, silent=False)
    model = payload.get("model") or "qwen3:4b"
    messages = payload.get("messages") or []
    messages = ensure_system_prompt(messages)

    # ollama-gui всегда шлёт stream=true, но на всякий случай читаем флаг
    want_stream = payload.get("stream", True)

    if not want_stream:
        # Не-стриминговый путь — для тестирования через curl
        try:
            MAX_TOOL_ROUNDS = 5
            resp = {}
            for _ in range(MAX_TOOL_ROUNDS):
                r = requests.post(
                    f"{OLLAMA_API}/api/chat",
                    json={
                        "model": model,
                        "messages": build_model_messages(messages),
                        "tools": TOOLS,
                        "stream": False,
                    },
                    timeout=300,
                )
                r.raise_for_status()
                resp = r.json()
                msg = resp.get("message") or {}
                tool_calls = msg.get("tool_calls") or []

                if not tool_calls:
                    break

                messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})

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
        stream_with_context(chat_stream_generator(model, messages)),
        content_type="application/x-ndjson",
    )


# ─── entry point ─────────────────────────────────────────────────────────────

init_tools()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)

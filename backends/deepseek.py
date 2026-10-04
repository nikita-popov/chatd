import json
import logging
import os
from typing import Any, Dict, Generator

import requests

log = logging.getLogger("chatd.backends.deepseek")

DEEPSEEK_API = os.environ.get(
    "CHATD_DEEPSEEK_API",
    "https://api.deepseek.com",
)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


class DeepSeekBackend:
    def _headers(self) -> Dict[str, str]:
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        return {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

    def chat_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = self._make_payload(payload, stream=False)
        response = requests.post(
            f"{DEEPSEEK_API}/chat/completions",
            headers=self._headers(),
            json=body,
            timeout=600,
        )
        response.raise_for_status()
        return self._normalize_response(response.json())

    def chat_stream(
        self,
        payload: Dict[str, Any],
    ) -> Generator[bytes, None, None]:
        body = self._make_payload(payload, stream=True)
        response = requests.post(
            f"{DEEPSEEK_API}/chat/completions",
            headers=self._headers(),
            json=body,
            timeout=600,
            stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            if line == b"data: [DONE]":
                break
            if line.startswith(b"data: "):
                yield line[6:] + b"\n"

    def _make_payload(
        self,
        payload: Dict[str, Any],
        stream: bool,
    ) -> Dict[str, Any]:
        body = {
            "model": payload["model"],
            "messages": payload["messages"],
            "stream": stream,
        }

        if payload.get("tools"):
            body["tools"] = payload["tools"]
            body["tool_choice"] = "auto"

        options = payload.get("options") or {}
        mapping = {
            "temperature": "temperature",
            "top_p": "top_p",
            "num_predict": "max_tokens",
        }
        for source, target in mapping.items():
            if source in options:
                body[target] = options[source]

        return body

    def _normalize_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}

        return {
            "message": {
                "role": message.get("role", "assistant"),
                "content": message.get("content") or "",
                "tool_calls": message.get("tool_calls") or [],
            },
            "done": True,
        }
    

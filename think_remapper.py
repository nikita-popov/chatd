#!/usr/bin/env python3
"""ThinkingRemapper — convert Ollama thinking tokens to <think>...</think> in content.

Ollama streams qwen3 thinking as a separate ``message.thinking`` field.
Frontends that only read ``message.content`` (e.g. ollama-gui) never see it.
This module re-wraps thinking tokens inside content so any client benefits.

State machine
-------------
  idle     → no thinking seen yet
  thinking → inside a thinking block, <think> already emitted
  closed   → </think> emitted, back to normal content
"""
import json
from typing import Optional


def _make_chunk(model: str, content: str, done: bool = False) -> bytes:
    """Minimal chunk builder (mirrors chatd.make_chunk without the import cycle)."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"
    obj = {
        "model": model,
        "created_at": ts,
        "message": {"role": "assistant", "content": content},
        "done": done,
    }
    if done:
        obj["done_reason"] = "stop"
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


class ThinkingRemapper:
    """Process one NDJSON line at a time, remapping thinking tokens into content.

    Usage::

        remapper = ThinkingRemapper(model)
        for line in ollama_response.iter_lines():
            yield remapper.feed(line)
        closing = remapper.close()
        if closing:
            yield closing

    Attributes
    ----------
    thinking_acc : str
        Accumulated raw thinking text (no tags).  Useful for logging/debugging.
    content_acc : str
        Accumulated real response text (no thinking).  Used for conversation history.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._state: str = "idle"      # idle | thinking | closed
        self.thinking_acc: str = ""
        self.content_acc: str = ""

    def feed(self, line: bytes) -> bytes:
        """Process one NDJSON line.  Returns the (possibly rewritten) line."""
        try:
            chunk = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return line

        msg      = chunk.get("message") or {}
        thinking = msg.get("thinking") or ""
        content  = msg.get("content")  or ""

        self.thinking_acc += thinking
        self.content_acc  += content

        # Fast path: nothing to remap.
        if not thinking and self._state == "idle":
            return line

        # Strip the separate thinking field — we fold it into content below.
        if "thinking" in msg:
            del msg["thinking"]

        if thinking:
            if self._state == "idle":
                self._state = "thinking"
                msg["content"] = "<think>" + thinking
            else:
                msg["content"] = thinking
        else:
            # thinking ended, first content chunk after the thinking block.
            if self._state == "thinking":
                self._state = "closed"
                msg["content"] = "</think>" + content

        chunk["message"] = msg
        return (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")

    def close(self) -> Optional[bytes]:
        """Emit a closing tag if the stream ended while still in thinking state."""
        if self._state == "thinking":
            self._state = "closed"
            return _make_chunk(self.model, "</think>")
        return None

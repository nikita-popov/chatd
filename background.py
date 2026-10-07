#!/usr/bin/env python3
"""background.py — background thinking (subconscious) for chatd.

Single-turn preview (PR B+). This module is deliberately additive and
isolated:

  - It does NOT write to mempalace (only reads via memory.wake_up()).
  - It does NOT touch session storage or the RAG store.
  - Its only side effects are inside $CHATD_BG_STATE_DIR:
        journal.jsonl — append-only log of ticks
        tick.lock     — flock-based concurrency lock

No tools are invoked in this preview. Planner / Executor / Reflector
phases (with tool loops, goals.json ledger and token budget) will be
added in a follow-up PR.

Why the lock:
  systemd timers with Persistent=true can trigger a new tick while the
  previous one is still running. Rather than queue or overlap, we skip
  and log — background work is idempotent by design, and overlapping
  ticks are never useful.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config import (
    BG_MODEL,
    BG_NUM_PREDICT,
    BG_SESSION_ID,
    BG_STATE_DIR,
)

log = logging.getLogger("chatd.bg")


# ── paths ─────────────────────────────────────────────────────────────────────

def _state_dir() -> Path:
    p = Path(os.path.expanduser(BG_STATE_DIR))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _journal_path() -> Path:
    return _state_dir() / "journal.jsonl"


def _lock_path() -> Path:
    return _state_dir() / "tick.lock"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"


def _append_journal(entry: Dict[str, Any]) -> None:
    path = _journal_path()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── lock ──────────────────────────────────────────────────────────────────────

class TickLock:
    """Non-blocking file lock — prevents overlapping ticks."""

    def __init__(self, path: Path):
        self.path = path
        self._fd: Optional[int] = None

    def acquire(self) -> bool:
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            os.close(self._fd)
            self._fd = None
            return False

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


# ── LLM call (single turn, no tools) ──────────────────────────────────────────

_DEFAULT_TICK_PROMPT = (
    "Background tick. Look at the current context and, in one short "
    "paragraph, note anything worth flagging. If nothing — reply "
    "with exactly: nothing."
)


def _build_bg_messages(payload: Dict[str, Any]) -> list:
    """Minimal message list for a single-turn background tick.

    System prompt = memory.wake_up() (L0+L1 from mempalace, read-only).
    User prompt   = payload["prompt"] or a default tick prompt.
    """
    import memory  # local import — avoid touching chatd.py's import order

    system = memory.wake_up()

    prompt = None
    if isinstance(payload, dict):
        prompt = payload.get("prompt")
    if not prompt:
        prompt = _DEFAULT_TICK_PROMPT

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]


def _one_shot(payload: Dict[str, Any]) -> str:
    """Single-turn LLM call. No tools, no session, no memory writes.

    This is deliberately the *only* place in background.py that talks to a
    backend. Planner/Executor/Reflector (next PR) will reuse it as the base
    primitive and layer tool loops on top.
    """
    import backends
    from config import DEFAULT_OPTIONS, THINKING

    messages = _build_bg_messages(payload)
    body = {
        "model":    BG_MODEL,
        "messages": messages,
        "stream":   False,
        "options":  dict(DEFAULT_OPTIONS),
        # No "tools" key — background tick never invokes MCP in PR B+.
    }

    if not THINKING:
        body["think"] = False

    backend = backends.get_backend(BG_MODEL)
    resp = backend.chat_sync(body)
    msg = resp.get("message") or {}
    return (msg.get("content") or "").strip()


# ── public API ────────────────────────────────────────────────────────────────

def run_tick(payload: Dict[str, Any], req_id: str) -> Dict[str, Any]:
    """One tick of the subconscious.

    Returns a JSON-serialisable result dict.
    Raises only on unexpected internal errors — caller (chatd /api/tick)
    converts them to HTTP 500.
    """
    started = time.time()
    ts = _now_iso()

    lock = TickLock(_lock_path())
    if not lock.acquire():
        log.info("[bg %s] tick skipped — previous tick still running", req_id)
        return {
            "req_id": req_id,
            "status": "skipped",
            "reason": "tick_already_running",
            "ts":     ts,
        }

    try:
        _append_journal({
            "req_id":  req_id,
            "ts":      ts,
            "session": BG_SESSION_ID,
            "model":   BG_MODEL,
            "payload": payload,
            "status":  "started",
        })

        answer = _one_shot(payload)
        elapsed_ms = int((time.time() - started) * 1000)

        result = {
            "req_id":     req_id,
            "status":     "ok",
            "answer":     answer,
            "elapsed_ms": elapsed_ms,
        }

        _append_journal({
            "req_id": req_id,
            "ts":     _now_iso(),
            "status": "done",
            "result": result,
        })

        log.info("[bg %s] tick done in %d ms (answer=%d chars)",
                 req_id, elapsed_ms, len(answer))
        return result

    except Exception as e:
        log.error("[bg %s] tick error: %s", req_id, e, exc_info=True)
        _append_journal({
            "req_id": req_id,
            "ts":     _now_iso(),
            "status": "error",
            "error":  str(e),
        })
        raise

    finally:
        lock.release()

#!/usr/bin/env python3
"""session.py — per-chat session state with L1.5 rolling summary.

Each session is identified by the chatId that ollama-gui sends in every
message.  State is persisted to a JSON file in CHATD_SESSION_DIR so it
survives chatd restarts.

Schema (JSON on disk)::

    {
        "session_id": "42",
        "summary":    "User prefers Emacs.  Working on chatd memory layers.",
        "updated_at": "2026-04-16T15:00:00Z"
    }
"""
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from config import CHATD_SESSION_DIR

log = logging.getLogger("chatd.session")

_SESSIONS: Dict[str, "Session"] = {}


@dataclass
class Session:
    session_id: str
    summary: str = ""
    # unsummarized turns accumulated since last compression
    raw_turns: list = field(default_factory=list, repr=False)

    # ── persistence ──────────────────────────────────────────────────────────

    def _path(self) -> Path:
        d = Path(os.path.expanduser(CHATD_SESSION_DIR))
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.session_id}.json"

    def save(self) -> None:
        """Persist summary to disk.  raw_turns are in-memory only."""
        data = {
            "session_id": self.session_id,
            "summary":    self.summary,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        try:
            self._path().write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log.debug("[session:%s] saved (%d chars)", self.session_id, len(self.summary))
        except Exception as e:
            log.warning("[session:%s] save failed: %s", self.session_id, e)

    @classmethod
    def load(cls, session_id: str) -> "Session":
        """Load session from disk, or return a blank one."""
        path = Path(os.path.expanduser(CHATD_SESSION_DIR)) / f"{session_id}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                s = cls(session_id=session_id, summary=data.get("summary", ""))
                log.info(
                    "[session:%s] loaded from disk (%d chars)",
                    session_id, len(s.summary),
                )
                return s
            except Exception as e:
                log.warning("[session:%s] load failed: %s", session_id, e)
        return cls(session_id=session_id)


def get_session(session_id: str) -> Session:
    """Return an in-memory Session, loading from disk on first access."""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = Session.load(session_id)
    return _SESSIONS[session_id]


def record_turn(session: Session, user_msg: str, assistant_msg: str) -> None:
    """Append a Q/A pair to the session's raw_turns buffer."""
    session.raw_turns.append({"user": user_msg[:1000], "assistant": assistant_msg[:1000]})
    log.debug("[session:%s] raw_turns=%d", session.session_id, len(session.raw_turns))

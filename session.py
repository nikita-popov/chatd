#!/usr/bin/env python3
"""session.py — per-chat session state with L1.5 rolling summary.

Each session is identified by the chatId that ollama-gui sends in every
message.  State is persisted to a JSON file in CHATD_SESSION_DIR so it
survives chatd restarts.

Schema (JSON on disk)::

    {
        "session_id": "42",
        "summary":    "User prefers Emacs.  Working on chatd memory layers.",
        "raw_turns":  [{"user": "...", "assistant": "..."}],
        "updated_at": "2026-04-16T15:00:00Z"
    }

Alongside each .json file a .jsonl sibling is written in Claude Code JSONL
format so the session can be ingested by ``mempalace mine --mode convos``
without any extra conversion step.
"""
import json
import logging
import os
import threading
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
    # Prevents two concurrent compress threads from running for the same
    # session (e.g. two requests arriving before the first compress finishes).
    # acquire(blocking=False) is used so the second caller just skips.
    _compress_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    # ── persistence ──────────────────────────────────────────────────────────

    def _path(self) -> Path:
        d = Path(os.path.expanduser(CHATD_SESSION_DIR))
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.session_id}.json"

    def save(self) -> None:
        """Persist summary and raw_turns to disk, then refresh the JSONL chatlog."""
        data = {
            "session_id": self.session_id,
            "summary":    self.summary,
            "raw_turns":  self.raw_turns,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        path = self._path()
        try:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log.debug(
                "[session:%s] saved to %s (summary=%d chars, raw_turns=%d)",
                self.session_id, path, len(self.summary), len(self.raw_turns),
            )
        except Exception as e:
            log.warning("[session:%s] save failed (path=%s): %s", self.session_id, path, e)
            return

        #self._save_chatlog()

    def _save_chatlog(self) -> None:
        """Write raw_turns as Claude Code JSONL for ``mempalace mine --mode convos``.

        Format: one JSON object per line, alternating human/assistant turns.
        normalize.py in mempalace detects this schema via the ``type`` field
        and applies strip_noise + tool_use capture automatically.
        """
        if not self.raw_turns:
            return
        log_path = self._path().with_suffix(".jsonl")
        lines = []
        for turn in self.raw_turns:
            lines.append(json.dumps(
                {"type": "human",
                 "message": {"role": "human", "content": turn["user"]}},
                ensure_ascii=False,
            ))
            lines.append(json.dumps(
                {"type": "assistant",
                 "message": {"role": "assistant", "content": turn["assistant"]}},
                ensure_ascii=False,
            ))
        try:
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            log.debug(
                "[session:%s] chatlog jsonl written to %s (%d turns)",
                self.session_id, log_path, len(self.raw_turns),
            )
        except Exception as e:
            log.warning(
                "[session:%s] chatlog write failed (path=%s): %s",
                self.session_id, log_path, e,
            )

    def append_chatlog(self, user: str, assistant: str) -> None:
        """Append one turn to the persistent chatlog — never truncates."""
        log_path = self._path().with_suffix(".jsonl")
        lines = [
            json.dumps({"type": "human",
                        "message": {"role": "human", "content": user}},
                       ensure_ascii=False),
            json.dumps({"type": "assistant",
                        "message": {"role": "assistant", "content": assistant}},
                       ensure_ascii=False),
        ]
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            log.warning("[session:%s] chatlog append failed: %s", self.session_id, e)

    @classmethod
    def load(cls, session_id: str) -> "Session":
        """Load session from disk, or return a blank one."""
        path = Path(os.path.expanduser(CHATD_SESSION_DIR)) / f"{session_id}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                s = cls(
                    session_id=session_id,
                    summary=data.get("summary", ""),
                    raw_turns=data.get("raw_turns", []),
                )
                log.info(
                    "[session:%s] loaded from %s (summary=%d chars, raw_turns=%d)",
                    session_id, path, len(s.summary), len(s.raw_turns),
                )
                return s
            except Exception as e:
                log.warning("[session:%s] load failed (path=%s): %s", session_id, path, e)
        log.debug("[session:%s] no file at %s, starting fresh", session_id, path)
        return cls(session_id=session_id)


def get_session(session_id: str) -> Session:
    """Return an in-memory Session, loading from disk on first access."""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = Session.load(session_id)
    return _SESSIONS[session_id]


def record_turn(session: Session, user_msg: str, assistant_msg: str) -> None:
    """Append a Q/A pair to raw_turns and immediately persist to disk."""
    session.raw_turns.append({"user": user_msg, "assistant": assistant_msg})
    log.debug(
        "[session:%s] recorded turn (raw_turns=%d, total_chars=%d)",
        session.session_id,
        len(session.raw_turns),
        sum(len(t["user"]) + len(t["assistant"]) for t in session.raw_turns),
    )
    session.append_chatlog(user_msg, assistant_msg)
    session.save()

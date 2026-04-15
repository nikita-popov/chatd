#!/usr/bin/env python3
"""memory.py — fast in-process memory abstraction for chatd.

Layered memory model:
  L0  identity.txt injected at startup (handled by chatd.py, not here)
  L1  FastMemory: KG triples read directly from SQLite — ~0 ms, no MCP
  L2  wake_up(): L0 + L1 text assembled for system prompt injection
  L3  mempalace_search MCP tool — ChromaDB semantic search, called on demand

This module owns the hot path.  MCP tools (mempalace_kg_query,
mempalace_search) remain available as LLM tool calls for the cold path and
for write operations.
"""
import functools
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

from config import MEMPALACE_KG_PATH, MEMPALACE_PALACE_PATH

log = logging.getLogger("chatd.memory")

# ── paths ─────────────────────────────────────────────────────────────────────

# Both paths are independent: palace dir (wings/rooms) and the KG SQLite
# file are not required to share a parent directory.
_KG_PATH: Path = Path(os.path.expanduser(MEMPALACE_KG_PATH))

# ── temporal filter helpers ───────────────────────────────────────────────────

# Mirrors the filter logic in MemPalace/mempalace knowledge_graph.py
# (query_entity method). Using >= so a triple whose valid_to = today is
# still considered current for the rest of the day, consistent with how
# mempalace.query_entity() behaves. Adding valid_from guard to exclude
# future-dated triples that are not yet effective.
_TEMPORAL_WHERE = (
    "(t.valid_from IS NULL OR t.valid_from <= date('now'))"
    " AND (t.valid_to IS NULL OR t.valid_to >= date('now'))"
)

# ── L1: in-process KG dict ────────────────────────────────────────────────────

# Populated on first access via _load_kg_dict().
# Keys: "subject:predicate" (lower-cased). Values: object string.
_KG_DICT: dict[str, str] = {}
_KG_LOADED: bool = False


def _load_kg_dict() -> None:
    """Read all current KG triples from SQLite into _KG_DICT."""
    global _KG_DICT, _KG_LOADED
    if not _KG_PATH.exists():
        log.warning("KG database not found: %s", _KG_PATH)
        _KG_LOADED = True
        return
    try:
        con = sqlite3.connect(str(_KG_PATH), timeout=5, check_same_thread=False)
        rows = con.execute(
            "SELECT s.name, t.predicate, o.name"
            " FROM triples t"
            " JOIN entities s ON t.subject = s.id"
            " JOIN entities o ON t.object  = o.id"
            f" WHERE {_TEMPORAL_WHERE}"
        ).fetchall()
        con.close()
        _KG_DICT = {
            f"{subj.lower()}:{pred.lower()}": obj
            for subj, pred, obj in rows
        }
        log.info("FastMemory: loaded %d KG triples from %s", len(_KG_DICT), _KG_PATH)
    except Exception as e:
        log.warning("FastMemory: KG load failed: %s", e)
    _KG_LOADED = True


def _ensure_loaded() -> None:
    if not _KG_LOADED:
        _load_kg_dict()


# ── Public API ────────────────────────────────────────────────────────────────

def kg_recall(entity: str) -> Optional[str]:
    """Return all current KG facts for *entity* as a compact text block.

    Hot path — dict lookup, no I/O, no subprocess.
    Returns None when no facts are found (caller decides whether to fall back
    to mempalace_kg_query MCP tool).
    """
    _ensure_loaded()
    prefix = entity.lower() + ":"
    lines = [
        f"{entity} {pred.split(':', 1)[1]} {obj}"
        for pred, obj in _KG_DICT.items()
        if pred.startswith(prefix)
    ]
    if not lines:
        return None
    return "\n".join(lines)


def kg_add(subject: str, predicate: str, obj: str) -> None:
    """Update in-process cache after a successful mempalace_kg_add MCP call.

    Call this right after call_tool('mempalace_kg_add', ...) returns so the
    hot cache stays consistent without a full reload.
    """
    key = f"{subject.lower()}:{predicate.lower()}"
    _KG_DICT[key] = obj
    log.debug("FastMemory: kg_add %s = %s", key, obj)


def invalidate() -> None:
    """Clear both the in-process KG dict and the wake-up lru_cache slot.

    Call whenever mempalace writes new data (kg_add, add_drawer) so the next
    wake_up() call rebuilds from fresh SQLite state.
    """
    global _KG_DICT, _KG_LOADED
    _KG_DICT = {}
    _KG_LOADED = False
    _wakeup_cached.cache_clear()
    log.info("FastMemory: cache invalidated")


# ── L2: wake-up assembly ──────────────────────────────────────────────────────

def _read_identity() -> str:
    """Read identity.txt relative to this file."""
    identity_path = Path(__file__).parent / "identity.txt"
    try:
        return identity_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        log.warning("identity.txt read failed: %s", e)
        return ""


def _read_kg_block(limit: int = 60) -> str:
    """Read the most recent KG triples directly from SQLite as a text block."""
    if not _KG_PATH.exists():
        return ""
    try:
        con = sqlite3.connect(str(_KG_PATH), timeout=5, check_same_thread=False)
        rows = con.execute(
            "SELECT s.name, t.predicate, o.name"
            " FROM triples t"
            " JOIN entities s ON t.subject = s.id"
            " JOIN entities o ON t.object  = o.id"
            f" WHERE {_TEMPORAL_WHERE}"
            " ORDER BY t.extracted_at DESC"
            " LIMIT ?",
            (limit,),
        ).fetchall()
        con.close()
        return "\n".join(f"{subj} {pred} {obj}" for subj, pred, obj in rows)
    except Exception as e:
        log.warning("FastMemory: KG block read failed: %s", e)
        return ""


def _run_wakeup_subprocess() -> str:
    """Fallback: execute 'python -m mempalace wake-up' via subprocess.

    Used only when the KG SQLite is absent or unreadable (e.g. first run).
    """
    import subprocess

    venv_python = os.environ.get("CHATD_MCP_MEMPALACE", "").split()[0] or "python3"
    env = {**os.environ, "MEMPALACE_PALACE_PATH": MEMPALACE_PALACE_PATH}
    try:
        result = subprocess.run(
            [venv_python, "-m", "mempalace", "wake-up"],
            capture_output=True, text=True, timeout=10, env=env,
        )
        if result.returncode != 0:
            log.warning("mempalace wake-up exited %d: %s", result.returncode, result.stderr.strip())
        return result.stdout.strip()
    except Exception as e:
        log.warning("mempalace wake-up subprocess failed: %s", e)
        return ""


@functools.lru_cache(maxsize=1)
def _wakeup_cached() -> str:
    """Assemble L0 + L1 context block (cached, single slot).

    Prefer direct SQLite reads over subprocess to avoid spawn overhead.
    Falls back to subprocess only when KG file is missing.
    """
    identity = _read_identity()
    kg_block = _read_kg_block()

    if not kg_block:
        log.info("FastMemory: KG empty or missing, falling back to subprocess wake-up")
        return _run_wakeup_subprocess()

    parts = [p for p in (identity, kg_block) if p]
    text = "\n\n".join(parts)
    log.info("FastMemory: wake-up assembled (%d chars, direct SQLite)", len(text))
    return text


def wake_up() -> str:
    """Return the system-prompt context block (L0 + L1).

    Cheap on repeated calls — result is cached until invalidate() is called.
    """
    return _wakeup_cached()


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def init() -> None:
    """Pre-load KG dict at startup so the first request pays no I/O cost."""
    _load_kg_dict()

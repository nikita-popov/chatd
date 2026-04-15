#!/usr/bin/env python3
"""memory.py — fast in-process memory abstraction for chatd.

Layered memory model:
  L0  identity.txt injected at startup (handled by chatd.py, not here)
  L1  FastMemory: KG triples via KnowledgeGraph.query_entity() — no MCP
  L2  wake_up(): L0 + L1 text assembled for system prompt injection
  L3  mempalace_search MCP tool — ChromaDB semantic search, called on demand

This module owns the hot path.  MCP tools (mempalace_kg_query,
mempalace_search) remain available as LLM tool calls for the cold path and
for write operations.
"""
import functools
import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional

from mempalace.knowledge_graph import KnowledgeGraph

from config import MEMPALACE_KG_PATH, MEMPALACE_PALACE_PATH

log = logging.getLogger("chatd.memory")

# ── L1: in-process KG ───────────────────────────────────────────────────────

_kg: Optional[KnowledgeGraph] = None


def _get_kg() -> Optional[KnowledgeGraph]:
    """Return a shared KnowledgeGraph instance, initialised lazily."""
    global _kg
    if _kg is None:
        kg_path = os.path.expanduser(MEMPALACE_KG_PATH)
        if not Path(kg_path).exists():
            log.warning("KG database not found: %s", kg_path)
            return None
        try:
            _kg = KnowledgeGraph(db_path=kg_path)
            log.info("FastMemory: KnowledgeGraph opened at %s", kg_path)
        except Exception as e:
            log.warning("FastMemory: KnowledgeGraph init failed: %s", e)
    return _kg


# ── Public API ────────────────────────────────────────────────────────────────

def kg_recall(entity: str) -> Optional[str]:
    """Return all current KG facts for *entity* as a compact text block.

    Hot path — uses KnowledgeGraph.query_entity() with today's date so
    temporal filtering is handled by mempalace itself.  Returns None when
    no facts are found (caller decides whether to fall back to
    mempalace_kg_query MCP tool).
    """
    kg = _get_kg()
    if kg is None:
        return None
    try:
        triples = kg.query_entity(entity, as_of=date.today().isoformat())
    except Exception as e:
        log.warning("FastMemory: kg_recall(%s) failed: %s", entity, e)
        return None
    if not triples:
        return None
    lines = [
        f"{t['subject']} {t['predicate']} {t['object']}"
        for t in triples
        if t.get("current")
    ]
    return "\n".join(lines) if lines else None


def invalidate() -> None:
    """Close the KnowledgeGraph connection and clear the wake-up cache.

    Call whenever mempalace writes new data (kg_add, add_drawer) so the next
    wake_up() call re-opens a fresh connection and sees the updated state.
    """
    global _kg
    if _kg is not None:
        try:
            _kg.close()
        except Exception:
            pass
        _kg = None
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
    """Return the most recent KG triples as a text block via KnowledgeGraph."""
    kg = _get_kg()
    if kg is None:
        return ""
    try:
        # timeline() returns triples ordered by valid_from ASC; we want
        # most-recent-first, so reverse and take the first *limit* entries.
        triples = kg.timeline()
        today = date.today().isoformat()
        current = [
            t for t in reversed(triples)
            if (t["valid_to"] is None or t["valid_to"] >= today)
            and (t["valid_from"] is None or t["valid_from"] <= today)
        ][:limit]
        return "\n".join(
            f"{t['subject']} {t['predicate']} {t['object']}" for t in current
        )
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
            log.warning(
                "mempalace wake-up exited %d: %s",
                result.returncode, result.stderr.strip(),
            )
        return result.stdout.strip()
    except Exception as e:
        log.warning("mempalace wake-up subprocess failed: %s", e)
        return ""


@functools.lru_cache(maxsize=1)
def _wakeup_cached() -> str:
    """Assemble L0 + L1 context block (cached, single slot).

    Prefer KnowledgeGraph reads over subprocess to avoid spawn overhead.
    Falls back to subprocess only when KG file is missing.
    """
    identity = _read_identity()
    kg_block = _read_kg_block()

    if not kg_block:
        log.info("FastMemory: KG empty or missing, falling back to subprocess wake-up")
        return _run_wakeup_subprocess()

    parts = [p for p in (identity, kg_block) if p]
    text = "\n\n".join(parts)
    log.info("FastMemory: wake-up assembled (%d chars, KnowledgeGraph)", len(text))
    return text


def wake_up() -> str:
    """Return the system-prompt context block (L0 + L1).

    Cheap on repeated calls — result is cached until invalidate() is called.
    """
    return _wakeup_cached()


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def init() -> None:
    """Open KnowledgeGraph at startup so the first request pays no init cost."""
    _get_kg()

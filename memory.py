#!/usr/bin/env python3
"""memory.py — fast in-process memory abstraction for chatd.

Layered memory model:
  L0   mempalace : model identity.txt — always loaded
  L0.5 chatd     : composed layer — per-chat compressed summary +
                   global compressed summary — always loaded
  L1   mempalace : Essential Story (wake-up context) — always loaded
  L1.5 chatd     : request-scoped sidecar — KG recall + external RAG
                   injected per request — always loaded
  L2   mempalace : Room Recall (filtered retrieval) — when topic comes up
  L3   mempalace : Deep Search (full semantic query) — when explicitly asked

mempalace layers (L0, L1, L2, L3) are never modified here.
chatd layers (L0.5, L1.5) are purely additive overlays.
"""
import functools
import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional

from mempalace.knowledge_graph import KnowledgeGraph

from config import (
    CHATD_GLOBAL_SUMMARY_PATH,
    MEMPALACE_KG_PATH,
    MEMPALACE_PALACE_PATH,
)

log = logging.getLogger("chatd.memory")

# ── L1: in-process KG ───────────────────────────────────────────────────────

_kg: Optional[KnowledgeGraph] = None

# Words to skip when extracting entity candidates from free text.
_STOPWORDS = frozenset({
    "что", "как", "где", "это", "мне", "ты", "я", "он", "она", "они",
    "мой", "моя", "моё", "мои", "the", "and", "for", "you", "are",
    "that", "this", "with", "from", "have", "what", "when", "where",
})


def _get_kg() -> Optional[KnowledgeGraph]:
    """Return a shared KnowledgeGraph instance, initialised lazily."""
    global _kg
    if _kg is None:
        kg_path = os.path.expanduser(MEMPALACE_KG_PATH)
        if not Path(kg_path).exists():
            log.warning("FastMemory: KG database not found: %s", kg_path)
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


def kg_recall_from_text(text: str, max_results: int = 5) -> Optional[str]:
    """Extract entity candidates from *text* and query KG for each.

    Scans the user message for words that might be KG subjects/entities,
    queries each one via kg_recall(), and returns a deduplicated fact block.
    Returns None when nothing is found — caller decides whether to fall back
    to mempalace_kg_query MCP tool.

    Heuristic: words longer than 3 chars that are not in _STOPWORDS.
    At most 8 entity candidates are checked to bound latency.
    """
    words = {
        w.lower().strip(".,!?:;\"'()")
        for w in text.split()
        if len(w) > 3 and w.lower().strip(".,!?:;\"'()") not in _STOPWORDS
    }
    if not words:
        return None

    kg = _get_kg()
    if kg is None:
        return None

    today = date.today().isoformat()
    lines: list[str] = []
    seen: set[str] = set()

    for word in sorted(words)[:8]:
        try:
            triples = kg.query_entity(word, as_of=today)
        except Exception as e:
            log.debug("FastMemory: kg_recall_from_text entity=%s failed: %s", word, e)
            continue
        for t in triples:
            key = f"{t['subject']}|{t['predicate']}|{t['object']}"
            if key not in seen and t.get("current"):
                seen.add(key)
                lines.append(f"{t['subject']} {t['predicate']} {t['object']}")
        if len(lines) >= max_results:
            break

    if not lines:
        return None
    result = "\n".join(lines[:max_results])
    log.debug("FastMemory: kg_recall_from_text found %d facts", len(lines))
    return result


def read_global_summary() -> str:
    """Read L0.5 global activity summary from disk."""
    path = Path(os.path.expanduser(CHATD_GLOBAL_SUMMARY_PATH))
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except Exception as e:
        log.warning("global summary read failed: %s", e)
        return ""


def write_global_summary(summary: str) -> None:
    """Persist L0.5 global activity summary to disk."""
    path = Path(os.path.expanduser(CHATD_GLOBAL_SUMMARY_PATH))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary.strip(), encoding="utf-8")
    except Exception as e:
        log.warning("global summary write failed: %s", e)


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


# ── wake-up assembly ──────────────────────────────────────────────────────

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
    """Assemble always-loaded mempalace block (cached, single slot).

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


def wake_up(per_chat_summary: str = "", global_summary: str = "") -> str:
    """Assemble the always-loaded system-prompt block.

    L0.5 is a composed layer: both per_chat_summary and global_summary
    are part of it. They are emitted separately so the model can
    distinguish chat-scoped context from broader activity context.

    Parameters
    ----------
    per_chat_summary:
        Per-chat rolling compressed summary (L0.5 chat part).
    global_summary:
        Cross-chat activity summary (L0.5 global part).
    """
    base = _wakeup_cached()   # L0 + L1, mempalace, cached
    parts = [base]
    if global_summary:
        parts.append(f"## Recent activity (global)\n{global_summary}")
    if per_chat_summary:
        parts.append(f"## Conversation summary\n{per_chat_summary}")
    return "\n\n".join(p for p in parts if p)


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def _check_palace(palace_path: str) -> bool:
    """Verify that the palace directory exists and contains at least one wing.

    A freshly initialised palace has at minimum a README or a wing subdirectory.
    Returns True when the palace looks healthy, False otherwise.
    """
    p = Path(palace_path)
    if not p.exists():
        log.warning("mempalace: palace directory missing: %s", palace_path)
        return False
    if not p.is_dir():
        log.warning("mempalace: palace path is not a directory: %s", palace_path)
        return False
    children = list(p.iterdir())
    if not children:
        log.warning("mempalace: palace directory is empty (not initialised?): %s", palace_path)
        return False
    log.info("mempalace: palace OK — %s (%d entries)", palace_path, len(children))
    return True


def _check_kg(kg_path: str) -> bool:
    """Verify that the KG SQLite file exists and KnowledgeGraph can open it.

    Returns True when the KG is healthy, False otherwise.
    """
    p = Path(kg_path)
    if not p.exists():
        log.warning("mempalace: KG database missing: %s", kg_path)
        log.warning(
            "mempalace: run 'python -m mempalace init' to create the database"
        )
        return False
    try:
        kg = KnowledgeGraph(db_path=kg_path)
        count = len(kg.timeline())
        kg.close()
        log.info("mempalace: KG OK — %s (%d triples)", kg_path, count)
        return True
    except Exception as e:
        log.warning("mempalace: KG open failed: %s — %s", kg_path, e)
        return False


def init() -> None:
    """Run startup health checks and open the KnowledgeGraph connection.

    Logs the status of every required path.  Emits WARNING (not an exception)
    when something is missing so the service can still start in degraded mode
    (e.g. answers without memory context until the user initialises mempalace).
    """
    from mempalace import __version__ as mp_version
    log.info("mempalace: version %s", mp_version)

    palace_path = os.path.expanduser(MEMPALACE_PALACE_PATH)
    kg_path     = os.path.expanduser(MEMPALACE_KG_PATH)

    palace_ok = _check_palace(palace_path)
    kg_ok     = _check_kg(kg_path)

    if palace_ok and kg_ok:
        log.info("mempalace: all checks passed, opening KnowledgeGraph")
        _get_kg()  # warm up the connection at startup
    else:
        issues = []
        if not palace_ok:
            issues.append(f"palace not found at {palace_path}")
        if not kg_ok:
            issues.append(f"KG not found at {kg_path}")
        log.warning(
            "mempalace: degraded mode — %s. "
            "Memory context will be empty until mempalace is initialised.",
            "; ".join(issues),
        )

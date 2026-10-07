#!/usr/bin/env python3
"""Central configuration module.

All runtime settings are read from environment variables so that the project
can be installed anywhere without modifying source files.  Copy .env.example
to .env (or set Environment= in the systemd unit) and adjust as needed.
"""
import configparser
import os
from pathlib import Path
from typing import Any, Dict, List, Set

# ── Thinking mode ────────────────────────────────────────────────────────────────────
THINKING: bool = os.environ.get("CHATD_THINKING", "false").lower() == "true"

# ── Tools ────────────────────────────────────────────────────────────────────────
TOOLS_FILTER: bool = os.environ.get("CHATD_TOOLS_FILTER", "false").lower() == "true"

_default_allowed = (
    "mempalace_status,mempalace_search,mempalace_add_drawer,"
    "mempalace_kg_query,mempalace_kg_add"
)
TOOLS_ALLOWED: Set[str] = set(
    os.environ.get("CHATD_TOOLS_ALLOWED", _default_allowed).split(",")
)

# ── Tool rounds ──────────────────────────────────────────────────────────────────
MAX_TOOL_ROUNDS: int = int(os.environ.get("CHATD_MAX_TOOL_ROUNDS", "5"))

# ── History window ───────────────────────────────────────────────────────────────
# Number of user+assistant pairs to keep when trimming history before sending
# to the backend.  Increase when using a model with a large context window.
MAX_HISTORY_TURNS: int = int(os.environ.get("CHATD_MAX_HISTORY_TURNS", "20"))

# ── Sampling ───────────────────────────────────────────────────────────────────────
DEFAULT_OPTIONS: Dict[str, Any] = {
    "num_predict":    int(os.environ.get("CHATD_NUM_PREDICT",    "768")),
    "num_ctx":        int(os.environ.get("CHATD_NUM_CTX",        "8192")),
    "temperature":    float(os.environ.get("CHATD_TEMPERATURE",  "0.3")),
    "top_p":          float(os.environ.get("CHATD_TOP_P",        "0.9")),
    "top_k":          int(os.environ.get("CHATD_TOP_K",          "40")),
    "repeat_penalty": float(os.environ.get("CHATD_REPEAT_PENALTY", "1.05")),
}

# ── MemPalace ────────────────────────────────────────────────────────────────────────
MEMPALACE_PALACE_PATH: str = os.environ.get(
    "MEMPALACE_PALACE_PATH", "~/.local/share/mempalace"
)

MEMPALACE_KG_PATH: str = os.environ.get(
    "MEMPALACE_KG_PATH", "~/.mempalace/knowledge_graph.sqlite3"
)

MEMPALACE_WRITE_TOOLS: Set[str] = {
    "mempalace_kg_add",
    "mempalace_add_drawer",
}

# ── Session ───────────────────────────────────────────────────────────────────────────────
CHATD_SESSION_DIR: str = os.environ.get(
    "CHATD_SESSION_DIR", "~/.local/share/chatd/sessions"
)

# ── External RAG store ───────────────────────────────────────────────────────────────────────────
RAG_ENABLED: bool = os.environ.get("CHATD_RAG_ENABLED", "true").lower() == "true"
RAG_DB_PATH: str = os.environ.get(
    "CHATD_RAG_DB_PATH", "~/.local/share/chatd/rag.sqlite3"
)
RAG_EMBED_MODEL: str = os.environ.get("CHATD_RAG_EMBED_MODEL", "nomic-embed-text")
RAG_TOP_K: int = int(os.environ.get("CHATD_RAG_TOP_K", "4"))
RAG_MAX_CHARS_PER_CHUNK: int = int(os.environ.get("CHATD_RAG_MAX_CHARS_PER_CHUNK", "4000"))
RAG_MIN_SCORE: float = float(os.environ.get("CHATD_RAG_MIN_SCORE", "0.25"))

# ── OpenRouter ───────────────────────────────────────────────────────────────────────────────────────
OPENROUTER_API_MODELS: List[str] = [
    m.strip()
    for m in os.environ.get("OPENROUTER_API_MODELS", "").split(",")
    if m.strip()
]

# ── Events ──────────────────────────────────────────────────────────────────────────
# Optional shared secret for /api/event.
# If set, caller must pass Authorization: Bearer <token>.
# If empty - endpoint is open (only for trusted internal networks).
CHATD_EVENT_TOKEN: str = os.environ.get("CHATD_EVENT_TOKEN", "")

# Model used to process incoming events (can differ from chat model).
CHATD_EVENT_MODEL: str = os.environ.get("CHATD_EVENT_MODEL", "")

# ── MCP auto-discovery prefix ───────────────────────────────────────────────────────────────────────
MCP_ENV_PREFIX: str = "CHATD_MCP_"

# ── Tool description overrides ──────────────────────────────────────────────────────────────────────────────────────────────────────
TOOL_OVERRIDE: bool = os.environ.get("CHATD_TOOL_OVERRIDE", "false").lower() == "true"

def _load_tool_descriptions() -> Dict[str, str]:
    cfg = configparser.ConfigParser()
    bundled = Path(__file__).parent / "tool_descriptions.ini"
    paths = [
        str(bundled),
        "/etc/chatd/tool_descriptions.ini",
        os.path.expanduser("~/.config/chatd/tool_descriptions.ini"),
        os.environ.get("CHATD_TOOL_DESCRIPTIONS", ""),
    ]
    cfg.read([p for p in paths if p])
    return {
        key: val.strip()
        for section in cfg.sections()
        for key, val in cfg.items(section)
    }


TOOL_DESCRIPTION_OVERRIDES: Dict[str, str] = _load_tool_descriptions()

DEEPSEEK_API = os.environ.get(
    "CHATD_DEEPSEEK_API",
    "https://api.deepseek.com",
)

DEEPSEEK_API_MODELS = [
    m.strip()
    for m in os.environ.get(
        "CHATD_DEEPSEEK_MODELS",
        "deepseek-flash",
    ).split(",")
    if m.strip()
]

# ── Background thinking (subconscious) ───────────────────────────────────────
# Disabled by default - enable per-installation if you want a periodic
# background worker ("subconscious") that observes and proposes, but never
# writes to canonical memory.
BG_ENABLED: bool = os.environ.get("CHATD_BG_ENABLED", "false").lower() == "true"

# Optional bearer token. If set, /api/tick requires Authorization: Bearer <token>.
# Keep separate from CHATD_EVENT_TOKEN - different triggers, different secrets.
BG_TOKEN: str = os.environ.get("CHATD_BG_TOKEN", "")

# Where the background worker keeps its own state (journal, ledger, locks).
# Never place this inside mempalace or session storage.
BG_STATE_DIR: str = os.environ.get(
    "CHATD_BG_STATE_DIR", "~/.local/share/chatd/bg"
)

# Session id used for background ticks - isolated from user chat sessions.
BG_SESSION_ID: str = os.environ.get("CHATD_BG_SESSION_ID", "background")

# Model used by the background worker. Same prefix routing as chat models
# (e.g. "qwen3:8b" → Ollama, "deepseek/deepseek-flash" → DeepSeek).
BG_MODEL: str = os.environ.get("CHATD_BG_MODEL", "qwen3:8b")

# Max output tokens per background tick. Background answers are short by
# design; keep this well below the chat default to avoid 30-minute ticks
# on CPU-only hosts.
BG_NUM_PREDICT: int = int(os.environ.get("CHATD_BG_NUM_PREDICT", "256"))

# Max tool rounds per tick (reserved for Planner/Executor - next PR).
BG_MAX_TOOL_ROUNDS: int = int(os.environ.get("CHATD_BG_MAX_TOOL_ROUNDS", "5"))

# Tools visible to the background worker - read-only subset.
# Filter is applied when building the tool list for /api/tick.
BG_TOOLS_ALLOWED: Set[str] = set(
    os.environ.get(
        "CHATD_BG_TOOLS_ALLOWED",
        "mempalace_search,mempalace_kg_query,mempalace_status,"
        "mempalace_wake_up,mempalace_kg_timeline,"
        "alerts_list,alerts_summary,monitor_query,notes_search",
    ).split(",")
)

# ── Background thinking - phases (PR C) ──────────────────────────────────────
# Per-phase model routing. If a phase var is empty, BG_MODEL is used as
# fallback - this preserves backwards compatibility with PR B+.
BG_MODEL_PLANNER: str = os.environ.get("CHATD_BG_MODEL_PLANNER", "")
BG_MODEL_EXECUTOR: str = os.environ.get("CHATD_BG_MODEL_EXECUTOR", "")
BG_MODEL_REFLECTOR: str = os.environ.get("CHATD_BG_MODEL_REFLECTOR", "")
# Reserved for PR D (escalation on repeated failure). Declared now, unused.
BG_MODEL_THINK: str = os.environ.get("CHATD_BG_MODEL_THINK", "")

# Per-phase timeouts (seconds). Executor gets more room for its tool loop.
BG_PHASE_TIMEOUT: int = int(os.environ.get("CHATD_BG_PHASE_TIMEOUT", "600"))
BG_PHASE_TIMEOUT_EXECUTOR: int = int(
    os.environ.get("CHATD_BG_PHASE_TIMEOUT_EXECUTOR", "900")
)
# Whole-tick timeout - everything above should fit inside this.
BG_TICK_TIMEOUT: int = int(os.environ.get("CHATD_BG_TICK_TIMEOUT", "2700"))

# Goal ledger limits.
BG_MAX_ATTEMPTS_PER_GOAL: int = int(
    os.environ.get("CHATD_BG_MAX_ATTEMPTS_PER_GOAL", "3")
)
# Maximum number of goals in status pending or active. blocked/done/cancelled
# do not count. Once reached, Planner cannot create new goals and Executor
# drops any new_goals from its output.
BG_MAX_OPEN_GOALS: int = int(os.environ.get("CHATD_BG_MAX_OPEN_GOALS", "5"))

# Exploratory goals (created by Executor when Planner returns skip).
#  - local executor model only
#  - read-only tools only
#  - one active at a time
#  - TTL: auto-cancelled if not closed within N hours
#  - hard cap per day
BG_EXPLORATORY_PER_DAY: int = int(
    os.environ.get("CHATD_BG_EXPLORATORY_PER_DAY", "3")
)
BG_EXPLORATORY_TTL_HOURS: int = int(
    os.environ.get("CHATD_BG_EXPLORATORY_TTL_HOURS", "24")
)

# Daily budget. 0 = no limit. Token counting relies on Ollama's
# prompt_eval_count + eval_count; OpenAI-compatible backends currently do
# not surface token usage through chatd, so their ticks count as 0.
BG_DAILY_TOKEN_BUDGET: int = int(
    os.environ.get("CHATD_BG_DAILY_TOKEN_BUDGET", "0")
)
BG_DAILY_TIME_BUDGET_SEC: int = int(
    os.environ.get("CHATD_BG_DAILY_TIME_BUDGET_SEC", "0")
)

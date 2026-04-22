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

# ── Thinking mode ────────────────────────────────────────────────────────────────────────
THINKING: bool = os.environ.get("CHATD_THINKING", "false").lower() == "true"

# ── Tools ────────────────────────────────────────────────────────────────────────────────
TOOLS_FILTER: bool = os.environ.get("CHATD_TOOLS_FILTER", "false").lower() == "true"

_default_allowed = (
    "mempalace_status,mempalace_search,mempalace_add_drawer,"
    "mempalace_kg_query,mempalace_kg_add"
)
TOOLS_ALLOWED: Set[str] = set(
    os.environ.get("CHATD_TOOLS_ALLOWED", _default_allowed).split(",")
)

# ── Sampling ─────────────────────────────────────────────────────────────────────────────────
DEFAULT_OPTIONS: Dict[str, Any] = {
    "num_predict":    int(os.environ.get("CHATD_NUM_PREDICT",    "768")),
    "num_ctx":        int(os.environ.get("CHATD_NUM_CTX",        "8192")),
    "temperature":    float(os.environ.get("CHATD_TEMPERATURE",  "0.3")),
    "top_p":          float(os.environ.get("CHATD_TOP_P",        "0.9")),
    "top_k":          int(os.environ.get("CHATD_TOP_K",          "40")),
    "repeat_penalty": float(os.environ.get("CHATD_REPEAT_PENALTY", "1.05")),
}

# ── MemPalace ─────────────────────────────────────────────────────────────────────────────────
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

# ── Session / summaries ─────────────────────────────────────────────────────────────────────────
CHATD_SESSION_DIR: str = os.environ.get(
    "CHATD_SESSION_DIR", "~/.local/share/chatd/sessions"
)

CHATD_GLOBAL_SUMMARY_PATH: str = os.environ.get(
    "CHATD_GLOBAL_SUMMARY_PATH", "~/.local/share/chatd/global_summary.txt"
)

CHATD_SUMMARY_MODEL: str = os.environ.get("CHATD_SUMMARY_MODEL", "qwen2.5:1.5b")

CHATD_COMPRESS_EVERY: int = int(os.environ.get("CHATD_COMPRESS_EVERY", "5"))

# ── External RAG store ────────────────────────────────────────────────────────────────────────────
RAG_DB_PATH: str = os.environ.get(
    "CHATD_RAG_DB_PATH", "~/.local/share/chatd/rag.sqlite3"
)
RAG_EMBED_MODEL: str = os.environ.get("CHATD_RAG_EMBED_MODEL", "nomic-embed-text")
RAG_TOP_K: int = int(os.environ.get("CHATD_RAG_TOP_K", "4"))
RAG_MAX_CHARS_PER_CHUNK: int = int(os.environ.get("CHATD_RAG_MAX_CHARS_PER_CHUNK", "4000"))
RAG_MIN_SCORE: float = float(os.environ.get("CHATD_RAG_MIN_SCORE", "0.25"))

# ── OpenRouter ───────────────────────────────────────────────────────────────────────────────────────
# Whitelist of OpenRouter models to expose via /api/tags.
# Format: comma-separated OR model IDs with the configured prefix.
# Example: OPENROUTER_API_MODELS=or/google/gemini-2.0-flash,or/anthropic/claude-3.5-sonnet
# If empty, no OpenRouter models are injected into /api/tags.
OPENROUTER_API_MODELS: List[str] = [
    m.strip()
    for m in os.environ.get("OPENROUTER_API_MODELS", "").split(",")
    if m.strip()
]

# ── MCP auto-discovery prefix ───────────────────────────────────────────────────────────────────────────────
MCP_ENV_PREFIX: str = "CHATD_MCP_"

# ── Tool description overrides ───────────────────────────────────────────────────────────────────────────────────────
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

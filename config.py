#!/usr/bin/env python3
"""Central configuration module.

All runtime settings are read from environment variables so that the project
can be installed anywhere without modifying source files.  Copy .env.example
to .env (or set Environment= in the systemd unit) and adjust as needed.
"""
import configparser
import os
from pathlib import Path
from typing import Any, Dict, Set

# ── Ollama ─────────────────────────────────────────────────────────────────────────────
OLLAMA_API: str = os.environ.get("OLLAMA_API", "http://127.0.0.1:11434")

# ── Thinking mode ─────────────────────────────────────────────────────────────────
# When True, ThinkingRemapper wraps model's thinking tokens in <think>...</think>.
# Keep False for qwen3 tool-calling (thinking mode degrades JSON fidelity).
THINKING: bool = os.environ.get("CHATD_THINKING", "false").lower() == "true"

# ── Tools ─────────────────────────────────────────────────────────────────
# Enable filtering of the list of tools available to the model.
TOOLS_FILTER: bool = os.environ.get("CHATD_TOOLS_FILTER", "false").lower() == "true"

# Tools to expose to the model.  All others are loaded into TOOL_REGISTRY
# (for call_tool) but hidden from the model context to save tokens.
_default_allowed = (
    "mempalace_status,mempalace_search,mempalace_add_drawer,"
    "mempalace_kg_query,mempalace_kg_add"
)
TOOLS_ALLOWED: Set[str] = set(
    os.environ.get("CHATD_TOOLS_ALLOWED", _default_allowed).split(",")
)

# ── Sampling ────────────────────────────────────────────────────────────────────
# Tuned for qwen3 8B non-thinking tool-calling.
# Raise temperature to 0.7–1.0 when THINKING=true.
DEFAULT_OPTIONS: Dict[str, Any] = {
    "num_predict":    int(os.environ.get("CHATD_NUM_PREDICT",    "768")),
    "num_ctx":        int(os.environ.get("CHATD_NUM_CTX",        "8192")),
    "temperature":    float(os.environ.get("CHATD_TEMPERATURE",  "0.3")),
    "top_p":          float(os.environ.get("CHATD_TOP_P",        "0.9")),
    "top_k":          int(os.environ.get("CHATD_TOP_K",          "40")),
    "repeat_penalty": float(os.environ.get("CHATD_REPEAT_PENALTY", "1.05")),
}

# ── MemPalace ─────────────────────────────────────────────────────────────────────
# Path to the palace directory (wings/rooms hierarchy).
MEMPALACE_PALACE_PATH: str = os.environ.get(
    "MEMPALACE_PALACE_PATH", "~/.local/share/mempalace"
)

# Path to knowledge_graph.sqlite3. The KG file is NOT necessarily a sibling
# of PALACE_PATH — set this explicitly when they live in different directories.
MEMPALACE_KG_PATH: str = os.environ.get(
    "MEMPALACE_KG_PATH", "~/.mempalace/knowledge_graph.sqlite3"
)

# Tools that write to memory → wake-up cache must be invalidated after.
MEMPALACE_WRITE_TOOLS: Set[str] = {
    "mempalace_kg_add",
    "mempalace_add_drawer",
}

# ── Session / L1.5 rolling summary ────────────────────────────────────────────
# Directory where per-chat session JSON files are stored.
CHATD_SESSION_DIR: str = os.environ.get(
    "CHATD_SESSION_DIR", "~/.local/share/chatd/sessions"
)

# Ollama model used for compressing the rolling summary (L1.5).
# A small, fast model is preferable — the main model is not needed here.
CHATD_SUMMARY_MODEL: str = os.environ.get("CHATD_SUMMARY_MODEL", "qwen2.5:1.5b")

# Number of new Q/A turns to accumulate before triggering a compression pass.
CHATD_COMPRESS_EVERY: int = int(os.environ.get("CHATD_COMPRESS_EVERY", "5"))

# ── MCP auto-discovery prefix ─────────────────────────────────────────────────────────
MCP_ENV_PREFIX: str = "CHATD_MCP_"

# ── Tool description overrides ──────────────────────────────────────────────────────────
# Loaded from INI files in this order (later files override earlier ones):
#   1. tool_descriptions.ini  (next to this file, shipped with the repo)
#   2. /etc/chatd/tool_descriptions.ini  (system-local, not in git)
#   3. ~/.config/chatd/tool_descriptions.ini  (user-local, not in git)
#   4. $CHATD_TOOL_DESCRIPTIONS  (runtime override, arbitrary path)

TOOL_OVERRIDE: bool = os.environ.get("CHATD_TOOL_OVERRIDE", "false").lower() == "true"

def _load_tool_descriptions() -> Dict[str, str]:
    cfg = configparser.ConfigParser()
    # Resolve bundled default relative to this file so it works from any cwd.
    bundled = Path(__file__).parent / "tool_descriptions.ini"
    paths = [
        str(bundled),
        "/etc/chatd/tool_descriptions.ini",
        os.path.expanduser("~/.config/chatd/tool_descriptions.ini"),
        os.environ.get("CHATD_TOOL_DESCRIPTIONS", ""),
    ]
    cfg.read([p for p in paths if p])  # skip empty strings
    return {
        key: val.strip()
        for section in cfg.sections()
        for key, val in cfg.items(section)
    }


TOOL_DESCRIPTION_OVERRIDES: Dict[str, str] = _load_tool_descriptions()

#!/usr/bin/env python3
"""Central configuration module.

All runtime settings are read from environment variables so that the project
can be installed anywhere without modifying source files.  Copy .env.example
to .env (or set Environment= in the systemd unit) and adjust as needed.
"""
import os
from typing import Any, Dict, Set

# ── Ollama ───────────────────────────────────────────────────────────────────────
OLLAMA_API: str = os.environ.get("OLLAMA_API", "http://127.0.0.1:11434")

# ── Thinking mode ───────────────────────────────────────────────────────────────
# When True, ThinkingRemapper wraps model's thinking tokens in <think>...</think>.
# Keep False for qwen3 tool-calling (thinking mode degrades JSON fidelity).
THINKING: bool = os.environ.get("CHATD_THINKING", "false").lower() == "true"

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

# ── MemPalace ───────────────────────────────────────────────────────────────────
# Tools to expose to the model.  All others are loaded into TOOL_REGISTRY
# (for call_tool) but hidden from the model context to save tokens.
_default_allowed = (
    "mempalace_status,mempalace_search,mempalace_add_drawer,"
    "mempalace_kg_query,mempalace_kg_add"
)
MEMPALACE_ALLOWED_TOOLS: Set[str] = set(
    os.environ.get("CHATD_MEMPALACE_TOOLS", _default_allowed).split(",")
)

# Tools that write to memory → wake-up cache must be invalidated after.
MEMPALACE_WRITE_TOOLS: Set[str] = {
    "mempalace_kg_add",
    "mempalace_add_drawer",
}

# ── MCP auto-discovery prefix ───────────────────────────────────────────────────────
MCP_ENV_PREFIX: str = "CHATD_MCP_"

# ── Tool description overrides ──────────────────────────────────────────────────────
# Replacing MCP-server descriptions with richer ones improves the model's
# tool-selection accuracy (it knows not just *what* a tool does but *when*).
TOOL_DESCRIPTION_OVERRIDES: Dict[str, str] = {
    "alerts_list": (
        "List Alertmanager alerts. "
        "Call ONLY when the user asks about server alerts, incidents, or firing rules."
    ),
    "alerts_summary": (
        "Summarize active Alertmanager alerts by name/severity/state. "
        "Call ONLY when asked for a monitoring overview or alert statistics."
    ),
    "monitor_query": (
        "Query VictoriaMetrics with PromQL. "
        "Call ONLY when the user asks for specific metrics (CPU, RAM, uptime, etc.)."
    ),
    "notes_search": (
        "Search Memos notes by text query. "
        "Call ONLY when the user asks to find or list their notes."
    ),
    "mempalace_status": (
        "Returns palace overview (wings, rooms, drawer count), AAAK dialect spec, "
        "and memory protocol instructions. "
        "Call ONCE at the start of every session before using any other memory tool. "
        "The returned protocol tells you how to correctly use add_drawer and search."
    ),
    "mempalace_search": (
        "Semantic vector search over all stored memories (drawers). "
        "Call when the user asks about their preferences, habits, past decisions, "
        "or any personal fact not present in the current conversation. "
        "Use the 'wing' and 'room' filters to narrow results when the topic is clear "
        "(e.g. wing='Person', room='preferences'). "
        "Fall back to this if mempalace_kg_query returned no results."
    ),
    "mempalace_add_drawer": (
        "File verbatim text into the palace (writes to ChromaDB, persists long-term). "
        "Use for ANY content longer than a short phrase: paragraphs, user bios, "
        "conversation summaries, decisions, preferences in natural language. "
        "Required args: "
        "'wing' (domain, e.g. 'wing_general' or 'wing_person'), "
        "'room' (topic, e.g. 'hall_preferences' or 'hall_facts'), "
        "'content' (the verbatim text to store, in AAAK dialect if possible). "
        "Call after EVERY turn where the user shared personal facts worth keeping. "
        "Do NOT call for greetings or transient context."
    ),
    "mempalace_kg_query": (
        "Query the structured knowledge graph for facts about a named entity. "
        "Call FIRST (before mempalace_search) when the user asks about their own "
        "attributes: favorite tools, location, job, language, etc. "
        "Required arg: 'entity' (e.g. 'user'). "
        "Returns subject\u2192predicate\u2192object triples with temporal validity. "
        "If result is empty, follow up with mempalace_search."
    ),
    "mempalace_kg_add": (
        "Add ONE atomic subject\u2192predicate\u2192object triple to the knowledge graph. "
        "Use for short, structured facts ONLY: single values, not sentences or paragraphs. "
        "Args MUST be ASCII, no Cyrillic, no spaces in values. "
        "Good: subject='user', predicate='favorite_editor', object='emacs'. "
        "Good: subject='user', predicate='location', object='Omsk'. "
        "BAD: object='Enthusiast of computer networks...' — "
        "use mempalace_add_drawer for multi-sentence content instead. "
        "When given a block of text with multiple facts: decompose into individual "
        "triples (one call per fact) AND also file the full text via add_drawer. "
        "Call ONLY when the user EXPLICITLY asks you to remember a specific fact."
    ),
}

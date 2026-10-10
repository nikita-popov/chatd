# subconscious-core

Tooling for autonomous neural agents that run in your infrastructure,
accumulate knowledge about their environment, and stay in the loop with you.

This is the core package: a chat interface, a background thinker, a memory
layer, and a router to LLMs and MCP tools. Bring your own infrastructure,
your own models, and your own goals - subconscious-core handles the rest.

**Docs:** [GOALS.md](docs/GOALS.md) · [ARCHITECTURE.md](docs/ARCHITECTURE.md)

## What it is

Not a framework. Not a SaaS. A ready-to-run application for one kind of
agent: the one that lives next to your servers, notices things, investigates
(read-only), remembers what it found, and asks before acting.

A typical deployment:

```
you  ──chat──►  subconscious-core  ──►  LLM (local or cloud)
                       │
                       ├──►  MCP tools  (monitor, alerts, notes, ssh, git…)
                       ├──►  memory     (KG, RAG, sessions)
                       └──►  subcortex  (planner → executor → reflector)
                                             ↑
                                        systemd timer
```

The agent is autonomous in:
- **investigation** - read-only everywhere;
- **writing** - to its own state files;
- **its memory**.

Everything else goes through you.

## Four zones

`subconscious-core` is organized in four zones, each with one job:

| Zone | Responsibility |
|---|---|
| **core** | data and state - memory, KG, ledger, sessions, budget |
| **cortex** | interfaces outward - chat, events, A2A, CLI |
| **thalamus** | bindings - LLM backends, MCP client, tool routing |
| **subcortex** | background thinking - planner, executor, reflector |

Chat is a *cortex* concern, not the core of the project. The subcortex is
where autonomy lives.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full picture.

## Features

- **Chat interface** - NDJSON streaming, tool loop, layered memory injection.
- **Background thinker** - hourly tick, three phases (planner → executor →
  reflector), goal ledger with attempts, history, archive.
- **Exploratory mode** - when there are no open goals, the agent performs a
  read-only observation, rotating through available tools.
- **Layered memory** - identity (L0), wake-up (L1), and a request-scoped
  sidecar (L1.5) built from KG recall and RAG.
- **MCP tool dispatch** - the agent sees MCP servers as tools; the list is
  filtered per phase (chat sees more than the background worker).
- **Multi-backend routing** - Ollama, OpenRouter, DeepSeek, ONNX. Selected
  by model name prefix.
- **Layered configuration** - code defaults, `/etc/subconscious/`,
  `~/.config/subconscious/`, then env vars. Later sources win.
- **A2A compatibility (in progress)** - Agent Card and Task mapping so other
  agents can talk to this one.

## Requirements

- Python 3.11+
- An LLM endpoint: Ollama (local), or API keys for cloud backends
- Optional: MCP servers for the tools you want the agent to use
- Optional: mempalace for the legacy memory layer

## Quick start

```sh
git clone https://github.com/subconscious-labs/subconscious-core
cd subconscious-core

python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
$EDITOR .env
```

Minimum to get a chat running:

```sh
OLLAMA_API=http://127.0.0.1:11434
SUBCONSCIOUS_MCP_MEMPALACE=/opt/subconscious/venv/bin/python -m mempalace.mcp_server
MEMPALACE_PALACE_PATH=/var/lib/mempalace
```

Run:

```sh
python chatd.py
```

Or via systemd - see `subconscious.service.example`.

## Background thinking

Enable and configure in `/etc/subconscious/subconscious.env`:

```sh
SUBCONSCIOUS_BG_ENABLED=true
SUBCONSCIOUS_BG_TOKEN=<random-secret>
SUBCONSCIOUS_BG_MODEL=qwen3.5:9b
SUBCONSCIOUS_BG_MODEL_PLANNER=qwen3.5:4b
SUBCONSCIOUS_BG_MODEL_REFLECTOR=qwen3.5:4b
```

Trigger manually:

```sh
curl -X POST http://127.0.0.1:5001/api/tick \
  -H "Authorization: Bearer $SUBCONSCIOUS_BG_TOKEN" \
  -d '{}'
```

Or install the timer: `subconscious-tick.timer` fires hourly, calls the
endpoint, and the agent does one tick of thinking.

All background state lives in `~/.local/share/subconscious/bg/`:

- `goals.json` - active goals (pending, active, blocked)
- `goals.archive.jsonl` - completed and cancelled goals
- `journal.jsonl` - append-only tick log
- `budget.json` - daily token and time counters
- `seed_rotation.json` - which exploratory tool to use next
- `tick.lock` - flock, prevents overlapping ticks

## Configuration

Layered lookup: `code defaults` < `/etc/subconscious/` < `~/.config/subconscious/` < `env`.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_API` | `http://127.0.0.1:11434` | Ollama endpoint |
| `SUBCONSCIOUS_THINKING` | `false` | Enable model extended thinking |
| `SUBCONSCIOUS_MAX_TOOL_ROUNDS` | `5` | Max tool rounds per chat request |
| `SUBCONSCIOUS_MAX_HISTORY_TURNS` | `20` | History window (user+assistant pairs) |
| `SUBCONSCIOUS_RAG_ENABLED` | `true` | Enable L1.5 RAG sidecar |
| `SUBCONSCIOUS_RAG_DB_PATH` | `~/.local/share/subconscious/rag.sqlite3` | RAG store |
| `MEMPALACE_PALACE_PATH` | `~/.local/share/mempalace` | mempalace palace directory |
| `MEMPALACE_KG_PATH` | `~/.mempalace/knowledge_graph.sqlite3` | mempalace KG |
| `DEEPSEEK_API_KEY` | *(unset)* | DeepSeek API key |
| `OPENROUTER_API_KEY` | *(unset)* | OpenRouter API key |
| `SUBCONSCIOUS_BG_ENABLED` | `false` | Enable background worker |
| `SUBCONSCIOUS_BG_TOKEN` | *(unset)* | Bearer token for `/api/tick` |
| `SUBCONSCIOUS_BG_MODEL` | `qwen3.5:9b` | Fallback model for all phases |
| `SUBCONSCIOUS_BG_MODEL_PLANNER` | *(unset)* | Planner model override |
| `SUBCONSCIOUS_BG_MODEL_EXECUTOR` | *(unset)* | Executor model override |
| `SUBCONSCIOUS_BG_MODEL_REFLECTOR` | *(unset)* | Reflector model override |

Full list: `.env.example`.

**Backward compatibility:** all `CHATD_*` env vars continue to work as
fallbacks. The agent logs a deprecation warning when it sees one.

## Overriding background prompts

Background prompts (planner, executor, exploratory, reflector) ship as
plain-text files in `prompts/`. They can be overridden per host or per user
without editing code:

1. `prompts/<name>.txt` - bundled default (in repo)
2. `/etc/subconscious/prompts/<name>.txt` - system override
3. `~/.config/subconscious/prompts/<name>.txt` - user override
4. `SUBCONSCIOUS_BG_PROMPT_<NAME>` env var - runtime override

Later sources win. Four names: `planner`, `executor`, `exploratory`,
`reflector`.

## Backends

Routing is by model prefix:

| Prefix | Backend | Notes |
|---|---|---|
| *(none)* | `OllamaBackend` | local Ollama over HTTP |
| `or/` | `OpenRouterBackend` | OpenAI-compatible cloud API |
| `deepseek/` | `DeepSeekBackend` | DeepSeek API |
| `onnx/` | `OnnxBackend` | ONNX encoder - embed only |

Add a backend by implementing `BackendProtocol` (`subconscious/thalamus/backends/base.py`).

## Related projects

Part of [`subconscious-labs`](https://github.com/subconscious-labs):

- [gonnx](https://github.com/subconscious-labs/gonnx) - ONNX runtime with Git-based model bundles
- [lazy-mcp](https://github.com/subconscious-labs/lazy-mcp) - lazy MCP proxy with meta-tools
- [tts-api](https://github.com/subconscious-labs/tts-api) - multi-engine TTS server (REST + MCP)
- [dav-mcp](https://github.com/subconscious-labs/dav-mcp) - CalDAV/CardDAV MCP server
- [time-mcp](https://github.com/subconscious-labs/time-mcp) - time/date MCP server

## Status

Early. The core works, the docs describe where it's going. See
[docs/GOALS.md](docs/GOALS.md) for the roadmap.

## License

MIT

# chatd

A minimal bridge that connects [Ollama](https://ollama.com) to MCP tool servers and a long-term memory system ([MemPalace](https://github.com/MemPalace/mempalace)).

Sits between a static frontend (e.g. [ollama-gui](https://github.com/ollama-webui/ollama-webui)) and Ollama, adding:

- **MCP tool dispatch** - monitor, alerts, notes, memory
- **Agentic tool loop** - up to 5 rounds of tool calls per request
- **Layered memory** - MemPalace wake-up + RAG injected into every system prompt
- **Multi-backend** - Ollama (local) or OpenRouter (cloud) per request, selected by model name prefix
- **Streaming** - NDJSON passthrough with keepalive chunks to prevent 499s

## Architecture

```
browser
  └── nginx (chat.example.com)
        ├── /          → ollama-gui  (static frontend, :8080)
        ├── /api/      → chatd       (:5001, this repo)
        │     ├── MCP tool loop
        │     ├── MemPalace (long-term memory, L0/L1/L2/L3)
        │     └── RAG store (L1.5, chatd-owned SQLite)
        └── /api/ollama/ → Ollama    (:11434)
```

## Memory stack

Every request assembles a system prompt from layered memory sources:

| Layer | Owner | Content | When |
|-------|-------|---------|------|
| L0 | mempalace | `identity.txt` — persona, rules | Always |
| L1 | mempalace | Essential Story (wake-up context) | Always |
| L1.5 | chatd | KG recall (entity facts) + RAG sidecar (semantic chunks) | Always |
| L2 | mempalace | Room Recall (filtered retrieval) | When topic comes up |
| L3 | mempalace | Deep Search (full semantic query) | When explicitly asked |

`mempalace` layers (L0, L1, L2, L3) are managed by the mempalace library.  
`chatd` layer (L1.5) is a purely additive overlays owned by chatd.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally on `:11434`
- For RAG (L1.5): `ollama pull nomic-embed-text`

## Setup

```sh
git clone https://github.com/nikita-popov/chatd
cd chatd

python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

Copy and edit the env file:

```sh
cp .env.example .env
$EDITOR .env
```

Key variables to set:

```sh
CHATD_MCP_MEMPALACE=/opt/chatd/venv/bin/python -m mempalace.mcp_server
MEMPALACE_PALACE_PATH=/var/lib/mempalace
```

## Running

```sh
cp chatd.service /etc/systemd/system/
systemctl enable --now chatd.service
```

The service binds to `0.0.0.0:5001` via gunicorn (1 worker, 4 threads).

**nginx:** see `example.nginx.conf`. Key points:

- `gzip off` on `/api/` — nginx buffers gzip and breaks streaming
- `proxy_buffering off` on `/api/`
- `proxy_set_header Connection ""` for HTTP/1.1 keepalive upstream

## API

### `POST /api/chat`

Ollama-compatible chat endpoint. Accepts the same payload as Ollama's `POST /api/chat`.

```json
{
  "model": "qwen3:8b",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true
}
```

To route a request to OpenRouter instead of Ollama, prefix the model name with `or/`:

```json
{ "model": "or/mistralai/mistral-7b-instruct:free", ... }
```

Returns NDJSON stream. The system prompt is assembled automatically from the full memory stack.

### `GET /api/tags`

Proxied directly to Ollama. Used by the frontend to list available models.

### `GET /api/health`

Returns `{"ok": true}`.

## Identity

The assistant identity and tool usage rules live in `identity.txt`.
Edit it to change the persona, language, or memory behaviour.
The file is loaded by MemPalace `wake-up` as L0 context on every request.

## Background thinking

`chatd` can run a background worker ("subconscious") on a timer. It reads
context and — when enabled — maintains its own goal ledger. It never writes
to canonical memory.

### State layout

Everything lives in `$CHATD_BG_STATE_DIR` (default `~/.local/share/chatd/bg`):

- `goals.json`     — goal ledger (atomic rewrite via tmp+rename)
- `budget.json`    — daily token and time counters
- `journal.jsonl`  — append-only tick log
- `tick.lock`      — flock; overlapping ticks are skipped

### Phases

Each tick runs up to three phases. Models are routed independently:

| Phase     | Model env                 | Tools | Purpose |
|-----------|---------------------------|-------|---------|
| Planner   | `CHATD_BG_MODEL_PLANNER`  | no    | pick existing goal, propose new, or skip |
| Executor  | `CHATD_BG_MODEL_EXECUTOR` | yes   | run the goal (tool loop) |
| Reflector | `CHATD_BG_MODEL_REFLECTOR`| no    | on blocked goal: retry / cancel / keep |

Empty per-phase env falls back to `CHATD_BG_MODEL`.

### Goal ledger

Goals have `status` ∈ {pending, active, done, blocked, cancelled} and
`kind` ∈ {normal, exploratory}. `blocked` and closed goals do not count
toward the open-goal cap (`CHATD_BG_MAX_OPEN_GOALS`). When the cap is
reached, Planner cannot create new goals and Executor drops extra
`new_goals` from its output.

### Exploratory ticks

When Planner returns `skip`, Executor performs a read-only exploratory tick:

- local executor model only
- tools limited to the read-only subset (`*_search`, `*_query`, `*_status`,
  `*_list`, `*_summary`, `*_timeline`, `*_wake_up`)
- one active exploratory goal at a time
- auto-cancelled after `CHATD_BG_EXPLORATORY_TTL_HOURS`
- hard cap `CHATD_BG_EXPLORATORY_PER_DAY` per day

### Timeouts and budget

- Per-phase timeout: `CHATD_BG_PHASE_TIMEOUT` (planner, reflector),
  `CHATD_BG_PHASE_TIMEOUT_EXECUTOR` (executor).
- Whole-tick timeout: `CHATD_BG_TICK_TIMEOUT`.
- Daily budget: `CHATD_BG_DAILY_TOKEN_BUDGET`, `CHATD_BG_DAILY_TIME_BUDGET_SEC`
  (0 = no limit). Token count relies on Ollama's `prompt_eval_count +
  eval_count`; OpenAI-compatible backends currently report 0.

### Enable and trigger

```sh
CHATD_BG_ENABLED=true
CHATD_BG_TOKEN=<random-secret>
CHATD_BG_MODEL=qwen3.5:9b
CHATD_BG_MODEL_PLANNER=qwen3.5:4b
CHATD_BG_MODEL_REFLECTOR=qwen3.5:4b
```

```sh
curl -X POST http://127.0.0.1:5001/api/tick \
  -H "Authorization: Bearer $CHATD_BG_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}'
```

When `CHATD_BG_ENABLED=false` (default), the endpoint returns `404`.
When `CHATD_BG_TOKEN` is set and the bearer doesn't match, `401`.

## Backends

Inference routing is handled by `backends/`:

| Prefix | Backend | Notes |
|--------|---------|-------|
| *(none)* | `OllamaBackend` | Default — local Ollama over HTTP |
| `or/` | `OpenRouterBackend` | Cloud — OpenAI-compatible API |
| `onnx/` | `OnnxBackend` | Local ONNX encoder — embed-only, not for chat |
| `deepseek/` | `DeepSeekBackend` | Cloud — DeepSeek API, OpenAI-compatible |

Add a new backend by implementing `BackendProtocol` (`backends/base.py`) and registering it in `backends/__init__.py`.

## RAG / L1.5

The L1.5 RAG store is a chatd-owned SQLite database, completely separate from mempalace.  
Conversation turns are embedded after each request and retrieved by cosine similarity for the next.

**Enable:**

```sh
# Pull the embed model (once)
ollama pull nomic-embed-text

# Add to .env
CHATD_RAG_DB_PATH=~/.local/share/chatd/rag.sqlite3
CHATD_RAG_EMBED_MODEL=nomic-embed-text
```

**Smoke-test:**

```sh
source venv/bin/activate
python - <<'EOF'
import rag
rag.init()
rag.index_turn("test", "My favourite editor is emacs", "Got it, you use emacs.")
result = rag.retrieve("what editor do I use?")
print(result)
assert result and "emacs" in result.lower()
print("OK")
EOF
```

All `CHATD_RAG_*` variables are documented in `.env.example`.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_API` | `http://127.0.0.1:11434` | Ollama endpoint |
| `CHATD_THINKING` | `false` | Enable model `<think>` extended thinking |
| `CHATD_TEMPERATURE` | `0.3` | Sampling temperature |
| `CHATD_NUM_CTX` | `8192` | Context window size |
| `CHATD_NUM_PREDICT` | `768` | Max tokens per response |
| `CHATD_RAG_DB_PATH` | `~/.local/share/chatd/rag.sqlite3` | L1.5 RAG store |
| `CHATD_RAG_EMBED_MODEL` | `nomic-embed-text` | Embed model for RAG |
| `CHATD_RAG_TOP_K` | `4` | Chunks returned per retrieval |
| `CHATD_RAG_MIN_SCORE` | `0.25` | Minimum cosine similarity |
| `MEMPALACE_PALACE_PATH` | `~/.local/share/mempalace` | Path to the palace directory |
| `MEMPALACE_KG_PATH` | `~/.mempalace/knowledge_graph.sqlite3` | Path to knowledge graph |
| `OPENROUTER_API_KEY` | *(unset)* | OpenRouter API key |
| `OPENROUTER_PREFIX` | `or/` | Model name prefix for OpenRouter routing |
| `DEEPSEEK_API_KEY` | *(unset)* | DeepSeek API key |
| `CHATD_DEEPSEEK_API` | `https://api.deepseek.com` | DeepSeek API base URL |
| `CHATD_DEEPSEEK_MODELS` | `deepseek-flash` | Comma-separated DeepSeek models to expose in `/api/tags` |
| `CHATD_BG_ENABLED` | `false` | Enable background worker (`/api/tick`) |
| `CHATD_BG_TOKEN` | *(unset)* | Bearer token for `/api/tick` |
| `CHATD_BG_STATE_DIR` | `~/.local/share/chatd/bg` | Background worker state |
| `CHATD_BG_SESSION_ID` | `background` | Isolated session id |
| `CHATD_BG_MODEL` | `qwen3:8b` | Model for background worker |
| `CHATD_BG_NUM_PREDICT` | `256` | Max output tokens per background tick |
| `CHATD_BG_MAX_TOOL_ROUNDS` | `5` | Max tool rounds per tick (reserved) |
| `CHATD_BG_TOOLS_ALLOWED` | *(read-only list)* | Tools visible to background |

Full list with comments: `.env.example`.

## License

MIT

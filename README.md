# chatd

A minimal bridge that connects [Ollama](https://ollama.com) to MCP tool servers and a long-term memory system ([MemPalace](https://github.com/MemPalace/mempalace)).

Sits between a static frontend (e.g. [ollama-gui](https://github.com/ollama-webui/ollama-webui)) and Ollama, adding:

- **MCP tool dispatch** — monitor, alerts, notes, memory
- **Agentic tool loop** — up to 5 rounds of tool calls per request
- **Layered memory** — MemPalace wake-up + rolling summaries + RAG injected into every system prompt
- **Multi-backend** — Ollama (local) or OpenRouter (cloud) per request, selected by model name prefix
- **Streaming** — NDJSON passthrough with keepalive chunks to prevent 499s

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
| L0.5 | chatd | Per-chat compressed summary + global summary | Always |
| L1 | mempalace | Essential Story (wake-up context) | Always |
| L1.5 | chatd | KG recall (entity facts) + RAG sidecar (semantic chunks) | Always |
| L2 | mempalace | Room Recall (filtered retrieval) | When topic comes up |
| L3 | mempalace | Deep Search (full semantic query) | When explicitly asked |

mempalace layers (L0, L1, L2, L3) are managed by the mempalace library.  
chatd layers (L0.5, L1.5) are purely additive overlays owned by chatd.

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

To route a request to OpenRouter instead of Ollama, prefix the model name with `or:`:

```json
{ "model": "or:mistralai/mistral-7b-instruct:free", ... }
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

## Backends

Inference routing is handled by `backends/`:

| Prefix | Backend | Notes |
|--------|---------|-------|
| *(none)* | `OllamaBackend` | Default — local Ollama over HTTP |
| `or:` | `OpenRouterBackend` | Cloud — OpenAI-compatible API |
| `onnx:` | `OnnxBackend` | Local ONNX encoder — embed-only, not for chat |

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
CHATD_RAG_EMBED_DIM=768
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
| `CHATD_SUMMARY_MODEL` | `qwen2.5:1.5b` | Model used for summary compression |
| `CHATD_COMPRESS_EVERY` | `5` | Turns between compressions |
| `CHATD_GLOBAL_SUMMARY_PATH` | `~/.local/share/chatd/global_summary.txt` | L0.5 global summary |
| `CHATD_RAG_DB_PATH` | `~/.local/share/chatd/rag.sqlite3` | L1.5 RAG store |
| `CHATD_RAG_EMBED_MODEL` | `nomic-embed-text` | Embed model for RAG |
| `CHATD_RAG_EMBED_DIM` | `768` | Embed vector dimension |
| `CHATD_RAG_TOP_K` | `4` | Chunks returned per retrieval |
| `CHATD_RAG_MIN_SCORE` | `0.25` | Minimum cosine similarity |
| `MEMPALACE_PALACE_PATH` | `~/.local/share/mempalace` | Path to the palace directory |
| `MEMPALACE_KG_PATH` | `~/.mempalace/knowledge_graph.sqlite3` | Path to knowledge graph |
| `OPENROUTER_API_KEY` | *(unset)* | OpenRouter API key |
| `OPENROUTER_PREFIX` | `or:` | Model name prefix for OpenRouter routing |

Full list with comments: `.env.example`.

## License

MIT

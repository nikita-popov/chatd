# chatd

A minimal bridge that connects [Ollama](https://ollama.com) to MCP tool servers and a long-term memory system ([MemPalace](https://github.com/MemPalace/mempalace)).

Sits between a static frontend (e.g. [ollama-gui](https://github.com/ollama-webui/ollama-webui)) and Ollama, adding:

- **MCP tool dispatch** - monitor, alerts, notes, memory
- **Agentic tool loop** - up to 5 rounds of tool calls per request
- **Persistent memory** - MemPalace `wake-up` context injected into every system prompt
- **Streaming** - NDJSON passthrough with keepalive chunks to prevent 499s

## Architecture

```
browser
  └── nginx (chat.example.com)
        ├── /          → ollama-gui  (static frontend, :8080)
        ├── /api/      → chatd       (:5001, this repo)
        │     ├── MCP
        │     └── MemPalace (long-term memory)
        └── /api/ollama/ → Ollama    (:11434)
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally on `:11434`

## Setup

```sh
git clone https://github.com/nikita-popov/chatd
cd chatd

python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

Edit `ENV` to point MCP commands:

```plain
CHATD_MCP_MONITOR=/path/to/venv/bin/python /path/to/mcp-monitor.py
```

And set environment variables required by the MCP servers.

## Running

```sh
cp chatd.service /etc/systemd/system/
systemctl enable --now chatd.service
```

The service binds to `0.0.0.0:5001` via gunicorn (1 worker, 4 threads).

**nginx:**

See `example.nginx.conf`. Key points:

- `gzip off` on `/api/` - mandatory, nginx buffers gzip and breaks streaming
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

Returns NDJSON stream. The system prompt is injected automatically from
MemPalace `wake-up` output (L0 + L1 context).

### `GET /api/tags`

Proxied directly to Ollama. Used by the frontend to list available models.

### `GET /api/health`

Returns `{"ok": true}`.

## Identity

The assistant identity and tool usage rules live in `identity.txt`.
Edit it to change the persona, language, or memory behaviour.
The file is loaded by MemPalace `wake-up` as L0 context on every request.

## Configuration

Key constants in `ENV`:

| Constant | Default | Description |
|----------|---------|-------------|
| `OLLAMA_API` | `http://127.0.0.1:11434` | Ollama endpoint |
| `CHATD_THINKING` | `False` | Enable model extended thinking (`<think>`) |
| `CHATD_TOOLS_FILTER` | `False` | Enable filtering of tools available to the model |
| `CHATD_TOOLS_ALLOWED` | see code | Tools visible to the model |
| `CHATD_TOOL_OVERRIDE` | `False` | Enable overwriting tool descriptions |
| `CHATD_TOOL_DESCRIPTIONS` | see code | Per-tool descriptions sent to the model |
| `MEMPALACE_PALACE_PATH` | `~/.local/share/mempalace` | Path to the palace directory |
| `MEMPALACE_KG_PATH` | `~/.mempalace/knowledge_graph.sqlite3` | Path to knowledge graph |

## License

MIT

# chatd

A minimal chat backend that connects [Ollama](https://ollama.com) to MCP tool servers and a long-term memory system ([MemPalace](https://github.com/MemPalace/mempalace)).

Sits between a static frontend (e.g. [ollama-gui](https://github.com/ollama-webui/ollama-webui)) and Ollama, adding:

- **MCP tool dispatch** — monitor, alerts, notes, memory
- **Agentic tool loop** — up to 5 rounds of tool calls per request
- **Persistent memory** — MemPalace `wake-up` context injected into every system prompt
- **Streaming** — NDJSON passthrough with keepalive chunks to prevent 499s

## Architecture

```
browser
  └── nginx (chat.example.com)
        ├── /          → ollama-gui  (static frontend, :8080)
        ├── /api/      → chatd       (:5001, this repo)
        │     ├── MCP: mcp-monitor.py   (VictoriaMetrics)
        │     ├── MCP: mcp-alerts.py    (Alertmanager)
        │     ├── MCP: mcp-notes.py     (Memos)
        │     └── MCP: mempalace.mcp_server (long-term memory)
        └── /api/ollama/ → Ollama    (:11434)
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally on `:11434`
- [MemPalace](https://github.com/MemPalace/mempalace) installed and initialised
- Memos instance (for notes MCP, optional)
- VictoriaMetrics + Alertmanager (for monitoring MCPs, optional)

## Setup

```sh
git clone https://github.com/nikita-popov/chatd /opt/chatd
cd /opt/chatd

python3 -m venv venv
. venv/bin/activate
pip install flask gunicorn requests mcp mempalace

# Initialise MemPalace palace
MEMPALACE_PALACE_PATH=/var/lib/mempalace mempalace init
```

Edit `mcp_client.py` to point MCP commands at your venv and scripts:

```python
MCP_ALERTS_CMD    = ["/opt/chatd/venv/bin/python", "/opt/chatd/mcp-alerts.py"]
MCP_MONITOR_CMD   = ["/opt/chatd/venv/bin/python", "/opt/chatd/mcp-monitor.py"]
MCP_NOTES_CMD     = ["/opt/chatd/venv/bin/python", "/opt/chatd/mcp-notes.py"]
MCP_MEMPALACE_CMD = ["/opt/chatd/venv/bin/python", "-m", "mempalace.mcp_server"]
```

Set environment variables required by the MCP servers (`MEMOS_URL`, `MEMOS_TOKEN`,
`ALERTMANAGER_URL`, `VICTORIA_URL`) in the systemd unit or shell environment.

## Running

**Development:**

```sh
. venv/bin/activate
MEMPALACE_PALACE_PATH=/var/lib/mempalace python chatd.py
```

**Production (systemd):**

```sh
cp chatd.service /etc/systemd/system/
systemctl enable --now chatd
```

The service binds to `0.0.0.0:5001` via gunicorn (1 worker, 4 threads).

**nginx:**

See `example.nginx.conf`. Key points:

- `gzip off` on `/api/` — mandatory, nginx buffers gzip and breaks streaming
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

## MCP Tools

Tools exposed to the model (filtered subset of what each MCP server provides):

| Tool | Server | When called |
|------|--------|-------------|
| `monitor_query` | mcp-monitor.py | User asks for metrics (CPU, RAM, uptime) |
| `alerts_list` | mcp-alerts.py | User asks about firing alerts |
| `alerts_summary` | mcp-alerts.py | User asks for alert overview |
| `notes_search` | mcp-notes.py | User asks to find notes |
| `mempalace_status` | mempalace | Once per session — loads memory protocol |
| `mempalace_search` | mempalace | Semantic search in long-term memory |
| `mempalace_add_drawer` | mempalace | Filing new facts into ChromaDB |
| `mempalace_kg_query` | mempalace | Structured fact lookup (fast) |
| `mempalace_kg_add` | mempalace | Adding triples on explicit user request |

Tool arguments for `mempalace_kg_add` are sanitised server-side:
Cyrillic is transliterated to Latin, spaces replaced with underscores.

## Identity

The assistant identity and tool usage rules live in `identity.txt`.
Edit it to change the persona, language, or memory behaviour.
The file is loaded by MemPalace `wake-up` as L0 context on every request.

## Configuration

Key constants in `chatd.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `THINKING` | `False` | Enable Qwen3 extended thinking (`<think>`) |
| `DEFAULT_OPTIONS` | see code | Ollama sampling params (temp, top_k, etc.) |
| `MEMPALACE_ALLOWED_TOOLS` | see code | MemPalace tools visible to the model |
| `MEMPALACE_WRITE_TOOLS` | see code | Tools that trigger wake-up cache invalidation |
| `TOOL_DESCRIPTION_OVERRIDES` | see code | Per-tool descriptions sent to the model |

## License

MIT

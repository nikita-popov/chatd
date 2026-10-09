# subconscious — архитектура

## Четыре зоны

Один Python-пакет `subconscious`, четыре подпакета. Каждый — одна
ответственность.

### core/ — данные и состояние

Что хранится, где, в каком формате. Не логика.

- `memory/` — KG (SQLite), vector store (sqlite-vec), identity, layers.
- `rag.py` — RAG-sidecar (L1.5).
- `session.py` — сессии чата (JSONL chatlog + JSON markers).
- `ledger.py` — goals.json + goals.archive.jsonl + journal.jsonl.
- `budget.py` — budget.json + new_goals_counter.json.
- `config.py` — layered конфиг (env > file > default).

### cortex/ — интерфейсы

Кто разговаривает с внешним миром. Всё, что видит пользователь или
другой агент.

- `api.py` — Flask app, `/api/chat`, `/api/tick`, `/api/event`.
- `a2a.py` — Agent Card, Task mapping (PR D).
- `cli.py` — команды `subconscious init`, `subconscious tick`, `subconscious status`.
- `event.py` — обработчик входящих событий.

**Чат — это cortex, не ядро.** Чат использует core, thalamus, subcortex,
но сам по себе не суть проекта.

### thalamus/ — связки

Роутер между нами и внешним миром.

- `backends/` — Ollama, OpenRouter, DeepSeek, ONNX.
- `mcp/client.py` — stdio-клиент к MCP-серверам.
- `mcp/router.py` — namespacing, фильтры, категории инструментов.
- `mcp/config.py` — загрузчик `mcp_servers.yaml`.
- `tools.py` — фильтр «что видит chat / executor / exploratory».

### subcortex/ — фоновое мышление

Главная фича. То, чего нет в других chatd-подобных проектах.

- `tick.py` — orchestration одного тика.
- `planner.py` — выбор цели / создание / skip.
- `executor.py` — tool loop выполнения цели.
- `reflector.py` — разбор blocked-целей.
- `assessor.py` — оценка done-целей (PR D).
- `prompts/` — planner.txt, executor.txt, exploratory.txt, reflector.txt.

## Поток данных

```
systemd timer
  → HTTP POST /api/tick
    → cortex/api.py
      → subcortex/tick.py
        → subcortex/planner.py  → core/ledger, core/memory, thalamus/backends
        → subcortex/executor.py → thalamus/mcp, thalamus/backends
        → subcortex/reflector.py → thalamus/backends
        → subcortex/assessor.py → thalamus/backends (PR D)
        → core/ledger (persist)
        → core/budget (accounting)
```

```
пользователь
  → HTTP POST /api/chat
    → cortex/api.py
      → core/memory.wake_up()
      → core/rag.retrieve()
      → thalamus/backends (streaming)
      → thalamus/mcp (tool calls)
      → core/session (persist)
```

## Зависимости (направление)

```
cortex    →  core, subcortex, thalamus
subcortex →  core, thalamus
thalamus  →  core (только конфиг)
core      →  ничего внутри проекта
```

**Никаких обратных зависимостей.** core не знает про subcortex. thalamus
не знает про cortex. Если появляется обратная — это ошибка дизайна.

## State

Всё мутабельное — в `$SUBCONSCIOUS_STATE_DIR` (по умолчанию
`~/.local/share/subconscious/`):

- `bg/` — цели, journal, budget, seed_rotation, new_goals_counter.
- `sessions/` — JSONL чатлоги.
- `rag.sqlite3` — RAG-индекс.
- `kg.sqlite3` — свой KG (PR E).
- `vectors.sqlite3` — свой vector store (PR E).

**Никогда** в `/opt/<svc>/`, никогда в home сервиса.
```

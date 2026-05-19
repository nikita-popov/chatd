# TODO

- add ring buffer for turns
- add all ollama routes
- context window strategy: sliding window or summarize-and-truncate for long sessions
  (L0+L1 already cached per session; L1.5 is per-request by design)
- /api/event has no session — wake-up is not cached there, consider a shared singleton

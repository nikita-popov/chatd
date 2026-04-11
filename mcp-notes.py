#!/usr/bin/env python3
import os
import json
import httpx
from mcp.server.fastmcp import FastMCP

#MEMOS_URL = os.environ.get("MEMOS_URL", "https://memos.example.com")
#MEMOS_TOKEN = os.environ.get("MEMOS_TOKEN", "")
MEMOS_URL = "https://notes.polyserv.xyz"
MEMOS_TOKEN = "eyJhbGciOiJIUzI1NiIsImtpZCI6InYxIiwidHlwIjoiSldUIn0.eyJuYW1lIjoicG9saXN0ZXJuIiwiaXNzIjoibWVtb3MiLCJzdWIiOiIxIiwiYXVkIjpbInVzZXIuYWNjZXNzLXRva2VuIl0sImlhdCI6MTc3NTkxNjI4Nn0.YeQyYhkUCb0YiZzI7J6Kwl3-zWTsDdoT39fVB7yPI9Q"

mcp = FastMCP("mcp-notes")


def memos_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if MEMOS_TOKEN:
        headers["Authorization"] = f"Bearer {MEMOS_TOKEN}"
    return headers


@mcp.tool()
async def notes_search(query: str, limit: int = 10) -> str:
    if not query.strip():
        raise ValueError("query is required")

    url = f"{MEMOS_URL.rstrip('/')}/api/v1/memos"
    params = {
        "content": query,
        "limit": limit,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params, headers=memos_headers())
        r.raise_for_status()
        data = r.json()

    # Пытаемся вытащить список заметок из нескольких возможных форматов
    if isinstance(data, list):
        notes_raw = data
    elif isinstance(data, dict):
        if isinstance(data.get("memos"), list):
            notes_raw = data["memos"]
        elif isinstance(data.get("data"), list):
            notes_raw = data["data"]
        elif isinstance(data.get("data"), dict) and isinstance(data["data"].get("memos"), list):
            notes_raw = data["data"]["memos"]
        else:
            return json.dumps({
                "notes": [],
                "raw": data,
                "error": "unsupported memos response format"
            }, ensure_ascii=False)
    else:
        return json.dumps({
            "notes": [],
            "raw": str(data),
            "error": "unexpected memos response type"
        }, ensure_ascii=False)

    norm = []
    for n in notes_raw[:limit]:
        if not isinstance(n, dict):
            continue
        content = n.get("content") or ""
        title = n.get("title") or ""
        norm.append({
            "id": n.get("id") or n.get("name"),
            "title": title,
            "excerpt": content[:200],
            "created_at": n.get("createTime") or n.get("createdTs") or n.get("created_at"),
        })

    return json.dumps({"notes": norm}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")

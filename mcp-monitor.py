#!/usr/bin/env python3
import os
import json
import httpx
from mcp.server.fastmcp import FastMCP

VM_URL = os.environ.get("VICTORIA_URL", "http://10.54.1.1:8428")

mcp = FastMCP("mcp-monitor")


@mcp.tool()
async def monitor_query(query: str, time: float | None = None) -> str:
    params = {"query": query}
    if time is not None:
        params["time"] = str(time)

    url = f"{VM_URL}/prometheus/api/v1/query"

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    norm = {
        "status": data.get("status"),
        "result_type": data.get("data", {}).get("resultType"),
        "series": [],
    }

    for item in data.get("data", {}).get("result", []):
        metric = item.get("metric", {})
        value = item.get("value") or []
        norm["series"].append({
            "labels": metric,
            "value": value[1] if len(value) > 1 else None,
            "timestamp": float(value[0]) if value else None,
        })

    return json.dumps(norm, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")

#!/usr/bin/env python3
import json
import os
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

ALERTMANAGER_URL = os.environ.get("ALERTMANAGER_URL", "http://10.54.1.1:9093")

mcp = FastMCP("mcp-alerts")


def _alertmanager_url(path: str) -> str:
    return f"{ALERTMANAGER_URL.rstrip('/')}{path}"


def _match_alert(alert: dict, matcher: Optional[str]) -> bool:
    if not matcher:
        return True

    matcher = matcher.strip()
    if not matcher:
        return True

    labels = alert.get("labels") or {}
    annotations = alert.get("annotations") or {}

    if "=" in matcher:
        key, value = matcher.split("=", 1)
        key = key.strip()
        value = value.strip()
        return str(labels.get(key, "")) == value

    haystack = []
    for k, v in labels.items():
        haystack.append(f"{k}={v}")
    for k, v in annotations.items():
        haystack.append(f"{k}={v}")

    text = "\n".join(haystack)
    return matcher in text


@mcp.tool()
async def alerts_list(
    active: bool = True,
    silenced: bool = True,
    inhibited: bool = True,
    unprocessed: bool = True,
    matcher: str = "",
    limit: int = 20,
) -> str:
    params = {
        "active": str(active).lower(),
        "silenced": str(silenced).lower(),
        "inhibited": str(inhibited).lower(),
        "unprocessed": str(unprocessed).lower(),
    }

    url = _alertmanager_url("/api/v2/alerts")

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    if not isinstance(data, list):
        return json.dumps(
            {
                "alerts": [],
                "error": "unexpected alertmanager response type",
                "raw": data,
            },
            ensure_ascii=False,
        )

    filtered = []
    for a in data:
        if not isinstance(a, dict):
            continue
        if not _match_alert(a, matcher):
            continue

        labels = a.get("labels") or {}
        annotations = a.get("annotations") or {}
        status = a.get("status") or {}

        filtered.append(
            {
                "fingerprint": a.get("fingerprint"),
                "status": {
                    "state": status.get("state"),
                    "silencedBy": status.get("silencedBy") or [],
                    "inhibitedBy": status.get("inhibitedBy") or [],
                },
                "labels": labels,
                "annotations": annotations,
                "startsAt": a.get("startsAt"),
                "endsAt": a.get("endsAt"),
                "updatedAt": a.get("updatedAt"),
                "generatorURL": a.get("generatorURL"),
            }
        )

    filtered = filtered[: max(0, limit)]

    return json.dumps(
        {
            "count": len(filtered),
            "alerts": filtered,
        },
        ensure_ascii=False,
    )


@mcp.tool()
async def alerts_summary(
    active: bool = True,
    silenced: bool = True,
    inhibited: bool = True,
    unprocessed: bool = True,
) -> str:
    params = {
        "active": str(active).lower(),
        "silenced": str(silenced).lower(),
        "inhibited": str(inhibited).lower(),
        "unprocessed": str(unprocessed).lower(),
    }

    url = _alertmanager_url("/api/v2/alerts")

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    if not isinstance(data, list):
        return json.dumps(
            {
                "error": "unexpected alertmanager response type",
                "raw": data,
            },
            ensure_ascii=False,
        )

    by_alertname = {}
    by_severity = {}
    by_state = {}

    for a in data:
        if not isinstance(a, dict):
            continue

        labels = a.get("labels") or {}
        status = a.get("status") or {}

        alertname = labels.get("alertname", "unknown")
        severity = labels.get("severity", "unknown")
        state = status.get("state", "unknown")

        by_alertname[alertname] = by_alertname.get(alertname, 0) + 1
        by_severity[severity] = by_severity.get(severity, 0) + 1
        by_state[state] = by_state.get(state, 0) + 1

    return json.dumps(
        {
            "count": len(data),
            "by_alertname": dict(sorted(by_alertname.items())),
            "by_severity": dict(sorted(by_severity.items())),
            "by_state": dict(sorted(by_state.items())),
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")

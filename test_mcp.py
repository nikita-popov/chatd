#!/usr/bin/env python3
import json
from mcp_client import MCPClient, MCP_ALERTS_CMD, MCP_MONITOR_CMD, MCP_NOTES_CMD


def tool_to_dict(tool):
    return {
        "name": getattr(tool, "name", None),
        "description": getattr(tool, "description", None),
        "inputSchema": getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None),
    }


def test_client(name, cmd):
    print(f"=== {name} ===")
    client = MCPClient(cmd)

    tools = client.list_tools()
    print("tools:")
    print(json.dumps([tool_to_dict(t) for t in tools], ensure_ascii=False, indent=2))

    if tools:
        t0 = tools[0]
        tname = getattr(t0, "name", None) or t0["name"]
        print(f"\ncalling tool: {tname}")

        if tname == "notes_search":
            args = {"query": "ollama", "limit": 3}
        elif tname == "monitor_query":
            args = {"query": "up"}
        else:
            args = {}

        res = client.call_tool(tname, args)
        print("result:")
        print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_client("alerts", MCP_ALERTS_CMD)
    print()
    test_client("monitor", MCP_MONITOR_CMD)
    print()
    test_client("notes", MCP_NOTES_CMD)

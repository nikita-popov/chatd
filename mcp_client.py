#!/usr/bin/env python3
import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MCP_ALERTS_CMD = ["/opt/chatd/venv/bin/python", "/opt/chatd/mcp-alerts.py"]
MCP_MONITOR_CMD = ["/opt/chatd/venv/bin/python", "/opt/chatd/mcp-monitor.py"]
MCP_NOTES_CMD = ["/opt/chatd/venv/bin/python", "/opt/chatd/mcp-notes.py"]


class MCPClient:
    def __init__(self, cmd: list[str]):
        self.cmd = cmd

    async def _with_session(self, fn):
        command = self.cmd[0]
        args = self.cmd[1:]

        async with AsyncExitStack() as stack:
            read, write = await stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=command,
                        args=args,
                        env=None,
                    )
                )
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            return await fn(session)

    def start(self):
        return None

    def stop(self):
        return None

    def list_tools(self):
        async def run(session):
            result = await session.list_tools()
            return result.tools
        return asyncio.run(self._with_session(run))

    def call_tool(self, name: str, arguments: dict[str, Any]):
        async def run(session):
            result = await session.call_tool(name, arguments=arguments)
            return result

        result = asyncio.run(self._with_session(run))
        contents = getattr(result, "content", None) or []
        if not contents:
            return None

        first = contents[0]
        text = getattr(first, "text", None)
        if text is None and isinstance(first, dict):
            text = first.get("text")

        if text is None:
            return str(first)

        try:
            return json.loads(text)
        except Exception:
            return text

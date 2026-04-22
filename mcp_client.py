#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import shlex
import signal
import subprocess
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import MCP_ENV_PREFIX

log = logging.getLogger("chatd.mcp_client")

# Seconds to wait for MCP server to respond.
# list_tools is called once at startup - generous budget.
# call_tool is called inline during streaming - shorter budget.
MCP_LIST_TOOLS_TIMEOUT: float = float(os.environ.get("CHATD_MCP_LIST_TOOLS_TIMEOUT", "30"))
MCP_CALL_TOOL_TIMEOUT:  float = float(os.environ.get("CHATD_MCP_CALL_TOOL_TIMEOUT",  "60"))

# ---------------------------------------------------------------------------
# MCP server auto-discovery
#
# Each CHATD_MCP_<NAME> environment variable registers one MCP server.
# The value is a shell command string, e.g.:
#
#   CHATD_MCP_MONITOR="/opt/chatd/venv/bin/python /opt/chatd/mcp-monitor.py"
#   CHATD_MCP_NOTES="/opt/chatd/venv/bin/python /opt/chatd/mcp-notes.py"
#   CHATD_MCP_MEMPALACE="/opt/chatd/venv/bin/python -m mempalace.mcp_server"
#
# Set these in .env or the systemd unit Environment= directives.
# ---------------------------------------------------------------------------


def discover_mcp_servers() -> dict[str, list[str]]:
    """Return {name: argv} for every CHATD_MCP_* variable in the environment."""
    servers: dict[str, list[str]] = {}
    for key, val in os.environ.items():
        if key.startswith(MCP_ENV_PREFIX) and val.strip():
            name = key[len(MCP_ENV_PREFIX):].lower()
            servers[name] = shlex.split(val)
    return servers


def _kill_process(cmd: list[str]) -> None:
    """Best-effort: find and SIGKILL child processes matching cmd.

    stdio_client does not expose the subprocess handle, so we use pgrep.
    Falls back silently on non-Linux systems where pgrep is unavailable.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", " ".join(cmd)],
            capture_output=True, text=True,
        )
        for pid_str in result.stdout.splitlines():
            try:
                pid = int(pid_str.strip())
                os.kill(pid, signal.SIGKILL)
                log.warning(
                    "[mcp] killed hung MCP process pid=%d cmd=%s",
                    pid, cmd[0],
                )
            except (ValueError, ProcessLookupError, PermissionError) as e:
                log.debug("[mcp] kill pid=%s failed: %s", pid_str.strip(), e)
    except FileNotFoundError:
        log.debug("[mcp] pgrep not available, cannot kill hung process")


class MCPClient:
    def __init__(self, cmd: list[str]):
        self.cmd = cmd

    async def _with_session(self, fn, timeout: float):
        """Run fn(session) inside a fresh stdio MCP session.

        Raises asyncio.TimeoutError if the whole operation
        (spawn + initialize + fn) exceeds *timeout* seconds.
        """
        command = self.cmd[0]
        args    = self.cmd[1:]

        async def _run():
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(
                    stdio_client(StdioServerParameters(command=command, args=args, env=os.environ.copy()))
                )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                return await fn(session)

        return await asyncio.wait_for(_run(), timeout=timeout)

    def start(self):  return None
    def stop(self):   return None

    def list_tools(self) -> list:
        async def run(session):
            result = await session.list_tools()
            return result.tools

        try:
            return asyncio.run(self._with_session(run, MCP_LIST_TOOLS_TIMEOUT))
        except asyncio.TimeoutError:
            log.error(
                "[mcp] list_tools timed out after %.0fs: %s",
                MCP_LIST_TOOLS_TIMEOUT, self.cmd,
            )
            _kill_process(self.cmd)
            raise RuntimeError(
                f"MCP list_tools timeout ({MCP_LIST_TOOLS_TIMEOUT:.0f}s): {self.cmd}"
            )

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        async def run(session):
            return await session.call_tool(name, arguments=arguments)

        try:
            result = asyncio.run(self._with_session(run, MCP_CALL_TOOL_TIMEOUT))
        except asyncio.TimeoutError:
            log.error(
                "[mcp] call_tool '%s' timed out after %.0fs: %s",
                name, MCP_CALL_TOOL_TIMEOUT, self.cmd,
            )
            _kill_process(self.cmd)
            # RuntimeError is caught by call_tool() in chatd.py and
            # returned as {"error": ...} dict so the model sees the timeout.
            raise RuntimeError(
                f"MCP call_tool timeout ({MCP_CALL_TOOL_TIMEOUT:.0f}s): {name}"
            )

        contents = getattr(result, "content", None) or []
        if not contents:
            return None

        first = contents[0]
        text  = getattr(first, "text", None)
        if text is None and isinstance(first, dict):
            text = first.get("text")
        if text is None:
            return str(first)

        try:
            return json.loads(text)
        except Exception:
            return text

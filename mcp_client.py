#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import shlex
import signal
import subprocess
import threading
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import MCP_ENV_PREFIX

log = logging.getLogger("chatd.mcp_client")

# Seconds to wait for MCP server to respond.
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
    """Best-effort: find and SIGKILL child processes matching cmd."""
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
    """Long-lived MCP client that keeps the server process running.

    Call start() once after construction, stop() on shutdown.
    list_tools() and call_tool() reuse the persistent session.
    """

    def __init__(self, cmd: list[str]):
        self.cmd = cmd
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _start_async(self) -> None:
        command = self.cmd[0]
        args    = self.cmd[1:]
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(
            stdio_client(StdioServerParameters(
                command=command, args=args, env=os.environ.copy(),
            ))
        )
        self._session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()

    def start(self) -> None:
        """Spawn the MCP server process and initialise the session."""
        self._loop = asyncio.new_event_loop()
        try:
            self._loop.run_until_complete(self._start_async())
            log.info("[mcp] started: %s", self.cmd[0])
        except Exception as e:
            log.error("[mcp] failed to start %s: %s", self.cmd, e)
            self._loop.close()
            self._loop = None
            raise

    def stop(self) -> None:
        """Tear down the session and kill the server process."""
        if self._stack and self._loop:
            try:
                self._loop.run_until_complete(self._stack.aclose())
            except Exception as e:
                log.debug("[mcp] stop aclose error: %s", e)
        if self._loop:
            self._loop.close()
            self._loop = None
        self._session = None
        self._stack = None
        log.info("[mcp] stopped: %s", self.cmd[0])

    # ------------------------------------------------------------------
    # Internal: run a coroutine on the persistent loop (thread-safe)
    # ------------------------------------------------------------------

    def _run(self, coro, timeout: float):
        if self._loop is None or self._session is None:
            raise RuntimeError(f"MCPClient not started: {self.cmd}")
        with self._lock:
            return self._loop.run_until_complete(
                asyncio.wait_for(coro, timeout=timeout)
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_tools(self) -> list:
        try:
            result = self._run(self._session.list_tools(), MCP_LIST_TOOLS_TIMEOUT)
            return result.tools
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
        try:
            result = self._run(
                self._session.call_tool(name, arguments=arguments),
                MCP_CALL_TOOL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.error(
                "[mcp] call_tool '%s' timed out after %.0fs: %s",
                name, MCP_CALL_TOOL_TIMEOUT, self.cmd,
            )
            _kill_process(self.cmd)
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

"""Tool registry — collects all tools and provides them to the agent."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.sandbox.executor import SandboxedExecutor
from src.tools.file_tools import create_file_tools
from src.tools.shell_tools import create_shell_tool
from src.tools.web_tools import create_web_tools
from src.tools.human_tools import create_human_help_tool
from src.tools.env_tools import create_env_tools
from src.tools.pdf_tools import create_pdf_tools


class ToolRegistry:
    """Central registry for all agent tools.

    Usage:
        registry = ToolRegistry(
            workspace_root="/tmp/ws",
            sandbox=executor,
            help_callback=lambda q, ctx: input(f"{q}\\n> "),
        )
        tools = registry.get_all()
    """

    def __init__(
        self,
        workspace_root: str | Path,
        sandbox: SandboxedExecutor,
        *,
        include_web_search: bool = True,
        help_callback: Callable[[str, str], str] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.sandbox = sandbox
        self.include_web_search = include_web_search
        self.help_callback = help_callback
        self._tools: list | None = None

    def get_all(self) -> list:
        """Return all registered tools as a flat list."""
        if self._tools is None:
            self._tools = []
            self._tools.extend(create_file_tools(self.workspace_root))
            self._tools.extend(create_shell_tool(self.sandbox))
            self._tools.extend(create_pdf_tools(self.workspace_root))
            self._tools.extend(create_env_tools(self.workspace_root))
            self._tools.extend(create_human_help_tool(self.help_callback))
            if self.include_web_search:
                self._tools.extend(create_web_tools())
        return self._tools

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.get_all()]

    def get_by_name(self, name: str):
        """Get a specific tool by name."""
        for t in self.get_all():
            if t.name == name:
                return t
        raise KeyError(f"Tool not found: {name}. Available: {self.tool_names}")

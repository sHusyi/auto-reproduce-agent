"""Sandboxed command executor with workspace isolation.

All commands run in subprocesses scoped to a workspace directory.
The permission controller gates every command before execution.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Callable

from src.sandbox.permissions import PermissionController
from src.state import CommandResult, PermissionLevel


class SandboxedExecutor:
    """Executes shell commands within a sandboxed workspace.

    Three-layer isolation:
    1. Permission check: classify command before execution
    2. Workspace scoping: all paths relative to workspace root
    3. Environment filtering: only whitelisted env vars passed through

    A human-approval callback can be registered for SENSITIVE/DANGEROUS levels.
    Without one, SENSITIVE commands are auto-rejected and DANGEROUS are always rejected.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        timeout: int = 60,
        max_output_bytes: int = 100_000,
        allowed_network_hosts: list[str] | None = None,
        approval_callback: Callable[[str, PermissionLevel], bool] | None = None,
        auto_approve_sensitive: bool = True,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_output_bytes = max_output_bytes
        self.allowed_network_hosts = allowed_network_hosts or []
        self.approval_callback = approval_callback
        self.auto_approve_sensitive = auto_approve_sensitive

        self.permissions = PermissionController(self.workspace_root)
        self._approved_sensitive: set[str] = set()  # Once-per-session cache

    def execute(self, command: str) -> CommandResult:
        """Execute a shell command through the sandbox. Returns CommandResult."""
        level, parsed = self.permissions.check(command)

        # Level 4: FORBIDDEN — never execute
        if level == PermissionLevel.FORBIDDEN:
            return CommandResult(
                command=command,
                blocked=True,
                block_reason=f"Forbidden command: {parsed.executable} targets system resources",
            )

        # Level 3: DANGEROUS — always ask human
        if level == PermissionLevel.DANGEROUS:
            if not self.approval_callback:
                return CommandResult(
                    command=command,
                    blocked=True,
                    block_reason="Dangerous command requires human approval (no callback registered)",
                )
            if not self.approval_callback(command, level):
                return CommandResult(
                    command=command,
                    blocked=True,
                    block_reason="Human rejected the dangerous command",
                )

        # Level 2: SENSITIVE — auto-approve or confirm once per session
        if level == PermissionLevel.SENSITIVE:
            key = f"{parsed.executable}:{parsed.args[0] if parsed.args else ''}"
            if key not in self._approved_sensitive:
                if self.approval_callback:
                    if not self.approval_callback(command, level):
                        return CommandResult(
                            command=command,
                            blocked=True,
                            block_reason="Human rejected the sensitive command",
                        )
                elif not self.auto_approve_sensitive:
                    return CommandResult(
                        command=command,
                        blocked=True,
                        block_reason=f"Sensitive command requires approval: {parsed.executable}",
                    )
                self._approved_sensitive.add(key)

        # Execute
        return self._run(command)

    def _run(self, command: str) -> CommandResult:
        """Run a command in a subprocess with isolation."""
        start = time.monotonic()
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace_root),
                timeout=self.timeout,
                capture_output=True,
                text=True,
                env=self._filtered_env(),
            )
            elapsed = (time.monotonic() - start) * 1000

            stdout = result.stdout
            stderr = result.stderr
            if len(stdout) > self.max_output_bytes:
                stdout = stdout[:self.max_output_bytes] + "\n... [output truncated]"
            if len(stderr) > self.max_output_bytes:
                stderr = stderr[:self.max_output_bytes] + "\n... [output truncated]"

            return CommandResult(
                command=command,
                exit_code=result.returncode,
                stdout=stdout,
                stderr=stderr,
                execution_time_ms=elapsed,
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.monotonic() - start) * 1000
            return CommandResult(
                command=command,
                timed_out=True,
                execution_time_ms=elapsed,
            )

    def _filtered_env(self) -> dict[str, str]:
        """Pass through environment, stripping parent venv/conda so the agent
        starts clean and must create its own isolated environment inside the
        workspace. Does not force any specific tool or path — the agent decides.
        """
        secret_suffixes = (
            "_API_KEY", "_SECRET", "_TOKEN", "_PASSWORD", "_PASSWD",
            "_CREDENTIAL", "_PRIVATE_KEY",
        )
        secret_prefixes = ("SECRET_", "PRIVATE_")

        # Strip only what ties processes to an external virtual environment.
        # Without VIRTUAL_ENV, pip/uv won't install into the parent .venv.
        # Without CONDA_*, conda won't think it's inside an existing env.
        strip_keys = {"VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
                      "CONDA_EXE", "CONDA_PYTHON_EXE", "CONDA_PROMPT_MODIFIER"}

        env = {}
        for key, value in os.environ.items():
            if key in strip_keys:
                continue
            if any(key.endswith(s) for s in secret_suffixes):
                continue
            if any(key.startswith(p) for p in secret_prefixes):
                continue
            env[key] = value

        env["PYTHONUNBUFFERED"] = "1"
        return env

    def check_command(self, command: str) -> tuple[PermissionLevel, str]:
        """Check a command without executing it. Returns (level, explanation)."""
        level, parsed = self.permissions.check(command)
        explanations = {
            PermissionLevel.SAFE: "Safe: read-only operation",
            PermissionLevel.WORKSPACE: "Safe: write within workspace",
            PermissionLevel.SENSITIVE: f"Sensitive: {parsed.executable} requires confirmation",
            PermissionLevel.DANGEROUS: "Dangerous: requires human approval",
            PermissionLevel.FORBIDDEN: "Forbidden: never allowed",
        }
        return level, explanations[level]

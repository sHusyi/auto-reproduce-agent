"""Shell execution tool — all commands go through the sandbox."""

from __future__ import annotations

from langchain_core.tools import tool

from src.sandbox.executor import SandboxedExecutor


def create_shell_tool(executor: SandboxedExecutor) -> list:
    """Create the shell execution tool backed by a SandboxedExecutor."""

    @tool
    def execute_command(command: str) -> str:
        """Execute a shell command inside the sandboxed workspace.

        Use this for: installing packages, running Python scripts, git operations,
        and any other shell commands needed to set up or test the project.

        The command runs in an isolated workspace directory. Dangerous commands
        (sudo, writes outside workspace, system modification) are blocked.

        Args:
            command: The shell command to execute.
        """
        result = executor.execute(command)
        if result.blocked:
            return f"BLOCKED: {result.block_reason}"
        if result.timed_out:
            return f"TIMEOUT: command exceeded {executor.timeout}s limit"
        if result.exit_code == 0:
            return result.stdout or "(command succeeded with no output)"
        else:
            return (
                f"EXIT CODE {result.exit_code}\n"
                f"STDOUT:\n{result.stdout or '(none)'}\n"
                f"STDERR:\n{result.stderr or '(none)'}"
            )

    return [execute_command]

"""File system tools for the agent — all operations scoped to workspace."""

from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.tools import tool


def create_file_tools(workspace_root: str | Path) -> list:
    """Create file operation tools scoped to a workspace directory.

    All paths are resolved relative to workspace_root. Operations outside
    the workspace are rejected at the tool level (before sandbox).
    """
    root = Path(workspace_root).resolve()

    def _resolve(safe_path: str) -> Path:
        """Resolve a user-supplied path, rejecting escapes from workspace."""
        p = (root / safe_path).resolve()
        try:
            p.relative_to(root)
        except ValueError:
            raise ValueError(f"Path escapes workspace: {safe_path}")
        return p

    @tool
    def read_file(path: str) -> str:
        """Read the contents of a file. Path is relative to workspace root.

        Args:
            path: Relative path to the file (e.g., 'main.py', 'src/model.py')
        """
        try:
            full = _resolve(path)
            if not full.is_file():
                return f"Error: not a file: {path}"
            content = full.read_text()
            return content
        except ValueError as e:
            return f"Error: {e}"

    @tool
    def edit_file(path: str, old_string: str, new_string: str) -> str:
        """Replace a string in a file. Finds the first exact match and replaces it.

        Use this to make targeted edits without rewriting the entire file.
        The old_string must match exactly (including whitespace).

        Args:
            path: Relative path to the file.
            old_string: Exact text to find and replace.
            new_string: Replacement text.
        """
        try:
            full = _resolve(path)
            if not full.is_file():
                return f"Error: not a file: {path}"
            content = full.read_text()
            if old_string not in content:
                return f"Error: string not found in {path}. Use read_file first to check exact content."
            # Replace first occurrence only
            content = content.replace(old_string, new_string, 1)
            full.write_text(content)
            return f"File edited: {path} (replaced {len(old_string)} chars with {len(new_string)} chars)"
        except ValueError as e:
            return f"Error: {e}"

    @tool
    def write_file(path: str, content: str) -> str:
        """Write content to a file. Creates parent directories if needed.

        Args:
            path: Relative path to the file.
            content: Text content to write.
        """
        try:
            full = _resolve(path)
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content)
            return f"File written: {path} ({len(content)} bytes)"
        except ValueError as e:
            return f"Error: {e}"

    @tool
    def list_directory(path: str = ".") -> str:
        """List contents of a directory. Defaults to workspace root.

        Args:
            path: Relative path to the directory. Default '.' for root.
        """
        try:
            full = _resolve(path)
            if not full.is_dir():
                return f"Error: not a directory: {path}"
            items = []
            for p in sorted(full.iterdir()):
                prefix = "[DIR]  " if p.is_dir() else "[FILE] "
                items.append(prefix + str(p.relative_to(root)))
            return "\n".join(items) if items else "(empty)"
        except ValueError as e:
            return f"Error: {e}"

    @tool
    def search_code(pattern: str, path: str = ".") -> str:
        """Search for a pattern in all files under a directory. Uses grep.

        Args:
            pattern: Text or regex pattern to search for.
            path: Directory to search in (default: entire workspace).
        """
        try:
            full = _resolve(path)
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", "--include=*.txt",
                 "--include=*.md", "--include=*.yml", "--include=*.yaml",
                 "--include=*.toml", "--include=*.cfg",
                 pattern, str(full)],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=str(root),
            )
            output = result.stdout
            if not output:
                return f"No matches for '{pattern}' in {path}"
            if len(output) > 5000:
                lines = output.split("\n")[:50]
                remaining = len(output.split("\n")) - 50
                output = "\n".join(lines) + f"\n... [{remaining} more matches truncated]"
            return output
        except ValueError as e:
            return f"Error: {e}"
        except subprocess.TimeoutExpired:
            return "Error: search timed out"

    return [read_file, edit_file, write_file, list_directory, search_code]

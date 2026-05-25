"""Permission controller with structured command parsing.

The key design principle: we parse commands into structured form BEFORE checking
permissions. We never trust the LLM's output — we analyze what the command
actually does, not what the LLM says it does.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

from src.state import ParsedCommand, PermissionLevel

# Commands that only read, never modify
SAFE_COMMANDS = frozenset({
    "ls", "cat", "head", "tail", "less", "file", "stat",
    "find", "grep", "wc", "du", "df", "tree",
    "which", "type", "echo", "printf", "date", "pwd", "whoami",
    "python3", "python",  # with --version or -c for inspection
})

# Commands that modify, but only within workspace
WORKSPACE_COMMANDS = frozenset({
    "mkdir", "cp", "mv", "touch", "chmod",
    "python3", "python",  # running scripts
    "pip", "pip3", "conda",
    "git", "curl", "wget", "tar", "unzip",
    "nano", "vim", "code",
})

# Commands that need extra scrutiny
SENSITIVE_COMMANDS = frozenset({
    "rm", "rmdir", "pip", "pip3", "conda",
    "curl", "wget", "git", "chmod", "chown",
})

# Never allowed
FORBIDDEN_PATTERNS = [
    re.compile(r"rm\s+.*(-rf?\s+)?/"),          # rm -rf /
    re.compile(r"rm\s+.*(-rf?\s+)?\*"),          # rm -rf *
    re.compile(r">\s*/dev/[a-z]+"),              # write to /dev/*
    re.compile(r"mkfs"),                          # format
    re.compile(r"dd\s+if="),                      # disk copy
    re.compile(r":\(\)\s*\{"),                    # fork bomb
    re.compile(r"chmod\s+.*777\s+/"),            # chmod 777 on system paths
]

FORBIDDEN_EXECUTABLES = frozenset({
    "shutdown", "reboot", "halt", "poweroff",
    "mkfs", "fdisk", "mount", "umount",
    "iptables", "ufw", "systemctl", "service",
})

SYSTEM_PATHS = frozenset({
    "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64",
    "/boot", "/dev", "/proc", "/sys", "/root", "/var",
    "/opt", "/home", "/tmp", "/private",
})


class PermissionController:
    """Classifies commands into permission levels based on structured analysis."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()

    def check(self, raw_command: str) -> tuple[PermissionLevel, ParsedCommand]:
        """Parse and classify a shell command. Returns (level, parsed)."""
        parsed = self._parse(raw_command)

        # Level 4: FORBIDDEN
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(parsed.raw):
                return PermissionLevel.FORBIDDEN, parsed
        if parsed.executable in FORBIDDEN_EXECUTABLES:
            return PermissionLevel.FORBIDDEN, parsed
        if parsed.targets_system_paths and parsed.executable in ("rm", "chmod", "chown"):
            return PermissionLevel.FORBIDDEN, parsed

        # Level 3: DANGEROUS
        if parsed.uses_sudo:
            return PermissionLevel.DANGEROUS, parsed
        if parsed.writes_outside_workspace:
            return PermissionLevel.DANGEROUS, parsed

        # Level 2: SENSITIVE
        if parsed.executable in SENSITIVE_COMMANDS:
            return PermissionLevel.SENSITIVE, parsed

        # Level 1: WORKSPACE (modifying within workspace)
        if parsed.executable in WORKSPACE_COMMANDS or parsed.executable not in SAFE_COMMANDS:
            return PermissionLevel.WORKSPACE, parsed

        # Level 0: SAFE
        return PermissionLevel.SAFE, parsed

    def _parse(self, raw: str) -> ParsedCommand:
        """Parse a shell command string into structured form."""
        raw = raw.strip()
        try:
            tokens = shlex.split(raw)
        except ValueError:
            tokens = raw.split()

        if not tokens:
            return ParsedCommand(executable="", args=[], raw=raw)

        executable = tokens[0]
        args = tokens[1:] if len(tokens) > 1 else []

        # Detect sudo
        if executable == "sudo":
            return ParsedCommand(
                executable="",
                args=[],
                raw=raw,
                uses_sudo=True,
                writes_outside_workspace=True,
            )

        # Resolve paths from arguments
        resolved_paths: list[Path] = []
        writes_outside = False
        targets_system = False
        has_redirect = ">" in raw
        has_pipe = "|" in raw

        for arg in args:
            # Skip flags
            if arg.startswith("-"):
                if arg.startswith("--prefix=") or arg.startswith("--target="):
                    path_str = arg.split("=", 1)[1]
                    resolved = self._resolve_path(path_str)
                    resolved_paths.append(resolved)
                    if not self._is_within_workspace(resolved):
                        writes_outside = True
                    if self._is_system_path(resolved):
                        targets_system = True
                continue

            # Check if arg looks like a path
            if self._looks_like_path(arg):
                resolved = self._resolve_path(arg)
                resolved_paths.append(resolved)
                if not self._is_within_workspace(resolved):
                    writes_outside = True
                if self._is_system_path(resolved):
                    targets_system = True

        # Handle redirect targets (> file, >> file)
        if has_redirect:
            redirect_match = re.search(r"[12]?>>?\s*(\S+)", raw)
            if redirect_match:
                path_str = redirect_match.group(1)
                resolved = self._resolve_path(path_str)
                resolved_paths.append(resolved)
                if not self._is_within_workspace(resolved):
                    writes_outside = True

        return ParsedCommand(
            executable=executable,
            args=args,
            raw=raw,
            resolved_paths=resolved_paths,
            writes_outside_workspace=writes_outside,
            uses_sudo="sudo" in tokens,
            targets_system_paths=targets_system,
            has_redirect=has_redirect,
            has_pipe=has_pipe,
        )

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a path string relative to the workspace root."""
        p = Path(path_str)
        if p.is_absolute():
            return p.resolve()
        return (self.workspace_root / p).resolve()

    def _is_within_workspace(self, path: Path) -> bool:
        """Check if a resolved path is within the workspace."""
        try:
            path.relative_to(self.workspace_root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_system_path(path: Path) -> bool:
        """Check if a path targets system directories."""
        path_str = str(path)
        return any(path_str.startswith(sp) for sp in SYSTEM_PATHS)

    @staticmethod
    def _looks_like_path(arg: str) -> bool:
        """Heuristic: does this argument look like a file path?"""
        return bool(
            "/" in arg
            or arg.startswith(".")
            or arg.startswith("~")
            or arg.endswith("/")
        )

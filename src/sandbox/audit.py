"""Audit logger for full traceability of all agent actions."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.state import AuditEntry


class AuditLogger:
    """Records every agent action to a JSONL file for traceability.

    Each line is a JSON object representing one action. This format is:
    - Append-only (safe for concurrent writes)
    - Human-readable (one action per line)
    - Machine-parseable (standard JSONL)
    """

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[AuditEntry] = []

    def log(self, entry: AuditEntry) -> None:
        """Record an audit entry to file and memory."""
        self._entries.append(entry)
        with open(self.log_path, "a") as f:
            f.write(entry.model_dump_json() + "\n")

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    @property
    def total_actions(self) -> int:
        return len(self._entries)

    def summary(self) -> str:
        """Return a human-readable summary of the audit trail."""
        if not self._entries:
            return "No actions recorded."

        lines = [f"Audit Trail ({self.total_actions} actions)", "=" * 50]
        for i, entry in enumerate(self._entries, 1):
            status = "✓" if entry.decision == "auto_approved" else "✗"
            cmd_preview = entry.command_raw[:80]
            lines.append(
                f"{i:3d}. [{entry.permission_level.name:12s}] {status} {cmd_preview}"
            )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Generate a markdown report of all actions."""
        lines = [
            "# Audit Trail\n",
            f"**Total Actions:** {self.total_actions}\n",
            "| # | Round | Decision | Level | Command | Exit | Time |",
            "|---|-------|----------|-------|---------|------|------|",
        ]
        for i, entry in enumerate(self._entries, 1):
            exit_code = entry.result.exit_code if entry.result else "-"
            ts = entry.timestamp.strftime("%H:%M:%S")
            cmd = entry.command_raw[:60]
            lines.append(
                f"| {i} | {entry.agent_round} | {entry.decision} "
                f"| {entry.permission_level.name} "
                f"| `{cmd}` | {exit_code} | {ts} |"
            )
        return "\n".join(lines)

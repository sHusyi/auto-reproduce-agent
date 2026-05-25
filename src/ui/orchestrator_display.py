"""Orchestrator display — renders node outputs and progress during the research loop.

Separated from orchestrator.py so the loop logic doesn't know about Rich.
"""

from __future__ import annotations

import json

from src.ui.terminal import Terminal


class OrchestratorDisplay:
    """Renders research loop progress using a Terminal instance."""

    def __init__(self, term: Terminal) -> None:
        self.term = term

    def show_node(self, node_name: str, output: dict, round_num: int) -> None:
        """Display a node's execution result."""
        content = self._extract_content(node_name, output)
        if not content:
            return
        if self.term.rich:
            self._show_rich(node_name, content, round_num)
        else:
            self._show_plain(node_name, content, round_num)

    # ── Private ──────────────────────────────────────────────────────────

    def _show_rich(self, node_name: str, content: str, round_num: int) -> None:
        from rich.panel import Panel as _RichPanel
        from rich import box as _rich_box

        spec = {
            "assess": ("cyan", "🔍"),
            "plan": ("yellow", "📋"),
            "execute": ("green", "⚡"),
            "reflect": ("magenta", "🤔"),
            "decide": ("blue", "🎯"),
        }
        color, icon = spec.get(node_name, ("white", "•"))
        panel = _RichPanel(
            content,
            title=f"[bold {color}]{icon} {node_name.upper()} (Round {round_num})[/bold {color}]",
            border_style=color,
            box=_rich_box.ROUNDED,
            width=min(100, self.term._console.width - 4) if self.term._console.width else 80,
        )
        self.term._console.print(panel)

    def _show_plain(self, node_name: str, content: str, round_num: int) -> None:
        print(f"\n── {node_name.upper()} (Round {round_num}) ──")
        for line in content.strip().split("\n")[:8]:
            print(f"  {line}")

    @staticmethod
    def _extract_content(node_name: str, output: dict) -> str:
        """Extract the human-readable content from a node's output."""
        if node_name == "assess":
            try:
                data = json.loads(output.get("assessment", "{}"))
                lines = [f"Situation: {data.get('situation', '...')}"]
                problems = data.get("problems_identified", [])
                if problems:
                    lines.append(f"Problems: {', '.join(problems)}")
                lines.append(f"Priority: {data.get('priority', '...')}")
                return "\n".join(lines)
            except (json.JSONDecodeError, AttributeError):
                return output.get("assessment", "")[:300]

        elif node_name == "plan":
            actions = output.get("planned_actions", [])
            if not actions:
                return "(no actions)"
            lines = []
            for a in actions[:5]:
                tn = a.get("tool_name", "?")
                ta = a.get("tool_args", {})
                desc = a.get("description", "")[:60]
                lines.append(f"  {tn}({json.dumps(ta) if ta else ''}) — {desc}")
            if len(actions) > 5:
                lines.append(f"  ... and {len(actions)-5} more")
            return "\n".join(lines)

        elif node_name == "execute":
            result = output.get("last_result", "")
            if "[FAILURE_TYPE:" in result:
                lines = result.split("\n")
                return f"{lines[0]}\n{lines[1]}\n{lines[2][:200] if len(lines) > 2 else ''}"
            return result[:500] if result else "(no output)"

        elif node_name == "reflect":
            try:
                data = json.loads(output.get("reflection", "{}"))
                return "\n".join([
                    f"Learned: {data.get('what_was_learned', '...')}",
                    f"Expectation met: {data.get('expectation_met', '?')}",
                    f"Strategy: {data.get('strategy_adjustment', '...')}",
                ])
            except (json.JSONDecodeError, AttributeError):
                return output.get("reflection", "")[:300]

        elif node_name == "decide":
            try:
                data = json.loads(output.get("decision", "{}"))
                decision = data.get("decision", "?")
                reasoning = data.get("reasoning", "")
                emoji = {"continue": "➡️", "success": "✅", "partial": "⚠️", "failed": "❌"}
                return f"{emoji.get(decision, '')} {decision.upper()}: {reasoning}"
            except (json.JSONDecodeError, AttributeError):
                return f"Decision: {output.get('verdict', '?')}"

        return ""

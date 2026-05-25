"""Terminal abstraction — all user I/O goes through this module.

Auto-detects Rich availability. Falls back to plain text + ANSI codes.
No other module should import Rich or use raw print() for UI.
"""

from __future__ import annotations

import os
import readline  # noqa: F401 — cursor movement in input()
from typing import Any

# ── ANSI constants (always available, no dependency) ─────────────────────

ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_GREEN = "\033[32m"
ANSI_CYAN = "\033[36m"
ANSI_YELLOW = "\033[33m"
ANSI_RED = "\033[31m"
ANSI_MAGENTA = "\033[35m"
ANSI_BLUE = "\033[34m"
ANSI_WHITE = "\033[37m"
ANSI_RESET = "\033[0m"

# ── Rich detection ────────────────────────────────────────────────────────

try:
    from rich.console import Console as _RichConsole
    from rich.panel import Panel as _RichPanel
    from rich.table import Table as _RichTable
    from rich.markdown import Markdown as _RichMarkdown
    from rich import box as _rich_box
    _RICH = True
except ImportError:
    _RICH = False


class Terminal:
    """All terminal I/O. Singleton-like — just instantiate once.

    Usage:
        term = Terminal()
        name = term.prompt("What's your name?")
        term.show_banner()
        term.show_panel("Content", title="Title", style="cyan")
        term.show_table("Title", ["Col1", "Col2"], [["a", "b"], ["c", "d"]])
    """

    def __init__(self) -> None:
        self.rich = _RICH
        self._console = _RichConsole() if _RICH else None

    # ── Low-level I/O ─────────────────────────────────────────────────

    def prompt(self, text: str, *, style: str = "bold") -> str:
        """Show a styled prompt and return user input."""
        color = {
            "bold": ANSI_BOLD, "green": ANSI_GREEN, "yellow": ANSI_YELLOW,
            "cyan": ANSI_CYAN, "red": ANSI_RED, "dim": ANSI_DIM,
        }.get(style, "")
        return input(f"{color}{text}{ANSI_RESET} ").strip()

    def confirm(self, text: str = "Proceed?", default_yes: bool = True) -> bool:
        """Ask a yes/no question."""
        hint = "[Y/n]" if default_yes else "[y/N]"
        resp = input(f"\n{ANSI_BOLD}{text}{ANSI_RESET} {ANSI_DIM}{hint}{ANSI_RESET} ").strip().lower()
        return resp != "n" if default_yes else resp == "y"

    def ask_questions(self, questions: list[str]) -> dict[str, str]:
        """Ask a list of questions and collect answers."""
        answers: dict[str, str] = {}
        if not questions:
            return answers
        self.status("Let me clarify a few things before we start:", "cyan")
        for q in questions:
            answer = input(f"  {ANSI_BOLD}{ANSI_CYAN}{q}{ANSI_RESET}\n  {ANSI_DIM}→{ANSI_RESET} ").strip()
            if answer:
                answers[q] = answer
        return answers

    # ── Status messages ────────────────────────────────────────────────

    def status(self, message: str, style: str = "dim") -> None:
        """Show a one-line status message."""
        if self.rich:
            self._console.print(f"[{style}]{message}[/{style}]")
        else:
            print(message)

    def error(self, message: str) -> None:
        """Show an error message."""
        if self.rich:
            self._console.print(f"[red]{message}[/red]")
        else:
            print(f"{ANSI_RED}{message}{ANSI_RESET}")

    def success(self, message: str) -> None:
        if self.rich:
            self._console.print(f"[green]{message}[/green]")
        else:
            print(f"{ANSI_GREEN}{message}{ANSI_RESET}")

    def warning(self, message: str) -> None:
        if self.rich:
            self._console.print(f"[yellow]{message}[/yellow]")
        else:
            print(f"{ANSI_YELLOW}{message}{ANSI_RESET}")

    # ── Rich panels & tables (no-op if no Rich) ─────────────────────────

    def show_panel(self, content: str, *, title: str = "", style: str = "white") -> None:
        """Show a bordered panel."""
        if self.rich:
            self._console.print(_RichPanel.fit(
                content, title=title, border_style=style,
                box=_rich_box.ROUNDED,
            ))
        else:
            if title:
                print(f"\n── {title} ──")
            for line in content.strip().split("\n"):
                print(f"  {line}")

    def show_table(self, title: str, columns: list[str], rows: list[list[str]]) -> None:
        """Show a formatted table."""
        if self.rich:
            table = _RichTable(title=title, border_style="blue", box=_rich_box.SIMPLE)
            for i, col in enumerate(columns):
                table.add_column(col, style="dim" if i == 0 else "", width=min(40, 100 // len(columns)))
            for row in rows:
                table.add_row(*[str(c) for c in row])
            self._console.print(table)
        else:
            print(f"\n── {title} ──")
            print(" | ".join(columns))
            for row in rows:
                print(" | ".join(str(c) for c in row))

    def show_markdown(self, text: str) -> None:
        """Render markdown text."""
        if self.rich:
            self._console.print(_RichMarkdown(text))
        else:
            print(text)

    def show_banner(self) -> None:
        """Show the startup banner."""
        if self.rich:
            self._console.print()
            self._console.print(_RichPanel.fit(
                "[bold cyan]Auto-Research Agent[/bold cyan]\n\n"
                "I autonomously reproduce ML paper results by exploring repositories,\n"
                "diagnosing issues, and iteratively fixing problems.\n\n"
                "[dim]Plan → Execute → Reflect → Replan loop[/dim]\n\n"
                "Type your request below (e.g., '帮我复现这篇论文的实验结果...')\n"
                "Type [bold]/help[/bold] for commands, [bold]/quit[/bold] to exit.",
                border_style="cyan",
                box=_rich_box.ROUNDED,
            ))
            self._console.print()
        else:
            print("\nAuto-Research Agent")
            print("Type your request or /help\n")

    def show_help(self) -> None:
        text = """**Commands:**
- `/help` — Show this help
- `/quit` or `/exit` — Exit the agent
- `/scenarios` — List demo scenarios

**Example requests:**
- "帮我复现一下这篇论文的实验结果，仓库是 https://github.com/kuangliu/pytorch-cifar"
- "Reproduce the results from https://github.com/xxx/yyy, target accuracy is 95%"

Just describe what you want in natural language. I'll figure out the details.
"""
        self.show_markdown(text)

    def show_intent(self, intent: dict) -> None:
        """Display parsed request intent."""
        repo_summary = intent.get("repo_summary", "")
        milestones = intent.get("milestones", [])
        criteria = intent.get("success_criteria", [])
        challenges = intent.get("potential_challenges", [])
        questions = intent.get("questions_for_user", [])

        if self.rich:
            if repo_summary:
                self._console.print(_RichPanel.fit(
                    repo_summary[:600],
                    title="[green]Repository Understanding[/green]",
                    border_style="green",
                ))
            if milestones:
                self.show_table(
                    "🎯 Planned Milestones",
                    ["#", "Goal", "Verification"],
                    [[str(m.get("step", "?")), m.get("goal", ""), m.get("verification", "")]
                     for m in milestones],
                )
            if criteria:
                self._console.print("\n[bold]Success criteria:[/bold]")
                for c in criteria:
                    self._console.print(f"  ✓ {c}")
            if challenges:
                self._console.print("\n[yellow]Potential challenges:[/yellow]")
                for c in challenges:
                    self._console.print(f"  ⚠ {c}")
            if questions:
                self._console.print("\n[bold red]Questions:[/bold red]")
                for q in questions:
                    self._console.print(f"  ❓ {q}")
        else:
            if repo_summary:
                print(f"\nRepository: {repo_summary[:300]}")
            for m in milestones:
                print(f"  Step {m.get('step')}: {m.get('goal')}")
            if criteria:
                print("\nSuccess criteria:")
                for c in criteria:
                    print(f"  - {c}")

    def show_parsed_request(self, parsed: dict) -> None:
        """Display the parsed user request."""
        repo_url = parsed.get("repo_url", "") or "NOT PROVIDED"
        paper_url = parsed.get("paper_url") or "NOT PROVIDED"
        target = parsed.get("target_metrics", {}) or "Will extract from paper/README"
        task = parsed.get("task_summary", "Unknown")

        if self.rich:
            self._console.print(_RichPanel.fit(
                f"[bold]Task:[/bold] {task}\n"
                f"[bold]Repo:[/bold] {repo_url}\n"
                f"[bold]Paper:[/bold] {paper_url}\n"
                f"[bold]Target:[/bold] {target}",
                title="Parsed Request",
                border_style="cyan",
            ))
        else:
            print(f"Task: {task}\nRepo: {repo_url}\nPaper: {paper_url}\nTarget: {target}")

    def show_verdict(self, verdict: str) -> None:
        """Show the final verdict with color."""
        styles = {"success": "green", "partial": "yellow", "interrupted": "yellow",
                  "failed": "red", "continue": "blue"}
        style = styles.get(verdict, "white")
        if self.rich:
            self._console.print(f"\n[bold {style}]Verdict: {verdict.upper()}[/bold {style}]")
        else:
            color = {"green": ANSI_GREEN, "yellow": ANSI_YELLOW, "red": ANSI_RED}.get(style, "")
            print(f"\n{color}{ANSI_BOLD}Verdict: {verdict.upper()}{ANSI_RESET}")

    def show_metrics(self, metrics_summary: str) -> None:
        """Display run metrics."""
        if self.rich:
            self._console.print(_RichMarkdown(f"```\n{metrics_summary}\n```"))
        else:
            print(f"\n{metrics_summary}")

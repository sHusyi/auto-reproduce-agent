"""Human-in-the-loop tools — let the agent request help, not just approval.

Unlike the sandbox approval callback (which only gates dangerous commands),
these tools let the agent proactively ask the human for guidance when stuck.
This is decision-level collaboration, not just command-level gating.

Key design choice:
- The callback is blocking — the agent pauses until the human responds
- If no callback is registered, the tool returns a message telling the agent
  to continue with its best guess (graceful degradation)
"""

from __future__ import annotations

from typing import Callable

from langchain_core.tools import tool

# Type for the help callback: (question, context) -> answer
HelpCallback = Callable[[str, str], str]


def create_human_help_tool(callback: HelpCallback | None = None) -> list:
    """Create the human help tool with an optional callback.

    Args:
        callback: Function called when agent requests help.
                  Receives (question, context) and returns human's answer.
                  If None, the tool tells the agent to proceed independently.
    """

    @tool
    def request_human_help(question: str, context: str = "") -> str:
        """Ask the human operator for help when you are stuck or uncertain.

        Use this tool when:
        - You've tried multiple approaches and all failed
        - You need domain expertise to interpret an error
        - You're unsure which strategy to pursue next
        - You want confirmation before a high-risk action

        Do NOT use this for routine questions you can answer yourself.

        Args:
            question: A clear, specific question for the human.
            context: What you've tried, what you know, why you're stuck.
        """
        if callback is None:
            return (
                "Human help is not available (no callback configured). "
                "Continue with your best judgment based on available information. "
                "If truly stuck, mark the task as 'failed' with an explanation."
            )

        return callback(question, context)

    @tool
    def report_progress(summary: str) -> str:
        """Report a milestone or important finding to the human operator.

        Use this when you've achieved something significant (e.g., fixed a bug,
        reached a metric threshold, confirmed a hypothesis).

        Args:
            summary: What was achieved and why it matters.
        """
        if callback is None:
            return "Progress noted (no callback configured)."
        callback(f"[Progress] {summary}", "")
        return "Progress reported to human."

    return [request_human_help, report_progress]

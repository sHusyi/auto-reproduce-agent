"""PARSE node — extracts structured task from natural language.

Before CLARIFY or the research loop, the user's free-form text is parsed
into structured fields: repo_url, paper_url, target_metrics, task_description.
"""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

PARSE_SYSTEM_PROMPT = """You are an AI assistant that extracts structured information from a user's
research reproduction request. Parse the user's message into a JSON object.

Output format:
{
    "repo_url": "GitHub repository URL (extract from message, or null if not provided)",
    "paper_url": "Paper URL (arxiv, openaccess, etc. — extract from message, or null)",
    "target_metrics": {"metric_name": target_value, ...},
    "task_summary": "One-sentence summary of what the user wants",
    "missing_info": ["What critical information is missing?"],
    "suggestions": []
}

Critical rule — "missing_info" should be EMPTY in these cases:
- The user provided a GitHub repo URL and asked to "reproduce results" or "复现实验结果"
  → The README contains the claimed results; the agent will extract targets later.
  → target_metrics = {} (empty), missing_info = []
- The user says "跑一下这个仓库" / "复现" / "reproduce" with a repo link → sufficient.
  → Do NOT ask for specific models, datasets, or accuracy targets — the repo has those.

Only mark missing_info if truly critical information is absent:
- No repo URL at all → "Please provide a GitHub repository URL."
- The user's request is completely ambiguous (not about reproduction) → ask for clarification.

Do NOT mark missing_info for things the repo README can answer (model names, metrics, datasets).
That's the agent's job to discover, not the user's job to specify."""


def create_parse_request_node(llm: BaseChatModel):
    """Create the request parsing node."""

    def parse_request(user_message: str) -> dict:
        """Parse a natural language request into structured fields.

        Returns a dict with repo_url, paper_url, target_metrics, etc.
        """
        response = llm.invoke([
            SystemMessage(content=PARSE_SYSTEM_PROMPT),
            HumanMessage(content=f"User request:\n{user_message}"),
        ])

        try:
            parsed = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback: try to find a URL in the message
            import re
            urls = re.findall(r'https?://[^\s]+', user_message)
            repo_url = next((u for u in urls if 'github.com' in u), None)
            paper_url = next((u for u in urls if u != repo_url), None)
            parsed = {
                "repo_url": repo_url,
                "paper_url": paper_url,
                "target_metrics": {},
                "task_summary": user_message[:200],
                "missing_info": [],
                "suggestions": [],
            }

        return parsed

    return parse_request

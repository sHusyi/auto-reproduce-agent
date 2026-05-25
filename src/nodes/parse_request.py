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
    "missing_info": ["What critical information is missing? (empty if sufficient)"],
    "suggestions": ["Suggestions for target metrics if user didn't specify any"]
}

Rules:
- If the user provides a GitHub URL, extract it as repo_url.
- If the user provides a paper link (arxiv.org, paperswithcode.com, etc.), extract it.
- If the user says "复现实验结果" or "reproduce results", target_metrics can be empty
  (the agent will extract them from the paper/README).
- If the user specifies metrics like "准确率达到95%" or "accuracy > 95%",
  parse them into target_metrics.
- If critical info is missing (no repo URL, unclear goal), list it in missing_info.
- Be helpful and specific in suggestions."""


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

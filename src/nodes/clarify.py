"""CLARIFY node — intent recognition and goal confirmation.

Runs ONCE before the research loop. Reads the repository, understands the user's
goal, breaks it into concrete milestones, and confirms with the user.
Only after confirmation does the research loop begin.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

CLARIFY_SYSTEM_PROMPT = """You are an AI research assistant clarifying a paper reproduction task.
Your job is to analyze the repository and user's goal, then produce a structured
understanding that the user can confirm or adjust.

First, analyze the repository structure and README (provided below).
Then output a JSON object:

{
    "repo_summary": "One paragraph summary of what this repository does",
    "paper_claim": "What results does the README/paper claim? (metrics, tables, etc.)",
    "user_goal": "Restate what the user wants to achieve",
    "success_criteria": [
        "Specific, measurable criteria for success (e.g., 'accuracy >= 95.47 on CIFAR-10 test set')"
    ],
    "milestones": [
        {"step": 1, "goal": "Set up Python environment with all dependencies", "verification": "import torch, torchvision succeeds"},
        {"step": 2, "goal": "Verify dataset is accessible", "verification": "Training script can load CIFAR-10"},
        {"step": 3, "goal": "Run baseline training", "verification": "Training completes without errors"},
        {"step": 4, "goal": "Compare results to paper claims", "verification": "Metrics meet or exceed paper claims"}
    ],
    "potential_challenges": [
        "What might go wrong? Missing dependencies, hardware requirements, data access, etc."
    ],
    "confidence": 0.0-1.0,
    "questions_for_user": [
        "Any clarifying questions for the user before starting (empty if clear)"
    ]
}

Be specific. Reference actual file names, claimed metrics, and dependencies from the repository.
If the README doesn't mention specific metrics, infer reasonable targets.
If you have questions, ask them — don't guess."""


def create_clarify_node(llm: BaseChatModel):
    """Create the CLARIFY node — runs once before the research loop."""

    def clarify(state: dict) -> dict:
        repo_url = state.get("repo_url", "")
        target_metrics = state.get("target_metrics", {})
        repo_exploration = state.get("repo_exploration", "")

        user_prompt = f"""Repository: {repo_url}
User's target metrics: {target_metrics}

Repository exploration results:
{repo_exploration or "(not yet explored)"}

Please analyze the repository and clarify the reproduction goal.
Output your analysis as JSON:"""

        response = llm.invoke([
            SystemMessage(content=CLARIFY_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        try:
            intent = json.loads(response.content)
        except json.JSONDecodeError:
            intent = {
                "repo_summary": response.content[:500],
                "paper_claim": "Unknown",
                "user_goal": str(target_metrics),
                "success_criteria": [f"Reach {target_metrics}"],
                "milestones": [
                    {"step": 1, "goal": "Set up environment", "verification": "Dependencies install"},
                    {"step": 2, "goal": "Run training", "verification": "Script executes"},
                    {"step": 3, "goal": "Compare results", "verification": f"Metrics >= {target_metrics}"},
                ],
                "potential_challenges": ["Unknown — will discover during execution"],
                "confidence": 0.5,
                "questions_for_user": [],
            }

        return {
            "intent": intent,
            "observations": [f"[Clarify] Goal understood: {intent.get('user_goal', '')}"],
        }

    return clarify

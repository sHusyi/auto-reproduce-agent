"""Memory compressor — prevents context explosion in long research loops.

When the agent runs many rounds, the observation list and experiment history
grow unboundedly. This module compresses old entries into summaries, keeping
the most recent items in full detail.
"""

from __future__ import annotations

import json
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import ResearchState

COMPRESS_SYSTEM_PROMPT = """You are summarizing the history of a research agent's actions.
Condense the provided observations and experiments into a concise summary.

Output a JSON object:
{
    "summary": "2-3 sentence summary of what happened and what was learned",
    "key_findings": ["finding 1", "finding 2", ...],
    "unresolved_issues": ["issue 1", ...],
    "successful_actions": ["action that worked", ...],
    "failed_actions": ["action that failed", ...]
}"""


def compress_observations(
    llm: BaseChatModel,
    state: ResearchState,
    *,
    keep_recent: int = 5,
    max_observations: int = 20,
) -> dict:
    """Compress old observations when the list grows too large.

    Args:
        llm: The LLM to use for summarization
        state: Current research state
        keep_recent: Number of recent observations to keep in full
        max_observations: Trigger compression when observations exceed this

    Returns:
        Dict with compressed observations list and updated knowledge notes
    """
    observations = state.get("observations", [])
    if len(observations) <= max_observations:
        return {}

    # Split: recent stays, old gets compressed
    old = observations[:-keep_recent]
    recent = observations[-keep_recent:]

    old_text = "\n".join(f"- {obs}" for obs in old[-30:])
    user_prompt = f"""Old observations to compress:
{old_text}

Current knowledge:
{json.dumps(state.get('knowledge', {}).__dict__ if state.get('knowledge') else {}, default=str)}

Output compression JSON:"""

    try:
        response = llm.invoke([
            SystemMessage(content=COMPRESS_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        compressed = json.loads(response.content)
    except (json.JSONDecodeError, Exception):
        compressed = {
            "summary": f"Compressed {len(old)} older observations",
            "key_findings": [],
            "unresolved_issues": [],
            "successful_actions": [],
            "failed_actions": [],
        }

    # Create a single summary observation that replaces all old ones
    summary_obs = (
        f"[Compressed {len(old)} observations] "
        f"{compressed.get('summary', '')}"
    )
    if compressed.get("key_findings"):
        summary_obs += " | Key findings: " + "; ".join(compressed["key_findings"][:5])

    new_observations = [summary_obs] + recent

    return {
        "observations": new_observations,
    }


def compress_experiments(
    state: ResearchState,
    *,
    keep_recent: int = 5,
) -> dict:
    """Keep only recent experiments in full; mark older ones for DB-only storage.

    Unlike observations, experiments are already persisted in SQLite.
    We just trim the in-memory list to avoid state bloat.
    """
    experiments = state.get("experiment_history", [])
    if len(experiments) <= keep_recent:
        return {}

    return {
        "experiment_history": experiments[-keep_recent:],
    }

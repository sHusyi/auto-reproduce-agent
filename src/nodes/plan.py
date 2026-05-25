"""PLAN node — action planning with hypothesis dedup, using shared context."""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.context import SHARED_SYSTEM_PROMPT, ContextBuilder
from src.state import Hypothesis, ResearchState


def create_plan_node(llm: BaseChatModel, tool_names: list[str] | None = None):
    def plan(state: ResearchState) -> dict:
        user_prompt = ContextBuilder.build(state, "plan", tool_names=tool_names)
        response = llm.invoke([
            SystemMessage(content=SHARED_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        try:
            plan_data = json.loads(response.content)
        except json.JSONDecodeError:
            plan_data = {
                "reasoning": "Fallback",
                "actions": [{"tool_name": "list_directory", "tool_args": {"path": "."},
                             "description": "Explore repo", "expected_outcome": "See structure"}],
            }

        # Support both old (single) and new (list) format
        actions = plan_data.get("actions", [])
        if not actions and plan_data.get("tool_name"):
            actions = [{
                "tool_name": plan_data["tool_name"],
                "tool_args": plan_data.get("tool_args", {}),
                "description": plan_data.get("action_description", ""),
                "expected_outcome": plan_data.get("expected_outcome", ""),
            }]

        # Dedup hypotheses
        new_hypotheses: list[Hypothesis] = []
        hypothesis_text = plan_data.get("hypothesis")
        if hypothesis_text:
            existing_statements = {
                h.statement.strip().lower()[:80]
                for h in state.get("hypotheses", [])
            }
            if hypothesis_text.strip().lower()[:80] not in existing_statements:
                outcomes = " → ".join(a.get("expected_outcome", "") for a in actions)
                new_hypotheses.append(Hypothesis(
                    statement=hypothesis_text,
                    confidence=0.6,
                    verification_method=outcomes or str(actions),
                ))

        return {
            "plan": [plan_data.get("reasoning", "")],
            "planned_actions": actions,
            "hypotheses": new_hypotheses,
        }
    return plan

"""DECIDE node — continue/stop decision, using shared context."""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.context import SHARED_SYSTEM_PROMPT, ContextBuilder
from src.state import ResearchState


def create_decide_node(llm: BaseChatModel):
    def decide(state: ResearchState) -> dict:
        user_prompt = ContextBuilder.build(state, "decide")
        response = llm.invoke([
            SystemMessage(content=SHARED_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        try:
            decision_data = json.loads(response.content)
        except json.JSONDecodeError:
            round_number = state.get("round_number", 0)
            max_rounds = state.get("max_rounds", 5)
            decision_data = {
                "decision": "continue" if round_number < max_rounds else "partial",
                "reasoning": "Default decision",
            }

        decision = decision_data.get("decision", "continue")
        round_number = state.get("round_number", 0)
        max_rounds = state.get("max_rounds", 5)
        should_continue = decision == "continue" and round_number < max_rounds
        next_round = round_number + 1 if should_continue else round_number

        return {
            "decision": json.dumps(decision_data, indent=2),
            "verdict": decision,
            "should_continue": should_continue,
            "round_number": next_round,
        }
    return decide

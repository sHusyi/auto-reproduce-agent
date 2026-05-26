"""DECIDE node — continue/stop decision with reasoning-chain history."""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.context import SHARED_SYSTEM_PROMPT, ContextBuilder
from src.state import ResearchState


def create_decide_node(llm: BaseChatModel):
    def decide(state: ResearchState) -> dict:
        round_num = state.get("round_number", 0)
        user_prompt = ContextBuilder.build(state, "decide")

        history = state.get("messages", [])
        response = llm.invoke([
            SystemMessage(content=SHARED_SYSTEM_PROMPT),
            *history,
            HumanMessage(content=user_prompt),
        ])
        try:
            decision_data = json.loads(response.content)
        except json.JSONDecodeError:
            max_rounds = state.get("max_rounds", 5)
            decision_data = {
                "decision": "continue" if round_num < max_rounds else "partial",
                "reasoning": "Default decision",
            }

        decision = decision_data.get("decision", "continue")
        max_rounds = state.get("max_rounds", 5)
        should_continue = decision == "continue" and round_num < max_rounds
        next_round = round_num + 1 if should_continue else round_num

        label = HumanMessage(content=f"[Round {round_num} DECIDE]")
        ai_msg = AIMessage(content=response.content)

        return {
            "decision": json.dumps(decision_data, indent=2),
            "verdict": decision,
            "should_continue": should_continue,
            "round_number": next_round,
            "messages": [label, ai_msg],
        }
    return decide

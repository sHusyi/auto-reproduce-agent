"""ASSESS node — situational awareness with reasoning-chain history."""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.context import SHARED_SYSTEM_PROMPT, ContextBuilder
from src.state import ResearchState


def create_assess_node(llm: BaseChatModel):
    def assess(state: ResearchState) -> dict:
        round_num = state.get("round_number", 0)
        user_prompt = ContextBuilder.build(state, "assess")

        # History: lightweight labels + LLM's own responses (the reasoning chain)
        history = state.get("messages", [])
        response = llm.invoke([
            SystemMessage(content=SHARED_SYSTEM_PROMPT),
            *history,
            HumanMessage(content=user_prompt),
        ])
        try:
            assessment = json.loads(response.content)
        except json.JSONDecodeError:
            assessment = {"situation": response.content, "priority": "unknown"}

        # Save to history: short label + full LLM response
        label = HumanMessage(content=f"[Round {round_num} ASSESS]")
        ai_msg = AIMessage(content=response.content)

        return {
            "assessment": json.dumps(assessment, indent=2),
            "observations": [f"[Assess R{round_num}] {assessment.get('situation', '')}"],
            "messages": [label, ai_msg],
        }
    return assess

"""ASSESS node — situational awareness using shared context."""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.context import SHARED_SYSTEM_PROMPT, ContextBuilder
from src.state import ResearchState


def create_assess_node(llm: BaseChatModel):
    def assess(state: ResearchState) -> dict:
        user_prompt = ContextBuilder.build(state, "assess")
        response = llm.invoke([
            SystemMessage(content=SHARED_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        try:
            assessment = json.loads(response.content)
        except json.JSONDecodeError:
            assessment = {"situation": response.content, "priority": "unknown"}

        return {
            "assessment": json.dumps(assessment, indent=2),
            "observations": [f"[Assess R{state.get('round_number', 0)}] {assessment.get('situation', '')}"],
        }
    return assess

"""REFLECT node — result analysis with reasoning-chain history."""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.context import SHARED_SYSTEM_PROMPT, ContextBuilder
from src.state import HypothesisStatus, ResearchState


def create_reflect_node(llm: BaseChatModel):
    def reflect(state: ResearchState) -> dict:
        round_num = state.get("round_number", 0)
        user_prompt = ContextBuilder.build(state, "reflect")

        history = state.get("messages", [])
        response = llm.invoke([
            SystemMessage(content=SHARED_SYSTEM_PROMPT),
            *history,
            HumanMessage(content=user_prompt),
        ])
        try:
            reflection = json.loads(response.content)
        except json.JSONDecodeError:
            reflection = {
                "result_summary": "Could not parse",
                "expectation_met": False,
                "what_was_learned": "Parsing failed",
                "strategy_adjustment": "Continue",
            }

        updated_hypotheses = []
        for h in state.get("hypotheses", []):
            for update in reflection.get("hypothesis_updates", []):
                if update.get("id") == h.id:
                    new_status = update.get("new_status")
                    evidence = update.get("evidence", "")
                    if new_status == "confirmed":
                        h.confirm(evidence)
                    elif new_status == "rejected":
                        h.reject(evidence)
                    elif new_status == "testing":
                        h.status = HypothesisStatus.TESTING
            updated_hypotheses.append(h)

        label = HumanMessage(content=f"[Round {round_num} REFLECT]")
        ai_msg = AIMessage(content=response.content)

        return {
            "reflection": json.dumps(reflection, indent=2),
            "hypotheses": updated_hypotheses,
            "observations": [
                f"[Reflect R{round_num}] {reflection.get('what_was_learned', '')} "
                f"→ {reflection.get('strategy_adjustment', '')}"
            ],
            "messages": [label, ai_msg],
        }
    return reflect

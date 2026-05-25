"""LangChain callback for collecting per-node metrics (tokens, latency).

Hooks into LangChain's callback system to capture token usage and timing
for every LLM call, without modifying any node code.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from src.tracker.tracing import RunMetrics


class MetricsCallback(BaseCallbackHandler):
    """Collects token usage and timing for every LLM call.

    Usage:
        metrics = RunMetrics()
        callback = MetricsCallback(metrics)
        llm = ChatOpenAI(callbacks=[callback])
        # Or: graph.invoke(state, {"callbacks": [callback]})
    """

    def __init__(self, metrics: RunMetrics) -> None:
        super().__init__()
        self.metrics = metrics
        self._start_times: dict[UUID, float] = {}

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str],
        *, run_id: UUID, parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_times[run_id] = time.monotonic()

    def on_llm_end(
        self, response: LLMResult, *, run_id: UUID,
        parent_run_id: UUID | None = None, **kwargs: Any,
    ) -> None:
        start = self._start_times.pop(run_id, time.monotonic())
        latency_ms = (time.monotonic() - start) * 1000

        # Extract token usage from response
        prompt_tokens = 0
        completion_tokens = 0
        model_name = ""

        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            model_name = response.llm_output.get("model_name", "")

        # Fallback: try to get from generations' response_metadata
        if not model_name and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    meta = gen.generation_info or {}
                    if not model_name:
                        model_name = meta.get("model_name", "")
                    if not prompt_tokens:
                        usage = meta.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)

        self.metrics.record_node(
            node_name="llm",
            round_number=0,  # Will be updated by orchestrator
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID,
        parent_run_id: UUID | None = None, **kwargs: Any,
    ) -> None:
        start = self._start_times.pop(run_id, time.monotonic())
        latency_ms = (time.monotonic() - start) * 1000
        self.metrics.record_node(
            node_name="llm",
            round_number=0,
            success=False,
            latency_ms=latency_ms,
            extra={"error": str(error)[:200]},
        )

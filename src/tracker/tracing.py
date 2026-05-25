"""LangSmith integration for automatic LLM call tracing and observability.

LangSmith auto-traces all LangChain/LangGraph calls when configured.
This module provides setup, metadata tagging, and user feedback collection.

Requires:
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=...
    LANGCHAIN_PROJECT=auto-research-agent

Without these env vars, LangSmith is a no-op (graceful degradation).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any


def setup_langsmith(project: str = "auto-research-agent") -> bool:
    """Configure LangSmith tracing. Returns True if enabled.

    Call once at startup. LangGraph nodes are automatically traced.
    """
    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    if not api_key:
        return False

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
    return True


def langsmith_enabled() -> bool:
    return bool(os.getenv("LANGCHAIN_API_KEY")) and os.getenv("LANGCHAIN_TRACING_V2") == "true"


def get_run_url() -> str | None:
    """Return the LangSmith run URL if available (from the latest trace)."""
    if not langsmith_enabled():
        return None
    # LangSmith sets LANGSMITH_RUN_ID in the context; we can't easily get it from
    # outside the callback, so we construct the project URL instead.
    project = os.getenv("LANGCHAIN_PROJECT", "auto-research-agent")
    return f"https://smith.langchain.com/projects/{project}"


def collect_user_feedback(run_id: str | None = None, score: float | None = None,
                          comment: str = "") -> bool:
    """Submit user feedback for a LangSmith run.

    Args:
        run_id: The LangSmith run ID. If None, uses LANGSMITH_RUN_ID env var.
        score: 0.0-1.0 rating, or None to skip.
        comment: Free-text feedback.
    """
    if not langsmith_enabled():
        return False

    try:
        from langsmith import Client
        client = Client()
        rid = run_id or os.getenv("LANGSMITH_RUN_ID")
        if rid and score is not None:
            client.create_feedback(
                run_id=rid,
                key="user_score",
                score=score,
                comment=comment,
            )
            return True
    except ImportError:
        pass
    except Exception:
        pass
    return False


# ── Run Metrics (local, always available) ──────────────────────────────────

class RunMetrics:
    """Collects per-node metrics during a research run.

    This works without LangSmith and provides the key observability data
    the user asked about: model, tokens, latency, success/failure per step.
    """

    def __init__(self) -> None:
        self.start_time = datetime.now()
        self.node_metrics: list[dict[str, Any]] = []
        self.total_llm_calls = 0
        self.total_tool_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.failed_steps: list[str] = []
        self.model_name: str = ""

    def record_node(
        self,
        node_name: str,
        round_number: int,
        *,
        success: bool = True,
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record metrics for one node execution."""
        if model:
            self.model_name = model
        self.node_metrics.append({
            "node": node_name,
            "round": round_number,
            "success": success,
            "model": model or self.model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": round(latency_ms, 1),
            "timestamp": datetime.now().isoformat(),
            **(extra or {}),
        })
        if not success:
            self.failed_steps.append(f"Round {round_number} / {node_name}")
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        if node_name == "execute":
            self.total_tool_calls += 1
        else:
            self.total_llm_calls += 1

    def summary(self) -> str:
        """Generate a human-readable metrics summary."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        lines = [
            "Run Metrics Summary",
            "=" * 50,
            f"Duration: {elapsed:.1f}s",
            f"Model: {self.model_name or 'unknown'}",
            f"LLM calls: {self.total_llm_calls}",
            f"Tool calls: {self.total_tool_calls}",
            f"Total tokens: {total_tokens:,} ({self.total_prompt_tokens:,} prompt / {self.total_completion_tokens:,} completion)",
            f"Failed steps: {len(self.failed_steps)}",
        ]
        if self.failed_steps:
            lines.append(f"  Failures: {', '.join(self.failed_steps)}")
        if total_tokens > 0:
            # Rough cost estimate for DeepSeek: $0.001/1K input, $0.002/1K output
            ds_cost = (self.total_prompt_tokens / 1000 * 0.001 +
                       self.total_completion_tokens / 1000 * 0.002)
            lines.append(f"Est. cost (DeepSeek): ${ds_cost:.4f}")

        lines.append(f"\nNode breakdown:")
        for m in self.node_metrics:
            status = "✓" if m["success"] else "✗"
            tokens = f"{m['total_tokens']:,} tokens" if m["total_tokens"] else ""
            lines.append(
                f"  {status} R{m['round']} {m['node']:8s} "
                f"{m['latency_ms']:7.0f}ms {tokens}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": (datetime.now() - self.start_time).total_seconds(),
            "model": self.model_name,
            "llm_calls": self.total_llm_calls,
            "tool_calls": self.total_tool_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "failed_steps": self.failed_steps,
            "node_metrics": self.node_metrics,
        }

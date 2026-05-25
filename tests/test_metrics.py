"""Test RunMetrics collection and LangSmith setup."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_run_metrics():
    from src.tracker.tracing import RunMetrics

    m = RunMetrics()

    # Record some nodes
    m.record_node("assess", 1, model="deepseek-v4-pro",
                  prompt_tokens=500, completion_tokens=100, latency_ms=1200.5)
    m.record_node("plan", 1, model="deepseek-v4-pro",
                  prompt_tokens=300, completion_tokens=150, latency_ms=800.0)
    m.record_node("execute", 1, success=True, extra={"tool": "list_directory"})
    m.record_node("reflect", 1, model="deepseek-v4-pro",
                  prompt_tokens=400, completion_tokens=80, latency_ms=900.0)
    m.record_node("decide", 1, model="deepseek-v4-pro",
                  prompt_tokens=200, completion_tokens=50, latency_ms=500.0)
    m.record_node("execute", 2, success=False, extra={"tool": "execute_command"})

    # Summary
    summary = m.summary()
    assert "deepseek-v4-pro" in summary
    assert "LLM calls:" in summary
    assert "Tool calls:" in summary
    assert "Failed steps:" in summary

    # Token counts
    assert m.total_prompt_tokens == 1400
    assert m.total_completion_tokens == 380
    assert m.total_llm_calls == 4
    assert m.total_tool_calls == 2
    assert len(m.failed_steps) == 1

    # Dict export
    d = m.to_dict()
    assert d["llm_calls"] == 4
    assert d["tool_calls"] == 2
    assert d["total_tokens"] == 1780
    assert len(d["failed_steps"]) == 1

    # Cost estimate check
    assert "Est. cost" in summary

    print("✓ run metrics: all tests passed")


def test_langsmith_setup():
    from src.tracker.tracing import setup_langsmith, langsmith_enabled

    # Without API key, should be disabled
    old_key = os.environ.pop("LANGCHAIN_API_KEY", None)
    try:
        enabled = setup_langsmith()
        assert enabled is False
        assert langsmith_enabled() is False
    finally:
        if old_key:
            os.environ["LANGCHAIN_API_KEY"] = old_key

    print("✓ langsmith setup: all tests passed")


def test_metrics_callback():
    from unittest.mock import MagicMock
    from src.tracker.tracing import RunMetrics
    from src.tracker.callbacks import MetricsCallback

    m = RunMetrics()
    cb = MetricsCallback(m)

    # Simulate on_llm_end
    mock_response = MagicMock()
    mock_response.llm_output = {
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 50},
        "model_name": "deepseek-v4-pro",
    }
    mock_response.generations = []

    import uuid
    rid = uuid.uuid4()
    cb.on_llm_start({"name": "test"}, ["prompt"], run_id=rid)
    cb.on_llm_end(mock_response, run_id=rid)

    assert m.total_prompt_tokens == 100
    assert m.total_completion_tokens == 50
    assert len(m.node_metrics) == 1

    print("✓ metrics callback: all tests passed")


if __name__ == "__main__":
    test_run_metrics()
    test_langsmith_setup()
    test_metrics_callback()
    print("\n" + "=" * 50)
    print("Metrics + LangSmith: ALL TESTS PASSED")
    print("=" * 50)

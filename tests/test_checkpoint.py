"""Test checkpoint persistence and failure classification."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_checkpoint_save_load():
    from src.tracker.checkpoint import CheckpointManager
    from src.state import KnowledgeState, Hypothesis, HypothesisStatus, ExperimentRecord

    ws = Path(tempfile.mkdtemp())
    cm = CheckpointManager(ws)

    # Build a realistic state
    state = {
        "repo_url": "https://github.com/test/repo",
        "target_metrics": {"accuracy": 0.95},
        "knowledge": KnowledgeState(repo_url="https://github.com/test/repo"),
        "hypotheses": [
            Hypothesis(
                statement="Missing dependency",
                confidence=0.9,
                status=HypothesisStatus.CONFIRMED,
                verification_method="Run pip install",
            )
        ],
        "observations": ["Obs 1", "Obs 2"],
        "round_number": 3,
        "assessment": '{"situation":"test"}',
        "plan": ["do something"],
        "planned_action": "test action",
        "planned_tool": "read_file",
        "planned_tool_args": {"path": "test.txt"},
        "last_result": "file content",
        "reflection": '{"what_was_learned":"test"}',
        "decision": '{"decision":"continue"}',
        "verdict": "continue",
        "experiment_history": [
            ExperimentRecord(round_number=1, action="test", exit_code=0)
        ],
        "audit_log": [],
        "max_rounds": 5,
        "should_continue": True,
    }

    # Save
    cm.save(state)
    assert cm.exists()

    # Load
    loaded = cm.load()
    assert loaded is not None
    assert loaded["round_number"] == 3
    assert loaded["verdict"] == "continue"
    assert loaded["should_continue"] is True
    assert len(loaded["hypotheses"]) == 1
    assert loaded["hypotheses"][0].statement == "Missing dependency"
    assert len(loaded["experiment_history"]) == 1

    # Summary
    summary = cm.summary()
    assert "Round 3" in summary
    assert "1 experiments" in summary

    # Clear
    cm.clear()
    assert not cm.exists()

    print("✓ checkpoint save/load: all tests passed")


def test_failure_classification():
    from src.nodes.execute import classify_error, FailureType

    # Transient
    assert classify_error("Connection timed out", "execute_command") == FailureType.TRANSIENT
    assert classify_error("Rate limit exceeded. Try again later.", "web_search") == FailureType.TRANSIENT
    assert classify_error("HTTP 503 Service Unavailable", "execute_command") == FailureType.TRANSIENT
    assert classify_error("Tunnel connection failed: 403 Forbidden", "execute_command") == FailureType.TRANSIENT

    # Blocked
    assert classify_error("BLOCKED: Forbidden command", "execute_command") == FailureType.BLOCKED
    assert classify_error("Permission denied", "execute_command") == FailureType.BLOCKED

    # Permanent
    assert classify_error("ModuleNotFoundError: No module named 'torchvision'", "execute_command") == FailureType.PERMANENT
    assert classify_error("FileNotFoundError: [Errno 2] No such file or directory: 'data'", "read_file") == FailureType.PERMANENT
    assert classify_error("SyntaxError: invalid syntax", "execute_command") == FailureType.PERMANENT
    assert classify_error("Error: not a file: /etc/passwd", "read_file") == FailureType.PERMANENT

    # Unknown
    assert classify_error("Something weird happened", "execute_command") == FailureType.UNKNOWN

    print("✓ failure classification: all tests passed")


def test_human_tools():
    from src.tools.human_tools import create_human_help_tool

    # No callback — graceful degradation
    tools = create_human_help_tool(callback=None)
    tool_map = {t.name: t for t in tools}

    result = tool_map["request_human_help"].invoke({
        "question": "What should I do?",
        "context": "Tried A, B, C, all failed.",
    })
    assert "not available" in result.lower() or "continue" in result.lower()

    # With callback
    def test_cb(q, ctx):
        return f"Human says: try D"
    tools2 = create_human_help_tool(callback=test_cb)
    tool_map2 = {t.name: t for t in tools2}
    result2 = tool_map2["request_human_help"].invoke({
        "question": "What now?",
        "context": "Stuck.",
    })
    assert "Human says: try D" in result2

    # Report progress
    result3 = tool_map["report_progress"].invoke({
        "summary": "Fixed the dependency issue!",
    })
    assert "noted" in result3.lower() or "progress" in result3.lower()

    print("✓ human tools: all tests passed")


if __name__ == "__main__":
    test_checkpoint_save_load()
    test_failure_classification()
    test_human_tools()
    print("\n" + "=" * 50)
    print("Checkpoint + failure + human tools: ALL TESTS PASSED")
    print("=" * 50)

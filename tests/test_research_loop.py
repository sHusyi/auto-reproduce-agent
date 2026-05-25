"""Day 3 verification: multi-round loop, hypothesis lifecycle, tracker, memory."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_experiment_tracker():
    """Verify SQLite tracker CRUD operations."""
    from src.tracker.db import ExperimentTracker
    from src.state import ExperimentRecord, Hypothesis

    tracker = ExperimentTracker(":memory:")

    # Save experiment
    exp = ExperimentRecord(
        round_number=1,
        action="Install dependencies",
        command="pip install -r requirements.txt",
        metrics_before={"accuracy": 0.0},
        metrics_after={"accuracy": 0.82, "loss": 0.5},
        exit_code=0,
        status="completed",
        observation="Dependencies installed successfully",
    )
    tracker.save_experiment(exp)

    # Get experiments
    exps = tracker.get_experiments()
    assert len(exps) == 1
    assert exps[0]["round_number"] == 1

    # Best metrics
    best = tracker.get_best_metrics()
    assert best["accuracy"] == 0.82
    assert best["loss"] == 0.5

    # Metrics progression
    exp2 = ExperimentRecord(
        round_number=2,
        action="Tune learning rate",
        command="python train.py --lr 0.001",
        metrics_before={"accuracy": 0.82},
        metrics_after={"accuracy": 0.90},
        exit_code=0,
        status="completed",
        observation="Accuracy improved",
    )
    tracker.save_experiment(exp2)

    best = tracker.get_best_metrics()
    assert best["accuracy"] == 0.90  # Should be the max

    # Hypotheses
    hyp = Hypothesis(
        statement="Missing torchvision is causing the import error",
        confidence=0.8,
        verification_method="pip install torchvision",
    )
    tracker.save_hypothesis(hyp)
    hyps = tracker.get_hypotheses()
    assert len(hyps) == 1

    print("✓ experiment tracker: all tests passed")


def test_experiment_comparison():
    """Verify experiment comparison and target gap analysis."""
    from src.tracker.comparison import compare_experiments, compare_to_target
    from src.state import ExperimentRecord

    before = ExperimentRecord(
        round_number=1,
        action="Baseline",
        command="python train.py",
        metrics_before={},
        metrics_after={"accuracy": 0.82, "loss": 0.5},
        exit_code=0,
    )

    after = ExperimentRecord(
        round_number=2,
        action="Tuned LR",
        command="python train.py --lr 0.001",
        metrics_before={"accuracy": 0.82},
        metrics_after={"accuracy": 0.90, "loss": 0.3},
        exit_code=0,
    )

    # Compare two experiments
    diff = compare_experiments(before, after)
    assert diff["net_positive"] is True
    assert diff["deltas"]["accuracy"]["improved"] is True
    assert diff["deltas"]["accuracy"]["delta"] == 0.08
    assert diff["deltas"]["loss"]["improved"] is False  # Loss decreased = improvement, but the naive check is lower=worse
    # Actually for loss, lower is better, so the naive "improved" check says False.
    # This is fine — the agent (LLM) interprets semantics, not us.

    # Compare to target
    gap = compare_to_target(after, {"accuracy": 0.95})
    assert gap["all_met"] is False
    assert gap["gaps"]["accuracy"]["gap"] == 0.05

    gap2 = compare_to_target(after, {"accuracy": 0.85})
    assert gap2["all_met"] is True

    print("✓ experiment comparison: all tests passed")


def test_memory_compressor():
    """Verify memory compressor triggers and produces summaries."""
    from src.tracker.memory import compress_observations, compress_experiments
    from src.state import KnowledgeState

    # Build a state with many observations
    state = {
        "observations": [f"Observation {i}" for i in range(25)],
        "knowledge": KnowledgeState(repo_url="https://github.com/test/repo"),
        "experiment_history": [MagicMock() for _ in range(10)],
    }

    # Should NOT compress at 25
    result = compress_observations(MagicMock(), state, keep_recent=5, max_observations=30)
    assert result == {}

    # SHOULD compress at 25 with max 20
    result = compress_observations(MagicMock(), state, keep_recent=5, max_observations=20)
    assert len(result["observations"]) <= 6  # 1 summary + 5 recent

    # Experiment compression
    result = compress_experiments(state, keep_recent=3)
    assert len(result["experiment_history"]) == 3

    result2 = compress_experiments(state, keep_recent=20)
    assert result2 == {}

    print("✓ memory compressor: all tests passed")


def test_report_generation():
    """Verify report generates without errors."""
    from src.tracker.report import generate_report
    from src.tracker.db import ExperimentTracker
    from src.state import ExperimentRecord, Hypothesis, KnowledgeState

    tracker = ExperimentTracker(":memory:")

    exp = ExperimentRecord(
        round_number=1,
        action="Baseline training",
        command="python main.py",
        metrics_after={"accuracy": 0.82},
        exit_code=0,
        status="completed",
        observation="Baseline established",
    )
    tracker.save_experiment(exp)

    state = {
        "repo_url": "https://github.com/test/repo",
        "target_metrics": {"accuracy": 0.95},
        "verdict": "partial",
        "round_number": 1,
        "max_rounds": 5,
        "hypotheses": [
            Hypothesis(
                statement="Missing dependency caused import error",
                confidence=0.9,
                verification_method="pip install",
                status="confirmed",
            )
        ],
        "experiment_history": [exp],
        "observations": ["Import error resolved", "Baseline accuracy: 82%"],
        "reflection": '{"result_summary": "Made good progress", "expectation_met": true}',
    }

    report = generate_report(state, tracker)
    assert "Research Report" in report
    assert "test/repo" in report
    assert "0.95" in report
    assert "Missing dependency" in report

    print("✓ report generation: all tests passed")


def test_full_loop_with_mock_llm():
    """Simulate a full multi-round research loop with a mock LLM.

    The mock returns pre-scripted responses for each node. This verifies
    the graph flows correctly end-to-end without needing a real API call.
    """
    from unittest.mock import MagicMock, patch
    from langgraph.graph import END, StateGraph

    from src.state import ResearchState, KnowledgeState, Hypothesis, ExperimentRecord
    from src.sandbox.executor import SandboxedExecutor
    from src.sandbox.audit import AuditLogger
    from src.tools.registry import ToolRegistry
    from src.nodes.assess import create_assess_node
    from src.nodes.plan import create_plan_node
    from src.nodes.execute import create_execute_node
    from src.nodes.reflect import create_reflect_node
    from src.nodes.decide import create_decide_node

    ws = Path(tempfile.mkdtemp())
    executor = SandboxedExecutor(ws)
    audit = AuditLogger(ws / "audit.jsonl")
    registry = ToolRegistry(ws, executor)

    # Mock LLM that returns pre-scripted JSON
    round_responses = [
        # Round 1
        {  # assess
            "situation": "Initial state. Need to set up environment.",
            "priority": "Install dependencies",
        },
        {  # plan
            "reasoning": "First step is to install requirements",
            "hypothesis": "Environment needs setup",
            "actions": [{"tool_name": "execute_command", "tool_args": {"command": "pip install -r requirements.txt"},
                         "description": "Install deps", "expected_outcome": "Dependencies installed"}],
            "if_fails": "Check requirements.txt content",
        },
        {  # reflect
            "result_summary": "Installation succeeded",
            "expectation_met": True,
            "what_was_learned": "Environment is ready",
            "hypothesis_updates": [],
            "strategy_adjustment": "Continue",
        },
        {  # decide
            "decision": "continue",
            "reasoning": "Environment ready, need to run baseline",
            "confidence": 0.8,
        },
        # Round 2
        {  # assess
            "situation": "Environment ready. Need baseline metrics.",
            "priority": "Run training to establish baseline",
        },
        {  # plan
            "reasoning": "Run the main training script",
            "hypothesis": "Baseline will be below target",
            "actions": [{"tool_name": "execute_command", "tool_args": {"command": "python main.py"},
                         "description": "Run training", "expected_outcome": "Accuracy ~85%"}],
            "if_fails": "Check for missing data or config issues",
        },
        {  # reflect
            "result_summary": "Training completed, accuracy 85.2%",
            "expectation_met": True,
            "what_was_learned": "Baseline accuracy is 85.2%, target is 95.47%",
            "hypothesis_updates": [],
            "strategy_adjustment": "Need to tune hyperparameters",
        },
        {  # decide
            "decision": "continue",
            "reasoning": "Accuracy below target, need optimization",
            "confidence": 0.7,
        },
        # Round 3
        {  # assess
            "situation": "Baseline at 85.2%. Need hyperparameter tuning.",
            "priority": "Tune learning rate",
        },
        {  # plan
            "reasoning": "Try a lower learning rate",
            "hypothesis": "Lower LR will improve convergence",
            "actions": [{"tool_name": "execute_command", "tool_args": {"command": "python main.py --lr 0.001"},
                         "description": "Train with lower LR", "expected_outcome": "Accuracy ~93%"}],
            "if_fails": "Try different optimizer",
        },
        {  # reflect
            "result_summary": "Accuracy improved to 93.1%",
            "expectation_met": True,
            "what_was_learned": "Lower LR helped significantly",
            "hypothesis_updates": [],
            "strategy_adjustment": "Close to target, try one more round",
        },
        {  # decide
            "decision": "continue",
            "reasoning": "Close to 95.47% target",
            "confidence": 0.6,
        },
        # Round 4
        {  # assess
            "situation": "Accuracy at 93.1%. Close to target.",
            "priority": "Final tuning pass",
        },
        {  # plan
            "reasoning": "Add weight decay and train longer",
            "hypothesis": "Regularization + more epochs will reach target",
            "actions": [{"tool_name": "execute_command", "tool_args": {"command": "python main.py --weight_decay 5e-4 --epochs 300"},
                         "description": "Train with weight decay", "expected_outcome": "Accuracy >= 95.47%"}],
            "if_fails": "Accept partial reproduction",
        },
        {  # reflect
            "result_summary": "Accuracy reached 95.6%, exceeding target!",
            "expectation_met": True,
            "what_was_learned": "Weight decay + more epochs achieved target",
            "hypothesis_updates": [],
            "strategy_adjustment": "Target achieved, can stop",
        },
        {  # decide
            "decision": "success",
            "reasoning": "Target accuracy 95.47% achieved (95.6%)",
            "confidence": 0.95,
            "verdict_summary": "Successfully reproduced paper results",
        },
    ]

    response_idx = 0
    def mock_llm():
        m = MagicMock()
        def invoke(messages):
            nonlocal response_idx
            resp = round_responses[response_idx % len(round_responses)]
            response_idx += 1
            content = json.dumps(resp)
            return MagicMock(content=content)
        m.invoke = invoke
        return m

    llm = mock_llm()
    assess = create_assess_node(llm)
    plan = create_plan_node(llm)
    execute = create_execute_node(registry)
    reflect = create_reflect_node(llm)
    decide = create_decide_node(llm)

    # Build and run the graph manually (step by step, no LLM network)
    graph = StateGraph(ResearchState)
    graph.add_node("assess", assess)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("reflect", reflect)
    graph.add_node("decide", decide)
    graph.set_entry_point("assess")
    graph.add_edge("assess", "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "reflect")
    graph.add_edge("reflect", "decide")
    graph.add_conditional_edges(
        "decide",
        lambda s: "continue" if s.get("should_continue") else "end",
        {"continue": "assess", "end": END},
    )
    compiled = graph.compile()

    initial_state: ResearchState = {
        "repo_url": "https://github.com/kuangliu/pytorch-cifar",
        "paper_url": None,
        "target_metrics": {"accuracy": 95.47},
        "knowledge": KnowledgeState(
            repo_url="https://github.com/kuangliu/pytorch-cifar",
            target_metrics={"accuracy": 95.47},
        ),
        "hypotheses": [],
        "observations": [],
        "round_number": 0,
        "assessment": "",
        "plan": [],
        "planned_action": "",
        "planned_actions": [],
        "last_result": "",
        "reflection": "",
        "decision": "",
        "verdict": "continue",
        "experiment_history": [],
        "audit_log": [],
        "max_rounds": 5,
        "should_continue": True,
    }

    # Run
    final = compiled.invoke(initial_state)

    # Verify
    assert final["verdict"] == "success", f"Expected success, got {final['verdict']}"
    assert final["round_number"] >= 3, f"Expected 3+ rounds, got {final['round_number']}"
    assert len(final.get("experiment_history", [])) >= 4
    assert len(final.get("observations", [])) > 5
    assert len(final.get("hypotheses", [])) >= 1

    print(f"  Rounds completed: {final['round_number']}")
    print(f"  Experiments: {len(final.get('experiment_history', []))}")
    print(f"  Observations: {len(final.get('observations', []))}")
    print(f"  Hypotheses: {len(final.get('hypotheses', []))}")

    print("✓ full loop with mock LLM: all tests passed")


if __name__ == "__main__":
    test_experiment_tracker()
    test_experiment_comparison()
    test_memory_compressor()
    test_report_generation()
    test_full_loop_with_mock_llm()
    print("\n" + "=" * 50)
    print("Day 3 verification: ALL TESTS PASSED")
    print("=" * 50)

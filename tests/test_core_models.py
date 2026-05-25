"""Day 1 verification: test sandbox, state models, and LLM config."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_state_models():
    """Verify all Pydantic models can be instantiated."""
    from src.state import (
        Hypothesis,
        HypothesisStatus,
        ExperimentRecord,
        CommandResult,
        ParsedCommand,
        PermissionLevel,
        AuditEntry,
        KnowledgeState,
    )

    # Hypothesis lifecycle
    h = Hypothesis(
        statement="The issue is missing torchvision dependency",
        confidence=0.8,
        verification_method="pip install torchvision && python -c 'import torchvision'",
    )
    assert h.status == HypothesisStatus.PROPOSED
    h.confirm("torchvision installed successfully")
    assert h.status == HypothesisStatus.CONFIRMED

    h2 = Hypothesis(statement="Wrong hypothesis", confidence=0.3, verification_method="check")
    h2.reject("evidence suggests otherwise")
    assert h2.status == HypothesisStatus.REJECTED

    # ExperimentRecord
    exp = ExperimentRecord(
        round_number=1,
        action="Install dependencies",
        command="pip install -r requirements.txt",
        metrics_before={"accuracy": 0.0},
        metrics_after={"accuracy": 0.82},
        exit_code=0,
        observation="Installation succeeded",
    )
    assert exp.exit_code == 0

    # CommandResult
    ok = CommandResult(command="ls", exit_code=0, stdout="file1\nfile2")
    assert ok.success

    blocked = CommandResult(command="rm -rf /", blocked=True, block_reason="Forbidden")
    assert not blocked.success

    # KnowledgeState
    ks = KnowledgeState(
        repo_url="https://github.com/kuangliu/pytorch-cifar",
        target_metrics={"accuracy": 95.47},
    )
    assert ks.environment_ready is False

    print("✓ state models: all tests passed")


def test_permission_controller():
    """Verify permission classification."""
    from src.sandbox.permissions import PermissionController
    from src.state import PermissionLevel

    ws = Path(tempfile.mkdtemp())
    pc = PermissionController(ws)

    # SAFE commands
    level, _ = pc.check("ls -la")
    assert level == PermissionLevel.SAFE, f"Expected SAFE, got {level}"

    level, _ = pc.check("cat README.md")
    assert level == PermissionLevel.SAFE, f"Expected SAFE, got {level}"

    # WORKSPACE commands
    level, _ = pc.check("mkdir test_dir")
    assert level == PermissionLevel.WORKSPACE, f"Expected WORKSPACE, got {level}"

    level, _ = pc.check("python train.py")
    assert level == PermissionLevel.WORKSPACE, f"Expected WORKSPACE, got {level}"

    # SENSITIVE commands
    level, _ = pc.check("rm -rf checkpoints")
    assert level == PermissionLevel.SENSITIVE, f"Expected SENSITIVE, got {level}"

    level, _ = pc.check("pip install torch")
    assert level == PermissionLevel.SENSITIVE, f"Expected SENSITIVE, got {level}"

    # DANGEROUS commands (write outside workspace)
    level, _ = pc.check("echo data > /etc/hosts")
    assert level == PermissionLevel.DANGEROUS, f"Expected DANGEROUS, got {level}"

    # FORBIDDEN commands
    level, _ = pc.check("rm -rf /")
    assert level == PermissionLevel.FORBIDDEN, f"Expected FORBIDDEN, got {level}"

    level, _ = pc.check("rm -rf /*")
    assert level == PermissionLevel.FORBIDDEN, f"Expected FORBIDDEN, got {level}"

    level, _ = pc.check("shutdown now")
    assert level == PermissionLevel.FORBIDDEN, f"Expected FORBIDDEN, got {level}"

    # sudo detection
    level, parsed = pc.check("sudo ls")
    assert parsed.uses_sudo
    assert level == PermissionLevel.DANGEROUS

    # Path resolution
    level, parsed = pc.check("cat /etc/passwd")
    assert parsed.targets_system_paths

    print("✓ permission controller: all tests passed")


def test_sandbox_executor():
    """Verify sandbox execution."""
    from src.sandbox.executor import SandboxedExecutor
    from src.state import PermissionLevel

    ws = Path(tempfile.mkdtemp())
    ex = SandboxedExecutor(ws)

    # Safe command should execute
    result = ex.execute("echo hello")
    assert result.success, f"Expected success, got {result.summary}"
    assert "hello" in result.stdout

    # Safe command touching workspace
    result = ex.execute("pwd")
    assert result.success

    # Forbidden command should be blocked
    result = ex.execute("rm -rf /")
    assert result.blocked
    assert "Forbidden" in result.block_reason

    # Check command without execution
    level, explanation = ex.check_command("ls")
    assert level == PermissionLevel.SAFE

    level, explanation = ex.check_command("sudo rm -rf /")
    assert level == PermissionLevel.FORBIDDEN

    print("✓ sandbox executor: all tests passed")


def test_llm_config():
    """Verify LLM config loading."""
    import os
    from src.llm.config import LLMConfig

    # Explicit config
    config = LLMConfig(
        provider="deepseek",
        model="deepseek-chat",
        api_key="sk-test",
        base_url="https://api.deepseek.com/v1",
    )
    assert config.provider == "deepseek"
    assert config.temperature == 0.0

    # Factory import check
    from src.llm.factory import LLMFactory
    assert hasattr(LLMFactory, "create")

    print("✓ LLM config: all tests passed")


def test_audit_logger():
    """Verify audit logging."""
    from src.sandbox.audit import AuditLogger
    from src.state import AuditEntry, PermissionLevel

    log_path = Path(tempfile.mkdtemp()) / "audit.jsonl"
    logger = AuditLogger(log_path)

    entry = AuditEntry(
        agent_round=1,
        tool_name="execute_command",
        command_raw="pip install torch",
        permission_level=PermissionLevel.SENSITIVE,
        decision="auto_approved",
    )
    logger.log(entry)

    assert logger.total_actions == 1
    assert log_path.exists()

    # Read back
    with open(log_path) as f:
        content = f.read()
    assert "pip install torch" in content

    print("✓ audit logger: all tests passed")


if __name__ == "__main__":
    test_state_models()
    test_permission_controller()
    test_sandbox_executor()
    test_llm_config()
    test_audit_logger()
    print("\n" + "=" * 50)
    print("Day 1 verification: ALL TESTS PASSED")
    print("=" * 50)

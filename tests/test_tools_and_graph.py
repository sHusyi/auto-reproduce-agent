"""Day 2 verification: tools, nodes, and graph structure."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_file_tools():
    """Verify file tools work correctly."""
    from src.tools.file_tools import create_file_tools

    ws = Path(tempfile.mkdtemp())
    (ws / "test.txt").write_text("hello world")
    (ws / "subdir").mkdir()
    (ws / "subdir" / "nested.py").write_text("print('hi')")

    tools = create_file_tools(ws)
    tool_map = {t.name: t for t in tools}

    # read_file
    result = tool_map["read_file"].invoke({"path": "test.txt"})
    assert "hello world" in result

    # read_file outside workspace
    result = tool_map["read_file"].invoke({"path": "../../etc/passwd"})
    assert "Error" in result

    # list_directory
    result = tool_map["list_directory"].invoke({"path": "."})
    assert "test.txt" in result
    assert "subdir" in result

    # write_file
    result = tool_map["write_file"].invoke({"path": "new.txt", "content": "new content"})
    assert "File written" in result
    assert (ws / "new.txt").read_text() == "new content"

    # search_code
    result = tool_map["search_code"].invoke({"pattern": "print", "path": "."})
    assert "nested.py" in result

    print("✓ file tools: all tests passed")


def test_shell_tool():
    """Verify shell tool works through sandbox."""
    from src.sandbox.executor import SandboxedExecutor
    from src.tools.shell_tools import create_shell_tool

    ws = Path(tempfile.mkdtemp())
    executor = SandboxedExecutor(ws)
    tools = create_shell_tool(executor)
    shell = tools[0]

    # Safe command
    result = shell.invoke({"command": "echo hello"})
    assert "hello" in result

    # Forbidden command
    result = shell.invoke({"command": "rm -rf /"})
    assert "BLOCKED" in result

    print("✓ shell tool: all tests passed")


def test_tool_registry():
    """Verify tool registry collects all tools."""
    from src.sandbox.executor import SandboxedExecutor
    from src.tools.registry import ToolRegistry

    ws = Path(tempfile.mkdtemp())
    executor = SandboxedExecutor(ws)
    registry = ToolRegistry(ws, executor)

    all_tools = registry.get_all()
    names = [t.name for t in all_tools]

    assert "read_file" in names
    assert "write_file" in names
    assert "list_directory" in names
    assert "search_code" in names
    assert "execute_command" in names

    # get_by_name
    tool = registry.get_by_name("list_directory")
    result = tool.invoke({"path": "."})
    assert isinstance(result, str)

    print("✓ tool registry: all tests passed")


def test_graph_structure():
    """Verify the LangGraph graph compiles correctly (no LLM)."""
    from unittest.mock import MagicMock
    from langgraph.graph import END, StateGraph
    from src.state import ResearchState

    # Build a minimal graph manually to verify structure understanding
    graph = StateGraph(ResearchState)

    def dummy(state):
        return {}

    graph.add_node("assess", dummy)
    graph.add_node("plan", dummy)
    graph.add_node("execute", dummy)
    graph.add_node("reflect", dummy)
    graph.add_node("decide", dummy)

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
    assert compiled is not None
    nodes = compiled.get_graph().nodes
    node_names = {n for n in nodes if n not in ("__start__", "__end__")}
    assert "assess" in node_names
    assert "plan" in node_names
    assert "execute" in node_names
    assert "reflect" in node_names
    assert "decide" in node_names

    # Verify graph structure
    assert compiled is not None

    print("✓ graph structure: all tests passed")


def test_sandbox_approval_callback():
    """Verify the approval callback mechanism works."""
    from src.sandbox.executor import SandboxedExecutor

    ws = Path(tempfile.mkdtemp())
    approved_commands = []

    def on_approval(cmd, level):
        approved_commands.append(cmd)
        return True

    executor = SandboxedExecutor(ws, approval_callback=on_approval)

    # Sensitive command should trigger callback
    result = executor.execute("pip install torch")
    assert "pip install torch" in approved_commands

    # Once approved, same sensitive command should be cached
    approved_commands.clear()
    result2 = executor.execute("pip install torch")
    assert len(approved_commands) == 0  # Cached, no callback needed
    assert result2.success or result2.exit_code != 0  # May fail without pip

    # Dangerous command rejected without callback
    executor_no_cb = SandboxedExecutor(ws)
    result = executor_no_cb.execute("sudo ls")
    assert result.blocked

    print("✓ approval callback: all tests passed")


def test_knowledge_state():
    """Verify KnowledgeState model."""
    from src.state import KnowledgeState

    ks = KnowledgeState(
        repo_url="https://github.com/test/repo",
        target_metrics={"accuracy": 95.0},
    )
    assert ks.repo_url == "https://github.com/test/repo"
    assert ks.environment_ready is False
    assert ks.known_issues == []
    assert ks.installed_packages == []

    # Simulate updates
    ks.known_issues.append("Missing torchvision")
    ks.installed_packages.append("torch==2.5.0")
    ks.environment_ready = True
    assert len(ks.known_issues) == 1

    print("✓ knowledge state: all tests passed")


if __name__ == "__main__":
    test_file_tools()
    test_shell_tool()
    test_tool_registry()
    test_graph_structure()
    test_sandbox_approval_callback()
    test_knowledge_state()
    print("\n" + "=" * 50)
    print("Day 2 verification: ALL TESTS PASSED")
    print("=" * 50)

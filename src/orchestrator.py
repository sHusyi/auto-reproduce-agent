"""Orchestrator — LangGraph StateGraph wiring the research loop.

The graph flow:

    ASSESS → PLAN → EXECUTE → REFLECT → DECIDE
       ↑                                   │
       └─────── (if continue) ─────────────┘
                       │
                 (if stop) → END

Display is delegated to src.ui.orchestrator_display.OrchestratorDisplay.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from langgraph.graph import END, StateGraph

from src.llm.factory import LLMFactory
from src.llm.config import LLMConfig
from src.nodes.assess import create_assess_node
from src.nodes.plan import create_plan_node
from src.nodes.execute import create_execute_node
from src.nodes.reflect import create_reflect_node
from src.nodes.decide import create_decide_node
from src.nodes.clarify import create_clarify_node
from src.sandbox.audit import AuditLogger
from src.sandbox.executor import SandboxedExecutor
from src.state import ResearchState
from src.tools.registry import ToolRegistry
from src.tracker.checkpoint import CheckpointManager
from src.tracker.tracing import RunMetrics
from src.tracker.callbacks import MetricsCallback
from src.ui.orchestrator_display import OrchestratorDisplay
from src.ui.terminal import Terminal


def create_research_graph(
    llm_config: LLMConfig,
    sandbox: SandboxedExecutor,
    audit_logger: AuditLogger,
    *,
    max_rounds: int = 5,
    help_callback: Callable[[str, str], str] | None = None,
    callbacks: list | None = None,
) -> StateGraph:
    """Build the research loop state graph."""
    llm = LLMFactory.create(llm_config, callbacks=callbacks)
    registry = ToolRegistry(
        workspace_root=sandbox.workspace_root,
        sandbox=sandbox,
        help_callback=help_callback,
    )

    graph = StateGraph(ResearchState)
    graph.add_node("assess", create_assess_node(llm))
    graph.add_node("plan", create_plan_node(llm, tool_names=registry.tool_names))
    graph.add_node("execute", create_execute_node(registry))
    graph.add_node("reflect", create_reflect_node(llm))
    graph.add_node("decide", create_decide_node(llm))

    graph.set_entry_point("assess")
    graph.add_edge("assess", "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "reflect")
    graph.add_edge("reflect", "decide")
    graph.add_conditional_edges(
        "decide",
        lambda state: "continue" if state.get("should_continue", False) else "end",
        {"continue": "assess", "end": END},
    )

    return graph.compile()


class ResearchOrchestrator:
    """High-level orchestrator with streaming display, checkpoint, human help, and metrics."""

    def __init__(
        self,
        llm_config: LLMConfig,
        sandbox: SandboxedExecutor,
        audit_logger: AuditLogger,
        *,
        max_rounds: int = 10,
        help_callback: Callable[[str, str], str] | None = None,
        checkpoint: CheckpointManager | None = None,
        metrics: RunMetrics | None = None,
        display: OrchestratorDisplay | None = None,
    ) -> None:
        self.llm_config = llm_config
        self.sandbox = sandbox
        self.audit = audit_logger
        self.max_rounds = max_rounds
        self.help_callback = help_callback
        self.checkpoint = checkpoint or CheckpointManager(sandbox.workspace_root)
        self.metrics = metrics or RunMetrics()
        self.display = display or OrchestratorDisplay(Terminal())

        self._metrics_callback = MetricsCallback(self.metrics)
        self._callbacks = [self._metrics_callback]

        self.graph = create_research_graph(
            llm_config, sandbox, audit_logger,
            max_rounds=max_rounds, help_callback=help_callback,
            callbacks=self._callbacks,
        )
        self.metrics.model_name = llm_config.model

    def clarify_intent(
        self,
        repo_url: str,
        target_metrics: dict[str, float],
        *,
        paper_url: str | None = None,
    ) -> dict:
        """Pre-loop phase: explore repo, clarify goal, return intent."""
        llm = LLMFactory.create(self.llm_config, callbacks=self._callbacks)
        clarify = create_clarify_node(llm)
        exploration = self._explore_repo(repo_url)

        result = clarify({
            "repo_url": repo_url,
            "target_metrics": target_metrics,
            "paper_url": paper_url,
            "repo_exploration": exploration,
        })
        return {
            "intent": result.get("intent", {}),
            "repo_exploration": exploration,
            "observations": result.get("observations", []),
        }

    def _explore_repo(self, repo_url: str) -> str:
        ws = self.sandbox.workspace_root
        parts = []
        try:
            items = sorted(ws.iterdir())
            parts.append("Repository structure:")
            for p in items[:30]:
                prefix = "[DIR] " if p.is_dir() else "[FILE]"
                parts.append(f"  {prefix} {p.name}")
        except Exception as e:
            parts.append(f"(Could not list directory: {e})")

        for readme_name in ("README.md", "README.rst", "README.txt", "README"):
            readme_path = ws / readme_name
            if readme_path.exists():
                try:
                    parts.append(f"\n{readme_name}:\n{readme_path.read_text()[:5000]}")
                except Exception:
                    pass
                break

        req_path = ws / "requirements.txt"
        if req_path.exists():
            try:
                parts.append(f"\nrequirements.txt:\n{req_path.read_text()[:1000]}")
            except Exception:
                pass

        return "\n".join(parts)

    def run(
        self,
        repo_url: str,
        target_metrics: dict[str, float],
        *,
        paper_url: str | None = None,
        resume: bool = False,
        user_answers: dict[str, str] | None = None,
    ) -> tuple[dict, RunMetrics]:
        """Run the full research loop with streaming progress display."""
        initial_state: ResearchState
        if resume and self.checkpoint.exists():
            saved = self.checkpoint.load()
            if saved:
                self.display.term.status(
                    f"Resumed from checkpoint: {self.checkpoint.summary()}", "cyan")
                initial_state = saved
            else:
                initial_state = self._build_initial_state(repo_url, target_metrics, paper_url, user_answers)
        else:
            self.checkpoint.clear()
            initial_state = self._build_initial_state(repo_url, target_metrics, paper_url, user_answers)

        current_round = initial_state.get("round_number", 0)
        accumulated_state: dict = dict(initial_state)
        last_checkpoint_round = current_round

        try:
            for chunk in self.graph.stream(initial_state):
                for node_name, node_output in chunk.items():
                    accumulated_state.update(node_output)

                    round_num = accumulated_state.get("round_number", 0)
                    if round_num > current_round and node_name == "assess":
                        current_round = round_num

                    # Record metrics
                    if node_name == "execute":
                        result = node_output.get("last_result", "")
                        success = "Error" not in result and "FAILED" not in result[:20]
                        self.metrics.record_node(
                            node_name=node_name, round_number=current_round,
                            success=success,
                            extra={"tools": [a.get("tool_name", "") for a in accumulated_state.get("planned_actions", [])]},
                        )
                    elif node_name in ("assess", "plan", "reflect", "decide"):
                        if self.metrics.node_metrics:
                            last = self.metrics.node_metrics[-1]
                            if last["node"] == "llm":
                                last["node"] = node_name
                                last["round"] = current_round

                    # Display progress (delegated to UI module)
                    self.display.show_node(node_name, node_output, current_round)
                    sys.stdout.flush()

                    # Checkpoint after each DECIDE completes a round
                    if node_name == "decide" and round_num > last_checkpoint_round:
                        self.checkpoint.save(accumulated_state)
                        last_checkpoint_round = round_num
        except KeyboardInterrupt:
            self.checkpoint.save(accumulated_state)
            self.display.term.warning(
                f"Interrupted. Checkpoint saved at round {last_checkpoint_round}.")
            accumulated_state["verdict"] = "interrupted"
            return accumulated_state, self.metrics
        except Exception as e:
            self.checkpoint.save(accumulated_state)
            self.display.term.error(f"Error: {e}")
            self.display.term.warning(f"Checkpoint saved at round {last_checkpoint_round}.")
            raise

        return accumulated_state, self.metrics

    def _build_initial_state(
        self, repo_url: str, target_metrics: dict[str, float],
        paper_url: str | None, user_answers: dict[str, str] | None = None,
    ) -> ResearchState:
        from src.state import KnowledgeState

        # Lightweight startup environment detection (one-time, fast)
        env_hint = self._startup_detect()
        initial_observations: list[str] = []
        if env_hint:
            initial_observations.append(env_hint)
        if user_answers:
            initial_observations.append("[User Clarification] The user answered pre-loop questions:")
            for q, a in user_answers.items():
                initial_observations.append(f"  Q: {q}\n  A: {a}")

        return {
            "repo_url": repo_url,
            "paper_url": paper_url,
            "target_metrics": target_metrics,
            "knowledge": KnowledgeState(
                repo_url=repo_url, paper_url=paper_url, target_metrics=target_metrics,
            ),
            "hypotheses": [],
            "observations": initial_observations,
            "round_number": 0,
            "assessment": "", "plan": [],
            "planned_actions": [],
            "last_result": "", "reflection": "", "decision": "",
            "verdict": "continue",
            "experiment_history": [], "audit_log": [],
            "max_rounds": self.max_rounds,
            "should_continue": True,
        }

    def _startup_detect(self) -> str:
        """Run lightweight detection. Returns a hint observation for the agent."""
        import platform, subprocess, sys as _sys
        lines = ["[Startup] Environment quick scan:"]
        lines.append(f"  OS: {platform.system()} {platform.release()} ({platform.machine()})")
        lines.append(f"  Python: {_sys.version.split()[0]}")

        # Detect available package managers
        pm_found = []
        for name, cmd in [("uv", "uv --version 2>/dev/null"), ("pip", f"{_sys.executable} -m pip --version 2>/dev/null"), ("conda", "conda --version 2>/dev/null")]:
            try:
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    pm_found.append(name)
            except Exception:
                pass
        lines.append(f"  Package managers: {', '.join(pm_found) if pm_found else 'pip'}")

        # GPU check
        try:
            r = subprocess.run("nvidia-smi 2>/dev/null | head -1", shell=True, capture_output=True, text=True, timeout=5)
            if r.stdout.strip():
                lines.append(f"  GPU: {r.stdout.strip()[:100]}")
        except Exception:
            pass
        if _sys.platform == "darwin" and platform.machine() == "arm64":
            lines.append("  GPU: Apple Silicon (MPS)")

        lines.append("  Tip: use detect_environment() for full details before installing packages.")
        return "\n".join(lines)

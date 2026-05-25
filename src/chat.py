"""Chat-based interface for the Auto-Research Agent.

Natural language input → parse intent → clarify → confirm → research loop.

All terminal I/O goes through src.ui.terminal.Terminal — no Rich/ANSI inline.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure project root is on sys.path for 'from src.xxx' imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.ui.terminal import Terminal


def setup_workspace(repo_url: str) -> Path:
    """Clone or use a repo, return workspace path."""
    project_dir = Path(__file__).resolve().parent.parent
    workspaces_dir = project_dir / "workspaces"
    workspaces_dir.mkdir(exist_ok=True)

    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    ws = workspaces_dir / repo_name

    if repo_url.startswith("http://") or repo_url.startswith("https://"):
        if ws.exists():
            shutil.rmtree(ws)
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(ws)],
            check=True, capture_output=True, timeout=60,
        )
    return ws


def run_chat_loop(llm_config):
    """Main chat loop: parse → clone → clarify → confirm → loop."""
    term = Terminal()

    from src.llm.factory import LLMFactory
    from src.nodes.parse_request import create_parse_request_node

    llm = LLMFactory.create(llm_config)
    parse_request = create_parse_request_node(llm)

    while True:
        try:
            user_input = term.prompt("You:", style="green")
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "/q"):
                term.status("Goodbye!", "dim")
                break
            elif cmd == "/help":
                term.show_help()
                continue
            elif cmd == "/scenarios":
                from demo.scenarios import list_scenarios
                print(list_scenarios())
                continue
            else:
                term.error(f"Unknown command: {user_input}")
                continue

        # ── Phase 0: Parse ──
        term.status("🔍 Understanding your request...", "cyan")
        parsed = parse_request(user_input)

        term.show_parsed_request(parsed)

        # Check missing info
        missing = parsed.get("missing_info", [])
        if missing:
            for m in missing:
                term.warning(m)
            term.warning("Please provide the missing information and try again.")
            continue

        repo_url = parsed.get("repo_url", "")
        if not repo_url:
            term.error("I need a repository URL to proceed.")
            continue

        # ── Phase 1: Clone ──
        term.status(f"📦 Cloning {repo_url}...", "dim")
        try:
            workspace = setup_workspace(repo_url)
        except subprocess.CalledProcessError as e:
            term.error(f"Failed to clone: {e}")
            continue
        term.status(f"Workspace: {workspace}", "dim")

        # ── Phase 2: Setup orchestrator ──
        from src.sandbox.executor import SandboxedExecutor
        from src.sandbox.audit import AuditLogger
        from src.tracker.db import ExperimentTracker
        from src.tracker.checkpoint import CheckpointManager
        from src.tracker.tracing import RunMetrics, setup_langsmith
        from src.orchestrator import ResearchOrchestrator
        from src.ui.orchestrator_display import OrchestratorDisplay

        setup_langsmith()
        sandbox = SandboxedExecutor(workspace)
        audit = AuditLogger(workspace / "audit.jsonl")
        tracker = ExperimentTracker(workspace / "experiments.db")
        checkpoint = CheckpointManager(workspace)
        metrics = RunMetrics()

        def help_cb(question: str, context: str) -> str:
            if term.rich:
                term.show_panel(
                    context or "(no context)",
                    title="[yellow]🤔 Agent needs help[/yellow]",
                    style="yellow",
                )
                print(f"\033[1;33mQ: {question}\033[0m")
            else:
                print(f"\n{'='*40}")
                print(f"Agent needs help: {question}")
                if context:
                    print(f"Context: {context}")
            return term.prompt("Your answer:", style="yellow")

        orch = ResearchOrchestrator(
            llm_config=llm_config,
            sandbox=sandbox,
            audit_logger=audit,
            max_rounds=10,
            help_callback=help_cb,
            checkpoint=checkpoint,
            metrics=metrics,
            display=OrchestratorDisplay(term),
        )

        # ── Phase 3: Clarify ──
        term.status("📖 Analyzing repository...", "cyan")
        intent_result = orch.clarify_intent(
            repo_url=str(workspace),
            target_metrics=parsed.get("target_metrics", {}),
            paper_url=parsed.get("paper_url"),
        )
        intent = intent_result.get("intent", {})
        term.show_intent(intent)

        # ── Phase 3.5: Answer questions & confirm ──
        questions = intent.get("questions_for_user", [])
        user_answers = term.ask_questions(questions)

        if not term.confirm("Proceed with this plan?"):
            term.status("Cancelled. Ask me something else.", "dim")
            continue

        # ── Phase 4: Research loop ──
        term.status("\nStarting research...", "cyan")
        result, run_metrics = orch.run(
            repo_url=str(workspace),
            target_metrics=parsed.get("target_metrics", {}),
            paper_url=parsed.get("paper_url"),
            user_answers=user_answers if user_answers else None,
        )

        # ── Phase 5: Report ──
        term.show_verdict(result.get("verdict", "unknown"))
        term.show_metrics(run_metrics.summary())

        for exp in result.get("experiment_history", []):
            tracker.save_experiment(exp)
        for hyp in result.get("hypotheses", []):
            tracker.save_hypothesis(hyp)

        from src.tracker.report import generate_report
        import json
        report = generate_report(result, tracker)
        report_path = workspace / "research_report.md"
        report_path.write_text(report)
        metrics_path = workspace / "metrics.json"
        metrics_path.write_text(json.dumps(run_metrics.to_dict(), indent=2, ensure_ascii=False))

        term.status(f"Report: {report_path}", "dim")
        term.status(f"Metrics: {metrics_path}", "dim")
        term.status("I'm ready for your next request.", "dim")


def main():
    term = Terminal()
    term.show_banner()

    from src.llm.config import LLMConfig
    config = LLMConfig.from_dotenv()
    if not config.api_key:
        term.error("No API key configured. Set DEEPSEEK_API_KEY in .env")
        sys.exit(1)

    run_chat_loop(config)


if __name__ == "__main__":
    main()

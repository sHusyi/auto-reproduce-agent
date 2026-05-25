"""Auto-Research Agent — entry point.

Usage:
    python -m src.main --repo https://github.com/kuangliu/pytorch-cifar --target accuracy=95.47
    python -m src.main --scenario missing_dependency
    python -m src.main --scenario missing_dependency --resume
"""

from __future__ import annotations

import argparse
import readline  # noqa: F401 — enables cursor movement in input()
import sys
from pathlib import Path

# Add project root to path so demo module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

# ANSI escape codes
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_GREEN = "\033[32m"
ANSI_CYAN = "\033[36m"
ANSI_YELLOW = "\033[33m"
ANSI_RED = "\033[31m"
ANSI_RESET = "\033[0m"

try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    RICH = True
except ImportError:
    RICH = False
    console = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-Research Agent — autonomously reproduce ML paper results"
    )
    parser.add_argument("--repo", help="URL or path to the repository")
    parser.add_argument("--scenario", help="Demo scenario (missing_dependency, hardcoded_path, wrong_hyperparam)")
    parser.add_argument("--target", nargs="+", help="Target metrics key=value (e.g., accuracy=95.47)")
    parser.add_argument("--paper", help="Optional URL to the paper")
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum research rounds (default: 10)")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "openai", "anthropic"])
    parser.add_argument("--workspace", help="Workspace directory")
    parser.add_argument("--interactive", action="store_true", help="Enable human-in-the-loop (approval + help)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--list-scenarios", action="store_true", help="List available demo scenarios")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_scenarios:
        from demo.scenarios import list_scenarios
        print(list_scenarios())
        return

    # Determine repo, target, and workspace
    repo_url: str
    target_metrics: dict[str, float]
    project_dir = Path(__file__).resolve().parent.parent
    workspaces_dir = project_dir / "workspaces"
    workspaces_dir.mkdir(exist_ok=True)

    if args.scenario:
        from demo.scenarios import SCENARIOS, prepare_scenario
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            from demo.scenarios import list_scenarios
            print(list_scenarios())
            sys.exit(1)
        s = SCENARIOS[args.scenario]
        if RICH:
            console.print(Panel.fit(
                f"[bold]{s.name}[/bold] [{s.difficulty}]\n{s.description}",
                title="Scenario", border_style="cyan"
            ))
        else:
            print(f"Scenario: {s.name} [{s.difficulty}]\n{s.description}")
        workspace = workspaces_dir / args.scenario
        repo_url = str(prepare_scenario(args.scenario, base_dir=workspace))
        workspace = Path(repo_url)
        target_metrics = s.target_metrics

    elif args.repo and args.target:
        repo_url = args.repo
        target_metrics = {}
        for t in args.target:
            k, _, v = t.partition("=")
            target_metrics[k] = float(v)
        workspace = Path(args.workspace) if args.workspace else Path(repo_url)
    else:
        print("Specify --repo + --target, or --scenario")
        sys.exit(1)

    # LLM
    from src.llm.config import LLMConfig
    llm_config = LLMConfig.from_dotenv(provider=args.provider)
    if not llm_config.api_key:
        print(f"Error: No API key for {args.provider}")
        sys.exit(1)

    # Human-in-the-loop callbacks
    approval_cb = None
    help_cb = None

    if args.interactive:
        def approval_cb(cmd, level):
            msg = f"[{level.name}] {cmd[:80]}"
            if RICH:
                console.print(f"  [yellow]⚠[/yellow] {msg}")
            else:
                print(f"  ⚠ {msg}")
            return input(f"  {ANSI_BOLD}{ANSI_YELLOW}Approve?{ANSI_RESET} [y/N] ").strip().lower() == "y"

        def help_cb(question: str, context: str) -> str:
            """Blocking callback: agent asks human for help."""
            if RICH:
                console.print()
                console.print(Panel.fit(
                    context or "(no additional context)",
                    title=f"[bold yellow]🤔 Agent needs help[/bold yellow]",
                    border_style="yellow",
                ))
                console.print(f"[bold]Q:[/bold] {question}")
            else:
                print(f"\n{'='*50}")
                print(f"Agent needs help:")
                if context:
                    print(f"Context: {context}")
                print(f"Q: {question}")
            answer = input(f"{ANSI_BOLD}{ANSI_YELLOW}Your answer:{ANSI_RESET} ")
            return answer

    # LangSmith
    from src.tracker.tracing import setup_langsmith, langsmith_enabled, get_run_url
    ls_enabled = setup_langsmith()

    # Sandbox
    from src.sandbox.executor import SandboxedExecutor
    sandbox = SandboxedExecutor(workspace, approval_callback=approval_cb)

    # Audit + Tracker + Checkpoint + Metrics
    from src.sandbox.audit import AuditLogger
    from src.tracker.db import ExperimentTracker
    from src.tracker.checkpoint import CheckpointManager
    from src.tracker.tracing import RunMetrics
    audit = AuditLogger(workspace / "audit.jsonl")
    tracker = ExperimentTracker(workspace / "experiments.db")
    checkpoint = CheckpointManager(workspace)
    metrics = RunMetrics()

    # Checkpoint info
    if checkpoint.exists():
        summary = checkpoint.summary()
        if RICH:
            console.print(f"[cyan]📦 Checkpoint found: {summary}[/cyan]")
        else:
            print(f"Checkpoint found: {summary}")
        if not args.resume:
            if RICH:
                console.print("[dim]Use --resume to continue, or start fresh (checkpoint will be cleared)[/dim]")
            else:
                print("Use --resume to continue, or start fresh (checkpoint will be cleared)")

    # Orchestrator
    from src.orchestrator import ResearchOrchestrator
    orch = ResearchOrchestrator(
        llm_config=llm_config,
        sandbox=sandbox,
        audit_logger=audit,
        max_rounds=args.max_rounds,
        help_callback=help_cb,
        checkpoint=checkpoint,
        metrics=metrics,
    )

    # ── Intent Recognition (pre-loop) ──
    # Skip if resuming from checkpoint (intent was already confirmed)
    if not args.resume:
        if RICH:
            console.print("[cyan]🔍 Analyzing repository and clarifying goal...[/cyan]")
        else:
            print("Analyzing repository and clarifying goal...")

        intent_result = orch.clarify_intent(
            repo_url=repo_url,
            target_metrics=target_metrics,
            paper_url=args.paper,
        )
        intent = intent_result.get("intent", {})

        # Display intent
        if RICH:
            from rich.table import Table
            console.print()
            console.print(Panel.fit(
                intent.get("repo_summary", "Unknown")[:500],
                title="[bold green]📖 Repository Understanding[/bold green]",
                border_style="green",
            ))

            if intent.get("paper_claim"):
                console.print(f"[yellow]Paper claims:[/yellow] {intent['paper_claim']}")

            # Milestones table
            milestones = intent.get("milestones", [])
            if milestones:
                table = Table(title="🎯 Planned Milestones", border_style="blue")
                table.add_column("#", style="dim")
                table.add_column("Goal")
                table.add_column("Verification")
                for m in milestones:
                    table.add_row(str(m.get("step", "?")), m.get("goal", ""), m.get("verification", ""))
                console.print(table)

            # Success criteria
            criteria = intent.get("success_criteria", [])
            if criteria:
                console.print("[bold]Success criteria:[/bold]")
                for c in criteria:
                    console.print(f"  ✓ {c}")

            # Challenges
            challenges = intent.get("potential_challenges", [])
            if challenges:
                console.print("[yellow]Potential challenges:[/yellow]")
                for c in challenges:
                    console.print(f"  ⚠ {c}")

            # Questions
            questions = intent.get("questions_for_user", [])
            if questions:
                console.print("[bold red]Questions for you:[/bold red]")
                for q in questions:
                    console.print(f"  ❓ {q}")

            console.print()
        else:
            print(f"\nRepository: {intent.get('repo_summary', 'Unknown')[:300]}")
            print(f"Paper claims: {intent.get('paper_claim', 'Unknown')}")
            for m in intent.get("milestones", []):
                print(f"  Step {m.get('step')}: {m.get('goal')}")
            for q in intent.get("questions_for_user", []):
                print(f"  Q: {q}")

        # Confirmation (interactive mode)
        if args.interactive:
            import readline  # noqa: F401
            confirmed = input(f"\n{ANSI_BOLD}Proceed with this plan?{ANSI_RESET} {ANSI_DIM}[Y/n]{ANSI_RESET} ").strip().lower() != "n"
            if not RICH:
                resp = input("Proceed with this plan? [Y/n] ").strip().lower()
                confirmed = resp != "n"
            if not confirmed:
                print("Aborted by user.")
                sys.exit(0)

    if RICH:
        console.print(f"[dim]Repo: {repo_url}[/dim]")
        console.print(f"[dim]Target: {target_metrics}[/dim]")
        console.print(f"[dim]LLM: {llm_config.provider}/{llm_config.model}[/dim]")
        console.print(f"[dim]Max rounds: {args.max_rounds}[/dim]")
        console.print(f"[dim]Interactive: {args.interactive}[/dim]")
        console.print(f"[dim]Resume: {args.resume}[/dim]")
        console.print(f"[dim]LangSmith: {'enabled' if ls_enabled else 'disabled'}[/dim]\n")
    else:
        print(f"Repo: {repo_url}\nTarget: {target_metrics}\nLLM: {llm_config.provider}/{llm_config.model}\n")

    result, run_metrics = orch.run(
        repo_url=repo_url,
        target_metrics=target_metrics,
        paper_url=args.paper,
        resume=args.resume,
    )

    # Persist experiments and hypotheses
    for exp in result.get("experiment_history", []):
        tracker.save_experiment(exp)
    for hyp in result.get("hypotheses", []):
        tracker.save_hypothesis(hyp)

    # Report
    verdict = result.get("verdict", "unknown")
    if RICH:
        style = "green" if verdict == "success" else ("yellow" if verdict in ("partial", "interrupted") else "red")
        console.print(f"\n[bold {style}]Verdict: {verdict.upper()}[/bold {style}]")
    else:
        print(f"\nVerdict: {verdict}")

    print(f"Rounds: {len(result.get('experiment_history', []))}")

    # Metrics summary
    print(f"\n{run_metrics.summary()}")

    # LangSmith link
    if ls_enabled:
        url = get_run_url()
        if url:
            print(f"\nLangSmith trace: {url}")

    print(f"\n{tracker.summary()}")

    # User feedback (interactive mode)
    if args.interactive:
        score_str = input(f"{ANSI_BOLD}Rate the result (0-10, or Enter to skip):{ANSI_RESET} ")
        if score_str.strip():
            try:
                score = float(score_str) / 10.0
                from src.tracker.tracing import collect_user_feedback
                collect_user_feedback(score=score, comment=f"User rated {score_str}/10")
            except ValueError:
                pass

    # Save report
    from src.tracker.report import generate_report
    report = generate_report(result, tracker)
    report_path = workspace / "research_report.md"
    report_path.write_text(report)
    print(f"\nReport: {report_path}")

    # Save metrics JSON
    metrics_path = workspace / "metrics.json"
    import json
    metrics_path.write_text(json.dumps(run_metrics.to_dict(), indent=2, ensure_ascii=False))
    print(f"Metrics: {metrics_path}")

    # Clear checkpoint on successful completion
    if verdict in ("success", "failed") and checkpoint.exists():
        checkpoint.clear()
        print("Checkpoint cleared (run complete).")


if __name__ == "__main__":
    main()

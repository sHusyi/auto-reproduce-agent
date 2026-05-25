"""Demo runner — runs the agent against a prepared scenario.

Usage:
    python -m demo.run_demo --scenario missing_dependency
    python -m demo.run_demo --scenario hardcoded_path --interactive
    python -m demo.run_demo --list
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.llm.config import LLMConfig
from src.orchestrator import ResearchOrchestrator
from src.sandbox.audit import AuditLogger
from src.sandbox.executor import SandboxedExecutor
from src.tracker.report import generate_report
from src.tracker.db import ExperimentTracker
from demo.scenarios import SCENARIOS, list_scenarios, prepare_scenario


def main():
    parser = argparse.ArgumentParser(description="Auto-Research Agent Demo Runner")
    parser.add_argument("--scenario", help="Scenario name to run")
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    parser.add_argument("--interactive", action="store_true", help="Approve sensitive commands")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "openai", "anthropic"])
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--repo", help="Run against a specific repo path (skips scenario setup)")
    parser.add_argument("--target", nargs="+", help="Target metrics key=value (e.g., accuracy=95.47)")
    args = parser.parse_args()

    if args.list:
        print(list_scenarios())
        return

    # Determine repo and target
    if args.repo:
        repo_path = args.repo
        target = {}
        if args.target:
            for t in args.target:
                k, _, v = t.partition("=")
                target[k] = float(v)
        scenario = None
    elif args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            print(list_scenarios())
            sys.exit(1)
        scenario = SCENARIOS[args.scenario]
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario.name} [{scenario.difficulty}]")
        print(f"{'='*60}")
        print(f"Description: {scenario.description}")
        print(f"Issue: {scenario.issue_description}")
        print(f"Target: {scenario.target_metrics}")
        print(f"{'='*60}\n")

        repo_path = str(prepare_scenario(args.scenario))
        target = scenario.target_metrics
    else:
        print("Specify --scenario or --repo")
        print(list_scenarios())
        sys.exit(1)

    # LLM
    llm_config = LLMConfig.from_dotenv(provider=args.provider)
    if not llm_config.api_key:
        print(f"Error: No API key for {args.provider}")
        sys.exit(1)

    # Sandbox
    approval_cb = None
    if args.interactive:
        def approval_cb(cmd, level):
            print(f"\n  [{level.name}] {cmd}")
            resp = input("  Approve? [y/N] ").strip().lower()
            return resp == "y"
        approval_cb_ = approval_cb
    else:
        approval_cb_ = None

    sandbox = SandboxedExecutor(repo_path, approval_callback=approval_cb_)
    audit = AuditLogger(Path(repo_path) / "audit.jsonl")
    tracker = ExperimentTracker(Path(repo_path) / "experiments.db")

    # Orchestrator
    from src.tracker.tracing import RunMetrics
    metrics = RunMetrics()
    orch = ResearchOrchestrator(
        llm_config=llm_config,
        sandbox=sandbox,
        audit_logger=audit,
        max_rounds=args.max_rounds,
        metrics=metrics,
    )

    print(f"Running agent with {llm_config.provider}/{llm_config.model}...")
    print(f"Max rounds: {args.max_rounds}")
    print()

    result, run_metrics = orch.run(repo_url=repo_path, target_metrics=target)
    print(f"\n{run_metrics.summary()}")

    # Save experiments to tracker
    for exp in result.get("experiment_history", []):
        tracker.save_experiment(exp)
    for hyp in result.get("hypotheses", []):
        tracker.save_hypothesis(hyp)

    # Report
    print("\n" + "=" * 60)
    print("RESEARCH COMPLETE")
    print("=" * 60)
    print(f"Verdict: {result.get('verdict', 'unknown')}")
    print(f"Rounds: {len(result.get('experiment_history', []))}")
    print(f"\n{tracker.summary()}")

    # Save report
    report = generate_report(result, tracker)
    report_path = Path(repo_path) / "research_report.md"
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

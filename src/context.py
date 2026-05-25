"""Context builder — assembles user prompts with shared system prompt.

All LLM nodes share ONE system prompt (cache-friendly). Role-specific
instructions are short markers in the user prompt. All context formatting
is deterministic string operations — no LLM compression calls.
"""

from __future__ import annotations

from src.state import ResearchState

# ── Shared system prompt (cached, same for all nodes) ──────────────────────

SHARED_SYSTEM_PROMPT = """You are an Auto-Research Agent. Your task is to autonomously reproduce
ML paper results by exploring repositories, diagnosing issues, and iteratively
fixing problems.

You operate in a loop: ASSESS → PLAN → EXECUTE → REFLECT → DECIDE.

## Decision principles (follow these, but use your judgment)
- Before installing packages, check the environment first (use detect_environment).
- Never repeat an action you've already taken. Check the tool call history.
- When a command fails, read the FULL error output before diagnosing.
- Don't propose the same hypothesis multiple times — check existing hypotheses.
- If stuck on the same problem for 3+ rounds, try a fundamentally different approach
  or ask the human for help.
- Be specific: reference actual file names, error messages, and metrics.
- Adapt to the specific project — different repos need different approaches.
- Requirements.txt can install all dependencies at once with a single command."""

# ── Role instructions ──────────────────────────────────────────────────────

ROLE_INSTRUCTIONS = {
    "assess": """## Current Phase: ASSESS

Analyze the current state. What do we know? What problems remain?
What is the gap between current state and the target? What should we do next?

Output a JSON object:
{
    "situation": "One-sentence summary of current state",
    "what_we_know": ["fact 1", "fact 2"],
    "what_we_dont_know": ["question 1"],
    "problems_identified": ["problem 1"],
    "progress_toward_target": "assessment of how close we are",
    "priority": "The single most important thing to address next"
}""",

    "plan": """## Current Phase: PLAN

Based on the assessment, plan the next action(s). You can batch multiple
independent actions together (e.g., list directory + read README + read
requirements.txt in one round). Don't over-batch — actions that depend
on previous results should be separate rounds.

{tool_list}

Output a JSON object:
{{
    "reasoning": "Why these actions",
    "hypothesis": "What I believe (null if none — check existing hypotheses first!)",
    "actions": [
        {{
            "tool_name": "one of the available tools",
            "tool_args": {{"arg": "value"}},
            "description": "What this action does",
            "expected_outcome": "What I expect"
        }}
    ],
    "if_fails": "What I'll try if these actions fail"
}}""",

    "reflect": """## Current Phase: REFLECT

Analyze the results of the last actions. Compare them to expectations.
Update your understanding. Should we continue the current strategy or pivot?

Output a JSON object:
{
    "result_summary": "One-sentence summary of what happened",
    "expectation_met": true/false,
    "what_was_learned": "Key insight gained",
    "surprise": "Anything unexpected (null if nothing)",
    "hypothesis_updates": [
        {"id": "hypothesis-id", "new_status": "confirmed|rejected|testing", "evidence": "why"}
    ],
    "beliefs_changed": "How has our understanding changed?",
    "strategy_adjustment": "continue current strategy or pivot? Why?"
}""",

    "decide": """## Current Phase: DECIDE

Decide whether to continue or stop. Be honest — a well-reasoned "failed"
is far more valuable than wasting rounds on hopeless attempts.

Decision criteria:
- "continue": Clear next step, making progress, issues are addressable
- "success": Target metrics met or exceeded → STOP
- "partial": Some progress but hit round limit → STOP
- "failed": Cannot proceed due to a blocker → STOP with explanation

When to choose "failed":
- Same error persists 3+ rounds despite different attempts
- Missing hardware requirement (GPU needed but not available)
- Repository has unfixable code bugs (syntax errors, missing critical files)
- Dependency conflicts that cannot be resolved
- The repo simply does not work as described

IMPORTANT: "failed" is NOT a failure of the agent. It is a valid, useful
conclusion. The user would rather hear "this can't work because X" than
watch you loop on the same problem.

## Stall indicators
{stall_section}

Output a JSON object:
{{
    "decision": "continue|success|partial|failed",
    "reasoning": "Why this decision — be specific about blockers if failing",
    "next_strategy": "What to focus on (if continuing)",
    "confidence": 0.0-1.0,
    "blockers": ["Specific reason(s) you cannot proceed (if failing)"]
}}""",
}


# ── Context builder ────────────────────────────────────────────────────────

TOKEN_BUDGET = 60000  # Soft cap for user prompt tokens (chars ≈ tokens for English)

class ContextBuilder:
    """Assembles the user prompt for each node from the shared ResearchState.

    All methods are deterministic — no LLM calls for compression.
    """

    @staticmethod
    def build(state: ResearchState, role: str, tool_names: list[str] | None = None) -> str:
        """Build the full user prompt for a given role.

        Args:
            state: Current research state.
            role: One of "assess", "plan", "reflect", "decide".
            tool_names: List of available tool names (for PLAN role).
        """
        parts: list[str] = []

        # 1. Role instruction
        role_text = ROLE_INSTRUCTIONS.get(role, f"## Current Phase: {role.upper()}")
        if role == "plan" and tool_names:
            role_text = role_text.format(tool_list="Available tools:\n- " + "\n- ".join(tool_names))
        elif role == "plan":
            role_text = role_text.format(tool_list="Available tools: (use the tools you have)")
        elif role == "decide":
            stall = ContextBuilder._stall_detection(state)
            role_text = role_text.format(stall_section=stall)
        parts.append(role_text)

        # 2. Task info
        target = state.get("target_metrics", {})
        target_str = ", ".join(f"{k}={v}" for k, v in target.items()) if target else "extract from README"
        parts.append(
            f"## Task\n"
            f"Repo: {state.get('repo_url', '')}\n"
            f"Target metrics: {target_str}\n"
            f"Round: {state.get('round_number', 0)}/{state.get('max_rounds', 5)}"
        )

        # 3. Knowledge state
        parts.append(ContextBuilder._knowledge(state))

        # 4. Hypotheses (deduplicated, by status)
        parts.append(ContextBuilder._hypotheses(state))

        # 5. Tool call history (recent full, old as summaries)
        parts.append(ContextBuilder._tool_history(state))

        # 6. Observations (all, but deduplicated)
        parts.append(ContextBuilder._observations(state))

        # 7. For REFLECT: full last_result for diagnosis
        if role == "reflect":
            last = state.get("last_result", "")
            if last:
                parts.append(f"## Last Action Result (FULL — use this to diagnose)\n{last[:5000]}")

        prompt = "\n\n".join(parts)

        # Hard truncation if over budget (trim oldest tool summaries first)
        if len(prompt) > TOKEN_BUDGET:
            prompt = prompt[:TOKEN_BUDGET] + "\n\n[... context truncated at token budget]"

        return prompt

    @staticmethod
    def _knowledge(state: ResearchState) -> str:
        k = state.get("knowledge")
        if not k:
            return "## Environment\n(not yet explored)"

        lines = ["## Environment"]
        lines.append(f"Ready: {k.environment_ready}")
        if k.installed_packages:
            lines.append(f"Packages: {', '.join(k.installed_packages[-10:])}")
        if k.key_files:
            lines.append(f"Key files: {', '.join(k.key_files)}")
        if k.known_issues:
            lines.append(f"Known issues: {', '.join(k.known_issues)}")
        if k.resolved_issues:
            lines.append(f"Resolved: {', '.join(k.resolved_issues)}")
        return "\n".join(lines)

    @staticmethod
    def _hypotheses(state: ResearchState) -> str:
        all_hypotheses = state.get("hypotheses", [])
        if not all_hypotheses:
            return "## Hypotheses\n(none yet)"

        # Deduplicate by statement similarity (exact match first, then prefix)
        seen: set[str] = set()
        unique: list = []
        for h in all_hypotheses:
            key = h.statement.strip().lower()[:80]
            if key not in seen:
                seen.add(key)
                unique.append(h)

        confirmed = [h for h in unique if hasattr(h, 'status') and h.status.value == "confirmed"]
        rejected = [h for h in unique if hasattr(h, 'status') and h.status.value == "rejected"]
        active = [h for h in unique if hasattr(h, 'status') and h.status.value not in ("confirmed", "rejected")]

        lines = ["## Hypotheses"]
        if confirmed:
            lines.append(f"✓ Confirmed ({len(confirmed)}):")
            for h in confirmed[-5:]:
                lines.append(f"  - {h.statement[:120]}")
        if rejected:
            lines.append(f"✗ Rejected ({len(rejected)}):")
            for h in rejected[-5:]:
                lines.append(f"  - {h.statement[:120]}")
        if active:
            lines.append(f"○ Active ({len(active)}):")
            for h in active[-8:]:
                lines.append(f"  - [{h.confidence:.0%}] {h.statement[:120]}")

        return "\n".join(lines)

    @staticmethod
    def _tool_history(state: ResearchState) -> str:
        experiments = state.get("experiment_history", [])
        if not experiments:
            return "## Tool Call History\n(none yet)"

        lines = ["## Tool Call History"]
        recent_count = 5

        # Recent: full detail
        recent = experiments[-recent_count:]
        for exp in recent:
            cmd = exp.command or exp.action
            status_icon = "✓" if exp.status.value == "completed" else "✗"
            obs = exp.observation[:120] if exp.observation else ""
            lines.append(f"  R{exp.round_number}: {status_icon} {cmd[:80]}")
            if obs:
                lines.append(f"       → {obs}")

        # Older: one-line summaries
        older = experiments[:-recent_count]
        if older:
            lines.append(f"  --- ({len(older)} earlier calls) ---")
            for exp in older:
                cmd = (exp.command or exp.action)[:60]
                status_icon = "✓" if exp.status.value == "completed" else "✗"
                lines.append(f"  R{exp.round_number}: {status_icon} {cmd}")

        return "\n".join(lines)

    @staticmethod
    def _observations(state: ResearchState) -> str:
        observations = state.get("observations", [])
        if not observations:
            return "## Observations\n(none yet)"

        # Keep all, but deduplicate consecutive similar ones
        deduped: list[str] = []
        for obs in observations:
            key = obs[:80].strip().lower()
            if not deduped or key != deduped[-1][:80].strip().lower():
                deduped.append(obs)

        lines = ["## Observations"]
        for obs in deduped:
            lines.append(f"  - {obs[:200]}")

        return "\n".join(lines)

    @staticmethod
    def _stall_detection(state: ResearchState) -> str:
        """Detect if the agent is stuck. Returns warnings or 'No stall detected.'."""
        experiments = state.get("experiment_history", [])
        if len(experiments) < 3:
            return "No stall detected."

        recent = experiments[-6:]

        # Consecutive failures
        consecutive_failures = 0
        for exp in reversed(recent):
            if exp.status.value == "failed":
                consecutive_failures += 1
            else:
                break

        # Repeated same action
        tool_counts: dict[str, int] = {}
        for exp in recent:
            key = exp.action[:60]
            tool_counts[key] = tool_counts.get(key, 0) + 1
        most_repeated = max(tool_counts.values()) if tool_counts else 0

        warnings = []
        if consecutive_failures >= 3:
            warnings.append(
                f"⚠️ {consecutive_failures} consecutive failures. "
                "Strongly consider \"failed\" — explain the blocker and stop."
            )
        elif consecutive_failures >= 2:
            warnings.append(
                f"⚠️ {consecutive_failures} consecutive failures. "
                "If next attempt also fails, stop."
            )

        if most_repeated >= 4:
            warnings.append(
                f"⚠️ Same action repeated {most_repeated}x. You may be looping. "
                "Try a different approach or stop."
            )

        last_actions = [exp.action[:30].lower() for exp in recent[-3:]]
        if all("read" in a or "list" in a for a in last_actions):
            warnings.append(
                "⚠️ Last 3 rounds were read-only. Stop exploring, act or conclude."
            )

        if not warnings:
            return "No stall detected — progress is being made."
        return "\n".join(warnings)

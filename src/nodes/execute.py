"""EXECUTE node — runs the planned tool call with retry and failure classification.

Failures are classified as:
- TRANSIENT: network issues, timeouts — retry with backoff
- PERMANENT: syntax errors, missing files — report immediately, don't retry
- BLOCKED: sandbox rejected the command — report and skip

This gives the LLM structured information about what went wrong, so the
REFLECT node can make better decisions about whether to retry or pivot.
"""

from __future__ import annotations

import time
from datetime import datetime
from enum import Enum

from src.state import AuditEntry, ExperimentRecord, PermissionLevel, ResearchState
from src.tools.registry import ToolRegistry


class FailureType(str, Enum):
    TRANSIENT = "transient"    # Network timeout, temporary unavailability
    PERMANENT = "permanent"    # Syntax error, missing file, wrong arguments
    BLOCKED = "blocked"        # Sandbox permission denied
    UNKNOWN = "unknown"        # Unexpected error


MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 2.0


def classify_error(error_str: str, tool_name: str) -> FailureType:
    """Classify an error string into a failure type.

    Heuristics based on common error patterns. The agent's REFLECT node
    also does its own semantic analysis; this provides a structured hint.
    """
    error_lower = error_str.lower()

    # Transient patterns
    transient_patterns = [
        "timeout", "timed out", "connection", "network",
        "rate limit", "too many requests", "temporarily unavailable",
        "try again", "retry", "503", "502", "429", "connection refused",
    ]
    for pattern in transient_patterns:
        if pattern in error_lower:
            return FailureType.TRANSIENT

    # Blocked patterns
    blocked_patterns = ["blocked", "forbidden", "permission denied"]
    for pattern in blocked_patterns:
        if pattern in error_lower:
            return FailureType.BLOCKED

    # Permanent patterns
    permanent_patterns = [
        "syntaxerror", "modulenotfounderror", "filenotfounderror",
        "no such file", "not found", "cannot find", "invalid",
        "error: ", "traceback", "attributeerror", "typeerror",
        "valueerror", "keyerror", "indexerror",
    ]
    for pattern in permanent_patterns:
        if pattern in error_lower:
            return FailureType.PERMANENT

    # Shell non-zero exit codes with specific stderr
    if "exit code" in error_lower and error_lower.count("exit code") == 1:
        return FailureType.PERMANENT

    return FailureType.UNKNOWN


def create_execute_node(registry: ToolRegistry):
    """Create the EXECUTE node with retry and failure classification."""

    def execute(state: ResearchState) -> dict:
        planned_actions = state.get("planned_actions", [])
        round_number = state.get("round_number", 0)

        if not planned_actions:
            return {"observations": [f"[Execute R{round_number}] No actions planned"]}

        experiments: list[ExperimentRecord] = []
        audits: list[AuditEntry] = []
        observations: list[str] = []
        all_results: list[str] = []
        knowledge = state.get("knowledge")

        for action in planned_actions:
            tool_name = action.get("tool_name", "list_directory")
            tool_args = action.get("tool_args", {"path": "."})
            action_desc = action.get("description", tool_name)
            result_str = ""
            failure_type = FailureType.UNKNOWN
            retries = 0

            # Find and invoke tool
            try:
                tool = registry.get_by_name(tool_name)
            except KeyError:
                result_str = f"Tool not found: {tool_name}"
                failure_type = FailureType.PERMANENT

            if failure_type == FailureType.UNKNOWN:
                for attempt in range(MAX_RETRIES + 1):
                    try:
                        result = tool.invoke(tool_args)
                        result_str = str(result)
                        failure_type = FailureType.UNKNOWN
                        break
                    except Exception as e:
                        result_str = f"Error: {e}"
                        failure_type = classify_error(result_str, tool_name)
                        if failure_type == FailureType.TRANSIENT and attempt < MAX_RETRIES:
                            retries += 1
                            time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
                            continue
                        break

            if retries:
                structured_result = f"[Retried {retries}x]\n{result_str}"
            else:
                structured_result = result_str

            success = failure_type == FailureType.UNKNOWN and "Error" not in result_str
            all_results.append(structured_result)

            # Command tracking
            command_raw = tool_args.get("command", "") if tool_name == "execute_command" else ""

            experiments.append(ExperimentRecord(
                round_number=round_number, action=action_desc, command=command_raw,
                observation=structured_result[:500],
                status="completed" if success else "failed",
                started_at=datetime.now(), completed_at=datetime.now(),
            ))
            audits.append(AuditEntry(
                agent_round=round_number, tool_name=tool_name,
                command_raw=command_raw or f"{tool_name}({tool_args})",
                permission_level=PermissionLevel.SAFE, decision="auto_approved",
            ))

            # Observation per action
            obs_parts = [f"[Execute R{round_number}]"]
            if tool_name == "execute_command":
                obs_parts.append(f"Ran: {command_raw[:100]}")
            else:
                obs_parts.append(f"{tool_name}({str(tool_args)[:80]})")
            if success:
                obs_parts.append(f"→ OK: {result_str[:150].replace(chr(10), ' ').strip()}")
            else:
                obs_parts.append("→ FAILED")
                if retries:
                    obs_parts.append(f"(retried {retries}x)")
                obs_parts.append(f"— {' '.join(result_str.split(chr(10))[-3:])[:200]}")
            observations.append(" ".join(obs_parts))

            # Track files read
            if knowledge and success and tool_name == "read_file" and "path" in tool_args:
                path = tool_args["path"]
                if path not in knowledge.key_files:
                    knowledge.key_files.append(path)

        return {
            "last_result": "\n\n".join(all_results),
            "experiment_history": experiments,
            "audit_log": audits,
            "observations": observations,
            "knowledge": knowledge,
        }

    return execute

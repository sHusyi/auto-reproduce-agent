"""Core data models for the Auto-Research Agent."""

from __future__ import annotations

import operator
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator


# ─── ID types ──────────────────────────────────────────────────────────────────

def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ─── Hypothesis ─────────────────────────────────────────────────────────────────

class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    TESTING = "testing"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class Hypothesis(BaseModel):
    """A structured hypothesis about what's wrong or what should be tried."""
    id: str = Field(default_factory=_new_id)
    statement: str = Field(..., description="What I believe is happening")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="0.0-1.0 confidence")
    verification_method: str = Field(..., description="How to test this hypothesis")
    status: HypothesisStatus = Field(default=HypothesisStatus.PROPOSED)
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    resolved_at: datetime | None = None

    def confirm(self, evidence: str = "") -> None:
        self.status = HypothesisStatus.CONFIRMED
        if evidence:
            self.evidence_for.append(evidence)
        self.resolved_at = datetime.now()

    def reject(self, evidence: str = "") -> None:
        self.status = HypothesisStatus.REJECTED
        if evidence:
            self.evidence_against.append(evidence)
        self.resolved_at = datetime.now()


# ─── Experiment ─────────────────────────────────────────────────────────────────

class ExperimentStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentRecord(BaseModel):
    """A single experiment run with its configuration and results."""
    id: str = Field(default_factory=_new_id)
    round_number: int
    hypothesis_id: str | None = None
    action: str = Field(..., description="What was done")
    command: str | None = Field(default=None, description="Shell command executed")
    metrics_before: dict[str, float] = Field(default_factory=dict)
    metrics_after: dict[str, float] = Field(default_factory=dict)
    stdout_preview: str = ""
    stderr_preview: str = ""
    exit_code: int | None = None
    status: ExperimentStatus = Field(default=ExperimentStatus.PLANNED)
    files_changed: list[str] = Field(default_factory=list)
    observation: str = Field(default="", description="Key observation from this experiment")
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None


# ─── Command Result ─────────────────────────────────────────────────────────────

class CommandResult(BaseModel):
    """Result of a sandboxed shell command execution."""
    command: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    blocked: bool = False
    block_reason: str = ""
    timed_out: bool = False
    execution_time_ms: float = 0.0

    @property
    def success(self) -> bool:
        return not self.blocked and self.exit_code == 0

    @property
    def summary(self) -> str:
        if self.blocked:
            return f"BLOCKED: {self.block_reason}"
        if self.timed_out:
            return "TIMED OUT"
        status = "OK" if self.exit_code == 0 else f"EXIT {self.exit_code}"
        preview = self.stdout[-200:] if self.stdout else self.stderr[-200:]
        return f"[{status}] {preview}"


# ─── Permission ─────────────────────────────────────────────────────────────────

class PermissionLevel(int, Enum):
    SAFE = 0         # read-only, auto-approve
    WORKSPACE = 1    # write within workspace, auto-approve
    SENSITIVE = 2    # pip install, rm, wget — confirm once per session
    DANGEROUS = 3    # sudo, write outside workspace — always ask
    FORBIDDEN = 4    # rm -rf /*, write /etc — never allow


class ParsedCommand(BaseModel):
    """Structured representation of a shell command after parsing."""
    executable: str
    args: list[str] = Field(default_factory=list)
    raw: str
    resolved_paths: list[Path] = Field(default_factory=list)
    writes_outside_workspace: bool = False
    uses_sudo: bool = False
    targets_system_paths: bool = False
    has_redirect: bool = False
    has_pipe: bool = False


# ─── Audit ──────────────────────────────────────────────────────────────────────

class AuditEntry(BaseModel):
    """A record of every action the agent takes, for traceability."""
    timestamp: datetime = Field(default_factory=datetime.now)
    agent_round: int
    tool_name: str
    command_raw: str
    command_parsed: ParsedCommand | None = None
    permission_level: PermissionLevel
    decision: Literal["auto_approved", "human_approved", "rejected"]
    result: CommandResult | None = None
    files_changed: list[str] = Field(default_factory=list)


# ─── Knowledge State ────────────────────────────────────────────────────────────

class KnowledgeState(BaseModel):
    """The agent's accumulated understanding of the environment."""
    repo_url: str = ""
    repo_path: str = ""
    paper_url: str | None = None
    target_metrics: dict[str, float] = Field(default_factory=dict)
    environment_ready: bool = False
    installed_packages: list[str] = Field(default_factory=list)
    known_issues: list[str] = Field(default_factory=list)
    resolved_issues: list[str] = Field(default_factory=list)
    key_files: list[str] = Field(default_factory=list)
    notes: str = ""


# ─── Research State (LangGraph) ─────────────────────────────────────────────────

class ResearchState(TypedDict, total=False):
    """The shared state that flows through the LangGraph research loop."""

    # Task
    repo_url: str
    paper_url: str | None
    target_metrics: dict[str, float]

    # Intent (pre-loop goal clarification)
    repo_exploration: str        # Raw output from reading repo (README, structure)
    intent: dict[str, Any]       # Structured goal understanding from CLARIFY
    intent_confirmed: bool       # User confirmed the intent

    # Knowledge
    knowledge: KnowledgeState
    hypotheses: Annotated[list[Hypothesis], operator.add]
    observations: Annotated[list[str], operator.add]

    # Current round
    round_number: int
    assessment: str
    plan: list[str]
    planned_actions: list[dict[str, Any]]
    last_result: str
    reflection: str
    decision: str
    verdict: str  # "continue" | "success" | "partial" | "failed"

    # History
    experiment_history: Annotated[list[ExperimentRecord], operator.add]
    audit_log: Annotated[list[AuditEntry], operator.add]

    # Limits
    max_rounds: int
    should_continue: bool

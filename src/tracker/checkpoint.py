"""State checkpoint manager — enables pause/resume of research runs.

After each completed round, the entire ResearchState is serialized to a JSON
checkpoint file. If the process crashes or is interrupted, the run can be
resumed from the last completed round.

Serialization strategy:
- Pydantic models → .model_dump() for Hypothesis, ExperimentRecord, etc.
- datetime → .isoformat()
- Enum → .value
- On restore, reconstruct Pydantic models from dicts
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.state import (
    AuditEntry,
    ExperimentRecord,
    Hypothesis,
    HypothesisStatus,
    KnowledgeState,
    ExperimentStatus,
)


CHECKPOINT_FILENAME = ".research_checkpoint.json"


class CheckpointManager:
    """Manages save/load of research state checkpoints."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)
        self.checkpoint_path = self.workspace / CHECKPOINT_FILENAME

    def save(self, state: dict[str, Any]) -> None:
        """Serialize and save the current research state."""
        serializable = self._serialize_state(state)
        # Atomic write: write to temp file, then rename
        tmp = self.checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))
        tmp.rename(self.checkpoint_path)

    def load(self) -> dict[str, Any] | None:
        """Load and deserialize a research state checkpoint. Returns None if no checkpoint."""
        if not self.checkpoint_path.exists():
            return None
        try:
            data = json.loads(self.checkpoint_path.read_text())
            return self._deserialize_state(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: checkpoint corrupted ({e}), starting fresh")
            return None

    def exists(self) -> bool:
        return self.checkpoint_path.exists()

    def clear(self) -> None:
        self.checkpoint_path.unlink(missing_ok=True)

    def summary(self) -> str | None:
        """Return a one-line summary of the checkpoint, or None."""
        state = self.load()
        if state is None:
            return None
        return (
            f"Round {state.get('round_number', '?')}, "
            f"{len(state.get('experiment_history', []))} experiments, "
            f"{len(state.get('hypotheses', []))} hypotheses, "
            f"verdict: {state.get('verdict', '?')}"
        )

    # ── Serialization ───────────────────────────────────────────────────────

    def _serialize_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Convert state dict with Pydantic models to plain JSON-serializable dict."""
        out: dict[str, Any] = {}

        # Simple fields
        for key in ("repo_url", "paper_url", "assessment", "plan",
                     "last_result", "reflection", "decision",
                     "verdict", "max_rounds", "should_continue"):
            if key in state:
                out[key] = state[key]

        # target_metrics (dict)
        out["target_metrics"] = state.get("target_metrics", {})

        # round_number
        out["round_number"] = state.get("round_number", 0)

        # planned_actions
        out["planned_actions"] = state.get("planned_actions", [])

        # KnowledgeState
        knowledge = state.get("knowledge")
        if knowledge:
            out["knowledge"] = knowledge.model_dump() if hasattr(knowledge, "model_dump") else dict(knowledge)
        else:
            out["knowledge"] = {}

        # Hypotheses (list of Pydantic models)
        out["hypotheses"] = []
        for h in state.get("hypotheses", []):
            if hasattr(h, "model_dump"):
                d = h.model_dump()
                d["created_at"] = d["created_at"].isoformat() if isinstance(d.get("created_at"), datetime) else d.get("created_at")
                d["resolved_at"] = d["resolved_at"].isoformat() if isinstance(d.get("resolved_at"), datetime) else d.get("resolved_at")
                out["hypotheses"].append(d)
            else:
                out["hypotheses"].append(h)

        # Observations
        out["observations"] = state.get("observations", [])

        # Experiment history
        out["experiment_history"] = []
        for exp in state.get("experiment_history", []):
            if hasattr(exp, "model_dump"):
                d = exp.model_dump()
                d["started_at"] = d["started_at"].isoformat() if isinstance(d.get("started_at"), datetime) else d.get("started_at")
                d["completed_at"] = d["completed_at"].isoformat() if isinstance(d.get("completed_at"), datetime) else d.get("completed_at")
                out["experiment_history"].append(d)
            else:
                out["experiment_history"].append(exp)

        # Audit log
        out["audit_log"] = []
        for entry in state.get("audit_log", []):
            if hasattr(entry, "model_dump"):
                d = entry.model_dump()
                d["timestamp"] = d["timestamp"].isoformat() if isinstance(d.get("timestamp"), datetime) else d.get("timestamp")
                out["audit_log"].append(d)
            else:
                out["audit_log"].append(entry)

        out["_checkpoint_time"] = datetime.now().isoformat()
        return out

    def _deserialize_state(self, data: dict[str, Any]) -> dict[str, Any]:
        """Reconstruct Pydantic models from serialized dict."""
        # KnowledgeState
        knowledge_dict = data.get("knowledge", {})
        if knowledge_dict:
            data["knowledge"] = KnowledgeState(**knowledge_dict)
        else:
            data["knowledge"] = KnowledgeState()

        # Hypotheses
        hypotheses = []
        for hd in data.get("hypotheses", []):
            try:
                hd["status"] = HypothesisStatus(hd.get("status", "proposed"))
                hypotheses.append(Hypothesis(**hd))
            except Exception:
                pass
        data["hypotheses"] = hypotheses

        # Experiment history
        experiments = []
        for ed in data.get("experiment_history", []):
            try:
                ed["status"] = ExperimentStatus(ed.get("status", "completed"))
                experiments.append(ExperimentRecord(**ed))
            except Exception:
                pass
        data["experiment_history"] = experiments

        # Audit log
        audits = []
        for ad in data.get("audit_log", []):
            try:
                audits.append(AuditEntry(**ad))
            except Exception:
                pass
        data["audit_log"] = audits

        # Restore should_continue if verdict was "continue" and not at max
        verdict = data.get("verdict", "continue")
        data["should_continue"] = (
            verdict == "continue"
            and data.get("round_number", 0) < data.get("max_rounds", 5)
        )

        return data

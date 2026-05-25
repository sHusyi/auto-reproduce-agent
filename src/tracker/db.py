"""Experiment tracker — SQLite-backed persistence for experiment history.

Stores experiments, hypotheses, and metrics so the agent can reference
past results and track progress over multiple rounds.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.state import ExperimentRecord, Hypothesis, HypothesisStatus


class ExperimentTracker:
    """SQLite-backed tracker for research experiments.

    Usage:
        tracker = ExperimentTracker(":memory:")  # or file path
        tracker.save_experiment(exp)
        history = tracker.get_experiments()
        tracker.save_hypothesis(hyp)
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._create_tables()
        return self._conn

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                round_number INTEGER NOT NULL,
                hypothesis_id TEXT,
                action TEXT NOT NULL,
                command TEXT,
                metrics_before TEXT DEFAULT '{}',
                metrics_after TEXT DEFAULT '{}',
                stdout_preview TEXT DEFAULT '',
                stderr_preview TEXT DEFAULT '',
                exit_code INTEGER,
                status TEXT DEFAULT 'planned',
                files_changed TEXT DEFAULT '[]',
                observation TEXT DEFAULT '',
                started_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY,
                statement TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                verification_method TEXT DEFAULT '',
                status TEXT DEFAULT 'proposed',
                evidence_for TEXT DEFAULT '[]',
                evidence_against TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                resolved_at TEXT
            );

            CREATE TABLE IF NOT EXISTS metrics_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );
        """)
        self.conn.commit()

    # ── Experiments ─────────────────────────────────────────────────────────

    def save_experiment(self, exp: ExperimentRecord) -> None:
        """Insert or update an experiment record."""
        self.conn.execute(
            """INSERT OR REPLACE INTO experiments
               (id, round_number, hypothesis_id, action, command,
                metrics_before, metrics_after, stdout_preview, stderr_preview,
                exit_code, status, files_changed, observation, started_at, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                exp.id, exp.round_number, exp.hypothesis_id, exp.action,
                exp.command,
                json.dumps(exp.metrics_before), json.dumps(exp.metrics_after),
                exp.stdout_preview, exp.stderr_preview,
                exp.exit_code, exp.status.value,
                json.dumps(exp.files_changed), exp.observation,
                exp.started_at.isoformat(),
                exp.completed_at.isoformat() if exp.completed_at else None,
            ),
        )

        # Insert metrics history rows
        if exp.metrics_after:
            for name, value in exp.metrics_after.items():
                self.conn.execute(
                    "INSERT INTO metrics_history (experiment_id, metric_name, metric_value, recorded_at) "
                    "VALUES (?, ?, ?, ?)",
                    (exp.id, name, value, (exp.completed_at or datetime.now()).isoformat()),
                )

        self.conn.commit()

    def get_experiments(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent experiments as dicts."""
        rows = self.conn.execute(
            "SELECT * FROM experiments ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_best_metrics(self) -> dict[str, float]:
        """Get the best value for each metric across all experiments."""
        rows = self.conn.execute(
            "SELECT metric_name, MAX(metric_value) as best "
            "FROM metrics_history GROUP BY metric_name"
        ).fetchall()
        return {r["metric_name"]: r["best"] for r in rows}

    def get_metrics_progression(self, metric_name: str) -> list[tuple[int, float]]:
        """Get metric values over rounds for a progression chart."""
        rows = self.conn.execute(
            "SELECT e.round_number, m.metric_value "
            "FROM metrics_history m "
            "JOIN experiments e ON m.experiment_id = e.id "
            "WHERE m.metric_name = ? "
            "ORDER BY e.round_number",
            (metric_name,),
        ).fetchall()
        return [(r["round_number"], r["metric_value"]) for r in rows]

    # ── Hypotheses ──────────────────────────────────────────────────────────

    def save_hypothesis(self, hyp: Hypothesis) -> None:
        """Insert or update a hypothesis."""
        self.conn.execute(
            """INSERT OR REPLACE INTO hypotheses
               (id, statement, confidence, verification_method, status,
                evidence_for, evidence_against, created_at, resolved_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                hyp.id, hyp.statement, hyp.confidence, hyp.verification_method,
                hyp.status.value,
                json.dumps(hyp.evidence_for), json.dumps(hyp.evidence_against),
                hyp.created_at.isoformat(),
                hyp.resolved_at.isoformat() if hyp.resolved_at else None,
            ),
        )
        self.conn.commit()

    def get_hypotheses(self) -> list[dict[str, Any]]:
        """Get all hypotheses."""
        rows = self.conn.execute(
            "SELECT * FROM hypotheses ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_confirmed_hypotheses(self) -> list[dict[str, Any]]:
        """Get only confirmed hypotheses — these are agent's validated knowledge."""
        rows = self.conn.execute(
            "SELECT * FROM hypotheses WHERE status = 'confirmed' ORDER BY resolved_at"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Summary ─────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of all tracked data."""
        exps = self.get_experiments()
        hyps = self.get_hypotheses()
        best = self.get_best_metrics()

        lines = ["Experiment Tracker Summary", "=" * 60]
        lines.append(f"Experiments: {len(exps)}")
        lines.append(f"Hypotheses: {len(hyps)}")
        for name, value in best.items():
            lines.append(f"Best {name}: {value:.4f}")

        if exps:
            lines.append("\nExperiment History:")
            for exp in exps[:10]:
                metrics_str = ", ".join(
                    f"{k}={v}" for k, v in json.loads(exp["metrics_after"]).items()
                ) if exp["metrics_after"] else "no metrics"
                lines.append(
                    f"  R{exp['round_number']}: {exp['action'][:60]} "
                    f"[{exp['status']}] {metrics_str}"
                )

        return "\n".join(lines)

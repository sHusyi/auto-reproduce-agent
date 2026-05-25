"""Experiment comparison — structured diff between experiment results.

Enables the agent to answer: "Did the change improve things?"
"""

from __future__ import annotations

from src.state import ExperimentRecord


def compare_experiments(
    before: ExperimentRecord,
    after: ExperimentRecord,
) -> dict:
    """Compare two experiment records and produce a structured diff.

    Args:
        before: The baseline experiment
        after: The experiment being compared against baseline

    Returns:
        Dict with metric deltas and summary assessment
    """
    deltas = {}
    all_metrics = set(before.metrics_after.keys()) | set(after.metrics_after.keys())
    for metric in sorted(all_metrics):
        v_before = before.metrics_after.get(metric)
        v_after = after.metrics_after.get(metric)
        if v_before is not None and v_after is not None:
            deltas[metric] = {
                "before": v_before,
                "after": v_after,
                "delta": round(v_after - v_before, 4),
                "pct_change": round((v_after - v_before) / abs(v_before) * 100, 2) if v_before != 0 else 0,
                "improved": v_after > v_before,
            }

    # Overall assessment
    improvements = sum(1 for d in deltas.values() if d["improved"])
    regressions = sum(1 for d in deltas.values() if not d["improved"] and d["delta"] != 0)

    return {
        "deltas": deltas,
        "summary": f"{improvements} metrics improved, {regressions} regressed",
        "net_positive": improvements >= regressions,
        "before_action": before.action,
        "after_action": after.action,
    }


def compare_to_target(
    experiment: ExperimentRecord,
    target_metrics: dict[str, float],
) -> dict:
    """Compare experiment metrics to paper target metrics.

    Returns:
        Dict with per-metric gap analysis and overall assessment.
    """
    gaps = {}
    for metric, target in target_metrics.items():
        current = experiment.metrics_after.get(metric)
        if current is not None:
            gap = abs(current - target)
            gaps[metric] = {
                "current": current,
                "target": target,
                "gap": round(gap, 4),
                "pct_to_target": round(current / target * 100, 2) if target != 0 else 100,
                "meets_target": current >= target,
            }

    if not gaps:
        return {"gaps": {}, "summary": "No target metrics available in experiment results"}

    all_met = all(g["meets_target"] for g in gaps.values())
    return {
        "gaps": gaps,
        "summary": "All targets met!" if all_met else "Some targets not yet met",
        "all_met": all_met,
    }

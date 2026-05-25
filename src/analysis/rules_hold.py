"""Generic scorer for timed-hold exercises. Exercise-agnostic — reads a
TargetPose and measured joint angles."""

import math

from analysis.types import HoldAnalysis, RuleViolation
from exercises.base import TargetPose


def score_frame(
    target: TargetPose,
    measured: dict[str, float],
) -> tuple[bool, list[RuleViolation]]:
    """Per-frame check: is the current pose within tolerance for every declared joint?

    Returns (in_target, violations). FSM consumes the bool; live LLM
    consumes the violations.

    A missing or NaN joint angle counts as out-of-target with severity 1.0.
    """
    violations: list[RuleViolation] = []
    in_target = True
    for j in target.joints:
        m = measured.get(j.name)
        if m is None or math.isnan(m):
            in_target = False
            violations.append(RuleViolation(j.name, 1.0, j.detail_th))
            continue
        deviation = abs(m - j.target_deg)
        if deviation > j.tolerance_deg:
            in_target = False
            # Severity ramps from 0 at the tolerance edge to 1.0 at 2x tolerance.
            severity = min(
                1.0, (deviation - j.tolerance_deg) / max(j.tolerance_deg, 1e-6)
            )
            violations.append(RuleViolation(j.name, severity, j.detail_th))
    return in_target, violations


def score_hold(
    exercise_name: str,
    meta: dict,
    target: TargetPose,
    max_severity_seen: dict[str, float],
) -> HoldAnalysis:
    """Final score on hold completion. 100-pt budget: 50 duration / 30 precision / 20 stability."""
    target_ms = int(target.hold_seconds * 1000)
    in_target_ms = int(meta["in_target_ms"])
    drift_count = int(meta["drift_count"])

    duration_ratio = min(1.0, in_target_ms / max(target_ms, 1))
    duration_pts = int(round(50 * duration_ratio))

    finite_sevs = [v for v in max_severity_seen.values() if not math.isnan(v)]
    mean_sev = sum(finite_sevs) / len(finite_sevs) if finite_sevs else 0.0
    precision_pts = int(round(30 * (1.0 - min(1.0, mean_sev))))

    # Stability: smooth decay. 0 drifts → 20. ~5+ drifts → 0.
    stability_pts = int(round(20 * math.exp(-0.4 * drift_count)))

    components = {
        "duration": duration_pts,
        "precision": precision_pts,
        "stability": stability_pts,
    }
    # Build a single violation list from the worst-offending joints (severity > 0.05).
    violations = [
        RuleViolation(
            name=jn,
            severity=sev,
            detail_th=next((j.detail_th for j in target.joints if j.name == jn), ""),
        )
        for jn, sev in max_severity_seen.items()
        if sev > 0.05
    ]
    return HoldAnalysis(
        exercise_name=exercise_name,
        score=sum(components.values()),
        components=components,
        violations=violations,
        in_target_ms=in_target_ms,
        drift_count=drift_count,
    )

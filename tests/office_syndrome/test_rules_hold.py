from analysis.rules_hold import score_frame, score_hold
from analysis.types import HoldAnalysis
from exercises.base import JointTarget, TargetPose


TP = TargetPose(
    joints=(
        JointTarget("a", target_deg=30.0, tolerance_deg=5.0, detail_th="adjust a"),
        JointTarget("b", target_deg=10.0, tolerance_deg=3.0, detail_th="adjust b"),
    )
)


def test_score_frame_in_target_when_all_joints_within_tolerance():
    in_target, violations = score_frame(TP, {"a": 28.0, "b": 11.0})
    assert in_target is True
    assert violations == []


def test_score_frame_out_of_target_when_one_joint_outside():
    in_target, violations = score_frame(TP, {"a": 28.0, "b": 20.0})
    assert in_target is False
    assert len(violations) == 1
    assert violations[0].name == "b"
    assert violations[0].detail_th == "adjust b"
    assert 0.0 < violations[0].severity <= 1.0


def test_score_frame_nan_is_out_of_target():
    in_target, violations = score_frame(TP, {"a": float("nan"), "b": 10.0})
    assert in_target is False
    assert any(v.name == "a" for v in violations)


def test_score_frame_missing_joint_is_out_of_target():
    in_target, violations = score_frame(TP, {"a": 30.0})
    assert in_target is False
    assert any(v.name == "b" for v in violations)


def test_score_hold_full_credit_when_clean():
    meta = {"in_target_ms": 20_000, "drift_count": 0, "completed_ts": 25.0}
    analysis = score_hold(
        exercise_name="x",
        meta=meta,
        target=TP,
        max_severity_seen={"a": 0.0, "b": 0.0},
    )
    assert isinstance(analysis, HoldAnalysis)
    assert analysis.components["duration"] == 50
    assert analysis.components["precision"] == 30
    assert analysis.components["stability"] == 20
    assert analysis.score == 100


def test_score_hold_duration_clip_when_under_target():
    # Target is 20s (hold_seconds default) but only 10s accumulated.
    meta = {"in_target_ms": 10_000, "drift_count": 0, "completed_ts": 30.0}
    analysis = score_hold("x", meta, TP, {"a": 0.0, "b": 0.0})
    assert analysis.components["duration"] == 25  # 50 * 0.5


def test_score_hold_precision_penalty_for_high_severity():
    meta = {"in_target_ms": 20_000, "drift_count": 0, "completed_ts": 25.0}
    analysis = score_hold("x", meta, TP, {"a": 0.5, "b": 0.5})
    # mean severity 0.5 → precision = 30 * 0.5 = 15
    assert analysis.components["precision"] == 15


def test_score_hold_stability_decays_with_drifts():
    meta = {"in_target_ms": 20_000, "drift_count": 4, "completed_ts": 30.0}
    analysis = score_hold("x", meta, TP, {"a": 0.0, "b": 0.0})
    # Some decay applied; exact value depends on formula but must drop from 20.
    assert 0 <= analysis.components["stability"] < 20


def test_score_hold_duration_caps_at_50_when_over_target():
    """If accumulated in-target ms exceeds target, duration component caps at 50."""
    # TP uses default hold_seconds=20.0; pass 30s of in-target time.
    meta = {"in_target_ms": 30_000, "drift_count": 0, "completed_ts": 35.0}
    analysis = score_hold("x", meta, TP, {"a": 0.0, "b": 0.0})
    assert analysis.components["duration"] == 50
    assert analysis.score <= 100

import numpy as np
from analysis.types import (
    PoseFrame,
    PhaseState,
    RepAnalysis,
    HoldState,
    HoldAnalysis,
    LiveSnapshot,
    RuleViolation,
)


def test_pose_frame_construction():
    keypoints = np.zeros((17, 2), dtype=np.float32)
    scores = np.zeros((17,), dtype=np.float32)
    pf = PoseFrame(
        timestamp=1.0, keypoints_2d=keypoints, scores=scores, frame_shape=(480, 640)
    )
    assert pf.timestamp == 1.0
    assert pf.keypoints_2d.shape == (17, 2)


def test_phase_state_enum():
    assert PhaseState.STANDING.value == "standing"
    assert PhaseState.BOTTOM.value == "bottom"


def test_rep_analysis_defaults():
    ra = RepAnalysis(
        rep_index=0,
        score=85,
        components={"depth": 25, "valgus": 20, "torso": 18, "symmetry": 14, "tempo": 8},
        violations=[],
        descent_ms=900,
        ascent_ms=800,
    )
    assert ra.score == 85
    assert ra.violations == []


def test_rep_analysis_has_default_metric_source_3d():
    rep = RepAnalysis(
        rep_index=0,
        score=85,
        components={"depth": 30, "valgus": 25, "torso": 20, "symmetry": 0, "tempo": 10},
        violations=[],
        descent_ms=1000,
        ascent_ms=1000,
    )
    assert rep.metric_source == "3d"


def test_rep_analysis_accepts_2d_metric_source():
    rep = RepAnalysis(
        rep_index=0,
        score=50,
        components={"depth": 0, "valgus": 25, "torso": 20, "symmetry": 0, "tempo": 5},
        violations=[],
        descent_ms=1000,
        ascent_ms=1000,
        metric_source="2d",
    )
    assert rep.metric_source == "2d"


def test_hold_state_enum_values():
    assert HoldState.IDLE.value == "idle"
    assert HoldState.ENTERING.value == "entering"
    assert HoldState.HOLDING.value == "holding"
    assert HoldState.DRIFTED.value == "drifted"
    assert HoldState.COMPLETE.value == "complete"


def test_hold_analysis_dataclass_roundtrip():
    a = HoldAnalysis(
        exercise_name="neck_stretch_left",
        score=87,
        components={"duration": 50, "precision": 25, "stability": 12},
        violations=[RuleViolation("head_lateral_tilt", 0.4, "เอียงคอเพิ่ม")],
        in_target_ms=18_000,
        drift_count=2,
    )
    assert a.score == 87
    assert a.components["duration"] == 50
    assert a.violations[0].name == "head_lateral_tilt"


def test_live_snapshot_dataclass_roundtrip():
    s = LiveSnapshot(
        exercise_name="neck_stretch_left",
        state=HoldState.HOLDING,
        progress_ratio=0.6,
        current_violations=[],
    )
    assert s.state is HoldState.HOLDING
    assert s.progress_ratio == 0.6
    assert s.current_violations == []

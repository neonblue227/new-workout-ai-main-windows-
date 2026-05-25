import math

import numpy as np
from analysis.rules_squat import score_rep
from analysis.types import PoseFrame


def make_pose_frame(
    knee_angle: float,
    hip_y: float,
    knee_y: float,
    valgus: float = 0.0,
    lean: float = 30.0,
) -> PoseFrame:
    """Manually construct a PoseFrame with hand-tuned keypoints for a given posture."""
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (100, hip_y)
    kps[12] = (150, hip_y)
    kps[13] = (100 + valgus * 50, knee_y)
    kps[14] = (150 - valgus * 50, knee_y)
    kps[15] = (100, knee_y + 100)
    kps[16] = (150, knee_y + 100)
    import math

    dx = math.tan(math.radians(lean)) * 100
    kps[5] = (100 + dx, hip_y - 100)
    kps[6] = (150 + dx, hip_y - 100)
    scores = np.ones((17,), dtype=np.float32)
    return PoseFrame(
        timestamp=1.0, keypoints_2d=kps, scores=scores, frame_shape=(480, 640)
    )


def test_perfect_rep_high_score():
    bottom = make_pose_frame(
        knee_angle=85.0, hip_y=260, knee_y=240, valgus=0.0, lean=35.0
    )
    result = score_rep(bottom_frame=bottom, descent_ms=1200, ascent_ms=1000)
    assert result.score >= 90
    assert result.violations == []


def test_shallow_squat_loses_depth_points():
    bottom = make_pose_frame(knee_angle=120.0, hip_y=180, knee_y=240, lean=30.0)
    result = score_rep(bottom_frame=bottom, descent_ms=900, ascent_ms=900)
    assert result.components["depth"] == 0
    assert any(v.name == "shallow_depth" for v in result.violations)


def _make_pose_at_knee_angle(knee_deg: float, lean_deg: float = 35.0) -> PoseFrame:
    """Construct a symmetric pose with the given mean knee flexion (degrees)."""
    knee_y = 240.0
    bone = 100.0
    theta = math.radians(knee_deg)
    hip_dx = bone * math.sin(theta)
    hip_dy = bone * math.cos(theta)
    lean = math.radians(lean_deg)
    sh_dx = bone * math.sin(lean)
    sh_dy = -bone * math.cos(lean)
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (100 + hip_dx, knee_y + hip_dy)
    kps[12] = (150 + hip_dx, knee_y + hip_dy)
    kps[13] = (100, knee_y)
    kps[14] = (150, knee_y)
    kps[15] = (100, knee_y + bone)
    kps[16] = (150, knee_y + bone)
    kps[5] = (kps[11][0] + sh_dx, kps[11][1] + sh_dy)
    kps[6] = (kps[12][0] + sh_dx, kps[12][1] + sh_dy)
    return PoseFrame(
        timestamp=1.0,
        keypoints_2d=kps,
        scores=np.ones((17,), dtype=np.float32),
        frame_shape=(480, 640),
    )


def test_near_parallel_gets_partial_depth_credit():
    """Mean knee ~110° (above parallel but not standing) should give partial depth credit."""
    bottom = _make_pose_at_knee_angle(knee_deg=110.0)
    result = score_rep(bottom_frame=bottom, descent_ms=1200, ascent_ms=1000)
    # 110° → ratio = (130-110)/40 = 0.5 → depth = 15
    assert 12 <= result.components["depth"] <= 18
    shallow = next(v for v in result.violations if v.name == "shallow_depth")
    assert 0.3 < shallow.severity < 0.7


def test_knee_valgus_detected():
    bottom = make_pose_frame(
        knee_angle=85.0, hip_y=260, knee_y=240, valgus=0.4, lean=30.0
    )
    result = score_rep(bottom_frame=bottom, descent_ms=1000, ascent_ms=1000)
    assert any(v.name == "knee_valgus" for v in result.violations)
    assert result.components["valgus"] < 25


def test_excessive_forward_lean():
    bottom = make_pose_frame(knee_angle=85.0, hip_y=260, knee_y=240, lean=70.0)
    result = score_rep(bottom_frame=bottom, descent_ms=1000, ascent_ms=1000)
    assert any(v.name == "excessive_forward_lean" for v in result.violations)
    assert result.components["torso"] < 20


def test_tempo_penalty_when_ascent_longer():
    bottom = make_pose_frame(knee_angle=85.0, hip_y=260, knee_y=240, lean=30.0)
    result = score_rep(bottom_frame=bottom, descent_ms=500, ascent_ms=1500)
    assert result.components["tempo"] < 10


def test_score_rep_uses_2d_fallback_when_no_3d_present():
    """Existing 2D-only fixtures should now report metric_source='2d'."""
    bottom = make_pose_frame(
        knee_angle=85.0, hip_y=260, knee_y=240, valgus=0.0, lean=35.0
    )
    # bottom.keypoints_3d is None by construction (make_pose_frame doesn't set it).
    result = score_rep(bottom_frame=bottom, descent_ms=1200, ascent_ms=1000)
    assert result.metric_source == "2d"


def _frame_with_3d(kps_3d: np.ndarray) -> PoseFrame:
    """Wrap a (17, 3) 3D pose in a PoseFrame. 2D placeholder is zeros — not read."""
    return PoseFrame(
        timestamp=1.0,
        keypoints_2d=np.zeros((17, 2), dtype=np.float32),
        scores=np.ones((17,), dtype=np.float32),
        frame_shape=(480, 640),
        keypoints_3d=kps_3d,
    )


def _bent_legs_at_90() -> np.ndarray:
    """Symmetric squat at the bottom: both knees at 90°, neutral knee tracking,
    upright torso."""
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[0] = (0.0, 0.0, 0.0)  # pelvis
    kps[1] = (0.2, 0.0, 0.0)  # R_hip
    kps[4] = (-0.2, 0.0, 0.0)  # L_hip
    kps[8] = (0.0, -1.0, 0.0)  # thorax (upright)
    kps[2] = (0.2, 0.3, 0.0)  # R_knee directly below R_hip
    kps[5] = (-0.2, 0.3, 0.0)  # L_knee
    kps[3] = (0.2, 0.3, 0.3)  # R_ankle forward of knee → 90° flexion
    kps[6] = (-0.2, 0.3, 0.3)  # L_ankle
    return kps


def test_3d_path_perfect_rep_full_score():
    import math

    kps = _bent_legs_at_90()
    # Add a 35° torso lean to land inside the [20, 55] good zone.
    theta = math.radians(35.0)
    kps[8] = (0.0, -math.cos(theta), math.sin(theta))
    result = score_rep(_frame_with_3d(kps), descent_ms=1200, ascent_ms=1000)
    assert result.metric_source == "3d"
    assert result.components["depth"] == 30
    assert result.components["valgus"] == 25
    assert result.components["torso"] == 20
    assert result.components["symmetry"] == 15
    assert result.components["tempo"] == 10
    assert result.score == 100
    assert result.violations == []


def test_3d_path_medial_knee_costs_valgus():
    kps = _bent_legs_at_90()
    # Push R knee toward midline.
    kps[2] = (0.05, 0.3, 0.0)
    result = score_rep(_frame_with_3d(kps), descent_ms=1200, ascent_ms=1000)
    assert result.metric_source == "3d"
    assert result.components["valgus"] < 25
    assert any(v.name == "knee_valgus" for v in result.violations)


def test_3d_path_lateral_knee_is_not_valgus():
    """Flaring the knee out is not valgus — should NOT trigger a violation.
    This is one of the two bugs we're fixing (the 2D code penalizes both directions)."""
    kps = _bent_legs_at_90()
    kps[2] = (0.5, 0.3, 0.0)  # R knee well outside the hip-ankle line
    result = score_rep(_frame_with_3d(kps), descent_ms=1200, ascent_ms=1000)
    assert result.metric_source == "3d"
    assert result.components["valgus"] == 25
    assert not any(v.name == "knee_valgus" for v in result.violations)


def test_3d_path_asymmetric_squat_penalized():
    import math

    kps = _bent_legs_at_90()
    # Stretch R knee toward 130° flexion (R leg less bent than L).
    theta = math.radians(130.0)
    ay = -0.3 * math.cos(theta)
    az = math.sqrt(max(0.0, 0.09 - ay * ay))
    kps[3] = (0.2, 0.3 + ay, az)
    result = score_rep(_frame_with_3d(kps), descent_ms=1200, ascent_ms=1000)
    assert result.metric_source == "3d"
    # L=90, R=130 → mean=110 → depth = 30 * (130-110)/40 = 15
    assert 12 <= result.components["depth"] <= 18
    # Delta = 40° → severity = min(1, (40-10)/20) = 1.0 → symmetry = 0
    assert result.components["symmetry"] == 0
    assert any(v.name == "asymmetric" for v in result.violations)


def test_3d_path_invariant_under_body_rotation():
    """The same 3D body posture rotated in world space should produce the same
    score — proving camera-angle independence for the 3D path."""
    kps_orig = _bent_legs_at_90()
    kps_orig[2] = (0.05, 0.3, 0.0)  # medial collapse on R

    # Rotate 90° about y axis: (x, y, z) → (z, y, -x).
    kps_rot = np.zeros_like(kps_orig)
    for i in range(kps_orig.shape[0]):
        x, y, z = kps_orig[i]
        kps_rot[i] = (z, y, -x)

    r1 = score_rep(_frame_with_3d(kps_orig), descent_ms=1000, ascent_ms=1000)
    r2 = score_rep(_frame_with_3d(kps_rot), descent_ms=1000, ascent_ms=1000)
    assert r1.components == r2.components

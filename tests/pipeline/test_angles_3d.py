import math

import numpy as np

from analysis.angles_3d import (
    body_frame_axes,
    head_lateral_tilt_3d,
    knee_flexion_3d,
    torso_lean_3d,
    valgus_offset_3d,
)

# H36M-17 indices for test use.
R_HIP_IDX = 1
R_KNEE_IDX = 2
R_ANKLE_IDX = 3
L_HIP_IDX = 4
L_KNEE_IDX = 5
L_ANKLE_IDX = 6


def _canonical_standing_pose() -> np.ndarray:
    """Pelvis at origin, thorax 1 unit up (image-y down → up = -y),
    hips along ±x. Other H36M joints zero-initialized (unused by this fn)."""
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[0] = (0.0, 0.0, 0.0)  # pelvis
    kps[1] = (0.2, 0.0, 0.0)  # R_hip
    kps[4] = (-0.2, 0.0, 0.0)  # L_hip
    kps[8] = (0.0, -1.0, 0.0)  # thorax
    return kps


def test_body_frame_axes_canonical_pose_is_orthonormal():
    kps = _canonical_standing_pose()
    up, lat, fwd = body_frame_axes(kps)
    np.testing.assert_allclose(up, [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(lat, [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(fwd, [0.0, 0.0, 1.0], atol=1e-6)


def test_body_frame_axes_are_unit_vectors():
    kps = _canonical_standing_pose()
    for v in body_frame_axes(kps):
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_body_frame_axes_lateral_is_orthogonal_to_up_even_when_hipline_tilted():
    kps = _canonical_standing_pose()
    # Tilt the hip line so it isn't already perpendicular to body_up.
    kps[1] = (0.2, -0.1, 0.0)
    kps[4] = (-0.2, 0.1, 0.0)
    up, lat, _ = body_frame_axes(kps)
    assert abs(np.dot(up, lat)) < 1e-6


def _both_legs_at_flexion(theta_deg: float) -> np.ndarray:
    """Symmetric pose with both knees at the given flexion (degrees).

    Geometry: thigh points straight up (knee→hip = (0, -1, 0) direction),
    shin rotated about the lateral axis so the angle between knee→hip and
    knee→ankle equals theta. theta=180° → straight leg; theta=90° → shin
    perpendicular to thigh."""
    import math

    kps = _canonical_standing_pose()
    bone = 0.3
    theta = math.radians(theta_deg)
    # knee→ankle direction in (y, z) sagittal plane:
    # at theta=180°, points (0, +1, 0) (down in image, opposite of knee→hip).
    # at theta=90°,  points (0, 0, +1) (forward).
    # general: (0, -cos(theta), sin(theta))
    shin_dir = np.array([0.0, -math.cos(theta), math.sin(theta)], dtype=np.float32)

    for hip_x in (0.2, -0.2):
        hip = np.array([hip_x, -bone, 0.0], dtype=np.float32)
        knee = np.array([hip_x, 0.0, 0.0], dtype=np.float32)
        ankle = knee + bone * shin_dir
        if hip_x > 0:
            kps[1] = hip  # R_HIP
            kps[2] = knee  # R_KNEE
            kps[3] = ankle  # R_ANKLE
        else:
            kps[4] = hip  # L_HIP
            kps[5] = knee  # L_KNEE
            kps[6] = ankle  # L_ANKLE
    return kps


def test_knee_flexion_3d_straight_leg_is_180():
    kps = _both_legs_at_flexion(180.0)
    left, right = knee_flexion_3d(kps)
    assert abs(left - 180.0) < 0.5
    assert abs(right - 180.0) < 0.5


def test_knee_flexion_3d_right_angle_is_90():
    kps = _both_legs_at_flexion(90.0)
    left, right = knee_flexion_3d(kps)
    assert abs(left - 90.0) < 0.5
    assert abs(right - 90.0) < 0.5


def test_knee_flexion_3d_degenerate_returns_nan():
    kps = _canonical_standing_pose()
    # All knee-related joints at origin — zero-length bones.
    left, right = knee_flexion_3d(kps)
    assert np.isnan(left) or np.isnan(right)


def test_torso_lean_3d_perfectly_upright_is_zero():
    kps = _canonical_standing_pose()
    assert abs(torso_lean_3d(kps)) < 0.5


def test_torso_lean_3d_thirty_degrees_forward():
    import math

    kps = _canonical_standing_pose()
    theta = math.radians(30.0)
    # Lean thorax 30° forward (sagittal +z), unit-length torso.
    kps[8] = (0.0, -math.cos(theta), math.sin(theta))
    assert abs(torso_lean_3d(kps) - 30.0) < 0.5


def test_torso_lean_3d_sixty_degrees_forward():
    import math

    kps = _canonical_standing_pose()
    theta = math.radians(60.0)
    kps[8] = (0.0, -math.cos(theta), math.sin(theta))
    assert abs(torso_lean_3d(kps) - 60.0) < 0.5


def _neutral_squat_3d() -> np.ndarray:
    """3D fixture: neutral squat at the bottom. Knees and ankles directly below
    hips in the body frame (no medial/lateral offset)."""
    kps = _canonical_standing_pose()
    # Knees at body-lat=±0.2, slightly forward, dropped 0.3 in image-y.
    kps[R_KNEE_IDX] = (0.2, 0.3, 0.0)
    kps[L_KNEE_IDX] = (-0.2, 0.3, 0.0)
    # Ankles further forward (z), same lateral as knees.
    kps[R_ANKLE_IDX] = (0.2, 0.3, 0.3)
    kps[L_ANKLE_IDX] = (-0.2, 0.3, 0.3)
    return kps


def test_valgus_offset_3d_neutral_squat_is_zero():
    kps = _neutral_squat_3d()
    left, right = valgus_offset_3d(kps)
    assert abs(left) < 0.05
    assert abs(right) < 0.05


def test_valgus_offset_3d_medial_knee_positive():
    kps = _neutral_squat_3d()
    # Push R knee toward midline (smaller lateral coord).
    kps[R_KNEE_IDX] = (0.05, 0.3, 0.0)
    left, right = valgus_offset_3d(kps)
    assert right > 0.10
    assert abs(left) < 0.05


def test_valgus_offset_3d_lateral_knee_negative():
    kps = _neutral_squat_3d()
    # Push R knee outward (larger lateral coord).
    kps[R_KNEE_IDX] = (0.5, 0.3, 0.0)
    _, right = valgus_offset_3d(kps)
    assert right < -0.10


def test_valgus_offset_3d_invariant_under_body_rotation():
    """Rotating the whole pose 90° about the image-y axis should not change
    the valgus signal — this is the camera-angle-independence guarantee."""
    kps_orig = _neutral_squat_3d()
    kps_orig[R_KNEE_IDX] = (0.05, 0.3, 0.0)  # medial collapse on R
    left1, right1 = valgus_offset_3d(kps_orig)

    # Rotate 90° about y axis: (x, y, z) → (z, y, -x).
    kps_rot = np.zeros_like(kps_orig)
    for i in range(kps_orig.shape[0]):
        x, y, z = kps_orig[i]
        kps_rot[i] = (z, y, -x)
    left2, right2 = valgus_offset_3d(kps_rot)

    assert abs(right1 - right2) < 0.01
    assert abs(left1 - left2) < 0.01


# ---------------------------------------------------------------------------
# head_lateral_tilt_3d tests
# ---------------------------------------------------------------------------


def _h36m_skeleton_with_head_offset(
    lat_offset: float, vertical: float = 1.0
) -> np.ndarray:
    """Build a minimal H36M-17 keypoint array with head offset laterally
    from a vertical thorax-pelvis axis.

    Body frame setup in MotionBERT's normalized coords (image-y down → up = -y):
      pelvis at origin
      thorax directly above pelvis (along -y)
      l_hip / r_hip on the x axis so body_lateral = +x
      head = thorax + (lat_offset, -vertical, 0)
    """
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[0] = (0.0, 0.0, 0.0)  # PELVIS
    kps[1] = (0.5, 0.0, 0.0)  # R_HIP  → lateral = R_HIP - L_HIP = +x ✓
    kps[4] = (-0.5, 0.0, 0.0)  # L_HIP
    kps[8] = (0.0, -1.0, 0.0)  # THORAX
    kps[10] = (lat_offset, -1.0 - vertical, 0.0)  # HEAD
    return kps


def test_head_lateral_tilt_zero_when_head_above_thorax():
    kps = _h36m_skeleton_with_head_offset(lat_offset=0.0)
    assert abs(head_lateral_tilt_3d(kps)) < 1.0  # within 1°


def test_head_lateral_tilt_positive_when_tilted_to_body_lateral_plus():
    kps = _h36m_skeleton_with_head_offset(lat_offset=0.5)
    angle = head_lateral_tilt_3d(kps)
    # Expected magnitude: atan2(0.5, 1.0) ≈ 26.57°
    assert 24.0 < angle < 30.0


def test_head_lateral_tilt_sign_flips_with_lateral_direction():
    pos = head_lateral_tilt_3d(_h36m_skeleton_with_head_offset(lat_offset=0.5))
    neg = head_lateral_tilt_3d(_h36m_skeleton_with_head_offset(lat_offset=-0.5))
    assert pos * neg < 0  # opposite signs


def test_head_lateral_tilt_nan_when_thorax_collapsed():
    kps = _h36m_skeleton_with_head_offset(lat_offset=0.0)
    kps[8] = kps[0]  # thorax coincides with pelvis → no body_up
    assert math.isnan(head_lateral_tilt_3d(kps))

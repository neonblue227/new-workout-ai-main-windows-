import math
import numpy as np
import pytest
from analysis.angles import (
    angle_between_3_points,
    craniovertebral_angle_2d,
    forward_head_offset_normalized_2d,
    head_lateral_tilt_2d,
    knee_angles,
    neck_flexion_2d,
    shoulder_elevation_asymmetry_2d,
    shoulder_protraction_ratio_2d,
    torso_lean_deg,
    hip_below_knee,
    knee_valgus_ratio,
    wrist_extension_2d,
)


def _shoulders_and_nose(nose_dx: float, mirrored: bool = False):
    """Build a (17, 2) keypoint array with shoulders 100 px apart at y=200 and
    nose 100 px above mid-shoulders, shifted laterally by `nose_dx`.

    `mirrored=True` swaps L/R shoulder x-coords to simulate a selfie-mirrored
    camera, where R_shoulder ends up image-left of L_shoulder.
    """
    kps = np.zeros((17, 2), dtype=np.float32)
    if mirrored:
        kps[5] = (200.0, 200.0)  # L_shoulder on image-right
        kps[6] = (100.0, 200.0)  # R_shoulder on image-left
    else:
        kps[5] = (100.0, 200.0)
        kps[6] = (200.0, 200.0)
    kps[0] = (150.0 + nose_dx, 100.0)
    scores = np.ones(17, dtype=np.float32)
    return kps, scores


def test_angle_180_degrees_straight():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([2.0, 0.0])
    assert angle_between_3_points(a, b, c) == pytest.approx(180.0, abs=0.5)


def test_angle_90_degrees():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([1.0, 1.0])
    assert angle_between_3_points(a, b, c) == pytest.approx(90.0, abs=0.5)


def test_knee_angles_standing():
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (100, 100)
    kps[13] = (100, 200)
    kps[15] = (100, 300)
    kps[12] = (150, 100)
    kps[14] = (150, 200)
    kps[16] = (150, 300)
    left, right = knee_angles(kps)
    assert left == pytest.approx(180.0, abs=1.0)
    assert right == pytest.approx(180.0, abs=1.0)


def test_knee_angles_squatting():
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (50, 200)
    kps[13] = (100, 200)
    kps[15] = (100, 300)
    kps[12] = kps[11]
    kps[14] = kps[13]
    kps[16] = kps[15]
    left, right = knee_angles(kps)
    assert left == pytest.approx(90.0, abs=2.0)


def test_torso_lean_upright():
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[5] = (100, 100)
    kps[6] = (150, 100)
    kps[11] = (100, 200)
    kps[12] = (150, 200)
    assert torso_lean_deg(kps) == pytest.approx(0.0, abs=1.0)


def test_torso_lean_45deg_forward():
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[5] = (200, 100)
    kps[6] = (250, 100)
    kps[11] = (100, 200)
    kps[12] = (150, 200)
    assert torso_lean_deg(kps) == pytest.approx(45.0, abs=2.0)


def test_hip_below_knee_true():
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (100, 250)
    kps[12] = (150, 250)
    kps[13] = (100, 200)
    kps[14] = (150, 200)
    assert hip_below_knee(kps) is True


def test_hip_below_knee_false():
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (100, 100)
    kps[12] = (150, 100)
    kps[13] = (100, 200)
    kps[14] = (150, 200)
    assert hip_below_knee(kps) is False


def test_knee_valgus_ratio_neutral_stance():
    # Knees directly above ankles -> zero valgus on both sides
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (100, 200)  # L hip
    kps[12] = (200, 200)  # R hip (hip_width = 100)
    kps[13] = (100, 250)  # L knee directly above L ankle
    kps[14] = (200, 250)  # R knee directly above R ankle
    kps[15] = (100, 300)  # L ankle
    kps[16] = (200, 300)  # R ankle
    left, right = knee_valgus_ratio(kps)
    assert left == pytest.approx(0.0, abs=0.001)
    assert right == pytest.approx(0.0, abs=0.001)


def test_knee_valgus_ratio_caved_inward():
    # Both knees shifted toward each other (caved in). Should produce positive values on both sides.
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (100, 200)  # L hip
    kps[12] = (200, 200)  # R hip (hip_width = 100)
    kps[13] = (130, 250)  # L knee shifted right by 30 (inward)
    kps[14] = (170, 250)  # R knee shifted left by 30 (inward)
    kps[15] = (100, 300)  # L ankle
    kps[16] = (200, 300)  # R ankle
    left, right = knee_valgus_ratio(kps)
    assert left == pytest.approx(0.3, abs=0.01)  # (130-100)/100 = 0.3
    assert right == pytest.approx(0.3, abs=0.01)  # (200-170)/100 = 0.3


# ---- head_lateral_tilt_2d ----


def test_head_tilt_2d_upright_is_zero():
    kps, scores = _shoulders_and_nose(nose_dx=0.0)
    assert head_lateral_tilt_2d(kps, scores) == pytest.approx(0.0, abs=0.5)


def test_head_tilt_2d_left_lean_is_negative():
    """Body-lateral = R - L = +x. Nose shifted to -x leans toward L_shoulder side."""
    kps, scores = _shoulders_and_nose(nose_dx=-30.0)
    assert head_lateral_tilt_2d(kps, scores) < -10.0


def test_head_tilt_2d_right_lean_is_positive():
    kps, scores = _shoulders_and_nose(nose_dx=+30.0)
    assert head_lateral_tilt_2d(kps, scores) > 10.0


def test_head_tilt_2d_45_degrees():
    """Nose 100 px above mid-shoulders and 100 px to image-right → atan2(100, 100) = 45°.
    With L on left and R on right (body_lateral = +x), this is a +45° tilt."""
    kps, scores = _shoulders_and_nose(nose_dx=+100.0)
    assert head_lateral_tilt_2d(kps, scores) == pytest.approx(45.0, abs=0.5)


def test_head_tilt_2d_mirror_invariant():
    """Selfie-mirrored camera flips R/L shoulder x but the user's tilt-direction
    label (left/right of the body) must stay the same. A real left-side tilt
    in a mirrored view: nose drifts toward image-LEFT, and R_shoulder is also
    on image-left, so body_lateral = R - L is negative → tilt stays negative."""
    # Non-mirrored: nose at -x → body_lateral = +x → negative angle.
    kps_a, scores_a = _shoulders_and_nose(nose_dx=-30.0, mirrored=False)
    angle_a = head_lateral_tilt_2d(kps_a, scores_a)

    # Mirrored selfie of the SAME physical pose: nose at +x (mirror flips it),
    # shoulders swapped so R is at image-left. body_lateral = R - L = -x.
    # The sign convention should still report left-side tilt as negative.
    kps_b, scores_b = _shoulders_and_nose(nose_dx=+30.0, mirrored=True)
    angle_b = head_lateral_tilt_2d(kps_b, scores_b)

    assert angle_a < 0
    assert angle_b < 0
    assert angle_a == pytest.approx(angle_b, abs=0.5)


def test_head_tilt_2d_nan_on_low_nose_score():
    kps, scores = _shoulders_and_nose(nose_dx=-30.0)
    scores[0] = 0.05
    assert math.isnan(head_lateral_tilt_2d(kps, scores))


def test_head_tilt_2d_nan_on_low_shoulder_score():
    kps, scores = _shoulders_and_nose(nose_dx=-30.0)
    scores[5] = 0.05  # L_shoulder
    assert math.isnan(head_lateral_tilt_2d(kps, scores))


def test_head_tilt_2d_nan_on_coincident_shoulders():
    """If RTMPose puts both shoulders at the same x, there is no usable lateral
    reference — must NaN out rather than divide by zero."""
    kps, scores = _shoulders_and_nose(nose_dx=-30.0)
    kps[6] = kps[5]  # R_shoulder collapsed onto L_shoulder
    assert math.isnan(head_lateral_tilt_2d(kps, scores))


def test_head_tilt_2d_works_without_scores():
    """`scores=None` skips gating — used when the caller already validated input."""
    kps, _ = _shoulders_and_nose(nose_dx=-30.0)
    assert head_lateral_tilt_2d(kps, scores=None) < 0


# ---- 2D office-syndrome cookbook (B2) ----


def _side_view_fixture(
    ear_dy_above_shoulder: float = 100.0,
    ear_dx_in_front: float = 0.0,
):
    """Build a (kps, scores) pair with ONE side fully clean (LEFT) — used for
    side-view metrics. Shoulder at (100, 200); ear positioned relative."""
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[5] = (100.0, 200.0)  # L_shoulder
    kps[3] = (100.0 + ear_dx_in_front, 200.0 - ear_dy_above_shoulder)  # L_ear
    # also place a right side so the auto-picker has a fallback
    kps[6] = (200.0, 200.0)
    kps[4] = (200.0 + ear_dx_in_front, 200.0 - ear_dy_above_shoulder)
    scores = np.ones(17, dtype=np.float32) * 0.9
    return kps, scores


def test_cva_upright_is_90deg():
    """Ear directly above shoulder → CVA = 90° (perfectly upright)."""
    kps, scores = _side_view_fixture(ear_dy_above_shoulder=100.0, ear_dx_in_front=0.0)
    assert craniovertebral_angle_2d(kps, scores, side="left") == pytest.approx(
        90.0, abs=0.5
    )


def test_cva_forward_head_drops_below_50():
    """Ear well in front of shoulder (forward-head posture) → CVA < 50° (unhealthy)."""
    kps, scores = _side_view_fixture(ear_dy_above_shoulder=80.0, ear_dx_in_front=80.0)
    cva = craniovertebral_angle_2d(kps, scores, side="left")
    assert cva < 50.0
    assert cva > 0  # still positive — ear is above shoulder


def test_cva_nan_when_ear_missing():
    kps, scores = _side_view_fixture()
    scores[3] = 0.05  # L_ear missing
    scores[4] = 0.05  # R_ear missing
    assert math.isnan(craniovertebral_angle_2d(kps, scores, side="auto"))


def test_cva_auto_picks_higher_confidence_side():
    kps, scores = _side_view_fixture(ear_dy_above_shoulder=100.0, ear_dx_in_front=0.0)
    # Wreck the right ear, leave left clean → auto must pick left and still report ~90°.
    scores[4] = 0.05  # R_ear bad
    assert craniovertebral_angle_2d(kps, scores, side="auto") == pytest.approx(
        90.0, abs=0.5
    )


def test_forward_head_offset_zero_when_aligned():
    """Ears directly above shoulders → offset 0."""
    kps, scores = _shoulders_and_nose(nose_dx=0.0)
    kps[3] = kps[5]  # L_ear directly above L_shoulder
    kps[4] = kps[6]
    assert forward_head_offset_normalized_2d(kps, scores) == pytest.approx(
        0.0, abs=0.001
    )


def test_forward_head_offset_normalized_by_shoulder_width():
    kps, scores = _shoulders_and_nose(nose_dx=0.0)
    # Shoulder width = 100 px. Move ears 30 px forward (away from midline) — offset = 0.30.
    kps[3] = (
        kps[5, 0] - 30.0,
        kps[5, 1] - 80.0,
    )  # ear shifted in image-left = forward for the user
    kps[4] = (kps[6, 0] + 30.0, kps[6, 1] - 80.0)
    val = forward_head_offset_normalized_2d(kps, scores)
    assert val == pytest.approx(0.30, abs=0.01)


def test_forward_head_offset_nan_when_no_clean_ear():
    kps, scores = _shoulders_and_nose(nose_dx=0.0)
    scores[3] = 0.05
    scores[4] = 0.05
    assert math.isnan(forward_head_offset_normalized_2d(kps, scores))


def test_neck_flexion_zero_when_upright():
    """Ear directly above shoulder (no forward/back lean) → flexion ≈ 0°."""
    kps, scores = _side_view_fixture(ear_dy_above_shoulder=100.0, ear_dx_in_front=0.0)
    assert neck_flexion_2d(kps, scores, side="left") == pytest.approx(0.0, abs=0.5)


def test_neck_flexion_positive_when_head_forward():
    """Ear forward of shoulder → positive (forward flexion)."""
    kps, scores = _side_view_fixture(ear_dy_above_shoulder=100.0, ear_dx_in_front=50.0)
    val = neck_flexion_2d(kps, scores, side="left")
    assert val > 10.0


def test_neck_flexion_nan_when_no_clean_input():
    kps, scores = _side_view_fixture()
    scores[3] = scores[4] = 0.05
    assert math.isnan(neck_flexion_2d(kps, scores, side="auto"))


def test_shoulder_asymmetry_zero_when_level():
    kps, scores = _shoulders_and_nose(nose_dx=0.0)
    assert shoulder_elevation_asymmetry_2d(kps, scores) == pytest.approx(0.0, abs=0.001)


def test_shoulder_asymmetry_positive_when_left_is_higher():
    """L_shoulder above R_shoulder (smaller image-y for L) → positive value."""
    kps, scores = _shoulders_and_nose(nose_dx=0.0)
    kps[5] = (kps[5, 0], kps[5, 1] - 10.0)  # L_shoulder 10 px higher
    assert shoulder_elevation_asymmetry_2d(kps, scores) > 0


def test_shoulder_asymmetry_normalized_by_shoulder_width():
    """Same y-delta on a narrower frame → bigger normalized value."""
    kps, scores = _shoulders_and_nose(nose_dx=0.0)
    kps[5] = (kps[5, 0], kps[5, 1] - 10.0)
    wide_val = shoulder_elevation_asymmetry_2d(kps, scores)

    kps2 = kps.copy()
    kps2[5] = (140.0, kps[5, 1])  # shrink shoulders to 60 px
    kps2[6] = (200.0, kps[6, 1])
    narrow_val = shoulder_elevation_asymmetry_2d(kps2, scores)
    assert abs(narrow_val) > abs(wide_val)


def test_shoulder_asymmetry_nan_when_shoulder_missing():
    kps, scores = _shoulders_and_nose(nose_dx=0.0)
    scores[5] = 0.05
    assert math.isnan(shoulder_elevation_asymmetry_2d(kps, scores))


def test_protraction_ratio_one_at_baseline():
    kps, scores = _shoulders_and_nose(nose_dx=0.0)
    # shoulders at 100 px → use 100 px as baseline.
    val = shoulder_protraction_ratio_2d(kps, scores, baseline_width_px=100.0)
    assert val == pytest.approx(1.0, abs=0.01)


def test_protraction_ratio_below_one_when_protracted():
    """Shoulders pulled forward → apparent width drops → ratio < 1.0."""
    kps, scores = _shoulders_and_nose(nose_dx=0.0)
    kps[5] = (120.0, kps[5, 1])  # shoulders now 80 px apart instead of 100
    kps[6] = (200.0, kps[6, 1])
    val = shoulder_protraction_ratio_2d(kps, scores, baseline_width_px=100.0)
    assert val < 0.95


def test_protraction_ratio_nan_without_baseline():
    kps, scores = _shoulders_and_nose(nose_dx=0.0)
    assert math.isnan(shoulder_protraction_ratio_2d(kps, scores, baseline_width_px=0.0))


def test_wrist_extension_zero_when_inline():
    """Wrist at same y as elbow → extension 0."""
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[7] = (100.0, 300.0)  # L_elbow
    kps[9] = (200.0, 300.0)  # L_wrist (same y)
    kps[8] = (110.0, 300.0)  # R_elbow
    kps[10] = (210.0, 300.0)  # R_wrist
    scores = np.ones(17, dtype=np.float32) * 0.9
    assert wrist_extension_2d(kps, scores, side="left") == pytest.approx(0.0, abs=0.001)


def test_wrist_extension_negative_when_wrist_above_elbow():
    """Wrist ABOVE elbow (smaller image-y) → negative extension (dorsiflexed)."""
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[7] = (100.0, 300.0)
    kps[9] = (200.0, 200.0)  # wrist 100 px above elbow
    scores = np.ones(17, dtype=np.float32) * 0.9
    val = wrist_extension_2d(kps, scores, side="left")
    assert val < 0
    assert val == pytest.approx(
        -1.0 / np.sqrt(2), abs=0.05
    )  # 100 dy / sqrt(100² + 100²)


def test_wrist_extension_nan_when_side_missing():
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[7] = (100.0, 300.0)
    kps[9] = (200.0, 300.0)
    kps[8] = (110.0, 300.0)
    kps[10] = (210.0, 300.0)
    scores = np.zeros(17, dtype=np.float32)  # ALL low confidence
    assert math.isnan(wrist_extension_2d(kps, scores, side="auto"))

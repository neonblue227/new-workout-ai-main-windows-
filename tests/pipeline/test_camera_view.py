import numpy as np

from analysis.camera_view import CameraView, classify_view


def _kps_with_visible_ears(
    shoulder_dx: float = 100.0,
    shoulder_dy: float = 0.0,
    l_ear_conf: float = 0.9,
    r_ear_conf: float = 0.9,
    shoulder_conf: float = 0.9,
):
    """Build a (kps, scores) fixture with controllable shoulder geometry +
    per-ear confidence. Centered at (150, 200)."""
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[5] = (150.0 - shoulder_dx / 2, 200.0 + shoulder_dy / 2)  # L_shoulder
    kps[6] = (150.0 + shoulder_dx / 2, 200.0 - shoulder_dy / 2)  # R_shoulder
    kps[3] = (140.0, 150.0)  # L_ear
    kps[4] = (160.0, 150.0)  # R_ear
    scores = np.ones(17, dtype=np.float32) * 0.9
    scores[3] = l_ear_conf
    scores[4] = r_ear_conf
    scores[5] = shoulder_conf
    scores[6] = shoulder_conf
    return kps, scores


def test_front_view_when_shoulders_level_and_both_ears_visible():
    kps, scores = _kps_with_visible_ears(shoulder_dx=100.0, shoulder_dy=2.0)
    assert classify_view(kps, scores) == CameraView.FRONT


def test_side_view_when_shoulders_almost_overlap():
    """Side view collapses the projected shoulder width: width_x / |Δy| is small."""
    kps, scores = _kps_with_visible_ears(
        shoulder_dx=15.0, shoulder_dy=8.0, l_ear_conf=0.05
    )
    assert classify_view(kps, scores) == CameraView.SIDE


def test_side_view_when_only_one_ear_visible():
    """Frontal-ish geometry but one ear gone → side or three-quarter, never FRONT."""
    kps, scores = _kps_with_visible_ears(
        shoulder_dx=100.0, shoulder_dy=0.0, r_ear_conf=0.05
    )
    result = classify_view(kps, scores)
    assert result in (CameraView.SIDE, CameraView.THREE_QUARTER)


def test_three_quarter_view_when_shoulders_partly_collapsed():
    """Mid-range shoulder ratio → THREE_QUARTER."""
    kps, scores = _kps_with_visible_ears(shoulder_dx=50.0, shoulder_dy=10.0)
    assert classify_view(kps, scores) == CameraView.THREE_QUARTER


def test_unknown_when_shoulders_unreliable():
    kps, scores = _kps_with_visible_ears(shoulder_conf=0.05)
    assert classify_view(kps, scores) == CameraView.UNKNOWN


def test_classify_view_enum_values():
    assert CameraView.FRONT.value == "front"
    assert CameraView.THREE_QUARTER.value == "three_quarter"
    assert CameraView.SIDE.value == "side"
    assert CameraView.UNKNOWN.value == "unknown"

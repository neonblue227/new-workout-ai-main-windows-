import math
import numpy as np

from analysis.camera_view import CameraView
from analysis.types import PoseFrame
from calibration import BaselinePose
from exercises.neck_stretch import NeckStretchLeft, NeckStretchRight


# Image-plane fixture: nose above mid-shoulders, shoulders horizontal. The
# `head_dx` parameter shifts the nose laterally; head_dx < 0 = head tilted toward
# L_shoulder (the body's -lateral direction), matching the NeckStretchLeft target.
def _kps2d_with_head_shifted(head_dx: float) -> tuple[np.ndarray, np.ndarray]:
    kps = np.zeros((17, 2), dtype=np.float32)
    # Shoulders 100 px apart, at image-y = 200; L is at x=100 (image-left), R at x=200.
    kps[5] = (100.0, 200.0)  # L_shoulder
    kps[6] = (200.0, 200.0)  # R_shoulder
    # Nose 100 px above mid-shoulders (smaller y = up in image coords).
    kps[0] = (150.0 + head_dx, 100.0)
    scores = np.ones(17, dtype=np.float32)
    return kps, scores


def _make_frame(kps_2d: np.ndarray, scores: np.ndarray) -> PoseFrame:
    return PoseFrame(
        timestamp=0.0,
        keypoints_2d=kps_2d,
        scores=scores,
        frame_shape=(720, 1280),
        keypoints_3d=None,
    )


def test_metadata():
    ex = NeckStretchLeft()
    assert ex.name == "neck_stretch_left"
    assert ex.target.side == "left"
    assert ex.target.hold_seconds == 25.0
    assert len(ex.target.joints) == 1
    assert ex.target.joints[0].name == "head_lateral_tilt"


def test_neck_stretch_valid_views_excludes_side_view():
    """Neck-tilt-2d depends on a usable shoulder lateral reference. Side view
    collapses both shoulders to nearly one point, making the measurement
    degenerate — so the exercise must refuse to score in side view."""
    ex = NeckStretchLeft()
    assert CameraView.FRONT in ex.target.valid_views
    assert CameraView.THREE_QUARTER in ex.target.valid_views
    assert CameraView.SIDE not in ex.target.valid_views
    assert CameraView.UNKNOWN not in ex.target.valid_views


def test_measure_returns_declared_joints():
    ex = NeckStretchLeft()
    kps, scores = _kps2d_with_head_shifted(-30.0)
    measured = ex.measure(_make_frame(kps, scores))
    assert set(measured.keys()) == {"head_lateral_tilt"}


def test_measure_nan_when_nose_confidence_low():
    """If RTMPose can't see the nose, the measurement must surface as NaN so the
    FSM stays IDLE rather than acting on garbage."""
    ex = NeckStretchLeft()
    kps, scores = _kps2d_with_head_shifted(-30.0)
    scores[0] = 0.05  # nose below threshold
    measured = ex.measure(_make_frame(kps, scores))
    assert math.isnan(measured["head_lateral_tilt"])


def test_measure_nan_when_shoulder_confidence_low():
    ex = NeckStretchLeft()
    kps, scores = _kps2d_with_head_shifted(-30.0)
    scores[5] = 0.05  # L_shoulder below threshold
    measured = ex.measure(_make_frame(kps, scores))
    assert math.isnan(measured["head_lateral_tilt"])


def test_prompt_template_renders_summary_without_keyerror():
    ex = NeckStretchLeft()
    rendered = ex.prompt.summary.format(
        exercise_th=ex.display_th,
        score=87,
        duration=50,
        precision=25,
        stability=12,
        violations="(none)",
    )
    assert "87" in rendered


def test_prompt_template_renders_live_without_keyerror():
    ex = NeckStretchLeft()
    rendered = ex.prompt.live.format(
        exercise_th=ex.display_th,
        state="holding",
        progress_pct=60,
        violations="(none)",
    )
    assert "60" in rendered


def test_measure_sign_for_left_tilt_is_negative():
    """Catch a regression if the head-tilt sign convention ever flips.

    Fixture has R_shoulder.x > L_shoulder.x (body_lateral = +x). A nose shifted
    to -x is leaning toward the body's L_shoulder side, which must produce a
    NEGATIVE tilt to match the NeckStretchLeft target of -35°.
    """
    ex = NeckStretchLeft()
    kps, scores = _kps2d_with_head_shifted(-30.0)
    result = ex.measure(_make_frame(kps, scores))
    assert result["head_lateral_tilt"] < 0, (
        f"expected negative tilt for left-leaning head, got {result['head_lateral_tilt']}"
    )


def _baseline(tilt_deg: float) -> BaselinePose:
    return BaselinePose(
        shoulder_width_px=100.0,
        head_lateral_tilt_deg=tilt_deg,
        shoulder_y_delta_norm=0.0,
        sample_count=30,
        captured_ts=0.0,
    )


def test_measure_without_baseline_returns_absolute_tilt():
    """Backward compatibility: existing callers that don't pass `baseline` see
    the same absolute-tilt value they always did."""
    ex = NeckStretchLeft()
    kps, scores = _kps2d_with_head_shifted(-30.0)
    no_baseline = ex.measure(_make_frame(kps, scores))["head_lateral_tilt"]
    explicit_none = ex.measure(_make_frame(kps, scores), baseline=None)[
        "head_lateral_tilt"
    ]
    assert no_baseline == explicit_none


def test_measure_with_baseline_subtracts_neutral():
    """A user whose neutral tilt is already −5° (habitual left lean) gets credit
    only for the *additional* tilt beyond their neutral — not the full absolute
    angle. delta = absolute − baseline, so at absolute −30° with baseline −5°,
    delta = −25° (less negative; closer to zero than absolute).
    """
    ex = NeckStretchLeft()
    kps, scores = _kps2d_with_head_shifted(-30.0)
    abs_tilt = ex.measure(_make_frame(kps, scores))["head_lateral_tilt"]

    baseline = _baseline(tilt_deg=-5.0)
    delta_tilt = ex.measure(_make_frame(kps, scores), baseline=baseline)[
        "head_lateral_tilt"
    ]

    # delta = abs - (-5.0) = abs + 5.0  (less negative than abs)
    assert delta_tilt == abs_tilt - (-5.0)
    assert (
        delta_tilt > abs_tilt
    )  # less negative — user only gets credit for tilt beyond neutral


def test_measure_with_zero_baseline_matches_absolute():
    """If the user's neutral is exactly 0°, baseline-aware measurement should
    be identical to absolute measurement."""
    ex = NeckStretchLeft()
    kps, scores = _kps2d_with_head_shifted(-30.0)
    abs_tilt = ex.measure(_make_frame(kps, scores))["head_lateral_tilt"]
    delta_tilt = ex.measure(_make_frame(kps, scores), baseline=_baseline(0.0))[
        "head_lateral_tilt"
    ]
    assert delta_tilt == abs_tilt


def test_measure_with_baseline_preserves_nan_propagation():
    """If the per-frame measurement is NaN (low confidence), baseline subtraction
    must not turn it into a number."""
    ex = NeckStretchLeft()
    kps, scores = _kps2d_with_head_shifted(-30.0)
    scores[0] = 0.05  # nose unreliable → NaN
    result = ex.measure(_make_frame(kps, scores), baseline=_baseline(-5.0))
    assert math.isnan(result["head_lateral_tilt"])


def test_right_metadata():
    ex = NeckStretchRight()
    assert ex.name == "neck_stretch_right"
    assert ex.target.side == "right"
    assert ex.target.hold_seconds == 25.0
    assert ex.target.joints[0].name == "head_lateral_tilt"
    assert ex.target.joints[0].target_deg == 35.0


def test_right_valid_views_exclude_side():
    ex = NeckStretchRight()
    assert CameraView.SIDE not in ex.target.valid_views
    assert CameraView.FRONT in ex.target.valid_views


def test_right_measure_sign_for_right_tilt_is_positive():
    """Nose shifted to +x (toward R_shoulder) must yield a POSITIVE tilt to
    match the NeckStretchRight target of +35°."""
    ex = NeckStretchRight()
    kps, scores = _kps2d_with_head_shifted(+30.0)
    result = ex.measure(_make_frame(kps, scores))
    assert result["head_lateral_tilt"] > 0


def test_left_hold_is_now_25s():
    assert NeckStretchLeft().target.hold_seconds == 25.0

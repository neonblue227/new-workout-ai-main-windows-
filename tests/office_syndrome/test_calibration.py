import math
import numpy as np
import pytest

from calibration import (
    BaselinePose,
    CalibrationError,
    calibrate_from_samples,
)


def _sample(
    nose_dx: float = 0.0,
    shoulder_dy: float = 0.0,
    nose_conf: float = 0.9,
    sh_conf: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (kps, scores) pair with controllable head-tilt and shoulder
    asymmetry. Shoulders sit 100 px apart at image y=200; nose 100 px above mid.
    """
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[5] = (100.0, 200.0 + shoulder_dy)  # L_shoulder (y shifted down by dy)
    kps[6] = (200.0, 200.0)  # R_shoulder
    kps[0] = (150.0 + nose_dx, 100.0)  # nose
    scores = np.ones(17, dtype=np.float32) * 0.9
    scores[0] = nose_conf
    scores[5] = sh_conf
    scores[6] = sh_conf
    return kps, scores


def test_baseline_pose_is_frozen():
    b = BaselinePose(
        shoulder_width_px=100.0,
        head_lateral_tilt_deg=0.0,
        shoulder_y_delta_norm=0.0,
        sample_count=30,
        captured_ts=0.0,
    )
    with pytest.raises(Exception):
        b.shoulder_width_px = 999.0  # type: ignore[misc]


def test_calibrate_averages_clean_samples():
    samples = [_sample(nose_dx=0.0) for _ in range(40)]
    base = calibrate_from_samples(samples, min_clean_frames=30)
    assert base.sample_count == 40
    assert base.shoulder_width_px == pytest.approx(100.0, abs=1e-3)
    assert base.head_lateral_tilt_deg == pytest.approx(0.0, abs=0.1)
    assert base.shoulder_y_delta_norm == pytest.approx(0.0, abs=0.01)


def test_calibrate_handles_consistent_left_tilt():
    """If the user is held-tilting throughout calibration, the baseline records
    that as the neutral — the whole point of per-user calibration."""
    samples = [_sample(nose_dx=-20.0) for _ in range(40)]
    base = calibrate_from_samples(samples, min_clean_frames=30)
    assert base.head_lateral_tilt_deg < -5.0


def test_calibrate_filters_low_confidence_samples():
    """Samples with bad confidence are dropped before averaging."""
    samples = [_sample(nose_dx=0.0) for _ in range(35)]
    # Mix in 10 samples where the nose conf is below threshold.
    samples += [_sample(nose_dx=-100.0, nose_conf=0.1) for _ in range(10)]
    base = calibrate_from_samples(samples, min_clean_frames=30)
    # The dropped samples would've yanked the average toward -100 dx if included.
    assert base.sample_count == 35
    assert base.head_lateral_tilt_deg == pytest.approx(0.0, abs=1.0)


def test_calibrate_raises_on_too_few_clean_samples():
    samples = [_sample(nose_conf=0.05) for _ in range(50)]
    with pytest.raises(CalibrationError):
        calibrate_from_samples(samples, min_clean_frames=30)


def test_calibrate_skips_frames_with_zero_shoulder_width():
    """If RTMPose collapses both shoulders onto one point on a sample, that
    sample is unusable (division by zero risk downstream)."""
    kps_bad = np.zeros((17, 2), dtype=np.float32)
    kps_bad[5] = (150.0, 200.0)
    kps_bad[6] = (150.0, 200.0)  # coincident
    kps_bad[0] = (150.0, 100.0)
    scores_bad = np.ones(17, dtype=np.float32) * 0.9
    bad = [(kps_bad, scores_bad) for _ in range(10)]
    good = [_sample() for _ in range(35)]
    base = calibrate_from_samples(bad + good, min_clean_frames=30)
    assert base.sample_count == 35  # only the good ones counted


def test_calibrate_records_captured_timestamp():
    samples = [_sample() for _ in range(35)]
    base = calibrate_from_samples(samples, min_clean_frames=30)
    assert base.captured_ts > 0
    assert not math.isnan(base.captured_ts)

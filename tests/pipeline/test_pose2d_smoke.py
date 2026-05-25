import cv2
import pytest
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "standing_person.jpg"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture image not downloaded")
def test_pose2d_detects_keypoints_on_standing_person():
    from pose2d import Pose2D

    img = cv2.imread(str(FIXTURE))
    assert img is not None

    detector = Pose2D()
    kps, scores = detector.infer(img)

    assert kps.shape == (17, 2)
    assert scores.shape == (17,)
    assert (scores > 0.3).sum() >= 8

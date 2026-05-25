from pathlib import Path

import cv2
import pytest

from analysis.rules_hold import score_frame
from analysis.types import PoseFrame
from exercises.neck_stretch import NeckStretchLeft

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "neck_stretch_left.jpg"
_MODELS = Path(__file__).resolve().parent.parent.parent / "models" / "rtmlib_cache"


@pytest.mark.skipif(
    not FIXTURE.exists() or not _MODELS.exists(),
    reason="fixture or pose weights missing",
)
def test_full_pipeline_on_static_image_runs_without_exception():
    from pose2d import Pose2D
    from pose3d import Pose3D, Pose3DBuffer, coco17_to_h36m17

    img = cv2.imread(str(FIXTURE))
    assert img is not None

    pose = Pose2D()
    kps, scores, _ = pose.infer_with_heatmaps(img)

    lifter = Pose3D()
    buf = Pose3DBuffer(lifter)
    h36m = coco17_to_h36m17(kps, scores)
    for _ in range(27):
        buf.push(h36m)
    rig_3d = buf.lift(img.shape[0], img.shape[1])
    assert rig_3d.shape == (17, 3)

    pf = PoseFrame(
        timestamp=0.0,
        keypoints_2d=kps,
        scores=scores,
        frame_shape=img.shape[:2],
        keypoints_3d=rig_3d,
    )

    ex = NeckStretchLeft()
    measured = ex.measure(pf)
    in_target, violations = score_frame(ex.target, measured)

    # We don't assert in_target — depends on the image content. Just no exceptions.
    assert "head_lateral_tilt" in measured
    assert in_target is True or in_target is False
    for v in violations:
        assert 0.0 <= v.severity <= 1.0

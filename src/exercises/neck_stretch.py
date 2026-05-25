import math
from typing import Optional

from analysis.angles import head_lateral_tilt_2d
from analysis.camera_view import CameraView
from analysis.types import PoseFrame
from calibration import BaselinePose
from exercises.base import JointTarget, PromptTemplate, TargetPose


# Target is expressed as a DELTA from the user's calibrated neutral tilt:
# "−35° below your own natural head position" rather than "−35° in absolute
# image space." When `measure()` is called without a baseline (e.g. legacy
# tests), the delta-subtraction is skipped — same absolute behavior as before.
_NECK_TILT_TARGET_LEFT_DEG = -35.0
_NECK_TILT_TARGET_RIGHT_DEG = 35.0
_NECK_TILT_TOLERANCE_DEG = 10.0
_HOLD_SECONDS = 25.0


_LIVE_TH = (
    "ผู้ใช้กำลังทำท่า {exercise_th} (สถานะ: {state}, ความคืบหน้า {progress_pct}%) "
    "ข้อสังเกตล่าสุด: {violations} "
    "ให้คำแนะนำสั้น ๆ 1 ประโยคเป็นภาษาไทยเพื่อช่วยให้ผู้ใช้ปรับท่าให้ถูกต้อง"
)

_SUMMARY_TH = (
    "ผลการทำท่า {exercise_th}: คะแนนรวม {score}/100\n"
    "  ระยะเวลา: {duration}/50\n"
    "  ความแม่นยำ: {precision}/30\n"
    "  ความนิ่ง: {stability}/20\n"
    "ข้อสังเกต: {violations}\n"
    "ช่วยสรุปสั้น ๆ เป็นภาษาไทยว่าทำดีตรงไหน ควรปรับตรงไหน"
)


def _measure_head_lateral_tilt(
    frame: PoseFrame, baseline: Optional[BaselinePose]
) -> dict[str, float]:
    """Shared measure for both neck-stretch sides: 2D head-lateral-tilt,
    baseline-subtracted when a BaselinePose is supplied (NaN-preserving)."""
    tilt = head_lateral_tilt_2d(frame.keypoints_2d, frame.scores)
    if (
        baseline is not None
        and not math.isnan(tilt)
        and not math.isnan(baseline.head_lateral_tilt_deg)
    ):
        tilt = tilt - baseline.head_lateral_tilt_deg
    return {"head_lateral_tilt": tilt}


class NeckStretchLeft:
    name = "neck_stretch_left"
    display_th = "ยืดคอด้านซ้าย"
    target = TargetPose(
        joints=(
            JointTarget(
                name="head_lateral_tilt",
                target_deg=_NECK_TILT_TARGET_LEFT_DEG,
                tolerance_deg=_NECK_TILT_TOLERANCE_DEG,
                detail_th="เอียงศีรษะไปทางซ้ายมากขึ้นอีกนิด",
            ),
        ),
        hold_seconds=_HOLD_SECONDS,
        side="left",
        # Neck-tilt-2d uses nose + shoulders, which are reliable in front and
        # three-quarter views. Side view collapses both shoulders onto one
        # point and makes the lateral reference degenerate, so refuse to score.
        valid_views=(CameraView.FRONT, CameraView.THREE_QUARTER),
    )
    prompt = PromptTemplate(live=_LIVE_TH, summary=_SUMMARY_TH)

    def measure(
        self,
        frame: PoseFrame,
        baseline: Optional[BaselinePose] = None,
    ) -> dict[str, float]:
        return _measure_head_lateral_tilt(frame, baseline)


class NeckStretchRight:
    name = "neck_stretch_right"
    display_th = "ยืดคอด้านขวา"
    target = TargetPose(
        joints=(
            JointTarget(
                name="head_lateral_tilt",
                target_deg=_NECK_TILT_TARGET_RIGHT_DEG,
                tolerance_deg=_NECK_TILT_TOLERANCE_DEG,
                detail_th="เอียงศีรษะไปทางขวามากขึ้นอีกนิด",
            ),
        ),
        hold_seconds=_HOLD_SECONDS,
        side="right",
        valid_views=(CameraView.FRONT, CameraView.THREE_QUARTER),
    )
    prompt = PromptTemplate(live=_LIVE_TH, summary=_SUMMARY_TH)

    def measure(
        self,
        frame: PoseFrame,
        baseline: Optional[BaselinePose] = None,
    ) -> dict[str, float]:
        return _measure_head_lateral_tilt(frame, baseline)

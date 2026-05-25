from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class PhaseState(Enum):
    STANDING = "standing"
    DESCENT = "descent"
    BOTTOM = "bottom"
    ASCENT = "ascent"


@dataclass
class PoseFrame:
    timestamp: float
    keypoints_2d: np.ndarray  # shape (17, 2) COCO-17
    scores: np.ndarray  # shape (17,)
    frame_shape: tuple[int, int]  # (H, W)
    keypoints_3d: Optional[np.ndarray] = (
        None  # shape (17, 3) when MotionBERT result is attached
    )
    attention: Optional[np.ndarray] = None  # shape (H, W) heatmap from RTMPose


@dataclass
class RuleViolation:
    name: str
    severity: float  # 0.0 .. 1.0
    detail_th: str  # short Thai phrase, used in LLM prompt as a hint


@dataclass
class RepAnalysis:
    rep_index: int
    score: int  # 0..100
    components: dict[str, int]  # depth/valgus/torso/symmetry/tempo
    violations: list[RuleViolation]
    descent_ms: int
    ascent_ms: int
    bottom_frame_keypoints_2d: Optional[np.ndarray] = None
    bottom_frame_keypoints_3d: Optional[np.ndarray] = None
    bottom_frame_image: Optional[np.ndarray] = None  # BGR HxWx3, optional for VLM
    metric_source: str = "3d"  # "3d" (primary) or "2d" (fallback)


class HoldState(Enum):
    IDLE = "idle"
    ENTERING = "entering"
    HOLDING = "holding"
    DRIFTED = "drifted"
    COMPLETE = "complete"


@dataclass
class HoldAnalysis:
    exercise_name: str
    score: int  # 0..100
    components: dict[str, int]  # duration / precision / stability
    violations: list[RuleViolation]
    in_target_ms: int
    drift_count: int


@dataclass
class LiveSnapshot:
    exercise_name: str
    state: HoldState
    progress_ratio: float  # 0.0 .. 1.0
    current_violations: list[RuleViolation]

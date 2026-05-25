from dataclasses import dataclass, field
from typing import Optional, Protocol

from analysis.camera_view import CameraView
from analysis.types import PoseFrame
from calibration import BaselinePose

_DEFAULT_VALID_VIEWS: tuple[CameraView, ...] = (
    CameraView.FRONT,
    CameraView.THREE_QUARTER,
    CameraView.SIDE,
)


@dataclass(frozen=True)
class JointTarget:
    """One joint angle that must be within range during the hold."""

    name: str
    target_deg: float
    tolerance_deg: float
    detail_th: str  # Thai hint shown if out-of-target


@dataclass(frozen=True)
class PromptTemplate:
    """Per-exercise Thai prompt templates."""

    live: str  # rendered with LiveSnapshot fields
    summary: str  # rendered with HoldAnalysis fields


@dataclass(frozen=True)
class TargetPose:
    joints: tuple[JointTarget, ...]
    hold_seconds: float = 20.0
    side: str | None = None  # "left" / "right" / None for symmetric
    valid_views: tuple[CameraView, ...] = field(
        default_factory=lambda: _DEFAULT_VALID_VIEWS
    )
    """Which camera framings the FSM is allowed to score in. Defaults to all
    three views for backward compatibility; exercises that depend on a
    specific framing (e.g. CVA needs side view) should narrow this. The FSM
    stays in IDLE with a "rotate your camera" coaching message when the
    current view falls outside this set. See `src/analysis/camera_view.py`."""


class Exercise(Protocol):
    name: str
    display_th: str
    target: TargetPose
    prompt: PromptTemplate

    def measure(
        self,
        frame: PoseFrame,
        baseline: Optional[BaselinePose] = None,
    ) -> dict[str, float]:
        """Return {joint_name: degrees} for every joint in self.target.joints.
        Pure function of one frame. May return NaN if keypoints unavailable.

        `baseline` (optional) is the user's per-session neutral pose. When
        provided, implementations should compute deltas-from-neutral instead
        of absolute angles, so `target.joints[*].target_deg` is interpreted
        as "N degrees away from the user's natural posture." When `None`,
        absolute-angle behavior is preserved (backward compatible)."""
        ...

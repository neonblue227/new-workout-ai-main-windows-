"""Camera-view classifier — distinguishes front / three-quarter / side framings
from a single 2D keypoint observation.

Some posture metrics only work in a specific view (CVA → side / three-quarter;
chest opening / shoulder elevation → front). Scoring an exercise in the wrong
view silently produces wrong answers. This module lets the FSM refuse to leave
IDLE when the view doesn't match what the active exercise needs.

Heuristic source: posture-coach prior-art synthesis 2026-05-23. Two signals:

1. **Shoulder geometry ratio** — `|R_shoulder.x - L_shoulder.x|` / `|Δy|`. In a
   front view this is large (shoulders project wide, almost flat in y). In a
   side view it collapses (one shoulder occludes the other, the projected
   width shrinks toward zero).
2. **Ear visibility** — front views show both ears at usable confidence; side
   views show at most one ear because the other is occluded by the head.

Plan: `docs/superpowers/plans/2026-05-23-cpu-and-2d-improvements.md` Task B3.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from analysis.angles import L_EAR, L_SHOULDER, R_EAR, R_SHOULDER

# Tunable thresholds — calibrated against synthetic fixtures (see
# tests/pipeline/test_camera_view.py). Adjust if real-camera observations
# show consistent mis-classification.
_EAR_CONFIDENCE_THRESHOLD = 0.3
_SHOULDER_CONFIDENCE_THRESHOLD = 0.3
_FRONT_SHOULDER_RATIO = 10.0  # |shoulder_dx / shoulder_dy| ≥ this → front
_SIDE_SHOULDER_RATIO = 4.0  # ≤ this → side (when ear visibility is also collapsed)


class CameraView(str, Enum):
    """Categorical camera-view label. `StrEnum` semantics so it formats as the
    string value when interpolated into Thai coaching messages."""

    FRONT = "front"
    THREE_QUARTER = "three_quarter"
    SIDE = "side"
    UNKNOWN = "unknown"


def classify_view(
    kps_2d: np.ndarray,
    scores: np.ndarray,
    shoulder_confidence_threshold: float = _SHOULDER_CONFIDENCE_THRESHOLD,
    ear_confidence_threshold: float = _EAR_CONFIDENCE_THRESHOLD,
) -> CameraView:
    """Classify the camera framing from a single COCO-17 keypoint observation.

    Returns `UNKNOWN` when shoulder confidence is too low for the heuristic to
    fire. Otherwise returns `FRONT` / `THREE_QUARTER` / `SIDE` based on the
    shoulder geometry ratio and ear visibility.
    """
    if (
        scores[L_SHOULDER] < shoulder_confidence_threshold
        or scores[R_SHOULDER] < shoulder_confidence_threshold
    ):
        return CameraView.UNKNOWN

    shoulder_dx = abs(float(kps_2d[R_SHOULDER, 0] - kps_2d[L_SHOULDER, 0]))
    shoulder_dy = abs(float(kps_2d[L_SHOULDER, 1] - kps_2d[R_SHOULDER, 1]))
    # Add a small epsilon so a perfectly horizontal line doesn't divide by zero.
    ratio = shoulder_dx / max(shoulder_dy, 1e-3)

    l_ear_visible = float(scores[L_EAR]) >= ear_confidence_threshold
    r_ear_visible = float(scores[R_EAR]) >= ear_confidence_threshold
    n_ears = int(l_ear_visible) + int(r_ear_visible)

    # Strong side-view signal: shoulders almost overlap horizontally AND
    # at most one ear is visible.
    if ratio <= _SIDE_SHOULDER_RATIO and n_ears <= 1:
        return CameraView.SIDE

    # Strong front signal: shoulders are wide and flat AND both ears visible.
    if ratio >= _FRONT_SHOULDER_RATIO and n_ears == 2:
        return CameraView.FRONT

    # Everything else is three-quarter — the middle ground where partial-view
    # posture metrics like neck-lateral-tilt-2d still work but CVA + chest
    # opening become unreliable.
    return CameraView.THREE_QUARTER

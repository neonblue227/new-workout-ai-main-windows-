"""Per-user neutral-pose calibration.

Captures a short window of clean frames at session start and computes a
`BaselinePose` containing the user's *own* neutral values (shoulder width in
pixels, neutral head-lateral tilt, neutral shoulder-y asymmetry). Downstream
exercise scoring expresses targets as offsets from this baseline rather than
as population-average absolute angles, so the same prescription works across
users / camera distances / sitting heights without re-tuning hard-coded
constants.

Pattern lifted from the 2D-direct posture-coach prior art reviewed on
2026-05-23 (NeckWatcher / Zen / pose-nudge — see synthesis agent #4). Plan:
`docs/superpowers/plans/2026-05-23-cpu-and-2d-improvements.md` Task B1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from analysis.angles import (
    L_SHOULDER,
    NOSE,
    R_SHOULDER,
    head_lateral_tilt_2d,
)

# Calibration is the one place where we want stricter-than-usual gates: a single
# bad frame inside the average can pull neutral measurements off by degrees.
CALIBRATION_CONFIDENCE_FLOOR = 0.6


class CalibrationError(RuntimeError):
    """Raised when calibration cannot produce a usable baseline."""


@dataclass(frozen=True)
class BaselinePose:
    """User-specific neutral-pose reference, captured once per session.

    All fields are computed in image-pixel space (matches the 2D-direct
    measurement convention in `analysis.angles`). Stored as a snapshot
    of the user's "natural" sitting/standing posture; downstream measurements
    are reported as deltas from this baseline.
    """

    shoulder_width_px: float
    """L→R shoulder pixel distance averaged across clean samples."""

    head_lateral_tilt_deg: float
    """`head_lateral_tilt_2d` averaged across clean samples — the user's
    natural neck tilt (often non-zero due to habitual posture)."""

    shoulder_y_delta_norm: float
    """(L_shoulder.y − R_shoulder.y) / shoulder_width, averaged. Captures
    the user's habitual shoulder asymmetry, used by shoulder-asymmetry
    metrics introduced in plan Task B2."""

    sample_count: int
    """Number of samples that survived the confidence filter."""

    captured_ts: float
    """`time.monotonic()` at the moment the baseline was committed."""


def _sample_is_clean(scores: np.ndarray, kps: np.ndarray) -> bool:
    """Return True if a single frame is usable for calibration."""
    if scores[NOSE] < CALIBRATION_CONFIDENCE_FLOOR:
        return False
    if scores[L_SHOULDER] < CALIBRATION_CONFIDENCE_FLOOR:
        return False
    if scores[R_SHOULDER] < CALIBRATION_CONFIDENCE_FLOOR:
        return False
    shoulder_dx = float(kps[R_SHOULDER, 0] - kps[L_SHOULDER, 0])
    if abs(shoulder_dx) < 1.0:
        return False  # shoulders coincident — division-by-zero risk downstream
    return True


def calibrate_from_samples(
    samples: list[tuple[np.ndarray, np.ndarray]],
    min_clean_frames: int = 30,
) -> BaselinePose:
    """Pure function: build a `BaselinePose` from a sequence of (kps, scores) pairs.

    Filters out frames where nose / shoulder confidence is below
    `CALIBRATION_CONFIDENCE_FLOOR` or where shoulders are coincident. Raises
    `CalibrationError` if fewer than `min_clean_frames` survive — calibration
    should fail loudly rather than silently produce a noisy baseline.
    """
    widths: list[float] = []
    tilts: list[float] = []
    shoulder_deltas: list[float] = []
    for kps, scores in samples:
        if not _sample_is_clean(scores, kps):
            continue
        sw = float(kps[R_SHOULDER, 0] - kps[L_SHOULDER, 0])
        widths.append(abs(sw))
        tilts.append(head_lateral_tilt_2d(kps, scores))
        sy_delta = float(kps[L_SHOULDER, 1] - kps[R_SHOULDER, 1])
        shoulder_deltas.append(sy_delta / abs(sw))

    if len(widths) < min_clean_frames:
        raise CalibrationError(
            f"only {len(widths)} clean calibration frames; need ≥ {min_clean_frames}. "
            "Make sure your head + shoulders are well-lit and within the camera frame."
        )

    return BaselinePose(
        shoulder_width_px=float(np.mean(widths)),
        head_lateral_tilt_deg=float(np.mean(tilts)),
        shoulder_y_delta_norm=float(np.mean(shoulder_deltas)),
        sample_count=len(widths),
        captured_ts=time.monotonic(),
    )


def calibrate(
    capture,
    pose2d,
    duration_s: float = 5.0,
    target_fps: int = 30,
    min_clean_frames: int = 30,
    sleep_fn=time.sleep,
    now_fn=time.monotonic,
) -> BaselinePose:
    """Live-camera calibration. Records frames for `duration_s` seconds at
    approximately `target_fps`, then defers to `calibrate_from_samples`.

    `capture` is anything that supports `read_latest(timeout) -> ndarray | None`
    (i.e. `WebcamCapture`). `pose2d` is the project's `Pose2D` instance — its
    `.infer(frame_bgr)` is called to produce (kps, scores). `sleep_fn` / `now_fn`
    are dependency-injection hooks for tests.
    """
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    interval_s = 1.0 / max(target_fps, 1)
    end_ts = now_fn() + duration_s
    while now_fn() < end_ts:
        frame = capture.read_latest(timeout=0.5)
        if frame is None:
            continue
        kps, scores = pose2d.infer(frame)
        samples.append((kps, scores))
        sleep_fn(interval_s)
    return calibrate_from_samples(samples, min_clean_frames=min_clean_frames)

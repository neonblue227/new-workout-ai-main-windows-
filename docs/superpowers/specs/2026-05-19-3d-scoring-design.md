# 3D Scoring Rewrite — Design

Date: 2026-05-19
Status: Draft, pending review

## Background and motivation

`analysis/rules_squat.py::score_rep` and the helpers in `analysis/angles.py` all consume `bottom_frame.keypoints_2d`. Image-pixel coordinates are fragile under camera-angle change:

| Metric | Failure mode |
|---|---|
| Valgus (`knee_valgus_ratio`) | Normalizes by projected hip width. In side view the projected width collapses, and normal forward knee travel registers as severe valgus. |
| Depth (recently switched to 2D `knee_angles`) | 2D knee flexion approximates 3D flexion only when the camera is near-perpendicular to the leg's plane of motion. Degrades off-axis. |
| Torso lean | Uses image vertical. Fine from side, meaningless from front. |
| Symmetry | Compares L vs R 2D knee angles. Collapses from front view (legs project on top of each other). |

The production target is a phone-streamed feed. Mounting angle is unpredictable; we cannot assume a perfect side or perfect front view.

The pipeline already computes 3D pose via MotionBERT-Lite (H36M-17 layout, pelvis-rooted, lifted every 5 frames into `PoseFrame.keypoints_3d`). This rewrite consumes that output for all spatial scoring.

## Goals

1. Depth, valgus, and symmetry become fully camera-angle-independent (depend only on body geometry, not on where the camera sits).
2. Torso lean becomes camera-yaw / pitch-independent. It still depends on the camera's roll being near zero (a "phone is mounted upright" assumption — see §Torso lean). Removing this last dependency is future work.
3. Same component budget (30 / 25 / 20 / 15 / 10 = 100) and the same `RepAnalysis` schema — drop-in replacement for the LLM prompt and any downstream consumer.
4. 2D fallback when `keypoints_3d` is None so scoring still works during MotionBERT's startup window or if the lift fails.

## Non-goals

- FSM phase detection (`analysis/phases.py`) stays on 2D `knee_angles`. (Decided 2026-05-19; captured in §Future work.)
- Tempo. Already temporal, unchanged.
- Empirical threshold calibration against a large dataset. Initial thresholds are eyeballed from a handful of sample frames; tuning is follow-up work.

## Coordinate frame

MotionBERT outputs H36M-17 in a pelvis-rooted normalized space with image-y down (a single frame's 3D y values therefore range roughly −0.5 to +0.5 with negative = up in the world, positive = down). Indices used in this doc:

```
0  pelvis      4  L_hip       7  spine
1  R_hip       5  L_knee      8  thorax
2  R_knee      6  L_ankle
3  R_ankle
```

We define body-frame unit axes derived per-frame from the 3D pose itself, independent of the camera frame:

- `body_up` = normalize(thorax − pelvis). Points head-ward along the torso.
- `body_lateral` = normalize((R_hip − L_hip) − ((R_hip − L_hip) · body_up) · body_up). The hip-line projected orthogonal to body_up so the basis is clean.
- `body_forward` = body_up × body_lateral. Sagittal axis (forward through the chest).

A body posture produces the same metric values regardless of where the camera is, because all spatial measurements project onto these body-frame axes rather than the camera's image axes. The single exception is torso lean (see below).

## Metric specifications

### Depth (30 pts) — 3D

3D knee flexion: angle at each knee between (knee → hip) and (knee → ankle) using full 3D vectors.

```
DEPTH_FULL_DEG = 90.0     # parallel or deeper: full credit
DEPTH_ZERO_DEG = 130.0    # very shallow: zero
mean_knee = (knee_flex_3d_L + knee_flex_3d_R) / 2
```

Linear ramp from full (≤ 90°) to zero (≥ 130°) — same shape as the existing 2D fix. The improvement is reliability when the camera is off-axis (the 2D knee angle distorts as the femur rotates relative to the image plane; the 3D angle does not).

Violation: `shallow_depth`, severity = 1 − ratio, `detail_th` unchanged.

### Valgus (25 pts) — 3D, frontal plane

For each leg, project hip, knee, ankle onto the frontal plane (the plane spanned by `body_lateral` and `body_up`). In that plane, compute the perpendicular signed offset from the knee to the line connecting hip and ankle. Positive sign = medial (knee toward body midline), negative = lateral (knee outside the line). Normalize by shin length.

```
VALGUS_THRESHOLD = 0.10   # |ratio| above this triggers a violation
VALGUS_SAT       = 0.30   # |ratio| at this saturates severity to 1.0
ratio_L, ratio_R = valgus_offset_3d(kps_3d)   # signed, normalized
worst = max(ratio_L, ratio_R)                  # SIGNED max — only medial counts
severity = clamp((worst − VALGUS_THRESHOLD) / (VALGUS_SAT − VALGUS_THRESHOLD), 0, 1)
points = round(25 * (1 − severity))
```

Two deliberate differences from the 2D version:

1. **Sign-aware.** Knees flaring outward (lateral, valgus < 0) is not valgus — it's the opposite. Today's `worst_valgus = max(abs(l), abs(r))` penalizes both directions, which is wrong. The 3D rewrite uses the signed max so only medial deviation counts.
2. **Normalized by shin length, not hip width.** Shin length is a stable body-frame scalar; projected hip width is not.

Threshold `0.10` is an initial guess. Needs calibration on a small set of squats with known valgus / neutral / lateral knee tracking.

Violation: `knee_valgus`, `detail_th` unchanged.

### Torso lean (20 pts) — 3D vs image up

Angle between `body_up` and image vertical (`(0, -1, 0)` in MotionBERT's frame).

```
world_up_proxy = (0.0, -1.0, 0.0)
lean_deg = angle_between(body_up, world_up_proxy)
```

Same ramp as today: full credit in [20°, 55°], smooth penalty outside.

**Caveat.** "Image up" is only "world up" when the camera roll is near zero (the typical phone-on-tripod case). If the camera is rotated, this metric drifts. An anatomical vertical reference (e.g., average ankle → hip direction) would remove that dependency but introduces its own instabilities and isn't necessary for the current target. Listed as future work.

Violations: `excessive_forward_lean` / `too_upright`, both unchanged.

### Symmetry (15 pts) — 3D

L vs R 3D knee flexion delta.

```
delta = abs(knee_flex_3d_L - knee_flex_3d_R)
```

Same ramp as today: full credit if delta < 10°, linear to 0 at delta = 30°. The 3D version is correct under front-view footage (where the 2D version collapses).

Violation: `asymmetric`, `detail_th` unchanged.

### Tempo (10 pts)

Unchanged. Already a function of timestamps, not coordinates.

## Module layout

**New:** `src/analysis/angles_3d.py`

```python
def body_frame_axes(kps_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (body_up, body_lateral, body_forward) as unit vectors in MotionBERT's 3D frame."""

def knee_flexion_3d(kps_3d: np.ndarray) -> tuple[float, float]:
    """Per-knee flexion angle in degrees, computed in full 3D. Returns (left, right)."""

def valgus_offset_3d(kps_3d: np.ndarray) -> tuple[float, float]:
    """Signed perpendicular offset of knee from hip-ankle line in the frontal plane,
    normalized by shin length. Positive = medial. Returns (left, right)."""

def torso_lean_3d(kps_3d: np.ndarray) -> float:
    """Angle in degrees between body_up and image up."""
```

**Modified:** `src/analysis/rules_squat.py`

- `score_rep` becomes a thin dispatcher: if `bottom_frame.keypoints_3d is not None`, call `_score_rep_3d`; otherwise fall back to today's logic, renamed `_score_rep_2d`.
- `_score_rep_3d` mirrors today's structure but reads the 3D metrics from `angles_3d` and uses the constants in §Metric specifications.
- The two paths return identically shaped `RepAnalysis` objects.

**Modified:** `src/analysis/types.py`

- Add `metric_source: str = "3d"` to `RepAnalysis`. Set to `"2d"` on the fallback path. Used only for observability — the LLM prompt does not see it.

**Unchanged:** `src/analysis/angles.py`, `src/analysis/phases.py`

- 2D helpers stay where they are; FSM continues to drive transitions off `knee_angles(kps_2d)`. Moving the FSM to 3D is future work.

## Fallback policy

If `bottom_frame.keypoints_3d is None`:

- Score via the existing 2D logic (preserved verbatim as `_score_rep_2d`).
- Set `metric_source = "2d"` on the returned `RepAnalysis`.

In practice, by the time a rep completes (multi-second descent + ascent through STANDING → DESCENT → BOTTOM → ASCENT → STANDING), MotionBERT's 27-frame buffer has long since filled and 3D is always present. The fallback exists for defensive reasons and to allow unit tests to exercise the 2D path without constructing 3D fixtures.

## Test plan

**`tests/test_angles_3d.py` (new) — unit tests on synthetic poses:**

- `body_frame_axes` returns an orthonormal triple for a canonical upright standing pose.
- `knee_flexion_3d` recovers 90°, 130°, and 180° on synthetic legs with known geometry.
- `valgus_offset_3d` is ≈ 0 for a symmetric perfect-form pose; positive for a hand-built medial-knee pose; negative for a lateral pose.
- `torso_lean_3d` recovers known angles on synthetic torsos at 0°, 30°, 60°.

**`tests/test_rules_squat.py` — updated:**

- Existing tests construct `PoseFrame` without `keypoints_3d`. They continue to pass through the 2D fallback.
- New tests construct `PoseFrame` with hand-built `keypoints_3d` and exercise the 3D path:
  - Side-view-fooled valgus: a pose where the 2D `knee_valgus_ratio` would fire but the 3D `valgus_offset_3d` does not. Asserts `valgus == 25` and no `knee_valgus` violation.
  - Genuine medial knee: 3D pose with knee inside hip-ankle line in the frontal plane. Asserts `valgus < 25` and `knee_valgus` violation present.
  - Asymmetric squat: L knee at 90°, R knee at 130° in 3D. Asserts `symmetry < 15` and `asymmetric` violation present.

## Calibration / open questions

- **Valgus threshold (0.10).** Initial guess. Should be calibrated by running the pipeline on (a) the existing side-view sample image, (b) a frontal-view squat with neutral knee tracking, (c) a frontal-view squat with known medial knee collapse. The threshold should fire on (c), not on (a) or (b).
- **Torso lean reference.** If we see phone-roll problems in real footage, swap the image-up reference for an anatomical-vertical reference (averaged ankle → hip direction). Out of scope for this rewrite.
- **3D pose accuracy on stills.** MotionBERT-Lite is the smallest variant; for the demo notebook (tiling a single frame across the 27-frame window), output is approximate. For live video this isn't a concern.

## Future work

- Move FSM (`analysis/phases.py`) to 3D knee flexion. Requires either per-frame 3D lift (costs FPS) or accepting ~5-frame lag in phase transitions.
- Anatomical vertical for torso lean (camera-roll robustness).
- Empirical threshold calibration against an annotated dataset.
- Confidence flag per metric (e.g., suppress valgus on the LLM prompt when the lateral axis is poorly resolved).

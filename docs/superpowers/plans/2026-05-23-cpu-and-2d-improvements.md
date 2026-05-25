# CPU Reduction + 2D-Direct Measurement Coverage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal.** Two workstreams informed by the four-agent research synthesis (2026-05-23):

- **Workstream A — CPU reduction.** Cut runtime CPU usage of `app.run()` / `src/test_2D_3D.py` by tuning the existing ONNX runtime path and (carefully) revisiting GPU offload, without changing the 2D detector stack.
- **Workstream B — Measurement coverage + UX.** Add per-user calibration, expand the 2D-direct measurement cookbook (CVA, neck flexion, shoulder asymmetry, etc.), and add a camera-view classifier so each exercise scores only in the views where its measurement is valid.

**Decisions (load-bearing — do not change without re-opening the synthesis):**

- **2D stack stays on `rtmlib` + YOLOX-tiny + RTMPose-s.** User has decided against swapping to MediaPipe Pose Landmarker or any other detector. All "improve the 2D signal" work in this plan is within the current detector's confidence-only output.
- **3D lifter stays on MotionBERT-Lite.** The 4-agent synthesis confirmed no 2D→3D lifter solves the seated/partial-body problem. The 3D rig remains a *visualization*; measurements stay 2D-direct.
- **Mlx-vlm + Qwen3.5-4B unchanged.**
- **Apple Silicon, macOS.** ONNX runtime 1.26.0; `CoreMLExecutionProvider` is available. PyTorch MPS already in use for MotionBERT. The eventual server pivot (CLAUDE.md "Long-term direction") is out of scope here but no decision in this plan should rule it out.

**Tech Stack:** Python 3.12 (uv), pytest, numpy, OpenCV, PyTorch (MPS), mlx-vlm, rtmlib, MotionBERT-Lite, Qwen3.5-4B.

---

## Pre-flight

- [ ] **Sanity check the working tree**

  Run: `cd "/Users/thatt/Dev/AI project/new-workout-ai" && uv run pytest -q`
  Expected: 107 tests passing. If not, stop and report — don't start this plan on red.

- [ ] **Read the synthesis once before starting**

  Re-read the synthesis message in conversation history that lists the four agent reports (2D detectors, 3D lifters, end-to-end SMPL, posture-coach prior art). This plan implements the subset of recommendations that *don't* require detector swap.

---

# Workstream A — CPU Reduction

## Task A1: Establish a CPU & latency baseline

**Why first:** Without numbers it's impossible to know whether later tasks actually helped. Most optimization wins are smaller than you'd guess and several are negative on Apple Silicon — measurement gates everything.

**Files:**
- Add: `scripts/profile_pipeline.py` (single-script profiler, not a test)

- [ ] **Step 1: Write the profiler script**

  The script should drive a `Pose2D` + `Pose3D` + `Renderer` pipeline against a saved fixture image (`data/neck_stretch/neck_stretch_01.jpg`) in a tight loop and print:
    - Mean & p95 wall-clock per stage: `pose2d.infer()`, `coco17_to_h36m17`, `Pose3DBuffer.lift()`, `head_lateral_tilt_2d()`, full canvas composition, `cv2.imshow` + `cv2.waitKey(1)`.
    - Total per-frame CPU time (sum of stages).
    - macOS-specific: optionally use `psutil.Process().cpu_percent()` over a 5 s window.
  Run for 600 frames warm. Print results as a markdown table on stdout.

- [ ] **Step 2: Record baseline numbers**

  Save the output to `docs/perf/2026-05-23-baseline.md`. Each subsequent A-task compares against this.

  Acceptance: a baseline table exists with at least the stages above timed.

---

## Task A2: Tune ONNX runtime thread count

**Why:** ONNX runtime defaults to `intra_op_num_threads = num_cpu_cores`. On an M-series with 8-12 cores this spins all cores even though RTMPose-s is tiny enough that 2-4 threads finish in nearly the same wall-clock time. This is consistently the single biggest CPU-usage drop in real-time ONNX pipelines on Apple Silicon, with near-zero latency cost.

**Files:**
- Modify: `src/pose2d.py`

- [ ] **Step 1: Inspect rtmlib's session-creation path**

  Read `<rtmlib package>/tools/base.py`. The `Body` / `RTMPose` / `YOLOX` classes each instantiate `onnxruntime.InferenceSession(...)` — but they do NOT expose `SessionOptions`. We will need to monkey-patch the session(s) after `Body()` constructs.

- [ ] **Step 2: Add a thread-count knob to `Pose2D.__init__`**

  Add a kwarg `onnx_threads: int = 4` to `Pose2D.__init__`. After `Body(...)` constructs, reach into `self._body.det_model.session` and `self._body.pose_model.session` and set `session.set_providers([...])` is **not** what we want; instead we need to recreate the session with `SessionOptions(intra_op_num_threads=onnx_threads, inter_op_num_threads=1)`. The simplest path: pull `session._model_path` and recreate via `onnxruntime.InferenceSession(model_path, sess_options=opts, providers=session.get_providers())`. Reassign back.

- [ ] **Step 3: Test 1, 2, 4, 6, 8 thread settings**

  Run the A1 profiler with each setting. Pick the lowest thread count whose p95 latency stays within +10% of the 8-thread baseline. Record the chosen value as a `_ONNX_THREADS_DEFAULT` constant in `pose2d.py`.

  Acceptance: CPU usage of `app.run()` drops by ≥30% over baseline. p95 inference latency stays within +10%. A table comparing thread counts is committed to `docs/perf/`.

---

## Task A3: Cap pose inference cadence

**Why:** Hold-mode stretches don't need 30 Hz detection — they need to see the user moving into / out of target, which is captured fine at 15 Hz. Camera capture stays at 30 Hz; pose inference runs every other frame.

**Files:**
- Modify: `src/app.py`, `src/test_2D_3D.py`

- [ ] **Step 1: Add a `pose_inference_hz` knob**

  In `app.run_session()`, track `last_pose_ts`. Skip `pose2d.infer()` when `now - last_pose_ts < 1 / pose_inference_hz`. Reuse the previous `kps` / `scores` for the FSM step.

  Defaults: `pose_inference_hz = 15` for hold mode. The `HoldFSM` stability gates already absorb the lower update rate — no FSM change needed.

  Same change applies to `test_2D_3D.py` for the diagnostic loop.

- [ ] **Step 2: Acceptance**

  Run A1 profiler with `pose_inference_hz=15`. ORT call count per second roughly halves. Total CPU% drops measurably. FSM transitions in `test_hold_fsm.py` still pass (this is a runtime-cadence change, no test impact expected).

---

## Task A4: Retry CoreML EP for RTMPose-only (gated experiment)

**Why:** Onnxruntime 1.26 may have fixed the dynamic-shape NMS bug that broke `device="mps"` for the YOLOX detector. Even if YOLOX still crashes, the RTMPose pose-estimator stage is heavier (per-joint simcc head) and might run cleanly under CoreML EP on its own, leaving YOLOX on CPU.

**This is an experiment.** It either delivers a measurable speedup, or it surfaces that the bug persists and we document and move on. Do not commit a CoreML-default change without a green A1 profile.

**Files:**
- Modify (experimental, behind flag): `src/pose2d.py`
- Add: `docs/perf/2026-05-23-coreml-experiment.md` documenting outcome

- [ ] **Step 1: Try `device="mps"` on the full Body first**

  Bypass the existing `Pose2D` warning comment temporarily: instantiate `Body(mode="lightweight", backend="onnxruntime", device="mps")` and run the A1 profiler with the standing-person fixture. Record whether YOLOX still crashes, and on which input.

- [ ] **Step 2: If full-Body CoreML crashes, try CoreML for pose-only**

  After `Body()` constructs with `device="cpu"`, reach into `body.pose_model` and recreate its ONNX session with `providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]`. Detection stays on CPU. Run profiler.

- [ ] **Step 3: Decide based on numbers**

  - If CoreML pose-only is ≥20% faster AND no crashes over 10k frames: add a `device_pose="coreml"` kwarg to `Pose2D`, default to CPU, document the option. Don't change defaults — we want to ship stability.
  - If no speedup or instability: write up the negative result in the perf doc, leave the existing warning comment in `pose2d.py` updated with the date/onnxruntime version retested, close task.

  Acceptance: a perf doc records exact numbers (latency, CPU%, frames-to-failure) and a clear go/no-go decision. CPU defaults stay green.

---

## Task A5: Reduce OpenCV display overhead

**Why:** `cv2.imshow` + `cv2.waitKey(1)` on macOS is known to be heavier than the actual rendering work, because every frame triggers a full GUI refresh through Cocoa. The 3-panel canvas in `test_2D_3D.py` (1920 × 560 BGR uint8) is also being re-allocated and re-drawn from scratch every frame.

**Files:**
- Modify: `src/test_2D_3D.py`, `src/render.py`

- [ ] **Step 1: Reuse the canvas buffer**

  Allocate the panel canvases once outside the loop (`np.zeros((H + PANEL_PAD, W, 3), uint8)`); in each frame, zero out only the regions that need redrawing instead of recreating with `_make_panel`.

- [ ] **Step 2: Decouple display rate from inference rate**

  Display can stay at 30 Hz (smooth feed); inference / panel-composition can run at 15 Hz (A3) and the display loop re-uses the last composed canvas in between.

- [ ] **Step 3: Acceptance**

  A1 profiler shows the canvas-composition + imshow stages drop by ≥30%. No visible UI regression — the live feed still feels smooth.

---

# Workstream B — Calibration + Measurement Coverage

## Task B1: Per-user neutral-baseline calibration

**Why:** Posture-coach research synthesis recommendation #2. Every reviewed open-source posture-coach (NeckWatcher, pose-nudge, Zen) uses per-user baseline, not population-average targets. Replaces hard-coded `_NECK_TILT_TARGET_LEFT_DEG = -35.0` with "your neutral + 35°", which closes the calibration TODO already sitting in `src/exercises/neck_stretch.py:6`.

**Files:**
- Add: `src/calibration.py` (`BaselinePose` dataclass + `calibrate()` function)
- Modify: `src/app.py` (insert calibration step before `HoldFSM`)
- Modify: `src/analysis/types.py` (add `BaselinePose` if not in `calibration.py`)
- Modify: `src/exercises/neck_stretch.py` (consume baseline; compute target as `baseline + offset` instead of absolute)
- Test: `tests/office_syndrome/test_calibration.py`

- [ ] **Step 1: Design `BaselinePose` dataclass**

  ```python
  @dataclass(frozen=True)
  class BaselinePose:
      shoulder_width_px: float
      head_lateral_tilt_deg: float       # neutral CVA-frontal proxy
      shoulder_y_delta_norm: float       # (L_sh.y - R_sh.y) / shoulder_width
      sample_count: int                  # frames averaged
      captured_ts: float
  ```

- [ ] **Step 2: Write `calibrate(capture, pose2d, duration_s=5) -> BaselinePose`**

  Capture `duration_s` of frames, average the per-frame measurements after filtering frames where any of nose / L_sh / R_sh confidence is below 0.6 (higher than measurement threshold — calibration should only see clean frames). If fewer than `0.5 * 30 * duration_s` clean frames captured, raise — user wasn't in frame.

- [ ] **Step 3: Wire into `app.run_session`**

  Before entering the FSM loop: render a "Sit naturally and hold still" coaching message, run `calibrate()`, then start the FSM with the baseline available to the exercise's `measure()`.

- [ ] **Step 4: Update `NeckStretchLeft.measure` to use the baseline**

  Add a `baseline: BaselinePose` kwarg to the `Exercise` protocol's `measure()`. Adjust `NeckStretchLeft` so its tilt is computed as `(absolute_tilt - baseline.head_lateral_tilt_deg)`. The `target_deg = -35` then means "−35° below your own neutral", which is much more robust per-user.

- [ ] **Step 5: Acceptance**

  Unit tests assert: calibration captures all four `BaselinePose` fields; raises when too few clean frames; `NeckStretchLeft.measure` returns 0° when input == baseline. `test_2D_3D.py` shows the baseline overlay in the camera panel.

---

## Task B2: Expand 2D measurement cookbook

**Why:** Synthesis recommendation #3 + the agent 4 cookbook. Adds the building blocks for the remaining 9 office-syndrome exercises without touching the 3D path. Each formula is from validated literature (CVA validated at Pearson r > 0.98 vs goniometer).

**Files:**
- Modify: `src/analysis/angles.py`
- Test: `tests/pipeline/test_angles.py`

Add these pure functions (each follows the `head_lateral_tilt_2d` shape: `(kps_2d, scores, score_threshold=0.3) -> float`, NaN-gated on the keypoints they read):

- [ ] **Step 1: `craniovertebral_angle_2d(kps_2d, scores, side="auto")`**

  Forward-head metric. `atan2(shoulder_y - ear_y, ear_x - shoulder_x)`. `side="auto"` picks the ear with higher confidence; `side="left"` / `"right"` forces. Healthy CVA ≥ 50°. Best in side / three-quarter view.

- [ ] **Step 2: `forward_head_offset_normalized_2d(kps_2d, scores)`**

  Frontal-view fallback. `(ear_x - shoulder_x) / shoulder_width`. Per-side; report max of both. Threshold ~0.30.

- [ ] **Step 3: `neck_flexion_2d(kps_2d, scores, side="auto")`**

  Angle between `shoulder → ear` and image vertical `(0, -1)`. Healthy < 25°.

- [ ] **Step 4: `shoulder_elevation_asymmetry_2d(kps_2d, scores)`**

  `(L_shoulder_y - R_shoulder_y) / shoulder_width`. Healthy < ±0.05.

- [ ] **Step 5: `shoulder_protraction_ratio_2d(kps_2d, scores, baseline_width_px)`**

  `current_shoulder_width / baseline_width`. Pulls shoulders forward → ratio drops. Requires `BaselinePose` from B1.

- [ ] **Step 6: `wrist_extension_2d(kps_2d, scores, side)`**

  `wrist_y - elbow_y` (signed; positive = wrist below elbow in image coords, i.e. neutral).

- [ ] **Step 7: Acceptance**

  Each function has at least three unit tests: neutral position returns expected value, exaggerated posture returns expected sign + magnitude, NaN gating fires when source keypoints are below threshold. Use the synthetic-fixture pattern already established in `tests/pipeline/test_angles.py` for `head_lateral_tilt_2d`.

---

## Task B3: Camera-view classifier

**Why:** Synthesis recommendation #4. Several measurements above only work in a specific view (CVA → side / three-quarter; chest opening → front). Scoring in the wrong view silently produces wrong answers. Explicit gating is a UX win.

**Files:**
- Add: `src/analysis/camera_view.py`
- Test: `tests/pipeline/test_camera_view.py`

- [ ] **Step 1: Define a `CameraView` enum**

  ```python
  class CameraView(StrEnum):
      FRONT = "front"
      THREE_QUARTER = "three_quarter"
      SIDE = "side"
      UNKNOWN = "unknown"
  ```

- [ ] **Step 2: Write `classify_view(kps_2d, scores) -> CameraView`**

  Heuristic using shoulder-width-to-shoulder-y-delta ratio plus ear visibility:
    - FRONT: shoulder_width_px / |L_shoulder.y - R_shoulder.y| > 8 AND both ears visible at conf ≥ 0.3
    - SIDE: shoulder_width_px / |L_shoulder.y - R_shoulder.y| < 3 OR exactly one ear visible
    - THREE_QUARTER: in between
    - UNKNOWN: when shoulders themselves are unreliable

- [ ] **Step 3: Acceptance**

  Unit tests with synthetic keypoints for each view. Visualization in `test_2D_3D.py` overlays the classified view in the camera panel.

---

## Task B4: Wire view-gating into the Exercise protocol

**Files:**
- Modify: `src/exercises/base.py` (add `valid_views: tuple[CameraView, ...]` to `TargetPose` or `Exercise`)
- Modify: `src/exercises/neck_stretch.py` (declare valid views: `(FRONT, THREE_QUARTER)`)
- Modify: `src/app.py` (skip FSM update + show "rotate" coaching message when current view is not in `valid_views`)

- [ ] **Step 1: Add the field**

  Default to `(FRONT, THREE_QUARTER, SIDE)` so existing exercises don't break.

- [ ] **Step 2: Coach when in wrong view**

  In `run_session`, if `classify_view(...)` ∉ `exercise.target.valid_views`, render a Thai coaching message ("หันด้านข้างเล็กน้อย" for "rotate slightly") and keep the FSM in `IDLE`.

- [ ] **Step 3: Acceptance**

  Integration test (synthetic): given a front-view fixture, `NeckStretchLeft` runs. Given a side-view fixture, FSM stays IDLE and coaching message is set. No existing test breaks.

---

## Task B5: Normalize distance metrics by shoulder width

**Why:** Synthesis recommendation #5. Standard practice across the surveyed posture-coach projects. Scale-invariant across user-camera distances. Small change with cumulative effect.

**Files:**
- Modify: `src/analysis/angles.py`

- [ ] **Step 1: Audit pixel-distance usage**

  Grep `analysis/angles.py` and the new functions from B2 for any place that uses a raw pixel distance. Confirm each is either (a) inside an `atan2` where it cancels, or (b) explicitly normalized by `shoulder_width`.

- [ ] **Step 2: Acceptance**

  No test fails. Functions that depend on shoulder width return NaN if shoulder confidence is below threshold (already implemented for most).

  **Do not touch `pose3d._normalize_2d`** — that's MotionBERT-specific and intentional per `CLAUDE.md`.

---

## Final integration: B1 → B4 in `app.run_session`

- [ ] **End-to-end run-through**

  Once B1–B4 are landed: run `uv run python main.py`, pick neck stretch, verify the flow: select exercise → "sit naturally" calibration message → 5s baseline capture → coaching message switches to "tilt your head left" → FSM enters HOLDING when in target → completes after 20 s in target.

  No more hard-coded `_NECK_TILT_TARGET_LEFT_DEG = -35.0` as the only knob. Camera view classified and displayed. CPU usage measurably lower than baseline.

---

## Out of scope

- **Swapping the 2D detector** (MediaPipe, MoveNet, Apple Vision, Sapiens). Decided against per user preference (2026-05-23).
- **Swapping the 3D lifter.** Four-agent synthesis confirmed no off-the-shelf 2D→3D lifter solves the seated/half-body OOD problem. Stay 2D-direct.
- **End-to-end SMPL methods** (HMR2.0 / 4D-Humans). Defer until a stretch actually requires 3D (chest/shoulder rotation, spinal twists) AND someone has demonstrated ≥5 fps on M-series via a CoreML/ONNX export.
- **GPU canvas / drawing** (Metal, CoreImage). Out of scope unless A5 doesn't deliver enough.
- **Server pivot.** No decision in this plan should make it harder — but the work itself stays local-desktop.

---

## Acceptance summary

When this plan is done:
1. `docs/perf/2026-05-23-baseline.md` + per-task perf deltas exist and show measurable CPU reduction.
2. `uv run python main.py` performs a calibration step before each session.
3. `src/analysis/angles.py` exposes the full 2D measurement cookbook (B2) with full unit-test coverage.
4. Camera-view classifier gates exercises in the wrong view with a coaching message.
5. CoreML EP is either adopted (with measurable wins, behind an opt-in flag) or its negative result is documented.
6. All 107 existing tests still pass, plus ~20 new tests covering B1–B4.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Real-time webcam-based form coach for macOS Apple Silicon. Streams from the webcam, runs 2D pose → **2D-direct angle measurement** → rule-based form analysis, and produces Thai-language coaching feedback from an on-device VLM. A 3D lift (MotionBERT-Lite) still runs for the on-screen rig, but is **visualization-only** — see "2D-direct measurement" below for why the measurements no longer come from the 3D pose.

**Two exercise modes coexist in the codebase:**

- **Guided neck-stretch routine (current entry point)** — `app.run()` loads pose + LLM + TTS once, then loops `choose_routine()` → `run_neck_stretch_routine()`. The routine is driven by `RoutineFSM` (`src/routine.py`): SETUP (clickable Start) → POSITIONING (humanoid outline; 3s clean-frame window doubles as calibration) → COUNTDOWN (3·2·1 + spoken side cue) → HOLD (fixed 25s wall-clock; form measured per frame via 2D-direct tilt + `score_frame`) → TRANSITION → repeat for 4 alternating sides (left, right, left, right) → SUMMARY. Per-set scoring reuses `rules_hold.py` (`score_frame` per frame + `score_hold` at set end) + `NeckStretchLeft`/`NeckStretchRight` exercise data; the app accumulates in-target time and counts drift edges directly rather than driving `HoldFSM` (the fixed-wall-clock set makes its IDLE→…→COMPLETE lifecycle unnecessary). Audio is spoken Thai via `feedback/tts.py` (`GeminiTTS` + macOS `say` fallback, played by `TTSWorker`). Screens are drawn by `src/screens.py`. The 3D rig is not used in this demo. Design spec: `docs/superpowers/specs/2026-05-23-neck-stretch-realtime-demo-design.md`; plan: `docs/superpowers/plans/2026-05-23-neck-stretch-realtime-demo.md`.
- **Squat coaching (rep-based, preserved)** — original `SquatFSM` (STANDING → DESCENT → BOTTOM → ASCENT) + `analysis/rules_squat.py::score_rep`. LLM fires on rep completion. The code is intact and the tests still cover it, but **`main.py` no longer routes to it** — the squat flow is parked while the stretch slice is in active development. To re-expose squats from the entry point, register a squat-shaped `Exercise` (or add a CLI flag in `app.run`).

Weights (Qwen, MotionBERT, RTMPose) are loaded once and reused across all sets and routine loops. Spec for the office-syndrome stretch architecture: `docs/superpowers/specs/2026-05-22-office-syndrome-stretches-design.md`.

## Commands

This project uses `uv` (Python 3.12) — always invoke Python via `uv run`.

```bash
uv sync --all-extras                          # install + dev deps
uv run python scripts/download_models.py      # one-time, ~6 GB into ./models/
uv run python main.py                         # guided neck-stretch routine: Start screen → positioning/calibration → 4×25s alternating holds with spoken Thai cues → summary; q/Esc quits
uv run python main.py --source ip --url http://<phone-ip>:8080/video   # same routine, frames from an Android IP Webcam app over the LAN
uv run python -m camera --url http://<phone-ip>:8080/video             # standalone IP webcam preview (verify the phone stream first)

uv run python src/test_2D_3D.py               # 3-panel diagnostic: camera / 2D pose + metrics / 3D rig — live, with per-stage profiling + 2D cookbook + camera-view label (mirrored display)
uv run python scripts/profile_pipeline.py     # headless per-stage timing harness (CPU%, ms/stage); flags: --onnx-threads, --pose-stride, --output

uv run pytest                                 # run full test suite (169)
uv run pytest tests/office_syndrome           # single section (pipeline / squat / office_syndrome)
uv run pytest tests/pipeline/test_angles.py   # single file
uv run pytest -k "not smoke"                  # skip everything that loads heavy weights
uv run ruff check                             # lint
uv run ruff format                            # format
```

`pytest.ini_options` sets `pythonpath = ["src"]`, so tests import as `from analysis.* import ...`, `from feedback.* import ...`, `from app import ...`, etc. without an editable install.

## Architecture

Single-process pipeline driven by `src/app.py::run()`. Top-level `run()` loads pose + LLM + TTS weights once, then loops: `choose_routine()` (launcher) → `run_neck_stretch_routine()` (guided multi-set routine) → repeat.

### Neck-stretch demo flow (current entry point)

`app.run()` loads pose + LLM + TTS weights once, then loops `choose_routine()` →
`run_neck_stretch_routine()`. The routine is driven by the pure `RoutineFSM`
(`src/routine.py`): SETUP (clickable Start) → POSITIONING (humanoid outline; the
3s clean-frame window doubles as calibration) → COUNTDOWN (3·2·1 + side cue) →
HOLD (fixed 25s wall-clock; form measured per frame via the 2D-direct tilt +
`score_frame`) → TRANSITION → repeat for 4 alternating sides → SUMMARY. Audio is
spoken Thai via `feedback/tts.py` (`GeminiTTS` + macOS `say` fallback, played by
`TTSWorker`). Screens are drawn by `src/screens.py`. The 3D rig is not used here.

Data flow per frame (hold mode — used within each set of the routine):

```
── session start: _calibration_phase (5 s) → BaselinePose (user's own neutral pose)
WebcamCapture (bg thread)
   → Pose2D (YOLOX-m + RTMPose-m via rtmlib/ONNX on CoreML/ANE)  kps:(17,2) + scores
       (2D inference gated on a NEW camera frame via cap.read_latest_with_ts —
        the loop runs faster than the camera, so duplicate frames reuse the last
        kps/scores instead of re-inferring; the UI loop is capped at ~30 fps)
   → classify_view(kps, scores) → view gate: if view ∉ exercise.target.valid_views,
                                   skip scoring + show "rotate" coaching, FSM stays IDLE
   → exercise.measure(PoseFrame, baseline)                 {joint_name: degrees}
       (2D-DIRECT: e.g. head_lateral_tilt_2d from nose+shoulders; baseline-subtracted)
       → rules_hold.score_frame(target, measured)          (in_target: bool, violations)
           → HoldFSM.update(in_target, ts)                 IDLE → ENTERING → HOLDING → COMPLETE (with DRIFTED branch)
   → (visualization only) coco17_to_h36m17 → Pose3DBuffer (27-frame) → Pose3D
                                   (MotionBERT-Lite, MPS) → rig for the on-screen panel
   → live (throttled ≥ 2.5s): LLMWorker.submit(LiveSnapshot, exercise=...)
   → on COMPLETE: rules_hold.score_hold(...) → LLMWorker.submit(HoldAnalysis, exercise=...)
                  → ThaiCoachLLM.generate(payload, exercise=...) — dispatches on payload type
   → Renderer.compose(..., hold_state=state.value, hold_progress=...) → cv2.imshow
```

**2D-direct measurement (why the 3D lift is visualization-only).** MotionBERT-Lite is finetuned on Human3.6M (full-body standing/walking) and produces structurally invalid output (crossed ankles, 2-3× elongated bones) for the seated / upper-body-only / desk-camera framing this project targets — confirmed even on clean ≥0.79-confidence inputs. No off-the-shelf 2D→3D lifter fixes this (the whole field finetunes on H36M). So measurements are computed **directly from 2D COCO keypoints** in `analysis/angles.py` (`head_lateral_tilt_2d` + the cookbook below), which only need nose/ears/shoulders — reliable in a desk frame. The 3D rig stays as a debug/visualization artifact. Use `src/test_2D_3D.py` (3-panel camera / 2D / 3D diagnostic) to see the divergence live.

Data flow for the parked squat mode (code still intact, no entry-point wiring):

```
... 2D + 3D pose stages identical ...
   → SquatFSM.update(kps, ts)                              STANDING → DESCENT → BOTTOM → ASCENT → STANDING
       → on rep complete: rules_squat.score_rep(bottom_frame) → RepAnalysis
           → LLMWorker.submit(RepAnalysis)
               → ThaiCoachLLM.generate(RepAnalysis) — same dispatch, squat branch
```

### Coordinate systems & keypoint layouts

- **2D keypoints throughout `analysis/`**: COCO-17 indices (see `analysis/angles.py` for the canonical index constants). Image-pixel coords with y growing downward — that's why `hip_below_knee` checks `hip_y > knee_y`.
- **3D keypoints**: Human3.6M-17 layout. `pose3d.coco17_to_h36m17` synthesizes pelvis, spine, and thorax from COCO joints (indices marked `-1`, `-2`, `-3` in `COCO_TO_H36M`). Normalization in `_normalize_2d` divides both x and y by frame **width** (not height) — this matches MotionBERT's expected input convention; do not "fix" it.
- **Heatmaps for attention overlay**: reconstructed from RTMPose's simcc outputs via outer product (see `pose2d.infer_with_heatmaps`). This is a monkey-patch of `rtmlib`'s pose estimator; if `rtmlib` API shifts, the patch silently degrades and `heatmaps` becomes `None`.

### Phase detection and scoring

Two FSMs live in `analysis/phases.py`. They share no state and aren't polymorphic — pick the right one for the exercise shape.

**`HoldFSM` (timed-hold stretches — preserved infrastructure; NOT used by the current neck-stretch routine, which uses fixed-wall-clock sets and counts drift directly in `app.py`).** Takes a single `bool in_target` per frame (computed upstream by `rules_hold.score_frame`) and tracks the lifecycle `IDLE → ENTERING → HOLDING → COMPLETE`, with a `DRIFTED` branch off `HOLDING`. Two parameters with defaults: `stability_window_s=0.5` (gate to filter accidental fly-throughs), `drift_grace_s=0.3` (forgive single-frame jitter). Pause-on-drift semantics — `in_target_ms` advances only while `HOLDING`. Drift past grace transitions back to `ENTERING` (preserving accumulated `in_target_ms` per spec §5.2) and increments `drift_count`, which feeds the stability score. Fires `on_hold_complete(meta)` with `in_target_ms` / `drift_count` / `completed_ts`.

`analysis/rules_hold.py` is the generic, exercise-agnostic scorer:
- `score_frame(target, measured) -> (in_target, violations)`: per-frame predicate driving the FSM. NaN or missing measurement → out-of-target with severity 1.0. Deviation past tolerance ramps severity from 0 at the edge to 1.0 at 2× tolerance.
- `score_hold(...) -> HoldAnalysis`: final 100-pt rubric — duration 50 (clamped `in_target_ms / target_ms`), precision 30 (driven by worst per-joint severity), stability 20 (smooth decay on `drift_count`).

Per-exercise data lives under `src/exercises/`. Each module declares `name`, `display_th`, `target: TargetPose` (joint angle ranges + tolerances + Thai hints + `valid_views`), `prompt: PromptTemplate` (Thai live + summary templates), and a `measure(frame, baseline=None) -> dict[str, float]` method. `measure` reads **2D keypoints** (`frame.keypoints_2d` + `frame.scores`) and, when a `BaselinePose` is passed, returns the angle **relative to the user's neutral** rather than absolute. Add a new exercise: drop a new module in, register it in `exercises/__init__.py::EXERCISES`.

### Calibration, camera-view gating, and the 2D measurement cookbook

- **`src/calibration.py`** — `_calibration_phase` (in `app.run_session`) captures ~5 s of clean frames at session start and builds a `BaselinePose` (neutral shoulder width, head-lateral-tilt, shoulder-y asymmetry). Targets like `NeckStretchLeft`'s `-35°` are then interpreted as a delta from the user's own neutral, which absorbs camera distance / sitting height / habitual posture. `calibrate_from_samples(samples, min_clean_frames)` is the pure, testable core; it raises `CalibrationError` if too few clean frames arrive (app falls back to a zero baseline = absolute-angle mode).
- **`src/analysis/camera_view.py`** — `classify_view(kps, scores) -> CameraView` (FRONT / THREE_QUARTER / SIDE / UNKNOWN), keyed off shoulder-width-to-Δy ratio + ear visibility. Each `TargetPose` declares `valid_views`; `run_session` keeps the FSM in IDLE with a Thai "rotate" coaching message when the live view falls outside that set. `NeckStretchLeft` allows `(FRONT, THREE_QUARTER)` — side view collapses the shoulder lateral reference.
- **2D cookbook in `analysis/angles.py`** — each is a pure `(kps_2d, scores, ...) -> float`, NaN-gated on the keypoints it reads, normalized by a body-frame scalar (shoulder width / forearm length) or computed inside an `atan2` where magnitude cancels: `head_lateral_tilt_2d`, `craniovertebral_angle_2d` (forward-head/CVA), `forward_head_offset_normalized_2d`, `neck_flexion_2d`, `shoulder_elevation_asymmetry_2d`, `shoulder_protraction_ratio_2d`, `wrist_extension_2d`. These cover the remaining office-syndrome exercises without touching the 3D path. `head_lateral_tilt_2d` is mirror-invariant (keys off the body's own shoulder vector), so display mirroring doesn't change the sign convention.

**`SquatFSM` (preserved squat mode, not wired to entry point).** Thresholds `STAND_THRESHOLD=160°` and `BOTTOM_THRESHOLD=100°` on the average of left/right knee angles. A rep commits on `ASCENT → STANDING`, firing `on_rep_complete(meta)` with `descent_ms` / `ascent_ms`.

`analysis/rules_squat.py::score_rep` is a pure function from a single `PoseFrame` (the bottom frame) + tempo to a `RepAnalysis`. Component budget totals 100: depth 30 / valgus 25 / torso 20 / symmetry 15 / tempo 10. Each violation also produces a Thai-language hint (`detail_th`) consumed by the LLM prompt — keep wording natural Thai when adding rules.

### LLM feedback

`feedback/llm.py::ThaiCoachLLM` wraps Qwen3.5-4B (mxfp4) via `mlx-vlm`. First `generate()` call is slow (compilation) — `app.run()` calls `warmup()` before entering the loop; preserve that order. `generate(payload, ..., exercise=None)` dispatches on payload type:
- `RepAnalysis` → squat system prompt + `build_user_prompt` (no `exercise` needed).
- `HoldAnalysis` → hold system prompt + `build_hold_summary_prompt(payload, exercise)` — `exercise=` is **required**, else `ValueError`.
- `LiveSnapshot` → hold system prompt + `build_live_prompt(payload, exercise)` — `exercise=` is **required**, else `ValueError`.

`feedback/worker.py::LLMWorker` runs in a background thread with **drop-stale** semantics: `submit(payload, **kwargs)` replaces any pending submission that hasn't been processed yet. There is no queue. Kwargs (including `exercise=`) are stored alongside the payload and forwarded to `generate()`. If submissions arrive faster than the LLM, older ones are silently dropped — by design, since stale feedback is worse than no feedback.

Hold-mode live cadence: `app.run_session` throttles submissions to ≥ 2.5 s apart (Qwen 4B mxfp4 on MPS produces a short Thai phrase in ~1–2 s, so faster submission is wasted work). Expect roughly one fresh nudge every 3–4 s during a 20 s hold.

### Model artifacts

All models live under `./models/` (gitignored) and are downloaded by `scripts/download_models.py`:

- `models/rtmlib_cache/` — YOLOX + RTMPose ONNX, all three rtmlib tiers cached (tiny/m/x detectors, s/m/x pose). `Pose2D` defaults to **`mode="balanced"` (YOLOX-m + RTMPose-m) + `accelerator="coreml"`** (Apple Neural Engine). Benchmarks (`docs/perf/2026-05-23-coreml-experiment.md`): balanced is 8.7 fps on CPU but 54 fps on CoreML, with noticeably better keypoint accuracy than lightweight — the upgrade is what makes the 2D-direct CVA / forward-head metrics reliable. CoreML uses `RequireStaticInputShapes=1` + `ModelFormat=MLProgram` so the dynamic-shape YOLOX NMS subgraph falls back to CPU (dodges the zero-detection crash). It also sets `ModelCacheDirectory=models/coreml_cache` (gitignored) + `SpecializationStrategy=FastPrediction` so the compiled `.mlmodelc` persists across launches instead of recompiling both models every start (Pose2D construction ~0.5 s cold → ~0.1 s warm). ORT never invalidates this cache — clear `models/coreml_cache/` if you swap a model file or change EP options. `Pose2D` also pins `intra_op_num_threads=2` (faster *and* ~80% less CPU than ORT's all-cores default on M-series).
- `models/motionbert/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin` — MotionBERT-Lite weights (HF: `walterzhu/MotionBERT`). Visualization-only (see "2D-direct measurement").
- `models/qwen3_5_4b_mxfp4/` — Qwen3.5-4B mxfp4 mlx-vlm snapshot

`vendor/motionbert/` is a checked-out copy of the upstream MotionBERT repo (gitignored as `vendor/`). `pose3d.py` does `sys.path.insert(0, MOTIONBERT_DIR)` and imports `lib.model.DSTformer` + `lib.utils.tools.get_config` from it. The config file at `vendor/motionbert/configs/pose3d/MB_ft_h36m_global_lite.yaml` is required at construction time. If `vendor/` is missing or wiped, `Pose3D` cannot be constructed — re-clone MotionBERT into `vendor/motionbert/`.

### Threading model

Three background threads, all daemonized:

1. **WebcamCapture._loop / IPWebcamCapture._loop** — pulls frames, keeps only the latest. `read_latest()` returns a copy with timeout. `IPWebcamCapture` (`src/camera/`) is a drop-in source that streams an Android IP Webcam app's MJPEG feed over the LAN (`iter_jpeg_frames` parses the stream; reconnects on drop); pick it with `--source ip --url ...` on `app.py` / `test_2D_3D.py` via `camera.build_capture`. It is the first step toward the streaming-server direction below.
2. **LLMWorker._loop** — condition-variable wait, drop-stale submission.
3. **Pose2D / Pose3D** themselves are not threaded; they run inline on the main loop.

The main loop is the only consumer of OpenCV's GUI (`cv2.imshow` / `cv2.waitKey`); don't call those from worker threads.

## Testing notes

- Tests are split into three sections (see `tests/README.md` for the full inventory), **169 total**: `tests/pipeline/` (shared pose pipeline, incl. `test_render`), `tests/squat/` (original rep-based mode), `tests/office_syndrome/` (current launcher mode — neck-stretch exercises, hold scoring, `RoutineFSM` (`test_routine`), TTS (`test_tts`), and screens (`test_screens`)). Run a single section with `uv run pytest tests/<section>`.
- `tests/conftest.py` adds `src/` to `sys.path` (pytest config also does this — both are intentional, removing one breaks `python -m pytest` invocations). Note: Pyright/Pylance does NOT honor pytest's `pythonpath`, so it flags `Import "calibration"/"analysis.camera_view"/... could not be resolved` for the `src/` modules — a known false positive; pytest resolves them fine.
- The `*_smoke.py` tests actually load the heavy models (RTMPose, MotionBERT, Qwen) and will be slow/fail without `scripts/download_models.py` having run. They use `pytest.mark.skipif` against `tests/fixtures/...` and `models/...` paths anchored from `__file__`, so they skip gracefully when assets are missing.
- Pure-logic tests are fast and have no model dependency — prefer these when iterating on `analysis/` or `exercises/`. The pure-logic set: `test_angles` (incl. the 2D measurement cookbook), `test_angles_3d`, `test_camera_view`, `test_phases`, `test_hold_fsm`, `test_rules_squat`, `test_rules_hold`, `test_attention`, `test_types`, `test_exercises_base`, `test_exercises_registry`, `test_neck_stretch`, `test_calibration`, `test_prompt_th`, `test_render`, `test_worker`, `test_routine`, `test_tts`, `test_screens`.
- `test_llm_smoke.py` loads ~4 GB of weights; only run when actually touching `feedback/`. It contains both squat (`RepAnalysis`) and hold (`HoldAnalysis`) generation checks.
- `test_exercise_pipeline_smoke.py` runs the full 2D → 3D → measure → score_frame glue for `NeckStretchLeft` against a fixture image — gated on RTMPose weights + the fixture being present.

## Development workflow

### Jupyter for per-stage debugging

Treat the pipeline as five separable stages and prefer iterating on each one in a notebook before wiring it back into `app.run()`. The natural cut points match the module boundaries:

1. **Capture** (`capture`) — read a few frames, save as PNGs, sanity-check shape/FPS.
2. **2D pose** (`pose2d`) — feed a still image, scatter the 17 keypoints with `matplotlib`, overlay the COCO skeleton from `render.SKELETON`.
3. **3D lift** (`pose3d`) — build a 27-frame window manually (or replay from a saved `.npy`), call `Pose3D.infer`, plot the result with `mpl_toolkits.mplot3d`.
4. **Analysis** (`analysis.*`) — these are pure functions; feed synthetic `PoseFrame`s and inspect `RepAnalysis` / FSM transitions step-by-step.
5. **LLM** (`feedback.llm`) — call `ThaiCoachLLM.generate(rep)` directly with a hand-built `RepAnalysis`; useful for prompt tweaking without running the camera loop.

Notebook conventions:

- Launch via `uv run jupyter lab` (add `jupyter` with `uv add --dev jupyter` if it isn't present yet) so the kernel sees the project's `.venv`.
- Keep exploratory notebooks under `notebooks/` (gitignored if they balloon with model outputs); promote stable cells into `tests/` as pure-logic tests once the behavior is settled.
- For attention/heatmap visualizations use `matplotlib.imshow` with `cmap="jet"` and alpha-blend over the frame — mirrors `Renderer._overlay_attention` so visuals stay consistent with the live app.
- For the 3D rig, draw bones with `SKELETON` from `render.py` so the topology matches what the user sees in the panel.
- Heavy models survive across cells — don't re-instantiate `Pose2D` / `Pose3D` / `ThaiCoachLLM` unless you actually need to; reuse the existing object to keep MPS memory pressure down.

### Tools available beyond the standard set

- **Python LSP** — use it to jump to definitions, find references, and rename symbols when navigating the codebase. Preferred over `grep` for symbol-level questions ("where is `coco17_to_h36m17` called from?", "what implements `on_rep_complete`?").
- **context7 MCP plugin** — fetch current docs for `mlx-vlm`, `rtmlib`, `torch` (MPS specifics), `mediapipe`/`opencv-python`, `huggingface_hub`, FastAPI/Starlette, WebRTC libs, etc. Use whenever an API surface might have drifted from training data, especially before writing integration code against `mlx-vlm` or MotionBERT internals.

## Long-term direction: streaming server for mobile client

The end-state target is **not** a desktop OpenCV app. The current `app.run()` is a local-debug harness; the production shape is a server that:

- Accepts a streaming video feed from a mobile client (WebRTC / WebSocket / HTTP-multipart — undecided; flag this when relevant).
- Runs the same per-frame pipeline (2D → 3D → FSM → score → LLM-on-rep-boundary) on the server.
- Streams back, per frame or per event:
  - 3D rig keypoints (the `(17, 3)` array currently passed to `Renderer._draw_rig_3d`) for the client to render.
  - Phase + live score updates.
  - Thai coaching text generated on rep completion.

Implications when making changes today:

- Keep the pipeline stages independent of OpenCV's GUI (`cv2.imshow` / `waitKey`) — those calls live only in `app.run()` and `Renderer`. New analysis logic must not assume a display is attached.
- Avoid coupling rep state or score state to the renderer; the renderer should remain a pure consumer.
- Frame I/O abstraction is currently `WebcamCapture`; a future `StreamCapture` (from a socket / WebRTC track) should be drop-in compatible — preserve `start()` / `read_latest(timeout)` / `stop()` semantics on any new capture source.
- `LLMWorker`'s drop-stale design is correct for the streaming case too; preserve it rather than queueing.

## Repo conventions

- Acceptance targets from the README (kept for orientation, not yet automated): 2D overlay ≥25 FPS, 3D rig ≥5 Hz, ±1 rep over a 10-rep set, Thai feedback within 3 s of rep completion.
- Thai feedback wording (`prompt_th.py`, `rules_squat.py` `detail_th`) is intentionally short and polite — match the existing tone.
- Design specs & plans:
  - Original squat coach: `docs/superpowers/specs/2026-05-18-pose-form-coach-design.md` (impl plan: `docs/superpowers/plans/2026-05-18-pose-form-coach.md`).
  - 3D scoring upgrade: `docs/superpowers/specs/2026-05-19-3d-scoring-design.md`. (Superseded for stretches by the 2D-direct decision — kept for the squat path.)
  - Office syndrome stretches: `docs/superpowers/specs/2026-05-22-office-syndrome-stretches-design.md` (impl plan: `docs/superpowers/plans/2026-05-22-office-syndrome-stretches.md`).
  - CPU reduction + 2D-direct measurement coverage: `docs/superpowers/plans/2026-05-23-cpu-and-2d-improvements.md`.
- Performance notes: `docs/perf/2026-05-23-baseline.md` (ONNX thread sweep + inference-cadence throttle for Workstream A — note its tables measure `lightweight+cpu`, not the app's `balanced+coreml`; the same doc records the real config numbers and Workstream B: frame-dedup + 30 fps cap, Thai-text sprite cache, CoreML model cache), `docs/perf/2026-05-23-coreml-experiment.md` (CoreML/ANE per-model-size benchmark + the balanced+CoreML default decision).

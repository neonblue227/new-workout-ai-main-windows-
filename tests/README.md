# Tests

Three sections, each runnable on its own with `uv run pytest tests/<section>`.

## `tests/pipeline/` — shared pose pipeline (82 tests)

The stages every exercise mode runs through. Owns:

| File | Module under test |
|---|---|
| `test_angles.py` | `analysis/angles.py` (2D joint angles, COCO-17, **incl. the 2D-direct measurement cookbook**: `head_lateral_tilt_2d`, CVA, forward-head, neck-flexion, shoulder asymmetry/protraction, wrist extension) |
| `test_angles_3d.py` | `analysis/angles_3d.py` (3D angles, H36M-17 — used by the visualization-only rig) |
| `test_camera_view.py` | `analysis/camera_view.py` (`classify_view` → FRONT / THREE_QUARTER / SIDE / UNKNOWN) |
| `test_attention.py` | `analysis/attention.py` (heatmap aggregation) |
| `test_capture.py` | `capture.py` (webcam thread) |
| `test_pose2d_smoke.py` | `pose2d.py` (RTMPose, gated on RTMPose weights) |
| `test_pose3d_smoke.py` | `pose3d.py` (MotionBERT lift, gated on MotionBERT weights + `vendor/motionbert/`) |
| `test_types.py` | `analysis/types.py` (`PoseFrame`, `RuleViolation`, `RepAnalysis`, `HoldAnalysis`, `LiveSnapshot`) |
| `test_render.py` | `render.py` (skeleton draw + panel compose, shared by both modes) |
| `test_worker.py` | `feedback/worker.py` (`LLMWorker`, drop-stale + kwargs forwarding) |
| `test_llm_smoke.py` | `feedback/llm.py` (gated on Qwen weights — covers both `RepAnalysis` and `HoldAnalysis` payloads) |

## `tests/squat/` — original squat-coaching mode (16 tests)

The rep-based FSM and scoring. The code under test is still in `src/analysis/phases.py::SquatFSM` and `src/analysis/rules_squat.py`, but the launcher (`main.py`) no longer routes to it. Tests still run.

| File | Module under test |
|---|---|
| `test_phases.py` | `SquatFSM` |
| `test_rules_squat.py` | `score_rep` |

## `tests/office_syndrome/` — timed-hold stretches mode (47 tests)

The active launcher mode. Owns the plug-in exercise architecture, per-user calibration, and the first registered exercise.

| File | Module under test |
|---|---|
| `test_hold_fsm.py` | `analysis/phases.py::HoldFSM` (state transitions, pause-on-drift, drift_count) |
| `test_rules_hold.py` | `analysis/rules_hold.py` (`score_frame`, `score_hold` rubric) |
| `test_exercises_base.py` | `exercises/base.py` (`JointTarget`, `TargetPose` incl. `valid_views`, `PromptTemplate`) |
| `test_exercises_registry.py` | `exercises/__init__.py::EXERCISES` (protocol conformance for all entries) |
| `test_neck_stretch.py` | `exercises/neck_stretch.py::NeckStretchLeft` (2D-direct measure, baseline-delta, `valid_views`) |
| `test_calibration.py` | `calibration.py` (`BaselinePose`, `calibrate_from_samples` filtering + `CalibrationError`) |
| `test_prompt_th.py` | `feedback/prompt_th.py` hold builders (`build_live_prompt`, `build_hold_summary_prompt`) |
| `test_exercise_pipeline_smoke.py` | End-to-end: image → Pose2D → Pose3D → `NeckStretchLeft.measure` → `score_frame` (gated on RTMPose weights + `tests/fixtures/neck_stretch_left.jpg`) |

## Shared fixtures

`tests/fixtures/` holds JPGs used by smoke tests:

- `standing_person.jpg` — used by `pipeline/test_pose2d_smoke.py`
- `neck_stretch_left.jpg` — used by `office_syndrome/test_exercise_pipeline_smoke.py`

Smoke tests reference fixtures via `Path(__file__).resolve().parent.parent / "fixtures" / "..."`. Model paths anchor from the project root via `parent.parent.parent`.

## Running

```bash
uv run pytest                          # everything (145 tests)
uv run pytest tests/pipeline           # shared infra only (82)
uv run pytest tests/squat              # squat mode only (16)
uv run pytest tests/office_syndrome    # stretches mode only (47)
uv run pytest -k "not smoke"           # skip everything that loads heavy weights
```

`tests/conftest.py` adds `src/` to `sys.path` so imports work as `from analysis.* import ...` without an editable install. `pythonpath = ["src"]` in `pyproject.toml` does the same for `python -m pytest`. Both are intentional; removing one breaks the other invocation.

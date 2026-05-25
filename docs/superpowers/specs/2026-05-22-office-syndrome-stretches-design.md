# Office Syndrome Stretches — Design Spec

**Date:** 2026-05-22
**Status:** Draft — awaiting user review

## 1. Purpose

Extend the existing squat coach pipeline to support six office-syndrome stretching exercises:

1. Neck stretch (left / right)
2. Shoulder stretch (left / right)
3. Chest and shoulder stretch
4. Front-hand stretch (left / right)
5. Back-hand stretch (left / right)
6. Neck flexion stretch

The current pipeline (`src/app.py::run`) is hard-coded to squats: it constructs `SquatFSM`, calls `score_rep`, and the Thai prompt template is squat-specific. This spec defines the structural changes that turn it into an exercise-agnostic pipeline with the six stretches as the first plug-in exercises. Squat support stays in place untouched (separate FSM, separate scorer); the new code lives alongside it.

## 2. Counting & scoring model

- **Timed hold**, not reps. User holds the target pose; a per-frame in-target predicate decides whether the hold timer accumulates.
- **Target-pose match**: each exercise declares one or more joint-angle targets with a tolerance band. A frame is "in-target" iff all declared joints are within their tolerance.
- **Live LLM coaching throughout the hold** via the existing `LLMWorker` drop-stale background thread, plus a final summary on hold completion.

## 3. Architecture (Approach A — protocol + per-exercise module)

### 3.1 Directory layout

```
src/
├── app.py                      # MODIFIED: parameterized run(exercise)
├── capture.py                  # UNCHANGED
├── pose2d.py                   # UNCHANGED
├── pose3d.py                   # UNCHANGED
├── render.py                   # MODIFIED: panel shows hold timer + state badge
├── selector.py                 # NEW: pre-loop OpenCV menu to pick an exercise
├── analysis/
│   ├── angles.py               # UNCHANGED (reused — generic 2D angles)
│   ├── angles_3d.py            # POSSIBLY MODIFIED: may add derived angles (e.g. neck tilt)
│   ├── attention.py            # UNCHANGED
│   ├── types.py                # MODIFIED: add HoldState, HoldAnalysis, LiveSnapshot
│   ├── phases.py               # MODIFIED: add HoldFSM next to SquatFSM
│   ├── rules_squat.py          # UNCHANGED (kept; squat still works)
│   └── rules_hold.py           # NEW: generic in-target scorer for all stretches
├── exercises/                  # NEW package
│   ├── __init__.py             # EXERCISES registry
│   ├── base.py                 # Exercise protocol, TargetPose, JointTarget
│   ├── neck_stretch.py
│   ├── shoulder_stretch.py
│   ├── chest_shoulder_stretch.py
│   ├── front_hand_stretch.py
│   ├── back_hand_stretch.py
│   └── neck_flexion_stretch.py
└── feedback/
    ├── llm.py                  # MODIFIED: dispatch on input type (snapshot vs analysis)
    ├── worker.py               # UNCHANGED (drop-stale already fits)
    └── prompt_th.py            # MODIFIED: per-exercise live + summary templates
```

### 3.2 Reuse summary

- **No edits**: `capture`, `pose2d`, `pose3d`, `analysis/angles`, `analysis/attention`, `feedback/worker`, `analysis/rules_squat`.
- **Possibly edited**: `analysis/angles_3d` — only if an exercise needs a derived angle (e.g. neck lateral tilt) that doesn't already exist there. Add helpers as required by the chosen vertical-slice exercise.
- **Small edits**: `app`, `render`, `analysis/types`, `analysis/phases`, `feedback/llm`, `feedback/prompt_th`.
- **New code**: `selector`, `analysis/rules_hold`, the whole `exercises/` package.

## 4. The `Exercise` protocol

`src/exercises/base.py`:

```python
from dataclasses import dataclass
from typing import Protocol
from analysis.types import PoseFrame

@dataclass(frozen=True)
class JointTarget:
    name: str                  # e.g. "neck_lateral_tilt"
    target_deg: float
    tolerance_deg: float
    detail_th: str             # Thai hint shown if out-of-target

@dataclass(frozen=True)
class PromptTemplate:
    live: str                  # for LiveSnapshot input
    summary: str               # for HoldAnalysis input

@dataclass(frozen=True)
class TargetPose:
    joints: tuple[JointTarget, ...]
    hold_seconds: float = 20.0
    side: str | None = None    # "left" / "right" / None

class Exercise(Protocol):
    name: str                  # internal id
    display_th: str            # Thai name shown in selector
    target: TargetPose
    prompt: PromptTemplate

    def measure(self, frame: PoseFrame) -> dict[str, float]:
        """Compute current angles. Returns {joint_name: degrees}. Pure."""
        ...
```

The protocol has one method (`measure`). Everything else is data. Adding a new exercise = filling in target angles + Thai strings + the angle computation.

### 4.1 The exercise registry

`src/exercises/__init__.py` exposes:

```python
EXERCISES: dict[str, Exercise] = {
    "neck_stretch_left": NeckStretchLeft(),
    "neck_stretch_right": NeckStretchRight(),
    "shoulder_stretch_left": ShoulderStretchLeft(),
    ...
}
```

The selector reads this dict; `app.run(exercise)` accepts any `Exercise`.

## 5. Hold FSM (`analysis/phases.py`)

Lives next to the unchanged `SquatFSM`.

### 5.1 States

```
IDLE → ENTERING → HOLDING → COMPLETE
                     ↑   ↓
                     └─DRIFTED
```

### 5.2 Transitions

Driven by a single `bool in_target` per frame (computed by `rules_hold.score_frame`):

- `IDLE → ENTERING` — first `in_target=True` frame. Timer not started yet.
- `ENTERING → HOLDING` — stable in-target for `stability_window_s` (default 0.5s). `hold_start_ts` set. Timer begins.
- `HOLDING → DRIFTED` — `in_target=False`. Timer pauses.
- `DRIFTED → HOLDING` — back in-target within `drift_grace_s` (default 0.3s). Timer resumes.
- `DRIFTED → ENTERING` — out-of-target longer than grace. Drift counted; hold continues.
- `* → COMPLETE` — accumulated in-target time ≥ `target.hold_seconds`. Fires `on_hold_complete(meta)` callback with `in_target_ms`, `total_ms`, `drift_count`, `quality_ratio`.

### 5.3 Semantics

**Pause-on-drift**: only in-target seconds count toward the hold. Honest semantic for what a "hold" means. (Alternative is wall-clock-with-penalty — captured as open question §11.1, not chosen.)

### 5.4 Interface

```python
class HoldState(Enum):
    IDLE = "idle"
    ENTERING = "entering"
    HOLDING = "holding"
    DRIFTED = "drifted"
    COMPLETE = "complete"

@dataclass
class HoldFSM:
    target_seconds: float
    stability_window_s: float = 0.5
    drift_grace_s: float = 0.3
    on_hold_complete: Callable[[dict], None] | None = None

    def update(self, in_target: bool, timestamp: float) -> HoldState: ...
```

The FSM takes a bool, not raw keypoints. This keeps `HoldFSM` exercise-agnostic — all exercise-specific math is in `exercise.measure()` + `rules_hold.score_frame()`.

## 6. Scoring (`analysis/rules_hold.py`)

### 6.1 Per-frame

```python
def score_frame(
    target: TargetPose,
    measured: dict[str, float],
) -> tuple[bool, list[RuleViolation]]:
    """Returns (in_target, per-joint violations).
    FSM uses the bool; live feedback uses the violations."""
```

In-target iff every declared joint's measured angle is within `tolerance_deg` of `target_deg`. Violations include severity (clamped 0–1, scaled by how far past tolerance).

### 6.2 Hold completion

```python
def score_hold(
    exercise_name: str,
    meta: dict,
    target: TargetPose,
    max_severity_seen: dict[str, float],
) -> HoldAnalysis:
    """100-point budget:
      duration    50 — clipped(in_target_ms / target_ms)
      precision   30 — 1 - mean(max severity per joint)
      stability   20 — drift-count-dependent decay
    """
```

`HoldAnalysis` (in `analysis/types.py`):

```python
@dataclass
class HoldAnalysis:
    exercise_name: str
    score: int                            # 0–100
    components: dict[str, int]            # {"duration": 50, "precision": 27, "stability": 18}
    violations: list[RuleViolation]
    in_target_ms: int
    drift_count: int
```

Mirrors the existing `RepAnalysis` shape so `LLMWorker` keeps working unchanged (duck-typed on `.score`, `.components`, `.violations`).

## 7. App flow

### 7.1 Top-level (`app.run()`)

```
run():
  load Pose2D, Pose3D, ThaiCoachLLM ONCE   # heavy weights
  loop:
    exercise = selector.choose_exercise()
    fsm = HoldFSM(target_seconds=exercise.target.hold_seconds,
                  on_hold_complete=on_complete)
    run_session(cap, pose, lifter, llm_worker, fsm, exercise)
    show summary panel for ~3s
    (return to selector unless user pressed Q to quit)
```

**Critical**: model objects load once. Switching exercises does NOT reload Qwen (~4 GB) or MotionBERT. This is the entire reason for "single app with selector."

### 7.2 Inner per-frame loop (`run_session`)

1. Read frame → `Pose2D` → `Pose3D` (every 5 frames, as today).
2. Build `PoseFrame`.
3. `measured = exercise.measure(pose_frame)`.
4. `in_target, violations = rules_hold.score_frame(exercise.target, measured)`.
5. `state = fsm.update(in_target, ts)`.
6. **Live LLM submission** (throttled — see §8): if ≥ N seconds since last submit and state ∈ {HOLDING, DRIFTED}, build a `LiveSnapshot` and `worker.submit(...)`.
7. On `state == COMPLETE` → `score_hold(...)` → `worker.submit(final_analysis)` → exit inner loop.
8. Render: camera + skeleton + 3D rig + hold timer + state badge + `worker.latest()` Thai text.

## 8. Live LLM feedback

The squat `LLMWorker` already has drop-stale semantics — reused unchanged. What's new:

- **Throttle in `run_session`**: don't `worker.submit()` more often than every ~2.5s. Qwen 4B mxfp4 on MPS generates a short Thai phrase in ~1–2s; faster submission is wasted work because drop-stale would discard it.
- **New input type `LiveSnapshot`** (`analysis/types.py`):

  ```python
  @dataclass
  class LiveSnapshot:
      exercise_name: str
      state: HoldState                  # HOLDING / DRIFTED
      progress_ratio: float             # in_target_ms / target_ms
      current_violations: list[RuleViolation]
  ```

- `ThaiCoachLLM.generate()` dispatches on input type:
  - `LiveSnapshot` → renders `exercise.prompt.live` template.
  - `HoldAnalysis` → renders `exercise.prompt.summary` template (mirrors current squat behavior).
  - `RepAnalysis` → renders the existing squat template (unchanged).

### 8.1 Performance ceiling

With ~2.5s throttle + ~1–2s generation, expect ~1 fresh Thai nudge every 3–4s — so 5–6 messages per 20s hold. This is the honest ceiling. See open question §11.2.

## 9. Selector (`src/selector.py`)

Pre-loop OpenCV window listing all `EXERCISES`. Number keys to pick, Esc/Q to cancel.

```python
def choose_exercise(default: str | None = None) -> Exercise:
    """Blocking. Returns selected Exercise instance. Esc → SystemExit."""
```

Lives outside `app.run()` so the main loop stays focused. OpenCV (not CLI) because the user is already at the screen and webcam permission flow naturally precedes/follows it. Future mobile-server target replaces this with a request parameter — same `EXERCISES` registry, different selector.

## 10. Renderer changes (`src/render.py`)

Minimal additions to the right panel:

- Hold progress arc / bar (`in_target_ms / target_ms`).
- State badge: `ENTERING` / `HOLDING` / `DRIFTED` / `COMPLETE` (Thai labels).
- Live in-target chip (green) / out-of-target chip (amber).
- Existing 3D rig display: unchanged.
- Existing attention overlay (toggled by `a`): unchanged.
- Existing Thai message from `worker.latest()`: unchanged plumbing.

## 11. Open questions / deferred decisions

### 11.1 Timer semantics — DECIDED: pause-on-drift
Alternative considered: wall-clock-with-penalty (timer always runs; drift just lowers the score). Going with pause-on-drift as the more honest "hold" semantic. Revisit if user testing shows people get stuck retrying drifty holds.

Drift past grace transitions DRIFTED → ENTERING (not IDLE) and preserves the accumulated `in_target_ms`. The stability penalty comes from `drift_count`, not from resetting the timer.

### 11.2 Live LLM throughput
~1 fresh nudge every 3–4s may feel sluggish. Three fallbacks if so:
  (a) shorter throttle, tolerate more drop-stale waste;
  (b) skip LLM during HOLDING — show rule-based Thai snippets live, LLM only on completion;
  (c) smaller/faster on-device model.
Not deciding now. Implementation will start with the throttled-LLM design; revisit after first vertical slice.

### 11.3 Target angle values per exercise
Each exercise needs concrete `target_deg` and `tolerance_deg` values per joint. These are content/calibration data, not architecture. Spec leaves placeholders. Suggested process: pick canonical reference photo per exercise → run `Pose3D` on it → use the measured angle as `target_deg` with ±8–12° tolerance to start; refine after user testing.

### 11.4 Per-side exercises
Decision: register each side as a separate `Exercise` (e.g. `NeckStretchLeft`, `NeckStretchRight`). Cleaner than runtime `side` switching, and the selector naturally shows both as distinct entries. Asymmetric exercises (have left/right): neck stretch, shoulder stretch, front-hand stretch, back-hand stretch — 4 movements × 2 sides = 8. Symmetric (no side): chest-and-shoulder stretch, neck flexion stretch — 2. **Total registered: 10 entries** in `EXERCISES`.

## 12. Build plan

**Vertical slice first**: build the framework + one exercise (proposed: `neck_stretch_left` — visually clear, well-defined target angle, single joint to measure) end-to-end. Once that works in the live app, the other 5 (≈10 with sides) are content-only additions — new files under `exercises/`, no framework changes.

## 13. Testing strategy

### Pure-logic tests (fast, no model load)

- `test_hold_fsm.py` — synthetic `in_target` sequences exercise every state transition: clean hold, jittery hold (drift within grace), drift beyond grace, accidental fly-through (in-target < stability window).
- `test_rules_hold.py` — `score_frame` with hand-built `TargetPose` + measured dicts (in-target, near-edge, far-off); `score_hold` math on synthetic meta.
- `test_exercises.py` — for each registered `Exercise`: protocol conformance, `measure()` returns the joint names declared in `target.joints`, prompt templates render without `KeyError`.
- `test_types.py` — extend with `HoldAnalysis` and `LiveSnapshot` round-trip.

### Smoke tests (gated, require downloaded models)

- `test_exercise_pipeline_smoke.py` — feed one canned image through `Pose2D → Pose3D → exercise.measure → rules_hold.score_frame` for one exercise. Validates glue, not correctness.
- `test_llm_smoke.py` — extend with one `LiveSnapshot` and one `HoldAnalysis` to confirm both templates produce Thai output.

### Not tested

- Selector UI (cv2 window) — manual.
- LLMWorker threading interaction — already exercised by squat code; same primitives.
- Real-world target-angle correctness — content/calibration, see §11.3.

## 14. Out of scope

- Mobile streaming server (the long-term direction in CLAUDE.md). Pipeline stages remain GUI-decoupled per existing convention; selector is replaceable.
- Rep-based exercises (already handled by `SquatFSM` + `rules_squat.py`).
- New pose model or alternative LLM.
- Localization beyond Thai.

## 15. Acceptance (informal, mirrors existing project conventions)

- Selector shows all registered exercises; picking one launches the session.
- For at least one exercise, holding the correct pose accumulates time and reaches COMPLETE within `target.hold_seconds` of real-world hold time.
- Drift out → time pauses, returns when back in-target (within grace).
- Drift beyond grace → drift_count increments, recorded in final `HoldAnalysis`.
- Live Thai nudges appear in the right panel during the hold; completion summary appears after.
- Existing squat flow still works (regression check).

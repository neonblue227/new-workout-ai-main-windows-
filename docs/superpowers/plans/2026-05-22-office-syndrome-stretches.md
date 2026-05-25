# Office Syndrome Stretches — Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a plug-in exercise architecture to the existing pose pipeline and ship one timed-hold stretch (NeckStretchLeft) end-to-end. After this slice works, adding the remaining 9 stretches is content-only (follow-up plan).

**Architecture:** Per spec `docs/superpowers/specs/2026-05-22-office-syndrome-stretches-design.md` §3 (Approach A — protocol + per-exercise module). New `HoldFSM` lives alongside `SquatFSM`; new generic `rules_hold.py` scorer reads a per-exercise `TargetPose`; new `exercises/` package holds one module per exercise; `app.py` is parameterized on the selected `Exercise`; `ThaiCoachLLM` dispatches on input type.

**Tech Stack:** Python 3.12 (uv), pytest, numpy, OpenCV, PyTorch (MPS), mlx-vlm, existing rtmlib/MotionBERT/Qwen weights.

---

## Pre-flight

- [ ] **Sanity check the working tree**

Run: `cd "/Users/thatt/Dev/AI project/new-workout-ai" && uv run pytest -q`
Expected: existing suite passes. If it doesn't, stop and report — don't start this plan on a red baseline.

- [ ] **Read the spec once before starting**

Open `docs/superpowers/specs/2026-05-22-office-syndrome-stretches-design.md` and skim §3 (architecture), §5 (HoldFSM), §6 (scoring), §8 (live LLM) so the rest of the plan slots in correctly.

---

## Task 1: Add new shared types

**Files:**
- Modify: `src/analysis/types.py`
- Test: `tests/test_types.py`

- [ ] **Step 1: Write failing tests for the new types**

Append to `tests/test_types.py`:

```python
import numpy as np
from analysis.types import (
    HoldState,
    HoldAnalysis,
    LiveSnapshot,
    RuleViolation,
)


def test_hold_state_enum_values():
    assert HoldState.IDLE.value == "idle"
    assert HoldState.ENTERING.value == "entering"
    assert HoldState.HOLDING.value == "holding"
    assert HoldState.DRIFTED.value == "drifted"
    assert HoldState.COMPLETE.value == "complete"


def test_hold_analysis_dataclass_roundtrip():
    a = HoldAnalysis(
        exercise_name="neck_stretch_left",
        score=87,
        components={"duration": 50, "precision": 25, "stability": 12},
        violations=[RuleViolation("head_lateral_tilt", 0.4, "เอียงคอเพิ่ม")],
        in_target_ms=18_000,
        drift_count=2,
    )
    assert a.score == 87
    assert a.components["duration"] == 50
    assert a.violations[0].name == "head_lateral_tilt"


def test_live_snapshot_dataclass_roundtrip():
    s = LiveSnapshot(
        exercise_name="neck_stretch_left",
        state=HoldState.HOLDING,
        progress_ratio=0.6,
        current_violations=[],
    )
    assert s.state is HoldState.HOLDING
    assert s.progress_ratio == 0.6
    assert s.current_violations == []
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_types.py -v`
Expected: ImportError on `HoldState` / `HoldAnalysis` / `LiveSnapshot`.

- [ ] **Step 3: Add the new types**

Append to `src/analysis/types.py`:

```python
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
    violations: list["RuleViolation"]
    in_target_ms: int
    drift_count: int


@dataclass
class LiveSnapshot:
    exercise_name: str
    state: HoldState
    progress_ratio: float  # 0.0 .. 1.0
    current_violations: list["RuleViolation"]
```

- [ ] **Step 4: Tests pass**

Run: `uv run pytest tests/test_types.py -v`
Expected: all tests pass, including pre-existing ones.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/types.py tests/test_types.py
git commit -m "feat(types): add HoldState, HoldAnalysis, LiveSnapshot"
```

---

## Task 2: Add `HoldFSM` next to `SquatFSM`

**Files:**
- Modify: `src/analysis/phases.py`
- Create: `tests/test_hold_fsm.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_hold_fsm.py`:

```python
from analysis.phases import HoldFSM
from analysis.types import HoldState


def test_starts_idle():
    fsm = HoldFSM(target_seconds=20.0)
    assert fsm.state is HoldState.IDLE


def test_enters_after_first_in_target_frame():
    fsm = HoldFSM(target_seconds=20.0)
    assert fsm.update(in_target=True, timestamp=0.0) is HoldState.ENTERING


def test_promotes_to_holding_after_stability_window():
    fsm = HoldFSM(target_seconds=20.0, stability_window_s=0.5)
    fsm.update(True, 0.0)            # ENTERING
    fsm.update(True, 0.3)            # still ENTERING (< 0.5)
    state = fsm.update(True, 0.6)    # crosses 0.5 → HOLDING
    assert state is HoldState.HOLDING


def test_fly_through_does_not_start_hold():
    fsm = HoldFSM(target_seconds=20.0, stability_window_s=0.5)
    fsm.update(True, 0.0)
    fsm.update(True, 0.3)
    state = fsm.update(False, 0.4)   # in-target window < 0.5 → reset
    assert state is HoldState.IDLE


def test_clean_hold_reaches_complete():
    fsm = HoldFSM(target_seconds=2.0, stability_window_s=0.5)
    completes = []
    fsm.on_hold_complete = lambda meta: completes.append(meta)

    t = 0.0
    state = None
    while state is not HoldState.COMPLETE and t < 10.0:
        state = fsm.update(True, t)
        t += 0.1
    assert state is HoldState.COMPLETE
    assert completes, "callback must fire on COMPLETE"
    meta = completes[0]
    assert meta["in_target_ms"] >= 2000
    assert meta["drift_count"] == 0


def test_drift_within_grace_resumes_holding():
    fsm = HoldFSM(target_seconds=5.0, stability_window_s=0.5, drift_grace_s=0.3)
    fsm.update(True, 0.0)
    fsm.update(True, 0.6)            # HOLDING
    fsm.update(False, 1.0)           # DRIFTED
    state = fsm.update(True, 1.2)    # within grace → HOLDING
    assert state is HoldState.HOLDING


def test_drift_beyond_grace_increments_count_and_resets_entry():
    fsm = HoldFSM(target_seconds=5.0, stability_window_s=0.5, drift_grace_s=0.3)
    fsm.update(True, 0.0)
    fsm.update(True, 0.6)            # HOLDING
    fsm.update(False, 1.0)           # DRIFTED
    state = fsm.update(False, 1.5)   # past grace
    # Either ENTERING-with-no-current-window or IDLE — either is acceptable
    # as long as drift_count tracked it.
    assert state in (HoldState.ENTERING, HoldState.IDLE)
    assert fsm.drift_count == 1


def test_timer_pauses_during_drift():
    """In-target ms accumulates only while HOLDING — not while DRIFTED."""
    fsm = HoldFSM(target_seconds=10.0, stability_window_s=0.5, drift_grace_s=0.3)
    fsm.update(True, 0.0)
    fsm.update(True, 0.6)            # HOLDING start
    fsm.update(True, 1.6)            # +1.0s in-target
    fsm.update(False, 1.7)           # DRIFTED — timer pauses
    fsm.update(False, 1.9)           # still in grace, still paused
    fsm.update(True, 2.0)            # resume HOLDING
    fsm.update(True, 2.5)            # +0.5s more in-target
    assert 1400 <= fsm.in_target_ms <= 1600
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_hold_fsm.py -v`
Expected: ImportError on `HoldFSM` (and `drift_count` / `in_target_ms` attribute access errors).

- [ ] **Step 3: Implement `HoldFSM`**

Append to `src/analysis/phases.py`:

```python
from analysis.types import HoldState


@dataclass
class HoldFSM:
    target_seconds: float
    stability_window_s: float = 0.5
    drift_grace_s: float = 0.3
    on_hold_complete: Optional[Callable[[dict], None]] = None

    state: HoldState = HoldState.IDLE
    entry_start_ts: Optional[float] = None      # when current ENTERING window began
    last_hold_ts: Optional[float] = None        # last frame in HOLDING (for accumulation)
    drift_start_ts: Optional[float] = None      # when current drift began
    in_target_ms: int = 0
    drift_count: int = 0

    def update(self, in_target: bool, timestamp: float) -> HoldState:
        if self.state is HoldState.IDLE:
            if in_target:
                self.state = HoldState.ENTERING
                self.entry_start_ts = timestamp

        elif self.state is HoldState.ENTERING:
            if not in_target:
                # Aborted entry — reset.
                self.state = HoldState.IDLE
                self.entry_start_ts = None
            elif timestamp - (self.entry_start_ts or timestamp) >= self.stability_window_s:
                self.state = HoldState.HOLDING
                self.last_hold_ts = timestamp

        elif self.state is HoldState.HOLDING:
            if in_target:
                if self.last_hold_ts is not None:
                    delta = max(0.0, timestamp - self.last_hold_ts)
                    self.in_target_ms += int(delta * 1000)
                self.last_hold_ts = timestamp
                if self.in_target_ms >= int(self.target_seconds * 1000):
                    self._fire_complete(timestamp)
            else:
                self.state = HoldState.DRIFTED
                self.drift_start_ts = timestamp

        elif self.state is HoldState.DRIFTED:
            if in_target:
                self.state = HoldState.HOLDING
                self.drift_start_ts = None
                self.last_hold_ts = timestamp
            elif timestamp - (self.drift_start_ts or timestamp) >= self.drift_grace_s:
                self.drift_count += 1
                self.state = HoldState.ENTERING
                self.entry_start_ts = None
                self.drift_start_ts = None
                self.last_hold_ts = None

        return self.state

    def _fire_complete(self, ts: float) -> None:
        self.state = HoldState.COMPLETE
        if self.on_hold_complete:
            self.on_hold_complete({
                "in_target_ms": self.in_target_ms,
                "drift_count": self.drift_count,
                "completed_ts": ts,
            })
```

Note: `dataclass` and `Optional` / `Callable` are already imported at the top of the file from the existing `SquatFSM` code. Add `from analysis.types import HoldState` near the existing `from analysis.types import PhaseState` import.

- [ ] **Step 4: Tests pass**

Run: `uv run pytest tests/test_hold_fsm.py tests/test_phases.py -v`
Expected: all new tests pass; existing `SquatFSM` tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/phases.py tests/test_hold_fsm.py
git commit -m "feat(phases): add HoldFSM for timed-hold exercises"
```

---

## Task 3: Add `exercises/base.py` (protocol + dataclasses)

**Files:**
- Create: `src/exercises/__init__.py`
- Create: `src/exercises/base.py`
- Create: `tests/test_exercises_base.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_exercises_base.py`:

```python
from exercises.base import JointTarget, TargetPose, PromptTemplate


def test_joint_target_immutable():
    j = JointTarget("head_lateral_tilt", -35.0, 10.0, "เอียงเพิ่ม")
    assert j.name == "head_lateral_tilt"
    assert j.target_deg == -35.0
    assert j.tolerance_deg == 10.0


def test_target_pose_defaults():
    t = TargetPose(joints=(JointTarget("k", 0.0, 5.0, "x"),))
    assert t.hold_seconds == 20.0
    assert t.side is None


def test_prompt_template_holds_both_strings():
    p = PromptTemplate(live="live {progress}", summary="sum {score}")
    assert p.live.format(progress=0.5) == "live 0.5"
    assert p.summary.format(score=88) == "sum 88"
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_exercises_base.py -v`
Expected: ImportError on `exercises.base`.

- [ ] **Step 3: Implement `exercises/base.py`**

Create `src/exercises/__init__.py` empty for now (registry comes in Task 6):

```python
"""Exercise registry. Populated by submodules at import time."""
```

Create `src/exercises/base.py`:

```python
from dataclasses import dataclass
from typing import Protocol

from analysis.types import PoseFrame


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
    live: str     # rendered with LiveSnapshot fields
    summary: str  # rendered with HoldAnalysis fields


@dataclass(frozen=True)
class TargetPose:
    joints: tuple[JointTarget, ...]
    hold_seconds: float = 20.0
    side: str | None = None  # "left" / "right" / None for symmetric


class Exercise(Protocol):
    name: str
    display_th: str
    target: TargetPose
    prompt: PromptTemplate

    def measure(self, frame: PoseFrame) -> dict[str, float]:
        """Return {joint_name: degrees} for every joint in self.target.joints.
        Pure function of one frame. May return NaN if keypoints unavailable."""
        ...
```

- [ ] **Step 4: Tests pass**

Run: `uv run pytest tests/test_exercises_base.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/exercises/__init__.py src/exercises/base.py tests/test_exercises_base.py
git commit -m "feat(exercises): add Exercise protocol + TargetPose dataclasses"
```

---

## Task 4: Add `analysis/rules_hold.py` (generic scorer)

**Files:**
- Create: `src/analysis/rules_hold.py`
- Create: `tests/test_rules_hold.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_rules_hold.py`:

```python
import math

from analysis.rules_hold import score_frame, score_hold
from analysis.types import HoldAnalysis
from exercises.base import JointTarget, TargetPose


TP = TargetPose(joints=(
    JointTarget("a", target_deg=30.0, tolerance_deg=5.0, detail_th="adjust a"),
    JointTarget("b", target_deg=10.0, tolerance_deg=3.0, detail_th="adjust b"),
))


def test_score_frame_in_target_when_all_joints_within_tolerance():
    in_target, violations = score_frame(TP, {"a": 28.0, "b": 11.0})
    assert in_target is True
    assert violations == []


def test_score_frame_out_of_target_when_one_joint_outside():
    in_target, violations = score_frame(TP, {"a": 28.0, "b": 20.0})
    assert in_target is False
    assert len(violations) == 1
    assert violations[0].name == "b"
    assert violations[0].detail_th == "adjust b"
    assert 0.0 < violations[0].severity <= 1.0


def test_score_frame_nan_is_out_of_target():
    in_target, violations = score_frame(TP, {"a": float("nan"), "b": 10.0})
    assert in_target is False
    assert any(v.name == "a" for v in violations)


def test_score_frame_missing_joint_is_out_of_target():
    in_target, violations = score_frame(TP, {"a": 30.0})
    assert in_target is False
    assert any(v.name == "b" for v in violations)


def test_score_hold_full_credit_when_clean():
    meta = {"in_target_ms": 20_000, "drift_count": 0, "completed_ts": 25.0}
    analysis = score_hold(
        exercise_name="x",
        meta=meta,
        target=TP,
        max_severity_seen={"a": 0.0, "b": 0.0},
    )
    assert isinstance(analysis, HoldAnalysis)
    assert analysis.components["duration"] == 50
    assert analysis.components["precision"] == 30
    assert analysis.components["stability"] == 20
    assert analysis.score == 100


def test_score_hold_duration_clip_when_under_target():
    # Target is 20s (hold_seconds default) but only 10s accumulated.
    meta = {"in_target_ms": 10_000, "drift_count": 0, "completed_ts": 30.0}
    analysis = score_hold("x", meta, TP, {"a": 0.0, "b": 0.0})
    assert analysis.components["duration"] == 25  # 50 * 0.5


def test_score_hold_precision_penalty_for_high_severity():
    meta = {"in_target_ms": 20_000, "drift_count": 0, "completed_ts": 25.0}
    analysis = score_hold("x", meta, TP, {"a": 0.5, "b": 0.5})
    # mean severity 0.5 → precision = 30 * 0.5 = 15
    assert analysis.components["precision"] == 15


def test_score_hold_stability_decays_with_drifts():
    meta = {"in_target_ms": 20_000, "drift_count": 4, "completed_ts": 30.0}
    analysis = score_hold("x", meta, TP, {"a": 0.0, "b": 0.0})
    # Some decay applied; exact value depends on formula but must drop from 20.
    assert 0 <= analysis.components["stability"] < 20
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_rules_hold.py -v`
Expected: ImportError on `analysis.rules_hold`.

- [ ] **Step 3: Implement `rules_hold.py`**

Create `src/analysis/rules_hold.py`:

```python
"""Generic scorer for timed-hold exercises. Exercise-agnostic — reads a
TargetPose and measured joint angles."""

import math

from analysis.types import HoldAnalysis, RuleViolation
from exercises.base import TargetPose


def score_frame(
    target: TargetPose,
    measured: dict[str, float],
) -> tuple[bool, list[RuleViolation]]:
    """Per-frame check: is the current pose within tolerance for every declared joint?

    Returns (in_target, violations). FSM consumes the bool; live LLM
    consumes the violations.

    A missing or NaN joint angle counts as out-of-target with severity 1.0.
    """
    violations: list[RuleViolation] = []
    in_target = True
    for j in target.joints:
        m = measured.get(j.name)
        if m is None or math.isnan(m):
            in_target = False
            violations.append(RuleViolation(j.name, 1.0, j.detail_th))
            continue
        deviation = abs(m - j.target_deg)
        if deviation > j.tolerance_deg:
            in_target = False
            # Severity ramps from 0 at the tolerance edge to 1.0 at 2x tolerance.
            severity = min(1.0, (deviation - j.tolerance_deg) / max(j.tolerance_deg, 1e-6))
            violations.append(RuleViolation(j.name, severity, j.detail_th))
    return in_target, violations


def score_hold(
    exercise_name: str,
    meta: dict,
    target: TargetPose,
    max_severity_seen: dict[str, float],
) -> HoldAnalysis:
    """Final score on hold completion. 100-pt budget: 50 duration / 30 precision / 20 stability."""
    target_ms = int(target.hold_seconds * 1000)
    in_target_ms = int(meta["in_target_ms"])
    drift_count = int(meta["drift_count"])

    duration_ratio = min(1.0, in_target_ms / max(target_ms, 1))
    duration_pts = int(round(50 * duration_ratio))

    if max_severity_seen:
        mean_sev = sum(max_severity_seen.values()) / len(max_severity_seen)
    else:
        mean_sev = 0.0
    precision_pts = int(round(30 * (1.0 - min(1.0, mean_sev))))

    # Stability: smooth decay. 0 drifts → 20. ~5+ drifts → 0.
    stability_pts = int(round(20 * math.exp(-0.4 * drift_count)))

    components = {
        "duration": duration_pts,
        "precision": precision_pts,
        "stability": stability_pts,
    }
    # Build a single violation list from the worst-offending joints (severity > 0.05).
    violations = [
        RuleViolation(name=jn, severity=sev,
                      detail_th=next((j.detail_th for j in target.joints if j.name == jn), ""))
        for jn, sev in max_severity_seen.items()
        if sev > 0.05
    ]
    return HoldAnalysis(
        exercise_name=exercise_name,
        score=sum(components.values()),
        components=components,
        violations=violations,
        in_target_ms=in_target_ms,
        drift_count=drift_count,
    )
```

- [ ] **Step 4: Tests pass**

Run: `uv run pytest tests/test_rules_hold.py -v`
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/rules_hold.py tests/test_rules_hold.py
git commit -m "feat(rules): add generic timed-hold scorer"
```

---

## Task 5: Add `head_lateral_tilt_3d` helper

This is the one new geometry function we need to score a neck stretch.

**Files:**
- Modify: `src/analysis/angles_3d.py`
- Modify: `tests/test_angles_3d.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_angles_3d.py`:

```python
import math
import numpy as np

from analysis.angles_3d import head_lateral_tilt_3d


def _h36m_skeleton_with_head_offset(lat_offset: float, vertical: float = 1.0) -> np.ndarray:
    """Build a minimal H36M-17 keypoint array with head offset laterally
    from a vertical thorax-pelvis axis.

    Body frame setup in MotionBERT's normalized coords (image-y down → up = -y):
      pelvis at origin
      thorax directly above pelvis (along -y)
      l_hip / r_hip on the x axis so body_lateral = +x
      head = thorax + (lat_offset, -vertical, 0)
    """
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[0] = (0.0, 0.0, 0.0)          # PELVIS
    kps[1] = (-0.5, 0.0, 0.0)         # R_HIP  (so L_hip→R_hip = -x; lateral = -x)
    kps[4] = (0.5, 0.0, 0.0)          # L_HIP
    kps[8] = (0.0, -1.0, 0.0)         # THORAX
    kps[10] = (lat_offset, -1.0 - vertical, 0.0)  # HEAD
    return kps


def test_head_lateral_tilt_zero_when_head_above_thorax():
    kps = _h36m_skeleton_with_head_offset(lat_offset=0.0)
    assert abs(head_lateral_tilt_3d(kps)) < 1.0  # within 1°


def test_head_lateral_tilt_positive_when_tilted_to_body_lateral_plus():
    kps = _h36m_skeleton_with_head_offset(lat_offset=0.5)
    angle = head_lateral_tilt_3d(kps)
    # Expected magnitude: atan2(0.5, 1.0) ≈ 26.57°
    assert 24.0 < abs(angle) < 30.0


def test_head_lateral_tilt_sign_flips_with_lateral_direction():
    pos = head_lateral_tilt_3d(_h36m_skeleton_with_head_offset(lat_offset=0.5))
    neg = head_lateral_tilt_3d(_h36m_skeleton_with_head_offset(lat_offset=-0.5))
    assert pos * neg < 0  # opposite signs


def test_head_lateral_tilt_nan_when_thorax_collapsed():
    kps = _h36m_skeleton_with_head_offset(lat_offset=0.0)
    kps[8] = kps[0]  # thorax coincides with pelvis → no body_up
    assert math.isnan(head_lateral_tilt_3d(kps))
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_angles_3d.py -k head_lateral -v`
Expected: ImportError on `head_lateral_tilt_3d`.

- [ ] **Step 3: Implement the helper**

Append to `src/analysis/angles_3d.py`:

```python
HEAD = 10


def head_lateral_tilt_3d(kps_3d: np.ndarray) -> float:
    """Signed head-tilt angle (degrees) in the body's frontal plane.

    Positive = head tilted toward body_lateral (+) direction (L_hip → R_hip);
    negative = tilted the opposite way. Use the sign convention defined by
    body_frame_axes(). Returns NaN if the body frame is degenerate.
    """
    up, lateral, _ = body_frame_axes(kps_3d)
    # Reject degenerate frames (zero vectors come back from _norm unchanged).
    if float(np.linalg.norm(up)) < 1e-6 or float(np.linalg.norm(lateral)) < 1e-6:
        return float("nan")
    head_vec = kps_3d[HEAD] - kps_3d[THORAX]
    if float(np.linalg.norm(head_vec)) < 1e-9:
        return float("nan")
    lat_component = float(np.dot(head_vec, lateral))
    up_component = float(np.dot(head_vec, -up))  # up is pelvis→thorax; head is further along -up
    return float(np.degrees(np.arctan2(lat_component, up_component)))
```

Note: in MotionBERT's frame the head sits at a more-negative y than thorax (image-y down). Projecting onto `-up` gives a positive "vertical" component for the typical case, which is what `arctan2` wants for "head-up" being the reference.

- [ ] **Step 4: Tests pass**

Run: `uv run pytest tests/test_angles_3d.py -v`
Expected: new tests pass; existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/angles_3d.py tests/test_angles_3d.py
git commit -m "feat(angles): add head_lateral_tilt_3d for neck stretches"
```

---

## Task 6: Add `NeckStretchLeft` exercise + registry

**Files:**
- Create: `src/exercises/neck_stretch.py`
- Modify: `src/exercises/__init__.py`
- Create: `tests/test_exercises_registry.py`
- Create: `tests/test_neck_stretch.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_neck_stretch.py`:

```python
import math
import numpy as np

from analysis.types import PoseFrame
from exercises.neck_stretch import NeckStretchLeft


def _make_frame(kps_3d: np.ndarray | None) -> PoseFrame:
    return PoseFrame(
        timestamp=0.0,
        keypoints_2d=np.zeros((17, 2), dtype=np.float32),
        scores=np.ones(17, dtype=np.float32),
        frame_shape=(720, 1280),
        keypoints_3d=kps_3d,
    )


def _h36m_with_head_at(lat_offset: float) -> np.ndarray:
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[0] = (0.0, 0.0, 0.0)
    kps[1] = (-0.5, 0.0, 0.0)
    kps[4] = (0.5, 0.0, 0.0)
    kps[8] = (0.0, -1.0, 0.0)
    kps[10] = (lat_offset, -2.0, 0.0)
    return kps


def test_metadata():
    ex = NeckStretchLeft()
    assert ex.name == "neck_stretch_left"
    assert ex.target.side == "left"
    assert ex.target.hold_seconds == 20.0
    assert len(ex.target.joints) == 1
    assert ex.target.joints[0].name == "head_lateral_tilt"


def test_measure_returns_declared_joints():
    ex = NeckStretchLeft()
    measured = ex.measure(_make_frame(_h36m_with_head_at(-0.5)))
    assert set(measured.keys()) == {"head_lateral_tilt"}


def test_measure_nan_when_no_3d_keypoints():
    ex = NeckStretchLeft()
    measured = ex.measure(_make_frame(None))
    assert math.isnan(measured["head_lateral_tilt"])


def test_prompt_template_renders_summary_without_keyerror():
    ex = NeckStretchLeft()
    rendered = ex.prompt.summary.format(
        exercise_th=ex.display_th,
        score=87,
        duration=50,
        precision=25,
        stability=12,
        violations="(none)",
    )
    assert "87" in rendered


def test_prompt_template_renders_live_without_keyerror():
    ex = NeckStretchLeft()
    rendered = ex.prompt.live.format(
        exercise_th=ex.display_th,
        state="holding",
        progress_pct=60,
        violations="(none)",
    )
    assert "60" in rendered
```

Create `tests/test_exercises_registry.py`:

```python
from exercises import EXERCISES


def test_registry_contains_neck_stretch_left():
    assert "neck_stretch_left" in EXERCISES


def test_every_exercise_has_required_attrs():
    for key, ex in EXERCISES.items():
        assert ex.name == key, f"key/name mismatch for {key}"
        assert ex.display_th, f"{key} missing display_th"
        assert ex.target.joints, f"{key} has no target joints"
        assert ex.prompt.live, f"{key} missing live prompt"
        assert ex.prompt.summary, f"{key} missing summary prompt"
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_neck_stretch.py tests/test_exercises_registry.py -v`
Expected: ImportError on `exercises.neck_stretch`, KeyError on `EXERCISES`.

- [ ] **Step 3: Implement the exercise**

Create `src/exercises/neck_stretch.py`:

```python
from analysis.angles_3d import head_lateral_tilt_3d
from analysis.types import PoseFrame
from exercises.base import JointTarget, PromptTemplate, TargetPose


# Initial guess. Calibrate per spec §11.3 (see Task 13).
_NECK_TILT_TARGET_LEFT_DEG = -35.0
_NECK_TILT_TOLERANCE_DEG = 10.0


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
        hold_seconds=20.0,
        side="left",
    )
    prompt = PromptTemplate(live=_LIVE_TH, summary=_SUMMARY_TH)

    def measure(self, frame: PoseFrame) -> dict[str, float]:
        if frame.keypoints_3d is None:
            return {"head_lateral_tilt": float("nan")}
        return {"head_lateral_tilt": head_lateral_tilt_3d(frame.keypoints_3d)}
```

Replace `src/exercises/__init__.py` with the registry:

```python
"""Exercise registry. Each submodule registers its exercises here."""

from exercises.base import Exercise
from exercises.neck_stretch import NeckStretchLeft


EXERCISES: dict[str, Exercise] = {
    "neck_stretch_left": NeckStretchLeft(),
}
```

- [ ] **Step 4: Tests pass**

Run: `uv run pytest tests/test_neck_stretch.py tests/test_exercises_registry.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/exercises/neck_stretch.py src/exercises/__init__.py tests/test_neck_stretch.py tests/test_exercises_registry.py
git commit -m "feat(exercises): add NeckStretchLeft and registry"
```

---

## Task 7: Extend `prompt_th.py` with hold templates

Refactor the prompt-building so the LLM can dispatch on input type without loading the model. Pure functions → cheap tests.

**Files:**
- Modify: `src/feedback/prompt_th.py`
- Modify: `tests/` (no existing prompt test — add one)
- Create: `tests/test_prompt_th.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_prompt_th.py`:

```python
from analysis.types import HoldAnalysis, HoldState, LiveSnapshot, RuleViolation
from exercises.neck_stretch import NeckStretchLeft
from feedback.prompt_th import build_hold_summary_prompt, build_live_prompt


def test_build_live_prompt_uses_exercise_template():
    ex = NeckStretchLeft()
    snap = LiveSnapshot(
        exercise_name=ex.name,
        state=HoldState.HOLDING,
        progress_ratio=0.5,
        current_violations=[RuleViolation("head_lateral_tilt", 0.3, "เอียงเพิ่ม")],
    )
    text = build_live_prompt(snap, ex)
    assert ex.display_th in text
    assert "50%" in text or "50" in text
    assert "เอียงเพิ่ม" in text


def test_build_summary_prompt_uses_exercise_template():
    ex = NeckStretchLeft()
    analysis = HoldAnalysis(
        exercise_name=ex.name,
        score=82,
        components={"duration": 50, "precision": 22, "stability": 10},
        violations=[],
        in_target_ms=20_000,
        drift_count=1,
    )
    text = build_hold_summary_prompt(analysis, ex)
    assert ex.display_th in text
    assert "82" in text
    assert "50/50" in text


def test_build_live_prompt_handles_no_violations():
    ex = NeckStretchLeft()
    snap = LiveSnapshot(ex.name, HoldState.HOLDING, 0.1, [])
    text = build_live_prompt(snap, ex)
    assert text  # renders without KeyError or empty-list crash
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_prompt_th.py -v`
Expected: ImportError on `build_hold_summary_prompt` / `build_live_prompt`.

- [ ] **Step 3: Add the builders**

Append to `src/feedback/prompt_th.py`:

```python
from analysis.types import HoldAnalysis, LiveSnapshot
from exercises.base import Exercise


SYSTEM_TH_HOLD = (
    "คุณเป็นโค้ชยืดเหยียดที่ให้คำแนะนำกระชับและสุภาพ "
    "ตอบเป็นภาษาไทย 1-2 ประโยคเท่านั้น ไม่ใช้ Markdown ไม่ใช้บุลเล็ต ไม่ใช้ภาษาอังกฤษ"
)


def _format_violations(vs) -> str:
    if not vs:
        return "(ไม่มี)"
    return "; ".join(f"{v.detail_th} (ระดับ {v.severity:.2f})" for v in vs)


def build_live_prompt(snapshot: LiveSnapshot, exercise: Exercise) -> str:
    return exercise.prompt.live.format(
        exercise_th=exercise.display_th,
        state=snapshot.state.value,
        progress_pct=int(round(snapshot.progress_ratio * 100)),
        violations=_format_violations(snapshot.current_violations),
    )


def build_hold_summary_prompt(analysis: HoldAnalysis, exercise: Exercise) -> str:
    return exercise.prompt.summary.format(
        exercise_th=exercise.display_th,
        score=analysis.score,
        duration=analysis.components.get("duration", 0),
        precision=analysis.components.get("precision", 0),
        stability=analysis.components.get("stability", 0),
        violations=_format_violations(analysis.violations),
    )
```

Note: the per-exercise summary template in `neck_stretch.py` writes the budget caps literally (`{duration}/50`, `{precision}/30`, `{stability}/20`). The builder passes the raw int and lets the template add the `/50` etc. — so the rendered string contains e.g. `"50/50"` exactly once. Don't double-format.

- [ ] **Step 4: Tests pass**

Run: `uv run pytest tests/test_prompt_th.py -v`
Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/feedback/prompt_th.py tests/test_prompt_th.py
git commit -m "feat(prompt): add live + hold-summary prompt builders"
```

---

## Task 8: Dispatch in `ThaiCoachLLM.generate`

Make `generate()` accept any of `RepAnalysis`, `HoldAnalysis`, or `LiveSnapshot`. The squat path stays unchanged.

**Files:**
- Modify: `src/feedback/llm.py`
- Modify: `tests/test_llm_smoke.py` (light extension — keep smoke-only since model loads)

- [ ] **Step 1: Read the current `generate()` signature**

Open `src/feedback/llm.py`. Confirm `generate(self, rep: RepAnalysis, ...)` exists and pulls `build_user_prompt(rep)`. The dispatch will sit above that.

- [ ] **Step 2: Update `generate` to dispatch on input type**

Replace the body of `ThaiCoachLLM.generate` (lines 26-55) with:

```python
def generate(
    self,
    payload,                         # RepAnalysis | HoldAnalysis | LiveSnapshot
    max_tokens: int = 160,
    frame_bgr: Optional[np.ndarray] = None,
    exercise=None,                   # required for HoldAnalysis / LiveSnapshot
) -> str:
    from mlx_vlm import generate as mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    from analysis.types import HoldAnalysis, LiveSnapshot, RepAnalysis
    from feedback.prompt_th import (
        SYSTEM_TH,
        SYSTEM_TH_HOLD,
        build_user_prompt,
        build_hold_summary_prompt,
        build_live_prompt,
    )

    if isinstance(payload, RepAnalysis):
        system = SYSTEM_TH
        user = build_user_prompt(payload)
    elif isinstance(payload, HoldAnalysis):
        if exercise is None:
            raise ValueError("exercise= required for HoldAnalysis")
        system = SYSTEM_TH_HOLD
        user = build_hold_summary_prompt(payload, exercise)
    elif isinstance(payload, LiveSnapshot):
        if exercise is None:
            raise ValueError("exercise= required for LiveSnapshot")
        system = SYSTEM_TH_HOLD
        user = build_live_prompt(payload, exercise)
    else:
        raise TypeError(f"Unsupported payload type: {type(payload).__name__}")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = apply_chat_template(
        self._processor, self._config, messages,
        num_images=0, enable_thinking=False,
    )
    result = mlx_generate(
        self._model, self._processor,
        prompt=prompt, max_tokens=max_tokens, verbose=False,
    )
    text = getattr(result, "text", str(result))
    return _THINK_BLOCK.sub("", text).strip()
```

Leave `warmup()` unchanged (still uses the squat `RepAnalysis` dummy — that's correct).

- [ ] **Step 3: Update the smoke test (optional but recommended)**

Append to `tests/test_llm_smoke.py`:

```python
# Smoke tests for hold/live payloads. Requires the Qwen weights.

import pytest


@pytest.mark.skipif(
    not __import__("pathlib").Path("models/qwen3_5_4b_mxfp4").exists(),
    reason="Qwen weights not downloaded",
)
def test_generate_accepts_hold_analysis():
    from analysis.types import HoldAnalysis
    from exercises.neck_stretch import NeckStretchLeft
    from feedback.llm import ThaiCoachLLM

    llm = ThaiCoachLLM()
    ex = NeckStretchLeft()
    a = HoldAnalysis(
        exercise_name=ex.name,
        score=88,
        components={"duration": 50, "precision": 25, "stability": 13},
        violations=[],
        in_target_ms=20_000,
        drift_count=1,
    )
    text = llm.generate(a, max_tokens=32, exercise=ex)
    assert text
```

(Don't run this in CI without the weights; the `skipif` keeps it harmless.)

- [ ] **Step 4: Run unit tests (smoke is skipped without weights)**

Run: `uv run pytest tests/test_prompt_th.py tests/test_llm_smoke.py -v`
Expected: prompt tests pass; smoke test either skips (no weights) or passes (weights present).

- [ ] **Step 5: Commit**

```bash
git add src/feedback/llm.py tests/test_llm_smoke.py
git commit -m "feat(llm): dispatch generate() on payload type"
```

---

## Task 9: Build the OpenCV selector

**Files:**
- Create: `src/selector.py`

No tests (manual UI — per spec §13).

- [ ] **Step 1: Implement selector**

Create `src/selector.py`:

```python
"""Pre-loop OpenCV menu to choose an exercise.

Returns the selected Exercise. Press 1..9 to pick, q/Esc to quit.
The window stays modal until the user picks or cancels.
"""

import cv2
import numpy as np

from exercises import EXERCISES
from exercises.base import Exercise


_PANEL_BG = (24, 24, 28)
_TEXT_FG = (240, 240, 240)
_HINT_FG = (160, 160, 180)


def choose_exercise(default: str | None = None) -> Exercise:
    items = list(EXERCISES.items())
    if not items:
        raise RuntimeError("No exercises registered")

    width, line_h = 780, 36
    height = 80 + line_h * (len(items) + 2)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    while True:
        canvas[:] = _PANEL_BG
        cv2.putText(canvas, "Choose an exercise (number to pick, q to quit):",
                    (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _TEXT_FG, 1, cv2.LINE_AA)
        for i, (key, ex) in enumerate(items):
            row = 80 + i * line_h
            label = f"{i + 1}. [{key}] {ex.display_th}"
            cv2.putText(canvas, label, (24, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _TEXT_FG, 1, cv2.LINE_AA)
        cv2.putText(canvas, "(English keys; Thai is for the panel display only)",
                    (16, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _HINT_FG, 1, cv2.LINE_AA)
        cv2.imshow("Workout AI — Select Exercise", canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow("Workout AI — Select Exercise")
            raise SystemExit("User canceled exercise selection")
        if ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if 0 <= idx < len(items):
                cv2.destroyWindow("Workout AI — Select Exercise")
                return items[idx][1]
```

- [ ] **Step 2: Sanity import**

Run: `uv run python -c "from selector import choose_exercise; print(choose_exercise)"`
Expected: prints the function object — no import errors.

- [ ] **Step 3: Commit**

```bash
git add src/selector.py
git commit -m "feat(selector): add OpenCV exercise picker"
```

---

## Task 10: Extend `Renderer` for hold UI

Add a hold progress bar and state badge to the right panel.

**Files:**
- Modify: `src/render.py`
- Modify: `tests/test_render.py`

- [ ] **Step 1: Read the existing `Renderer.compose` to find an insertion point**

Open `src/render.py`. Locate `compose(...)`. The hold UI must accept optional `hold_state` and `hold_progress` keyword args and draw them on the right panel when present.

- [ ] **Step 2: Write a failing test**

Append to `tests/test_render.py`:

```python
import numpy as np
from render import Renderer


def test_compose_accepts_hold_kwargs_without_error():
    r = Renderer(panel_width=320)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = r.compose(
        frame,
        score=None,
        running_avg=0.0,
        rep_count=0,
        phase="holding",
        thai_text="",
        hold_state="holding",
        hold_progress=0.5,
    )
    assert out.shape == (480, 640 + 320, 3)
```

- [ ] **Step 3: Run to confirm failure**

Run: `uv run pytest tests/test_render.py -v`
Expected: TypeError — `compose()` got unexpected keyword args.

- [ ] **Step 4: Extend `compose`**

Modify `Renderer.compose` to accept and render hold UI. Add to the signature (alongside existing kwargs):

```python
def compose(
    self,
    frame: np.ndarray,
    score: int | None,
    running_avg: float,
    rep_count: int,
    phase: str,
    thai_text: str,
    rig_3d_kps: np.ndarray | None = None,
    attention: np.ndarray | None = None,
    hold_state: str | None = None,       # NEW: "idle"/"entering"/"holding"/"drifted"/"complete"
    hold_progress: float | None = None,  # NEW: 0.0 .. 1.0
) -> np.ndarray:
```

Inside the function, after the existing panel-drawing code, add:

```python
if hold_state is not None:
    panel_x = w + 12
    y = h - 90
    cv2.putText(canvas, f"Hold: {hold_state}", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    if hold_progress is not None:
        bar_y = y + 14
        bar_w = self.panel_width - 24
        cv2.rectangle(canvas, (panel_x, bar_y),
                      (panel_x + bar_w, bar_y + 14),
                      (60, 60, 70), 1)
        fill = int(bar_w * max(0.0, min(1.0, hold_progress)))
        color = (90, 220, 90) if hold_state == "holding" else (220, 200, 90)
        if hold_state == "drifted":
            color = (90, 150, 220)
        cv2.rectangle(canvas, (panel_x, bar_y),
                      (panel_x + fill, bar_y + 14),
                      color, -1)
```

- [ ] **Step 5: Tests pass**

Run: `uv run pytest tests/test_render.py -v`
Expected: passes.

- [ ] **Step 6: Commit**

```bash
git add src/render.py tests/test_render.py
git commit -m "feat(render): hold state badge + progress bar"
```

---

## Task 11: Refactor `app.py` — parameterize on exercise + selector loop

This is the integration step. Existing squat behavior stays accessible via `run_squat()`; new entry point is `run()` (the selector loop).

**Files:**
- Modify: `src/app.py`

No unit tests (integration only — covered manually in Task 12).

- [ ] **Step 1: Read current `app.py`**

Open `src/app.py` to recall the current structure. The whole `run()` body becomes either `run_squat()` (kept intact) or migrates pieces into a generic `run_session(...)`.

- [ ] **Step 2: Rewrite `app.py`**

Replace the contents of `src/app.py` with:

```python
import time
import cv2
import numpy as np

from capture import WebcamCapture
from pose2d import Pose2D
from pose3d import Pose3D, Pose3DBuffer, coco17_to_h36m17
from render import Renderer
from selector import choose_exercise
from analysis.phases import HoldFSM
from analysis.rules_hold import score_frame, score_hold
from analysis.types import HoldState, LiveSnapshot, PoseFrame
from exercises.base import Exercise
from feedback.llm import ThaiCoachLLM
from feedback.worker import LLMWorker


_LIVE_SUBMIT_INTERVAL_S = 2.5  # see spec §8.1


def run():
    """Top-level: load heavy weights once, then loop selector → session."""
    cap = WebcamCapture(device=0, width=1280, height=720)
    pose = Pose2D()
    lifter = Pose3D()
    renderer = Renderer(panel_width=320)

    print("Loading Qwen3.5-4B (this takes ~10 seconds the first time)...")
    llm = ThaiCoachLLM()
    print("Warming up LLM...")
    llm.warmup()
    worker = LLMWorker(llm)
    worker.start()
    print("Ready.")

    cap.start()
    try:
        while True:
            try:
                exercise = choose_exercise()
            except SystemExit:
                break
            run_session(cap, pose, lifter, renderer, worker, exercise)
    finally:
        worker.stop()
        cap.stop()
        cv2.destroyAllWindows()


def run_session(
    cap: WebcamCapture,
    pose: Pose2D,
    lifter: Pose3D,
    renderer: Renderer,
    worker: LLMWorker,
    exercise: Exercise,
) -> None:
    """One held-pose session. Returns when the hold completes or the user quits."""
    buf3d = Pose3DBuffer(lifter)
    fsm = HoldFSM(target_seconds=exercise.target.hold_seconds)
    max_severity_seen: dict[str, float] = {j.name: 0.0 for j in exercise.target.joints}
    last_live_submit_ts = 0.0
    frame_idx = 0
    last_rig_3d: np.ndarray | None = None

    completion_holder: dict = {}

    def on_complete(meta: dict) -> None:
        completion_holder["meta"] = meta

    fsm.on_hold_complete = on_complete

    while True:
        frame = cap.read_latest(timeout=2.0)
        if frame is None:
            return

        ts = time.monotonic()
        kps, scores, _hms = pose.infer_with_heatmaps(frame)
        pf = PoseFrame(
            timestamp=ts,
            keypoints_2d=kps,
            scores=scores,
            frame_shape=frame.shape[:2],
        )

        h36m = coco17_to_h36m17(kps, scores)
        buf3d.push(h36m)
        if frame_idx % 5 == 0 and buf3d.ready():
            try:
                last_rig_3d = buf3d.lift(frame.shape[0], frame.shape[1])
                pf.keypoints_3d = last_rig_3d
            except Exception as e:
                print(f"3D lift error: {e}")

        measured = exercise.measure(pf)
        in_target, violations = score_frame(exercise.target, measured)
        for v in violations:
            max_severity_seen[v.name] = max(max_severity_seen.get(v.name, 0.0), v.severity)

        state = fsm.update(in_target, ts)

        # Live LLM submission, throttled.
        if state in (HoldState.HOLDING, HoldState.DRIFTED) and \
           (ts - last_live_submit_ts) >= _LIVE_SUBMIT_INTERVAL_S:
            target_ms = int(exercise.target.hold_seconds * 1000) or 1
            snap = LiveSnapshot(
                exercise_name=exercise.name,
                state=state,
                progress_ratio=min(1.0, fsm.in_target_ms / target_ms),
                current_violations=violations,
            )
            worker.submit(snap, exercise=exercise)
            last_live_submit_ts = ts

        if state is HoldState.COMPLETE:
            meta = completion_holder.get("meta") or {
                "in_target_ms": fsm.in_target_ms,
                "drift_count": fsm.drift_count,
                "completed_ts": ts,
            }
            analysis = score_hold(exercise.name, meta, exercise.target, max_severity_seen)
            worker.submit(analysis, exercise=exercise)
            print(f"[hold {exercise.name}] score={analysis.score} components={analysis.components}")
            # Display the final state for ~3 seconds before returning.
            end_ts = ts + 3.0
            while time.monotonic() < end_ts:
                _show_frame(cap, renderer, exercise, fsm, worker, last_rig_3d)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
            return

        # Render this frame.
        target_ms = int(exercise.target.hold_seconds * 1000) or 1
        thai_text = worker.latest() or ""
        frame_drawn = renderer.draw_skeleton(frame, kps, scores)
        display = renderer.compose(
            frame_drawn,
            score=None,
            running_avg=0.0,
            rep_count=0,
            phase=exercise.display_th,
            thai_text=thai_text,
            rig_3d_kps=last_rig_3d,
            hold_state=state.value,
            hold_progress=min(1.0, fsm.in_target_ms / target_ms),
        )
        cv2.imshow("Workout AI", display)
        frame_idx += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return


def _show_frame(cap, renderer, exercise, fsm, worker, last_rig_3d):
    """Render-only helper used during the post-completion delay."""
    frame = cap.read_latest(timeout=0.5)
    if frame is None:
        return
    thai_text = worker.latest() or ""
    target_ms = int(exercise.target.hold_seconds * 1000) or 1
    display = renderer.compose(
        frame,
        score=None,
        running_avg=0.0,
        rep_count=0,
        phase=exercise.display_th,
        thai_text=thai_text,
        rig_3d_kps=last_rig_3d,
        hold_state="complete",
        hold_progress=min(1.0, fsm.in_target_ms / target_ms),
    )
    cv2.imshow("Workout AI", display)


if __name__ == "__main__":
    run()
```

- [ ] **Step 3: Update `LLMWorker.submit` if needed**

Inspect `src/feedback/worker.py`. The current `submit(rep)` takes one argument. Since this plan calls `worker.submit(snap, exercise=exercise)`, the worker must forward kwargs to `llm.generate(...)`.

Open the file and either:
- if `submit` already accepts `**kwargs` → no change.
- if it doesn't → change `submit(self, rep)` to `submit(self, payload, **kwargs)` and store kwargs alongside the payload; pass them through in the worker loop's `self._llm.generate(payload, **kwargs)`.

Add this confirmation as its own commit-able sub-step. Sketch (only if needed):

```python
def submit(self, payload, **kwargs):
    with self._cv:
        self._pending = (payload, kwargs)
        self._cv.notify()
```

…and in the loop:

```python
payload, kwargs = self._pending
self._pending = None
text = self._llm.generate(payload, **kwargs)
```

- [ ] **Step 4: Existing squat test suite still green**

Run: `uv run pytest -q`
Expected: all tests pass. (The new `app.py` no longer drives squats directly. If a test or notebook still imports the old `app.run()` flow, leave it for now; the squat path can be restored as a flag later. This plan's scope is the stretches slice.)

- [ ] **Step 5: Commit**

```bash
git add src/app.py src/feedback/worker.py
git commit -m "feat(app): selector loop + run_session parameterized on Exercise"
```

---

## Task 12: End-to-end smoke test

A single smoke test that runs the pose pipeline against a static image and exercises `score_frame` for `NeckStretchLeft`. Gated on the presence of model weights.

**Files:**
- Create: `tests/test_exercise_pipeline_smoke.py`
- Use: `tests/fixtures/` (existing — already has at least one image; add one if needed)

- [ ] **Step 1: Confirm a fixture image exists**

Run: `ls "/Users/thatt/Dev/AI project/new-workout-ai/tests/fixtures"`
Expected: at least one `.jpg` or `.png`. If empty, save one screenshot from any neck-stretch-looking pose into `tests/fixtures/neck_stretch_left.jpg` before continuing.

- [ ] **Step 2: Write the test**

Create `tests/test_exercise_pipeline_smoke.py`:

```python
from pathlib import Path
import cv2
import pytest

from analysis.rules_hold import score_frame
from analysis.types import PoseFrame
from exercises.neck_stretch import NeckStretchLeft

FIXTURE = Path(__file__).parent / "fixtures" / "neck_stretch_left.jpg"


@pytest.mark.skipif(
    not FIXTURE.exists() or not Path("models/rtmlib_cache").exists(),
    reason="fixture or pose weights missing",
)
def test_full_pipeline_on_static_image_runs_without_exception():
    from pose2d import Pose2D
    from pose3d import Pose3D, Pose3DBuffer, coco17_to_h36m17

    img = cv2.imread(str(FIXTURE))
    assert img is not None

    pose = Pose2D()
    kps, scores, _ = pose.infer_with_heatmaps(img)

    lifter = Pose3D()
    buf = Pose3DBuffer(lifter)
    h36m = coco17_to_h36m17(kps, scores)
    for _ in range(27):
        buf.push(h36m)
    rig_3d = buf.lift(img.shape[0], img.shape[1])

    pf = PoseFrame(
        timestamp=0.0,
        keypoints_2d=kps,
        scores=scores,
        frame_shape=img.shape[:2],
        keypoints_3d=rig_3d,
    )

    ex = NeckStretchLeft()
    measured = ex.measure(pf)
    in_target, violations = score_frame(ex.target, measured)

    # We don't assert in_target — depends on the image content. Just no exceptions.
    assert "head_lateral_tilt" in measured
    assert isinstance(in_target, bool)
    for v in violations:
        assert 0.0 <= v.severity <= 1.0
```

- [ ] **Step 3: Run the smoke test**

Run: `uv run pytest tests/test_exercise_pipeline_smoke.py -v`
Expected: skipped if weights/fixture missing; passes otherwise (no exceptions).

- [ ] **Step 4: Commit**

```bash
git add tests/test_exercise_pipeline_smoke.py tests/fixtures
git commit -m "test(exercises): add static-image smoke test for NeckStretchLeft"
```

---

## Task 13: Manual verification + calibrate target angles

This task is **not** code — it's a real-world sanity check and the calibration loop the spec calls out in §11.3.

- [ ] **Step 1: Run the app**

Run: `uv run python main.py`
Expected: a selector window appears listing "1. [neck_stretch_left] ยืดคอด้านซ้าย". Pressing `1` opens the camera with the live overlay, hold state badge, and progress bar.

- [ ] **Step 2: Verify FSM transitions in the live UI**

Stand in front of the camera. Walk through:
- Neutral pose → state should read `idle`.
- Tilt head left slowly → `entering` for ~0.5s, then `holding`. The progress bar fills.
- Return to neutral briefly (<0.3s) → `drifted` then back to `holding`.
- Hold long enough to complete → state badge reads `complete`, panel shows a Thai summary.

If any transition feels wrong (e.g. requires unreasonable tilt to reach `in_target`), the calibration is off → go to Step 3.

- [ ] **Step 3: Calibrate `target_deg` if needed**

Take a single reference photo of yourself in the correct neck-stretch-left posture. Run a one-off script to measure:

```bash
uv run python -c "
import cv2, sys
from pose2d import Pose2D
from pose3d import Pose3D, Pose3DBuffer, coco17_to_h36m17
from analysis.angles_3d import head_lateral_tilt_3d

img = cv2.imread('tests/fixtures/neck_stretch_left.jpg')
pose, lifter = Pose2D(), Pose3D()
kps, scores, _ = pose.infer_with_heatmaps(img)
buf = Pose3DBuffer(lifter)
h36m = coco17_to_h36m17(kps, scores)
for _ in range(27): buf.push(h36m)
rig = buf.lift(img.shape[0], img.shape[1])
print('head_lateral_tilt_3d =', head_lateral_tilt_3d(rig))
"
```

Update `_NECK_TILT_TARGET_LEFT_DEG` in `src/exercises/neck_stretch.py` to that measured value. Keep tolerance at ~10° to start; tighten if the in/out-of-target signal feels too lenient.

- [ ] **Step 4: Re-run and re-verify**

`uv run python main.py` again. Confirm the in-target band feels correct.

- [ ] **Step 5: Commit the calibrated value (if changed)**

```bash
git add src/exercises/neck_stretch.py
git commit -m "tune(neck_stretch_left): calibrate target tilt from reference photo"
```

---

## Out of scope for this plan

These come in a **follow-up plan** once the vertical slice ships:

- The remaining 9 exercise entries (right side of neck, both shoulders, chest-and-shoulder, both front-hand, both back-hand, neck-flexion). Each is a new module under `src/exercises/` + a registry entry + any missing angle helper (e.g. shoulder abduction in `angles_3d.py`) + tests. Templating the same shape as Task 5 + Task 6.
- Restoring the standalone squat run mode (the spec keeps squat working but this slice replaces `app.run()`'s top-level squat-only flow with the selector loop; if you still want a one-shot squat launcher, add `run_squat()` and a CLI flag in the follow-up).
- Performance tuning for the live LLM cadence — spec §11.2 open question.
- Mobile streaming server — CLAUDE.md long-term direction; unaffected by this slice as long as the selector remains replaceable.

---

## Self-review notes (already applied)

- Verified `analysis.types` already imports `dataclass`, `Enum`, `Optional` — no extra imports needed in Task 1.
- Verified `analysis.phases` already imports `dataclass`, `Optional`, `Callable` — only `HoldState` import needed in Task 2.
- Confirmed the body-frame axis convention in `angles_3d.body_frame_axes` (`body_lateral = L_hip → R_hip` projected orthogonal to body_up) — drives the sign convention in Task 5 + Task 13 calibration.
- `LLMWorker.submit` may already accept `**kwargs`; Task 11 Step 3 makes the change conditional so the plan works either way.
- `head_lateral_tilt_3d` is the only new geometry helper needed for this slice. Other stretches will likely need shoulder abduction / elbow angles — captured in the out-of-scope note.

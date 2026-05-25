# Real-Time Squat Form Coach Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a macOS Apple-Silicon desktop app that captures webcam video, estimates 2D pose with RTMPose-l, lifts to 3D with MotionBERT, scores squat form using joint-angle rules + phase FSM + model attention, renders a live skeleton + 3D rig + score overlay, and generates async Thai-language feedback with Qwen3.5-4B via mlx-vlm.

**Architecture:** A single Python process runs a 30 FPS OpenCV loop. Per-frame: 2D pose + angle rules + phase FSM + render. Every 5 frames: MotionBERT 3D lift over a sliding 27-frame window. At rep boundaries: structured analysis is enqueued to a background LLM worker that returns Thai text 1–3 s later. All ML weights are downloaded into a project-local `models/` directory.

**Tech Stack:** Python 3.12, `uv`, OpenCV, `rtmlib` (ONNX RTMPose-l), PyTorch + MPS (MotionBERT), `mlx-vlm` (Qwen3.5-4B mxfp4), `huggingface_hub`, `pytest`.

---

## File Structure

```
src/workout_ai/
├── __init__.py
├── capture.py              # webcam capture thread
├── pose2d.py               # rtmlib RTMPose-l wrapper
├── pose3d.py               # MotionBERT wrapper + sliding window buffer
├── analysis/
│   ├── __init__.py
│   ├── angles.py           # joint-angle computation (pure functions)
│   ├── phases.py           # squat FSM
│   ├── attention.py        # RTMPose heatmap aggregation (model attention proxy for GradCAM)
│   ├── rules_squat.py      # squat rules + scoring
│   └── types.py            # PoseFrame, RepAnalysis, PhaseState dataclasses
├── feedback/
│   ├── __init__.py
│   ├── llm.py              # mlx-vlm Qwen3.5-4B wrapper
│   ├── worker.py           # async LLM worker thread
│   └── prompt_th.py        # Thai prompt template
├── render.py               # OpenCV overlay (skeleton, 3D rig, score, attention, Thai text)
└── app.py                  # main loop
scripts/
└── download_models.py      # downloads RTMPose-l, MotionBERT, Qwen3.5-4B
tests/
├── test_angles.py
├── test_phases.py
├── test_rules_squat.py
├── test_pose2d_smoke.py
├── test_pose3d_smoke.py
└── test_llm_smoke.py
models/                     # gitignored; populated by download_models.py
vendor/motionbert/          # cloned upstream code (gitignored)
docs/superpowers/specs/2026-05-18-pose-form-coach-design.md
docs/superpowers/plans/2026-05-18-pose-form-coach.md   # this file
```

**Key boundaries:**
- `analysis/*` is pure-Python, no ML deps, fully unit-testable with synthetic keypoints.
- `pose2d.py` / `pose3d.py` / `feedback/llm.py` each wrap one model and expose a narrow interface.
- `render.py` only draws — never reasons about form.
- `app.py` is the only file that touches the camera + threads + the user.

---

## Task 1: Project dependencies and structure

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `src/workout_ai/__init__.py`
- Create: `src/workout_ai/analysis/__init__.py`
- Create: `src/workout_ai/feedback/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Update `pyproject.toml` with dependencies**

```toml
[project]
name = "new-workout-ai"
version = "0.1.0"
description = "Real-time squat form coach with Thai feedback"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "opencv-python>=4.10.0",
    "numpy>=1.26,<2.0",
    "rtmlib>=0.0.13",
    "onnxruntime>=1.20.0",
    "torch>=2.4.0",
    "torchvision>=0.19.0",
    "mlx>=0.21.0",
    "mlx-vlm>=0.1.0",
    "huggingface-hub>=0.26.0",
    "pillow>=10.0.0",
    "tqdm>=4.66.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-mock>=3.12.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/workout_ai"]
```

- [ ] **Step 2: Update `.gitignore`**

Append:

```
# Project-specific
models/
vendor/
*.mp4
*.mov
.pytest_cache/
.ruff_cache/
```

- [ ] **Step 3: Create empty package init files**

Create `src/workout_ai/__init__.py`:

```python
"""Real-time squat form coach."""
__version__ = "0.1.0"
```

Create `src/workout_ai/analysis/__init__.py`:

```python
```

Create `src/workout_ai/feedback/__init__.py`:

```python
```

Create `tests/__init__.py`:

```python
```

Create `tests/conftest.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

- [ ] **Step 4: Install deps and verify**

Run: `uv sync --all-extras`
Expected: completes without error, `.venv/` populated.

Run: `uv run python -c "import cv2, numpy, rtmlib, torch, mlx, mlx_vlm, huggingface_hub; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore src/ tests/ uv.lock
git commit -m "feat: scaffold project structure and dependencies"
```

---

## Task 2: Shared types

**Files:**
- Create: `src/workout_ai/analysis/types.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: Write failing test for type construction**

`tests/test_types.py`:

```python
import numpy as np
from workout_ai.analysis.types import PoseFrame, PhaseState, RepAnalysis, RuleViolation


def test_pose_frame_construction():
    keypoints = np.zeros((17, 2), dtype=np.float32)
    scores = np.zeros((17,), dtype=np.float32)
    pf = PoseFrame(timestamp=1.0, keypoints_2d=keypoints, scores=scores, frame_shape=(480, 640))
    assert pf.timestamp == 1.0
    assert pf.keypoints_2d.shape == (17, 2)


def test_phase_state_enum():
    assert PhaseState.STANDING.value == "standing"
    assert PhaseState.BOTTOM.value == "bottom"


def test_rep_analysis_defaults():
    ra = RepAnalysis(
        rep_index=0,
        score=85,
        components={"depth": 25, "valgus": 20, "torso": 18, "symmetry": 14, "tempo": 8},
        violations=[],
        descent_ms=900,
        ascent_ms=800,
    )
    assert ra.score == 85
    assert ra.violations == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'workout_ai.analysis.types'`.

- [ ] **Step 3: Implement types**

Create `src/workout_ai/analysis/types.py`:

```python
from dataclasses import dataclass, field
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
    scores: np.ndarray        # shape (17,)
    frame_shape: tuple[int, int]  # (H, W)
    keypoints_3d: Optional[np.ndarray] = None  # shape (17, 3) when MotionBERT result is attached
    attention: Optional[np.ndarray] = None     # shape (H, W) heatmap from RTMPose


@dataclass
class RuleViolation:
    name: str
    severity: float  # 0.0 .. 1.0
    detail_th: str   # short Thai phrase, used in LLM prompt as a hint


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_types.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/workout_ai/analysis/types.py tests/test_types.py
git commit -m "feat(types): add PoseFrame, PhaseState, RepAnalysis dataclasses"
```

---

## Task 3: Joint-angle computation (pure functions)

**Files:**
- Create: `src/workout_ai/analysis/angles.py`
- Create: `tests/test_angles.py`

Background: COCO-17 keypoint indices used here:

```
0  nose
5  L_shoulder    6  R_shoulder
11 L_hip         12 R_hip
13 L_knee        14 R_knee
15 L_ankle       16 R_ankle
```

- [ ] **Step 1: Write failing tests**

`tests/test_angles.py`:

```python
import numpy as np
import pytest
from workout_ai.analysis.angles import (
    angle_between_3_points,
    knee_angles,
    torso_lean_deg,
    hip_below_knee,
)


def test_angle_180_degrees_straight():
    # three collinear points -> 180
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([2.0, 0.0])
    assert angle_between_3_points(a, b, c) == pytest.approx(180.0, abs=0.5)


def test_angle_90_degrees():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([1.0, 1.0])
    assert angle_between_3_points(a, b, c) == pytest.approx(90.0, abs=0.5)


def test_knee_angles_standing():
    # Synthetic standing pose: hip above knee above ankle, all roughly aligned
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (100, 100)  # L hip
    kps[13] = (100, 200)  # L knee
    kps[15] = (100, 300)  # L ankle
    kps[12] = (150, 100)  # R hip
    kps[14] = (150, 200)  # R knee
    kps[16] = (150, 300)  # R ankle
    left, right = knee_angles(kps)
    assert left == pytest.approx(180.0, abs=1.0)
    assert right == pytest.approx(180.0, abs=1.0)


def test_knee_angles_squatting():
    # Knee bent ~90deg
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (50, 200)
    kps[13] = (100, 200)  # knee
    kps[15] = (100, 300)  # ankle (below knee)
    kps[12] = kps[11]
    kps[14] = kps[13]
    kps[16] = kps[15]
    left, right = knee_angles(kps)
    assert left == pytest.approx(90.0, abs=2.0)


def test_torso_lean_upright():
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[5] = (100, 100)   # L shoulder
    kps[6] = (150, 100)
    kps[11] = (100, 200)  # L hip
    kps[12] = (150, 200)
    assert torso_lean_deg(kps) == pytest.approx(0.0, abs=1.0)


def test_torso_lean_45deg_forward():
    kps = np.zeros((17, 2), dtype=np.float32)
    # Shoulders shifted forward relative to hips by same distance as vertical separation
    kps[5] = (200, 100)
    kps[6] = (250, 100)
    kps[11] = (100, 200)
    kps[12] = (150, 200)
    assert torso_lean_deg(kps) == pytest.approx(45.0, abs=2.0)


def test_hip_below_knee_true():
    # OpenCV coords: y increases downward; "hip below knee" means hip_y > knee_y
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (100, 250)  # L hip y=250
    kps[12] = (150, 250)
    kps[13] = (100, 200)  # L knee y=200
    kps[14] = (150, 200)
    assert hip_below_knee(kps) is True


def test_hip_below_knee_false():
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (100, 100)
    kps[12] = (150, 100)
    kps[13] = (100, 200)
    kps[14] = (150, 200)
    assert hip_below_knee(kps) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_angles.py -v`
Expected: all FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement angles**

Create `src/workout_ai/analysis/angles.py`:

```python
import numpy as np

L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def angle_between_3_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees, B is the vertex."""
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def knee_angles(kps: np.ndarray) -> tuple[float, float]:
    left = angle_between_3_points(kps[L_HIP], kps[L_KNEE], kps[L_ANKLE])
    right = angle_between_3_points(kps[R_HIP], kps[R_KNEE], kps[R_ANKLE])
    return left, right


def torso_lean_deg(kps: np.ndarray) -> float:
    """Angle between the vector (mid-hip -> mid-shoulder) and vertical (up)."""
    mid_shoulder = (kps[L_SHOULDER] + kps[R_SHOULDER]) / 2.0
    mid_hip = (kps[L_HIP] + kps[R_HIP]) / 2.0
    v = mid_shoulder - mid_hip  # points from hip to shoulder
    # In image coords, "up" is -y. Vertical reference vector:
    vertical = np.array([0.0, -1.0])
    cos = np.dot(v, vertical) / (np.linalg.norm(v) + 1e-9)
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def hip_below_knee(kps: np.ndarray) -> bool:
    """In image coords (y down), hip below knee means hip_y > knee_y."""
    mid_hip_y = (kps[L_HIP, 1] + kps[R_HIP, 1]) / 2.0
    mid_knee_y = (kps[L_KNEE, 1] + kps[R_KNEE, 1]) / 2.0
    return bool(mid_hip_y > mid_knee_y)


def knee_valgus_ratio(kps: np.ndarray) -> tuple[float, float]:
    """Signed normalized distance: (knee_x - ankle_x) / hip_width.
    Positive = knee inward of ankle on that side. Returns (left, right).
    Used as a valgus signal; abs value > 0.15 considered valgus."""
    hip_width = abs(kps[R_HIP, 0] - kps[L_HIP, 0]) + 1e-6
    l = (kps[L_KNEE, 0] - kps[L_ANKLE, 0]) / hip_width
    r = (kps[R_ANKLE, 0] - kps[R_KNEE, 0]) / hip_width
    return float(l), float(r)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_angles.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/workout_ai/analysis/angles.py tests/test_angles.py
git commit -m "feat(analysis): joint-angle and posture-geometry pure functions"
```

---

## Task 4: Squat phase FSM

**Files:**
- Create: `src/workout_ai/analysis/phases.py`
- Create: `tests/test_phases.py`

The FSM transitions:

```
STANDING --(knee angle drops below 160)--> DESCENT
DESCENT  --(knee angle below 100)--> BOTTOM
BOTTOM   --(knee angle rises above 100)--> ASCENT
ASCENT   --(knee angle above 160)--> STANDING (+1 rep)
```

- [ ] **Step 1: Write failing tests**

`tests/test_phases.py`:

```python
import numpy as np
from workout_ai.analysis.phases import SquatFSM
from workout_ai.analysis.types import PhaseState


def make_kps(knee_angle_deg: float) -> np.ndarray:
    """Build a synthetic keypoint array with a target knee angle."""
    kps = np.zeros((17, 2), dtype=np.float32)
    # Place hip, knee, ankle so the L knee angle matches knee_angle_deg.
    # Hip at origin, knee at (0, 100), ankle at angle theta from -y axis.
    import math
    theta = math.radians(180 - knee_angle_deg)
    kps[11] = (0.0, 0.0)
    kps[13] = (0.0, 100.0)
    kps[15] = (100.0 * math.sin(theta), 100.0 + 100.0 * math.cos(theta))
    # Mirror for right side
    kps[12] = kps[11] + (50, 0)
    kps[14] = kps[13] + (50, 0)
    kps[16] = kps[15] + (50, 0)
    # Shoulders straight up from hips
    kps[5] = kps[11] + (0, -50)
    kps[6] = kps[12] + (0, -50)
    return kps


def test_starts_standing():
    fsm = SquatFSM()
    assert fsm.state == PhaseState.STANDING


def test_full_rep_cycle():
    fsm = SquatFSM()
    rep_completed = []

    def on_rep(meta):
        rep_completed.append(meta)

    fsm.on_rep_complete = on_rep
    # Standing
    fsm.update(make_kps(175.0), timestamp=0.0)
    assert fsm.state == PhaseState.STANDING
    # Descent
    fsm.update(make_kps(140.0), timestamp=0.5)
    assert fsm.state == PhaseState.DESCENT
    # Bottom
    fsm.update(make_kps(85.0), timestamp=1.0)
    assert fsm.state == PhaseState.BOTTOM
    # Ascent
    fsm.update(make_kps(110.0), timestamp=1.5)
    assert fsm.state == PhaseState.ASCENT
    # Back to standing -> rep complete
    fsm.update(make_kps(175.0), timestamp=2.0)
    assert fsm.state == PhaseState.STANDING
    assert len(rep_completed) == 1
    meta = rep_completed[0]
    assert meta["descent_ms"] == 1000  # 0.0 -> 1.0
    assert meta["ascent_ms"] == 1000   # 1.0 -> 2.0


def test_descent_then_back_up_no_rep():
    fsm = SquatFSM()
    rep_completed = []
    fsm.on_rep_complete = lambda m: rep_completed.append(m)
    fsm.update(make_kps(175.0), timestamp=0.0)
    fsm.update(make_kps(140.0), timestamp=0.5)   # descent
    fsm.update(make_kps(175.0), timestamp=1.0)   # standing again, never hit bottom
    assert fsm.state == PhaseState.STANDING
    assert len(rep_completed) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_phases.py -v`
Expected: FAIL — `SquatFSM` not defined.

- [ ] **Step 3: Implement FSM**

Create `src/workout_ai/analysis/phases.py`:

```python
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

from workout_ai.analysis.angles import knee_angles
from workout_ai.analysis.types import PhaseState


STAND_THRESHOLD = 160.0   # knee angle (deg) above this = standing
BOTTOM_THRESHOLD = 100.0  # below this = bottom


@dataclass
class SquatFSM:
    state: PhaseState = PhaseState.STANDING
    descent_start_ts: Optional[float] = None
    bottom_ts: Optional[float] = None
    on_rep_complete: Optional[Callable[[dict], None]] = None

    def update(self, kps: np.ndarray, timestamp: float) -> PhaseState:
        l, r = knee_angles(kps)
        angle = (l + r) / 2.0

        if self.state == PhaseState.STANDING:
            if angle < STAND_THRESHOLD:
                self.state = PhaseState.DESCENT
                self.descent_start_ts = timestamp
        elif self.state == PhaseState.DESCENT:
            if angle < BOTTOM_THRESHOLD:
                self.state = PhaseState.BOTTOM
                self.bottom_ts = timestamp
            elif angle >= STAND_THRESHOLD:
                # Went back up without hitting bottom; abort rep
                self.state = PhaseState.STANDING
                self.descent_start_ts = None
        elif self.state == PhaseState.BOTTOM:
            if angle >= BOTTOM_THRESHOLD:
                self.state = PhaseState.ASCENT
        elif self.state == PhaseState.ASCENT:
            if angle >= STAND_THRESHOLD:
                meta = {
                    "descent_ms": int((self.bottom_ts - self.descent_start_ts) * 1000),
                    "ascent_ms": int((timestamp - self.bottom_ts) * 1000),
                    "bottom_ts": self.bottom_ts,
                    "completed_ts": timestamp,
                }
                self.state = PhaseState.STANDING
                self.descent_start_ts = None
                self.bottom_ts = None
                if self.on_rep_complete:
                    self.on_rep_complete(meta)
        return self.state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_phases.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/workout_ai/analysis/phases.py tests/test_phases.py
git commit -m "feat(analysis): squat phase FSM with rep detection"
```

---

## Task 5: Squat rules + scoring

**Files:**
- Create: `src/workout_ai/analysis/rules_squat.py`
- Create: `tests/test_rules_squat.py`

- [ ] **Step 1: Write failing tests**

`tests/test_rules_squat.py`:

```python
import numpy as np
from workout_ai.analysis.rules_squat import score_rep
from workout_ai.analysis.types import PoseFrame


def make_pose_frame(knee_angle: float, hip_y: float, knee_y: float, valgus: float = 0.0, lean: float = 30.0) -> PoseFrame:
    """Manually construct a PoseFrame with hand-tuned keypoints for a given posture."""
    kps = np.zeros((17, 2), dtype=np.float32)
    # Hip + knee with desired y
    kps[11] = (100, hip_y)
    kps[12] = (150, hip_y)
    kps[13] = (100 + valgus * 50, knee_y)
    kps[14] = (150 - valgus * 50, knee_y)
    # Ankle: place to give the requested knee angle approximately
    # For simplicity in this test, we don't constrain knee angle precisely;
    # we pass the angle separately to score_rep.
    kps[15] = (100, knee_y + 100)
    kps[16] = (150, knee_y + 100)
    # Shoulders: tilt forward by `lean` degrees
    import math
    dx = math.tan(math.radians(lean)) * 100
    kps[5] = (100 + dx, hip_y - 100)
    kps[6] = (150 + dx, hip_y - 100)
    scores = np.ones((17,), dtype=np.float32)
    return PoseFrame(timestamp=1.0, keypoints_2d=kps, scores=scores, frame_shape=(480, 640))


def test_perfect_rep_high_score():
    bottom = make_pose_frame(knee_angle=85.0, hip_y=260, knee_y=240, valgus=0.0, lean=35.0)
    result = score_rep(bottom_frame=bottom, descent_ms=1200, ascent_ms=1000)
    assert result.score >= 90
    assert result.violations == []


def test_shallow_squat_loses_depth_points():
    # Hip is ABOVE knee at bottom -> depth fails
    bottom = make_pose_frame(knee_angle=120.0, hip_y=180, knee_y=240, lean=30.0)
    result = score_rep(bottom_frame=bottom, descent_ms=900, ascent_ms=900)
    assert result.components["depth"] == 0
    assert any(v.name == "shallow_depth" for v in result.violations)


def test_knee_valgus_detected():
    bottom = make_pose_frame(knee_angle=85.0, hip_y=260, knee_y=240, valgus=0.4, lean=30.0)
    result = score_rep(bottom_frame=bottom, descent_ms=1000, ascent_ms=1000)
    assert any(v.name == "knee_valgus" for v in result.violations)
    assert result.components["valgus"] < 25


def test_excessive_forward_lean():
    bottom = make_pose_frame(knee_angle=85.0, hip_y=260, knee_y=240, lean=70.0)
    result = score_rep(bottom_frame=bottom, descent_ms=1000, ascent_ms=1000)
    assert any(v.name == "excessive_forward_lean" for v in result.violations)
    assert result.components["torso"] < 20


def test_tempo_penalty_when_ascent_longer():
    bottom = make_pose_frame(knee_angle=85.0, hip_y=260, knee_y=240, lean=30.0)
    result = score_rep(bottom_frame=bottom, descent_ms=500, ascent_ms=1500)
    assert result.components["tempo"] < 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rules_squat.py -v`
Expected: FAIL — `score_rep` not defined.

- [ ] **Step 3: Implement rules**

Create `src/workout_ai/analysis/rules_squat.py`:

```python
from workout_ai.analysis.angles import (
    knee_angles,
    torso_lean_deg,
    hip_below_knee,
    knee_valgus_ratio,
)
from workout_ai.analysis.types import PoseFrame, RepAnalysis, RuleViolation

VALGUS_THRESHOLD = 0.15  # |knee_x - ankle_x| / hip_width above this = valgus


def score_rep(bottom_frame: PoseFrame, descent_ms: int, ascent_ms: int, rep_index: int = 0) -> RepAnalysis:
    kps = bottom_frame.keypoints_2d
    violations: list[RuleViolation] = []
    components: dict[str, int] = {}

    # --- Depth (30 pts) ---
    if hip_below_knee(kps):
        components["depth"] = 30
    else:
        components["depth"] = 0
        violations.append(RuleViolation(
            name="shallow_depth",
            severity=1.0,
            detail_th="ลงไม่ลึกพอ สะโพกยังสูงกว่าหัวเข่า",
        ))

    # --- Valgus (25 pts) ---
    l_v, r_v = knee_valgus_ratio(kps)
    worst_valgus = max(abs(l_v), abs(r_v))
    if worst_valgus < VALGUS_THRESHOLD:
        components["valgus"] = 25
    else:
        severity = min(1.0, (worst_valgus - VALGUS_THRESHOLD) / 0.3)
        components["valgus"] = int(25 * (1.0 - severity))
        violations.append(RuleViolation(
            name="knee_valgus",
            severity=severity,
            detail_th="หัวเข่าเข้าด้านใน ควรกางหัวเข่าออกตามแนวปลายเท้า",
        ))

    # --- Torso (20 pts) ---
    lean = torso_lean_deg(kps)
    if 20.0 <= lean <= 55.0:
        components["torso"] = 20
    elif lean > 55.0:
        components["torso"] = max(0, int(20 * (1 - (lean - 55) / 30)))
        violations.append(RuleViolation(
            name="excessive_forward_lean",
            severity=min(1.0, (lean - 55) / 30),
            detail_th="ลำตัวโน้มไปข้างหน้ามากเกินไป",
        ))
    else:
        components["torso"] = max(0, int(20 * (1 - (20 - lean) / 20)))
        violations.append(RuleViolation(
            name="too_upright",
            severity=min(1.0, (20 - lean) / 20),
            detail_th="ลำตัวตั้งตรงเกินไป ควรโน้มไปข้างหน้าเล็กน้อย",
        ))

    # --- Symmetry (15 pts) ---
    l_ang, r_ang = knee_angles(kps)
    delta = abs(l_ang - r_ang)
    if delta < 10.0:
        components["symmetry"] = 15
    else:
        severity = min(1.0, (delta - 10.0) / 20.0)
        components["symmetry"] = int(15 * (1.0 - severity))
        violations.append(RuleViolation(
            name="asymmetric",
            severity=severity,
            detail_th="ซ้ายและขวาไม่สมมาตร",
        ))

    # --- Tempo (10 pts) ---
    if descent_ms >= ascent_ms:
        components["tempo"] = 10
    else:
        ratio = descent_ms / max(1, ascent_ms)
        components["tempo"] = int(10 * ratio)
        violations.append(RuleViolation(
            name="fast_descent",
            severity=1.0 - ratio,
            detail_th="ลงเร็วกว่าขึ้น ควรลงช้าๆ ควบคุมการเคลื่อนไหว",
        ))

    total = sum(components.values())
    return RepAnalysis(
        rep_index=rep_index,
        score=total,
        components=components,
        violations=violations,
        descent_ms=descent_ms,
        ascent_ms=ascent_ms,
        bottom_frame_keypoints_2d=kps.copy(),
        bottom_frame_keypoints_3d=bottom_frame.keypoints_3d.copy() if bottom_frame.keypoints_3d is not None else None,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rules_squat.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/workout_ai/analysis/rules_squat.py tests/test_rules_squat.py
git commit -m "feat(analysis): squat rule set with weighted 0-100 scoring"
```

---

## Task 6: Model download script

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/download_models.py`

This downloads three model artifacts into `models/`:
1. RTMPose-l ONNX (Halpe-26 or COCO-17 — we use COCO-17 from rtmlib's built-in)
2. MotionBERT checkpoint (`MB_ft_h36m_global_lite.bin`) from the official Hugging Face mirror
3. Qwen3.5-4B mxfp4 MLX-VLM

Note: `rtmlib` auto-downloads RTMPose to its own cache on first use. We pre-warm it explicitly here so it lives where we expect.

- [ ] **Step 1: Create download script**

Create `scripts/__init__.py`:

```python
```

Create `scripts/download_models.py`:

```python
"""Download all model artifacts into ./models/ at the project root."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def download_rtmpose():
    """Pre-warm rtmlib's RTMPose-l download by instantiating it once."""
    print("[1/3] Downloading RTMPose-l via rtmlib...")
    os.environ.setdefault("RTMLIB_CACHE", str(MODELS_DIR / "rtmlib_cache"))
    from rtmlib import Body
    _ = Body(mode="performance", to_openpose=False, backend="onnxruntime", device="cpu")
    print("       RTMPose-l ready.")


def download_motionbert():
    print("[2/3] Downloading MotionBERT checkpoint...")
    from huggingface_hub import hf_hub_download
    target_dir = MODELS_DIR / "motionbert"
    target_dir.mkdir(parents=True, exist_ok=True)
    ckpt = hf_hub_download(
        repo_id="walterzhu/MotionBERT-Lite",
        filename="MB_ft_h36m_global_lite.bin",
        local_dir=str(target_dir),
    )
    print(f"       MotionBERT at {ckpt}")


def download_qwen():
    print("[3/3] Downloading Qwen3.5-4B (mxfp4 mlx-vlm)...")
    from huggingface_hub import snapshot_download
    target_dir = MODELS_DIR / "qwen3_5_4b_mxfp4"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id="RepublicOfKorokke/Qwen3.5-4B-mlx-vlm-mxfp4",
        local_dir=str(target_dir),
    )
    print(f"       Qwen at {path}")


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    download_rtmpose()
    download_motionbert()
    download_qwen()
    print("\nAll models downloaded.")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the download (long-running, can be deferred)**

Run: `uv run python scripts/download_models.py`
Expected: 3 download phases complete, `models/` populated. Total ~6 GB.

If the MotionBERT repo `walterzhu/MotionBERT-Lite` doesn't resolve, fall back to manual download from the upstream Google Drive link in the MotionBERT README and place the file at `models/motionbert/MB_ft_h36m_global_lite.bin`.

If the Qwen mxfp4 repo unavailable, swap to `mlx-community/Qwen3.5-4B-MLX-8bit` and update the path used in `feedback/llm.py` (Task 16).

- [ ] **Step 3: Smoke-check files exist**

Run: `ls -la models/`
Expected: directories `rtmlib_cache/`, `motionbert/`, `qwen3_5_4b_mxfp4/` present and non-empty.

- [ ] **Step 4: Commit**

```bash
git add scripts/__init__.py scripts/download_models.py
git commit -m "feat(scripts): unified model downloader into project-local models/"
```

---

## Task 7: 2D pose wrapper around rtmlib

**Files:**
- Create: `src/workout_ai/pose2d.py`
- Create: `tests/test_pose2d_smoke.py`
- Create: `tests/fixtures/standing_person.jpg` (downloaded; instructions in step)

- [ ] **Step 1: Add a fixture image**

Run:

```bash
mkdir -p tests/fixtures
curl -L -o tests/fixtures/standing_person.jpg \
  "https://images.pexels.com/photos/3076509/pexels-photo-3076509.jpeg?auto=compress&w=640"
```

Expected: a ~640px JPEG of a standing person at `tests/fixtures/standing_person.jpg`.

If that URL fails, substitute any public-domain photo of a clearly visible standing person. If no network is available, generate a synthetic image with a stick figure drawn in OpenCV and skip the pose detection assertion (mark the test `@pytest.mark.skip`).

- [ ] **Step 2: Write smoke test**

`tests/test_pose2d_smoke.py`:

```python
import os
import cv2
import numpy as np
import pytest
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "standing_person.jpg"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture image not downloaded")
def test_pose2d_detects_keypoints_on_standing_person():
    from workout_ai.pose2d import Pose2D

    img = cv2.imread(str(FIXTURE))
    assert img is not None

    detector = Pose2D()
    kps, scores = detector.infer(img)

    assert kps.shape == (17, 2)
    assert scores.shape == (17,)
    # At least 8 keypoints should have a confidence above 0.3
    assert (scores > 0.3).sum() >= 8
```

- [ ] **Step 3: Run test to confirm it fails**

Run: `uv run pytest tests/test_pose2d_smoke.py -v`
Expected: FAIL — `Pose2D` not defined.

- [ ] **Step 4: Implement Pose2D**

Create `src/workout_ai/pose2d.py`:

```python
import os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("RTMLIB_CACHE", str(PROJECT_ROOT / "models" / "rtmlib_cache"))


class Pose2D:
    """Wraps rtmlib's RTMPose-l. Single-person inference: returns the highest-score person."""

    def __init__(self, device: str = "cpu"):
        from rtmlib import Body
        # "performance" mode uses RTMPose-l
        self._body = Body(mode="performance", to_openpose=False, backend="onnxruntime", device=device)

    def infer(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (keypoints (17,2) float32, scores (17,) float32) for the most prominent person."""
        keypoints, scores = self._body(image_bgr)
        if len(keypoints) == 0:
            return (
                np.zeros((17, 2), dtype=np.float32),
                np.zeros((17,), dtype=np.float32),
            )
        # rtmlib returns shape (N_persons, N_kpts, 2) and (N_persons, N_kpts)
        # Pick the person with the highest summed score.
        idx = np.argmax(scores.sum(axis=1))
        return keypoints[idx].astype(np.float32), scores[idx].astype(np.float32)
```

- [ ] **Step 5: Run smoke test**

Run: `uv run pytest tests/test_pose2d_smoke.py -v`
Expected: PASS (≥ 8 keypoints with score > 0.3 on the fixture image).

- [ ] **Step 6: Commit**

```bash
git add src/workout_ai/pose2d.py tests/test_pose2d_smoke.py tests/fixtures/standing_person.jpg
git commit -m "feat(pose2d): RTMPose-l wrapper via rtmlib (ONNX)"
```

---

## Task 8: Webcam capture

**Files:**
- Create: `src/workout_ai/capture.py`
- Create: `tests/test_capture.py`

- [ ] **Step 1: Write failing test using a mock video source**

`tests/test_capture.py`:

```python
import cv2
import numpy as np
import time
import pytest
from workout_ai.capture import WebcamCapture


def test_capture_thread_yields_frames(monkeypatch):
    # Patch cv2.VideoCapture with a fake that returns synthetic frames
    class FakeVC:
        def __init__(self, idx):
            self.opened = True

        def isOpened(self):
            return self.opened

        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self):
            self.opened = False

        def set(self, *args, **kwargs):
            return True

    monkeypatch.setattr(cv2, "VideoCapture", FakeVC)
    cap = WebcamCapture(device=0)
    cap.start()
    try:
        frame = cap.read_latest(timeout=1.0)
        assert frame is not None
        assert frame.shape == (480, 640, 3)
    finally:
        cap.stop()
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `uv run pytest tests/test_capture.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement WebcamCapture**

Create `src/workout_ai/capture.py`:

```python
import threading
import time
from typing import Optional
import cv2
import numpy as np


class WebcamCapture:
    """Background thread that pulls frames from a webcam and keeps the latest one available."""

    def __init__(self, device: int = 0, width: int = 1280, height: int = 720):
        self._device = device
        self._width = width
        self._height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._running = False

    def start(self):
        self._cap = cv2.VideoCapture(self._device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera device {self._device}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest = frame
                self._latest_ts = time.monotonic()

    def read_latest(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None:
                    return self._latest.copy()
            time.sleep(0.005)
        return None

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_capture.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/workout_ai/capture.py tests/test_capture.py
git commit -m "feat(capture): threaded webcam reader"
```

---

## Task 9: 2D skeleton renderer

**Files:**
- Create: `src/workout_ai/render.py`
- Create: `tests/test_render.py`

- [ ] **Step 1: Write failing test**

`tests/test_render.py`:

```python
import numpy as np
from workout_ai.render import Renderer


def test_draw_skeleton_does_not_crash():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    kps = np.array([[100 + i * 5, 100 + i * 5] for i in range(17)], dtype=np.float32)
    scores = np.ones((17,), dtype=np.float32) * 0.9
    r = Renderer(panel_width=320)
    out = r.draw_skeleton(frame, kps, scores)
    assert out.shape == frame.shape


def test_compose_with_panel_returns_wider_image():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    r = Renderer(panel_width=320)
    out = r.compose(frame, score=85, running_avg=72.5, rep_count=4, phase="bottom", thai_text="")
    assert out.shape == (480, 640 + 320, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_render.py -v`
Expected: FAIL — `Renderer` not defined.

- [ ] **Step 3: Implement Renderer (initial 2D + composition; 3D rig panel added in Task 12)**

Create `src/workout_ai/render.py`:

```python
import cv2
import numpy as np

# COCO-17 skeleton connections (pairs of keypoint indices)
SKELETON = [
    (5, 7), (7, 9),         # left arm
    (6, 8), (8, 10),        # right arm
    (5, 6),                 # shoulders
    (5, 11), (6, 12),       # torso
    (11, 12),               # hips
    (11, 13), (13, 15),     # left leg
    (12, 14), (14, 16),     # right leg
]


class Renderer:
    def __init__(self, panel_width: int = 320):
        self.panel_width = panel_width

    def draw_skeleton(self, frame: np.ndarray, kps: np.ndarray, scores: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        out = frame.copy()
        for i, (x, y) in enumerate(kps):
            if scores[i] < threshold:
                continue
            cv2.circle(out, (int(x), int(y)), 4, (0, 255, 0), -1)
        for a, b in SKELETON:
            if scores[a] < threshold or scores[b] < threshold:
                continue
            pa = (int(kps[a, 0]), int(kps[a, 1]))
            pb = (int(kps[b, 0]), int(kps[b, 1]))
            cv2.line(out, pa, pb, (255, 200, 0), 2)
        return out

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
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        canvas = np.zeros((h, w + self.panel_width, 3), dtype=np.uint8)

        # Overlay attention on frame if provided
        display_frame = frame
        if attention is not None:
            display_frame = self._overlay_attention(frame, attention)

        canvas[:, :w] = display_frame

        # HUD on the frame
        cv2.putText(canvas, f"Reps: {rep_count}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Avg: {running_avg:.1f}", (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Phase: {phase}", (12, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
        if score is not None:
            cv2.putText(canvas, f"Last: {score}", (12, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # Right panel
        cv2.rectangle(canvas, (w, 0), (w + self.panel_width, h), (30, 30, 30), -1)
        cv2.putText(canvas, "Coach", (w + 12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if rig_3d_kps is not None:
            self._draw_rig_3d(canvas, rig_3d_kps, top_left=(w + 10, 40), size=(self.panel_width - 20, 220))

        if thai_text:
            self._draw_thai(canvas, thai_text, top_left=(w + 12, 280), max_width=self.panel_width - 24)

        return canvas

    def _overlay_attention(self, frame: np.ndarray, attention: np.ndarray) -> np.ndarray:
        # attention is (H, W) float in [0, 1]
        att = (attention * 255).astype(np.uint8)
        att = cv2.resize(att, (frame.shape[1], frame.shape[0]))
        heat = cv2.applyColorMap(att, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.7, heat, 0.3, 0)

    def _draw_rig_3d(self, canvas: np.ndarray, kps_3d: np.ndarray, top_left: tuple[int, int], size: tuple[int, int]):
        # Orthographic projection: drop Z; auto-scale to fit panel
        x0, y0 = top_left
        w, h = size
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (50, 50, 50), 1)

        xy = kps_3d[:, :2].copy()
        # Centre and scale
        mins = xy.min(axis=0)
        maxs = xy.max(axis=0)
        span = max((maxs - mins).max(), 1e-3)
        xy = (xy - mins) / span  # [0,1]
        xy[:, 0] = x0 + 10 + xy[:, 0] * (w - 20)
        xy[:, 1] = y0 + 10 + xy[:, 1] * (h - 20)

        for i, (x, y) in enumerate(xy):
            cv2.circle(canvas, (int(x), int(y)), 3, (0, 200, 255), -1)
        for a, b in SKELETON:
            if a >= len(xy) or b >= len(xy):
                continue
            pa = (int(xy[a, 0]), int(xy[a, 1]))
            pb = (int(xy[b, 0]), int(xy[b, 1]))
            cv2.line(canvas, pa, pb, (0, 200, 255), 1)

    def _draw_thai(self, canvas: np.ndarray, text: str, top_left: tuple[int, int], max_width: int):
        # OpenCV's default Hershey fonts don't include Thai glyphs. Use PIL.
        from PIL import Image, ImageDraw, ImageFont
        x, y = top_left
        pil_canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_canvas)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Ayuthaya.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        # Naive word wrap
        words = text.split()
        line = ""
        cy = y
        for word in words:
            test = (line + " " + word).strip()
            w, _ = draw.textbbox((0, 0), test, font=font)[2:]
            if w > max_width and line:
                draw.text((x, cy), line, fill=(255, 255, 255), font=font)
                cy += 20
                line = word
            else:
                line = test
        if line:
            draw.text((x, cy), line, fill=(255, 255, 255), font=font)
        # Copy back
        np.copyto(canvas, cv2.cvtColor(np.array(pil_canvas), cv2.COLOR_RGB2BGR))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_render.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/workout_ai/render.py tests/test_render.py
git commit -m "feat(render): OpenCV+PIL overlay with skeleton, HUD, 3D rig panel, Thai text"
```

---

## Task 10: First end-to-end run (camera + 2D + render only)

**Files:**
- Create: `src/workout_ai/app.py`
- Modify: `main.py`

This is the first milestone where the app actually shows you something. No analysis or LLM yet.

- [ ] **Step 1: Create `app.py`**

Create `src/workout_ai/app.py`:

```python
import time
import cv2

from workout_ai.capture import WebcamCapture
from workout_ai.pose2d import Pose2D
from workout_ai.render import Renderer


def run():
    cap = WebcamCapture(device=0, width=1280, height=720)
    pose = Pose2D()
    renderer = Renderer(panel_width=320)
    cap.start()

    rep_count = 0
    running_avg = 0.0
    last_score = None

    try:
        while True:
            frame = cap.read_latest(timeout=2.0)
            if frame is None:
                break

            kps, scores = pose.infer(frame)
            frame = renderer.draw_skeleton(frame, kps, scores)
            display = renderer.compose(
                frame,
                score=last_score,
                running_avg=running_avg,
                rep_count=rep_count,
                phase="standing",
                thai_text="",
            )
            cv2.imshow("Workout AI", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Wire `main.py` to it**

Replace `main.py` with:

```python
from workout_ai.app import run


def main():
    run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Manual smoke run**

Run: `uv run python main.py`
Expected: webcam window opens, green keypoint dots and cyan skeleton lines track your body. Press `q` to quit.

If macOS prompts for camera permission, grant it.

- [ ] **Step 4: Commit**

```bash
git add src/workout_ai/app.py main.py
git commit -m "feat(app): first end-to-end run with 2D pose overlay"
```

---

## Task 11: Wire analysis (FSM + rules + score) into the live loop

**Files:**
- Modify: `src/workout_ai/app.py`

- [ ] **Step 1: Update `app.py` to drive FSM and score reps**

Replace `src/workout_ai/app.py` with:

```python
import time
import cv2
import numpy as np

from workout_ai.capture import WebcamCapture
from workout_ai.pose2d import Pose2D
from workout_ai.render import Renderer
from workout_ai.analysis.phases import SquatFSM
from workout_ai.analysis.rules_squat import score_rep
from workout_ai.analysis.types import PoseFrame, PhaseState


def run():
    cap = WebcamCapture(device=0, width=1280, height=720)
    pose = Pose2D()
    renderer = Renderer(panel_width=320)
    fsm = SquatFSM()

    rep_count = 0
    running_sum = 0
    last_score: int | None = None

    last_bottom_frame: PoseFrame | None = None

    def on_rep_complete(meta: dict):
        nonlocal rep_count, running_sum, last_score
        if last_bottom_frame is None:
            return
        analysis = score_rep(last_bottom_frame, meta["descent_ms"], meta["ascent_ms"], rep_index=rep_count)
        rep_count += 1
        running_sum += analysis.score
        last_score = analysis.score
        print(f"[rep {analysis.rep_index}] score={analysis.score} components={analysis.components}")
        for v in analysis.violations:
            print(f"    violation: {v.name} severity={v.severity:.2f}")

    fsm.on_rep_complete = on_rep_complete
    cap.start()

    try:
        while True:
            frame = cap.read_latest(timeout=2.0)
            if frame is None:
                break

            ts = time.monotonic()
            kps, scores = pose.infer(frame)
            pf = PoseFrame(timestamp=ts, keypoints_2d=kps, scores=scores, frame_shape=frame.shape[:2])
            prev_state = fsm.state
            state = fsm.update(kps, ts)

            # Capture the bottom-of-rep PoseFrame
            if state == PhaseState.BOTTOM:
                last_bottom_frame = pf

            frame = renderer.draw_skeleton(frame, kps, scores)
            avg = (running_sum / rep_count) if rep_count else 0.0
            display = renderer.compose(
                frame,
                score=last_score,
                running_avg=avg,
                rep_count=rep_count,
                phase=state.value,
                thai_text="",
            )
            cv2.imshow("Workout AI", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Manual smoke run, perform a squat**

Run: `uv run python main.py`
Expected: do a slow squat in front of the camera. Phase text changes standing → descent → bottom → ascent → standing. At rep completion, console prints a score 0–100 and component breakdown. HUD `Reps:` increments. Press `q` to quit.

- [ ] **Step 3: Commit**

```bash
git add src/workout_ai/app.py
git commit -m "feat(app): wire FSM, rules, and scoring into live loop"
```

---

## Task 12: MotionBERT 3D lift integration

**Files:**
- Create: `vendor/motionbert/` (cloned from upstream)
- Create: `src/workout_ai/pose3d.py`
- Create: `tests/test_pose3d_smoke.py`
- Modify: `src/workout_ai/app.py`
- Modify: `.gitignore` (already covers `vendor/`)

- [ ] **Step 1: Vendor MotionBERT minimal inference code**

Run:

```bash
git clone --depth 1 https://github.com/Walter0807/MotionBERT.git vendor/motionbert
```

Expected: `vendor/motionbert/lib/model/DSTformer.py` and `vendor/motionbert/configs/pose3d/MB_ft_h36m_global_lite.yaml` are present.

- [ ] **Step 2: Write smoke test**

`tests/test_pose3d_smoke.py`:

```python
import numpy as np
import pytest
from pathlib import Path

MOTIONBERT_PATH = Path(__file__).resolve().parent.parent / "vendor" / "motionbert"
CKPT = Path(__file__).resolve().parent.parent / "models" / "motionbert" / "MB_ft_h36m_global_lite.bin"


@pytest.mark.skipif(not (MOTIONBERT_PATH.exists() and CKPT.exists()),
                    reason="MotionBERT vendor code or checkpoint missing")
def test_pose3d_returns_17x3():
    from workout_ai.pose3d import Pose3D

    lifter = Pose3D()
    # 27 frames of dummy 2D keypoints (H36M order, 17 joints, (x, y, score))
    window = np.zeros((27, 17, 3), dtype=np.float32)
    window[..., 2] = 1.0  # full confidence
    out = lifter.infer(window)
    assert out.shape == (17, 3)
```

- [ ] **Step 3: Run test (expect fail)**

Run: `uv run pytest tests/test_pose3d_smoke.py -v`
Expected: FAIL — `Pose3D` not defined.

- [ ] **Step 4: Implement Pose3D**

Create `src/workout_ai/pose3d.py`:

```python
import sys
from pathlib import Path
from collections import deque
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MOTIONBERT_DIR = PROJECT_ROOT / "vendor" / "motionbert"
CKPT_PATH = PROJECT_ROOT / "models" / "motionbert" / "MB_ft_h36m_global_lite.bin"
CONFIG_PATH = MOTIONBERT_DIR / "configs" / "pose3d" / "MB_ft_h36m_global_lite.yaml"

# COCO-17 -> Human3.6M-17 reordering. MotionBERT was trained on H36M ordering.
# H36M order: pelvis, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle, spine, thorax, neck/nose, head, l_shoulder, l_elbow, l_wrist, r_shoulder, r_elbow, r_wrist
COCO_TO_H36M = {
    0: -1,    # pelvis: synthesized as (L_hip + R_hip) / 2
    1: 12,    # R hip
    2: 14,    # R knee
    3: 16,    # R ankle
    4: 11,    # L hip
    5: 13,    # L knee
    6: 15,    # L ankle
    7: -2,    # spine: midpoint of pelvis and thorax
    8: -3,    # thorax: midpoint of shoulders
    9: 0,     # neck/nose -> nose
    10: 0,    # head -> nose (coarse)
    11: 5,    # L shoulder
    12: 7,    # L elbow
    13: 9,    # L wrist
    14: 6,    # R shoulder
    15: 8,    # R elbow
    16: 10,   # R wrist
}


def coco17_to_h36m17(kps_coco: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Convert (17, 2) COCO + (17,) scores -> (17, 3) H36M with (x, y, score)."""
    out = np.zeros((17, 3), dtype=np.float32)
    l_hip, r_hip = kps_coco[11], kps_coco[12]
    pelvis = (l_hip + r_hip) / 2.0
    l_sh, r_sh = kps_coco[5], kps_coco[6]
    thorax = (l_sh + r_sh) / 2.0
    spine = (pelvis + thorax) / 2.0

    for h_idx, c_idx in COCO_TO_H36M.items():
        if c_idx == -1:
            out[h_idx, :2] = pelvis
            out[h_idx, 2] = min(scores[11], scores[12])
        elif c_idx == -2:
            out[h_idx, :2] = spine
            out[h_idx, 2] = min(scores[5], scores[6], scores[11], scores[12])
        elif c_idx == -3:
            out[h_idx, :2] = thorax
            out[h_idx, 2] = min(scores[5], scores[6])
        else:
            out[h_idx, :2] = kps_coco[c_idx]
            out[h_idx, 2] = scores[c_idx]
    return out


def _normalize_2d(kps: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
    """MotionBERT expects keypoints normalised to [-1, 1] using image width/height."""
    out = kps.copy()
    out[..., 0] = out[..., 0] / frame_w * 2 - 1
    out[..., 1] = out[..., 1] / frame_w * 2 - frame_h / frame_w  # MotionBERT convention
    return out


class Pose3D:
    """MotionBERT-Lite wrapper for 2D->3D lifting on a sliding 27-frame window."""

    def __init__(self, window_size: int = 27, device: str | None = None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        sys.path.insert(0, str(MOTIONBERT_DIR))
        from lib.utils.tools import get_config
        from lib.model.loss import loss_pose3d
        from lib.model.DSTformer import DSTformer

        cfg = get_config(str(CONFIG_PATH))
        self._model = DSTformer(
            dim_in=3, dim_out=3, dim_feat=cfg.dim_feat, dim_rep=cfg.dim_rep,
            depth=cfg.depth, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
            num_joints=cfg.num_joints, maxlen=cfg.maxlen,
        )
        ckpt = torch.load(CKPT_PATH, map_location=self.device, weights_only=False)
        state = ckpt.get("model_pos", ckpt)
        # Strip "module." prefixes if present
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self._model.load_state_dict(state, strict=False)
        self._model.eval().to(self.device)
        self.window_size = window_size

    @torch.no_grad()
    def infer(self, window: np.ndarray, frame_h: int = 720, frame_w: int = 1280) -> np.ndarray:
        """window: (T, 17, 3) H36M-ordered (x, y, score). Returns (17, 3) for the centre frame."""
        norm = _normalize_2d(window, frame_h, frame_w)
        x = torch.from_numpy(norm).float().unsqueeze(0).to(self.device)  # (1, T, 17, 3)
        out = self._model(x)  # (1, T, 17, 3)
        out = out.squeeze(0).cpu().numpy()
        centre = out[self.window_size // 2]
        return centre


class Pose3DBuffer:
    """Rolling 2D buffer that yields a 3D pose on demand."""

    def __init__(self, lifter: Pose3D):
        self._buf: deque[np.ndarray] = deque(maxlen=lifter.window_size)
        self._lifter = lifter

    def push(self, h36m_kps: np.ndarray):
        self._buf.append(h36m_kps)

    def ready(self) -> bool:
        return len(self._buf) == self._buf.maxlen

    def lift(self, frame_h: int, frame_w: int) -> np.ndarray:
        if not self.ready():
            raise RuntimeError("buffer not full")
        window = np.stack(self._buf, axis=0)
        return self._lifter.infer(window, frame_h, frame_w)
```

- [ ] **Step 5: Run smoke test**

Run: `uv run pytest tests/test_pose3d_smoke.py -v`
Expected: PASS (returns `(17, 3)` array).

If `weights_only=False` triggers a warning or the checkpoint format differs, inspect with `python -c "import torch; print(torch.load('models/motionbert/MB_ft_h36m_global_lite.bin', map_location='cpu', weights_only=False).keys())"` and adjust the key lookup.

- [ ] **Step 6: Wire Pose3D into the app**

Modify `src/workout_ai/app.py`: replace the imports and the loop body to include 3D lift every 5 frames. Replace the file with:

```python
import time
import cv2
import numpy as np

from workout_ai.capture import WebcamCapture
from workout_ai.pose2d import Pose2D
from workout_ai.pose3d import Pose3D, Pose3DBuffer, coco17_to_h36m17
from workout_ai.render import Renderer
from workout_ai.analysis.phases import SquatFSM
from workout_ai.analysis.rules_squat import score_rep
from workout_ai.analysis.types import PoseFrame, PhaseState


def run():
    cap = WebcamCapture(device=0, width=1280, height=720)
    pose = Pose2D()
    lifter = Pose3D()
    buf3d = Pose3DBuffer(lifter)
    renderer = Renderer(panel_width=320)
    fsm = SquatFSM()

    rep_count = 0
    running_sum = 0
    last_score: int | None = None
    last_bottom_frame: PoseFrame | None = None
    last_rig_3d: np.ndarray | None = None
    frame_idx = 0

    def on_rep_complete(meta: dict):
        nonlocal rep_count, running_sum, last_score
        if last_bottom_frame is None:
            return
        analysis = score_rep(last_bottom_frame, meta["descent_ms"], meta["ascent_ms"], rep_index=rep_count)
        rep_count += 1
        running_sum += analysis.score
        last_score = analysis.score
        print(f"[rep {analysis.rep_index}] score={analysis.score} components={analysis.components}")
        for v in analysis.violations:
            print(f"    violation: {v.name} severity={v.severity:.2f}")

    fsm.on_rep_complete = on_rep_complete
    cap.start()

    try:
        while True:
            frame = cap.read_latest(timeout=2.0)
            if frame is None:
                break

            ts = time.monotonic()
            kps, scores = pose.infer(frame)
            pf = PoseFrame(timestamp=ts, keypoints_2d=kps, scores=scores, frame_shape=frame.shape[:2])

            # Feed 3D buffer
            h36m = coco17_to_h36m17(kps, scores)
            buf3d.push(h36m)
            if frame_idx % 5 == 0 and buf3d.ready():
                try:
                    last_rig_3d = buf3d.lift(frame.shape[0], frame.shape[1])
                    pf.keypoints_3d = last_rig_3d
                except Exception as e:
                    print(f"3D lift error: {e}")

            state = fsm.update(kps, ts)
            if state == PhaseState.BOTTOM:
                last_bottom_frame = pf

            frame = renderer.draw_skeleton(frame, kps, scores)
            avg = (running_sum / rep_count) if rep_count else 0.0
            display = renderer.compose(
                frame,
                score=last_score,
                running_avg=avg,
                rep_count=rep_count,
                phase=state.value,
                thai_text="",
                rig_3d_kps=last_rig_3d,
            )
            cv2.imshow("Workout AI", display)
            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
```

- [ ] **Step 7: Manual smoke run**

Run: `uv run python main.py`
Expected: 3D rig appears in the right panel after ~1 s of warmup and updates as you move.

- [ ] **Step 8: Commit**

```bash
git add vendor/motionbert src/workout_ai/pose3d.py src/workout_ai/app.py tests/test_pose3d_smoke.py
git commit -m "feat(pose3d): MotionBERT-Lite 2D->3D lifting on rolling window"
```

(Note: depending on `vendor/motionbert` being gitignored by Task 1, the `vendor/motionbert` path here will be skipped — that is intentional.)

---

## Task 13: Model-attention overlay (GradCAM stand-in)

**Files:**
- Create: `src/workout_ai/analysis/attention.py`
- Create: `tests/test_attention.py`
- Modify: `src/workout_ai/pose2d.py` (expose intermediate heatmaps)
- Modify: `src/workout_ai/app.py`

True gradient-based GradCAM on RTMPose requires running the PyTorch checkpoint with hooks, which is expensive for every frame. v1 uses a simpler proxy: aggregate RTMPose's joint heatmaps to produce a "where is the model looking" map. This is implementation-cheap and visually meaningful. If a true GradCAM is needed later, swap `attention.compute()` for a hook-based version.

- [ ] **Step 1: Write failing test**

`tests/test_attention.py`:

```python
import numpy as np
from workout_ai.analysis.attention import aggregate_heatmaps


def test_aggregate_heatmaps_normalised_to_0_1():
    hm = np.random.rand(17, 64, 48).astype(np.float32) * 5.0
    out = aggregate_heatmaps(hm)
    assert out.shape == (64, 48)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_aggregate_focuses_on_joint_peaks():
    hm = np.zeros((17, 64, 48), dtype=np.float32)
    # Single hot pixel on the left knee (idx 13) heatmap
    hm[13, 30, 24] = 1.0
    out = aggregate_heatmaps(hm)
    assert out[30, 24] == out.max()
```

- [ ] **Step 2: Run test, expect fail**

Run: `uv run pytest tests/test_attention.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement attention**

Create `src/workout_ai/analysis/attention.py`:

```python
import numpy as np


LOWER_BODY_JOINTS = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles


def aggregate_heatmaps(heatmaps: np.ndarray, joints: list[int] | None = None) -> np.ndarray:
    """heatmaps: (N_kpts, H, W). Returns a single (H, W) attention map in [0, 1].

    Squat-specific: weight lower-body joints heavier.
    """
    if joints is None:
        joints = LOWER_BODY_JOINTS
    weights = np.ones(heatmaps.shape[0], dtype=np.float32)
    for j in joints:
        weights[j] = 3.0
    agg = (heatmaps * weights[:, None, None]).sum(axis=0)
    agg = agg - agg.min()
    rng = agg.max() - agg.min()
    if rng < 1e-9:
        return np.zeros_like(agg)
    return (agg / rng).astype(np.float32)
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_attention.py -v`
Expected: PASS.

- [ ] **Step 5: Expose heatmaps from Pose2D**

Modify `src/workout_ai/pose2d.py`: rtmlib's `Body` returns only keypoints by default. We need to call the underlying `RTMPose` predictor directly to get heatmaps.

Replace the file with:

```python
import os
from pathlib import Path
from typing import Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("RTMLIB_CACHE", str(PROJECT_ROOT / "models" / "rtmlib_cache"))


class Pose2D:
    """Wraps rtmlib's RTMPose-l. Single-person inference: returns the highest-score person.
    Optionally returns simcc-decoded heatmaps via `infer_with_heatmaps`."""

    def __init__(self, device: str = "cpu"):
        from rtmlib import Body
        self._body = Body(mode="performance", to_openpose=False, backend="onnxruntime", device=device)
        # The internal pose model has access to simcc outputs.
        self._pose = self._body.pose_estimator

    def infer(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keypoints, scores = self._body(image_bgr)
        if len(keypoints) == 0:
            return (
                np.zeros((17, 2), dtype=np.float32),
                np.zeros((17,), dtype=np.float32),
            )
        idx = int(np.argmax(scores.sum(axis=1)))
        return keypoints[idx].astype(np.float32), scores[idx].astype(np.float32)

    def infer_with_heatmaps(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Returns (keypoints, scores, heatmaps) where heatmaps is (17, H, W) reconstructed from simcc.
        If rtmlib version does not expose simcc outputs, heatmaps will be None."""
        kps, scores = self.infer(image_bgr)
        heatmaps = None
        try:
            simcc_x, simcc_y = getattr(self._pose, "_last_simcc", (None, None))
            if simcc_x is not None and simcc_y is not None:
                # simcc_x: (N, K, W*2), simcc_y: (N, K, H*2). Outer product per joint -> heatmap.
                sx = simcc_x[0]
                sy = simcc_y[0]
                hms = []
                for k in range(sx.shape[0]):
                    hm = np.outer(sy[k], sx[k])
                    hms.append(hm)
                heatmaps = np.stack(hms, axis=0).astype(np.float32)
        except Exception:
            heatmaps = None
        return kps, scores, heatmaps
```

If rtmlib's predictor does not stash `_last_simcc`, monkey-patch its `forward` after instantiation. Concretely, add this after `self._pose = ...`:

```python
        # Monkey-patch to capture simcc tensors for attention map
        _orig_forward = self._pose.forward

        def _capturing_forward(image):
            simcc_x, simcc_y = _orig_forward(image)
            self._pose._last_simcc = (simcc_x, simcc_y)
            return simcc_x, simcc_y

        self._pose.forward = _capturing_forward
```

- [ ] **Step 6: Wire attention into app**

Modify `src/workout_ai/app.py` to call `infer_with_heatmaps` and pass attention to renderer. Add a hotkey `a` to toggle the overlay.

Change in `app.py`:

Replace:

```python
            kps, scores = pose.infer(frame)
```

with:

```python
            kps, scores, hms = pose.infer_with_heatmaps(frame)
```

Add near the top of `run()`:

```python
    show_attention = False
```

In the `compose(...)` call, add:

```python
                attention=aggregate_heatmaps(hms) if (show_attention and hms is not None) else None,
```

(Import `aggregate_heatmaps` from `workout_ai.analysis.attention`.)

In the key-handling block:

```python
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                show_attention = not show_attention
```

- [ ] **Step 7: Manual smoke run**

Run: `uv run python main.py`
Expected: press `a` — a jet-colored attention overlay appears on top of the frame, concentrated around joints. Press `a` again to toggle off.

If heatmaps remain `None` (rtmlib version doesn't expose simcc), this is acceptable — the overlay simply doesn't appear and we log a one-time warning. Continue with the rest of the plan.

- [ ] **Step 8: Commit**

```bash
git add src/workout_ai/analysis/attention.py src/workout_ai/pose2d.py src/workout_ai/app.py tests/test_attention.py
git commit -m "feat(attention): RTMPose joint-heatmap aggregation overlay (toggle: a)"
```

---

## Task 14: Qwen3.5-4B LLM wrapper

**Files:**
- Create: `src/workout_ai/feedback/llm.py`
- Create: `src/workout_ai/feedback/prompt_th.py`
- Create: `tests/test_llm_smoke.py`

- [ ] **Step 1: Write the Thai prompt template**

Create `src/workout_ai/feedback/prompt_th.py`:

```python
from workout_ai.analysis.types import RepAnalysis

SYSTEM_TH = (
    "คุณเป็นโค้ชฟิตเนสที่ให้คำแนะนำเกี่ยวกับท่าสควอตอย่างกระชับและสุภาพ "
    "ตอบเป็นภาษาไทย 2-3 ประโยค ระบุสิ่งที่ทำถูกและสิ่งที่ควรแก้ไข"
)


def build_user_prompt(rep: RepAnalysis) -> str:
    lines = [
        f"ผลสควอตรอบที่ {rep.rep_index + 1}: คะแนนรวม {rep.score}/100",
        f"  ความลึก: {rep.components['depth']}/30",
        f"  หัวเข่า: {rep.components['valgus']}/25",
        f"  ลำตัว: {rep.components['torso']}/20",
        f"  สมมาตร: {rep.components['symmetry']}/15",
        f"  จังหวะ: {rep.components['tempo']}/10",
        f"เวลาลง: {rep.descent_ms} ms | เวลาขึ้น: {rep.ascent_ms} ms",
    ]
    if rep.violations:
        lines.append("ข้อสังเกต:")
        for v in rep.violations:
            lines.append(f"  - {v.detail_th} (ระดับ {v.severity:.2f})")
    else:
        lines.append("ไม่พบข้อสังเกตที่ต้องแก้ไข")
    lines.append("ช่วยสรุปสั้นๆ ว่าทำดีตรงไหน ควรปรับตรงไหน")
    return "\n".join(lines)
```

- [ ] **Step 2: Write smoke test**

`tests/test_llm_smoke.py`:

```python
import os
from pathlib import Path
import numpy as np
import pytest

QWEN_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen3_5_4b_mxfp4"


@pytest.mark.skipif(not QWEN_DIR.exists(), reason="Qwen model not downloaded")
def test_llm_generates_thai_text():
    from workout_ai.feedback.llm import ThaiCoachLLM
    from workout_ai.analysis.types import RepAnalysis

    llm = ThaiCoachLLM()
    rep = RepAnalysis(
        rep_index=0,
        score=78,
        components={"depth": 30, "valgus": 18, "torso": 20, "symmetry": 12, "tempo": 8},
        violations=[],
        descent_ms=1200,
        ascent_ms=900,
    )
    text = llm.generate(rep, max_tokens=120)
    assert isinstance(text, str)
    assert len(text) > 5
    # Contains at least one Thai codepoint
    assert any("฀" <= c <= "๿" for c in text)
```

- [ ] **Step 3: Run test, expect fail**

Run: `uv run pytest tests/test_llm_smoke.py -v`
Expected: FAIL (`ThaiCoachLLM` not defined) — or SKIP if Qwen weights aren't downloaded yet.

- [ ] **Step 4: Implement ThaiCoachLLM**

Create `src/workout_ai/feedback/llm.py`:

```python
from pathlib import Path
from typing import Optional
import numpy as np

from workout_ai.feedback.prompt_th import SYSTEM_TH, build_user_prompt
from workout_ai.analysis.types import RepAnalysis

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "qwen3_5_4b_mxfp4"


class ThaiCoachLLM:
    """Wraps Qwen3.5-4B (vision-language) via mlx-vlm. v1 sends text-only prompts."""

    def __init__(self, model_dir: Path | str | None = None):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        model_path = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self._model, self._processor = load(str(model_path))
        self._config = load_config(str(model_path))

    def generate(self, rep: RepAnalysis, max_tokens: int = 160, frame_bgr: Optional[np.ndarray] = None) -> str:
        from mlx_vlm import generate as mlx_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        user = build_user_prompt(rep)
        # v1: text only. To add the bottom-of-rep image later, pass image=frame_bgr.
        messages = [
            {"role": "system", "content": SYSTEM_TH},
            {"role": "user", "content": user},
        ]
        prompt = apply_chat_template(self._processor, self._config, messages, num_images=0)
        result = mlx_generate(
            self._model,
            self._processor,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        # mlx-vlm.generate returns a GenerationResult; .text holds the string
        return getattr(result, "text", str(result)).strip()

    def warmup(self):
        """First call is slow due to compilation. Run once at app start."""
        dummy = RepAnalysis(
            rep_index=-1,
            score=50,
            components={"depth": 10, "valgus": 10, "torso": 10, "symmetry": 10, "tempo": 10},
            violations=[],
            descent_ms=0,
            ascent_ms=0,
        )
        _ = self.generate(dummy, max_tokens=16)
```

- [ ] **Step 5: Run smoke test**

Run: `uv run pytest tests/test_llm_smoke.py -v`
Expected: PASS (≥ 5 chars Thai text, may take 5–15 s on first run).

If `mlx_vlm.generate` signature differs (mlx-vlm evolves quickly), inspect with `uv run python -c "import mlx_vlm; help(mlx_vlm.generate)"` and adjust the kwargs.

- [ ] **Step 6: Commit**

```bash
git add src/workout_ai/feedback/llm.py src/workout_ai/feedback/prompt_th.py tests/test_llm_smoke.py
git commit -m "feat(feedback): Qwen3.5-4B mlx-vlm wrapper with Thai prompt template"
```

---

## Task 15: Async LLM worker

**Files:**
- Create: `src/workout_ai/feedback/worker.py`
- Create: `tests/test_worker.py`

- [ ] **Step 1: Write failing test using a fake LLM**

`tests/test_worker.py`:

```python
import time
from workout_ai.feedback.worker import LLMWorker
from workout_ai.analysis.types import RepAnalysis


class FakeLLM:
    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.calls = 0

    def generate(self, rep, max_tokens: int = 120, frame_bgr=None) -> str:
        time.sleep(self.delay)
        self.calls += 1
        return f"feedback for rep {rep.rep_index}"


def make_rep(idx: int) -> RepAnalysis:
    return RepAnalysis(
        rep_index=idx,
        score=80,
        components={"depth": 30, "valgus": 20, "torso": 15, "symmetry": 10, "tempo": 5},
        violations=[],
        descent_ms=1000,
        ascent_ms=1000,
    )


def test_worker_returns_feedback_async():
    llm = FakeLLM(delay=0.05)
    w = LLMWorker(llm)
    w.start()
    try:
        w.submit(make_rep(0))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            text = w.latest()
            if text:
                break
            time.sleep(0.01)
        assert text == "feedback for rep 0"
    finally:
        w.stop()


def test_worker_drops_stale_submissions():
    llm = FakeLLM(delay=0.2)
    w = LLMWorker(llm)
    w.start()
    try:
        for i in range(5):
            w.submit(make_rep(i))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if w.latest() == "feedback for rep 4":
                break
            time.sleep(0.05)
        # Last submission must win; earlier ones are dropped, so calls < 5
        assert llm.calls <= 2
        assert w.latest() == "feedback for rep 4"
    finally:
        w.stop()
```

- [ ] **Step 2: Run test, expect fail**

Run: `uv run pytest tests/test_worker.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement worker**

Create `src/workout_ai/feedback/worker.py`:

```python
import threading
from typing import Optional
from workout_ai.analysis.types import RepAnalysis


class LLMWorker:
    """Background thread that processes the most recent RepAnalysis at a time, dropping older ones."""

    def __init__(self, llm):
        self._llm = llm
        self._lock = threading.Lock()
        self._pending: Optional[RepAnalysis] = None
        self._latest_text: Optional[str] = None
        self._cv = threading.Condition(self._lock)
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, rep: RepAnalysis):
        with self._cv:
            self._pending = rep  # newer overwrites older
            self._cv.notify()

    def latest(self) -> Optional[str]:
        with self._lock:
            return self._latest_text

    def _loop(self):
        while self._running:
            with self._cv:
                while self._running and self._pending is None:
                    self._cv.wait(timeout=0.1)
                if not self._running:
                    return
                rep = self._pending
                self._pending = None
            try:
                text = self._llm.generate(rep)
            except Exception as e:
                text = f"[LLM error: {e}]"
            with self._lock:
                self._latest_text = text

    def stop(self):
        with self._cv:
            self._running = False
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_worker.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/workout_ai/feedback/worker.py tests/test_worker.py
git commit -m "feat(feedback): async worker that keeps only the latest rep"
```

---

## Task 16: Wire LLM into the live loop

**Files:**
- Modify: `src/workout_ai/app.py`

- [ ] **Step 1: Update `app.py` to spin up the LLM worker and submit each rep**

Replace `src/workout_ai/app.py` with:

```python
import time
import cv2
import numpy as np

from workout_ai.capture import WebcamCapture
from workout_ai.pose2d import Pose2D
from workout_ai.pose3d import Pose3D, Pose3DBuffer, coco17_to_h36m17
from workout_ai.render import Renderer
from workout_ai.analysis.phases import SquatFSM
from workout_ai.analysis.rules_squat import score_rep
from workout_ai.analysis.attention import aggregate_heatmaps
from workout_ai.analysis.types import PoseFrame, PhaseState
from workout_ai.feedback.llm import ThaiCoachLLM
from workout_ai.feedback.worker import LLMWorker


def run():
    cap = WebcamCapture(device=0, width=1280, height=720)
    pose = Pose2D()
    lifter = Pose3D()
    buf3d = Pose3DBuffer(lifter)
    renderer = Renderer(panel_width=320)
    fsm = SquatFSM()

    print("Loading Qwen3.5-4B (this takes ~10 seconds the first time)...")
    llm = ThaiCoachLLM()
    print("Warming up LLM...")
    llm.warmup()
    worker = LLMWorker(llm)
    worker.start()
    print("Ready.")

    rep_count = 0
    running_sum = 0
    last_score: int | None = None
    last_bottom_frame: PoseFrame | None = None
    last_rig_3d: np.ndarray | None = None
    frame_idx = 0
    show_attention = False

    def on_rep_complete(meta: dict):
        nonlocal rep_count, running_sum, last_score
        if last_bottom_frame is None:
            return
        analysis = score_rep(last_bottom_frame, meta["descent_ms"], meta["ascent_ms"], rep_index=rep_count)
        rep_count += 1
        running_sum += analysis.score
        last_score = analysis.score
        worker.submit(analysis)
        print(f"[rep {analysis.rep_index}] score={analysis.score} components={analysis.components}")

    fsm.on_rep_complete = on_rep_complete
    cap.start()

    try:
        while True:
            frame = cap.read_latest(timeout=2.0)
            if frame is None:
                break

            ts = time.monotonic()
            kps, scores, hms = pose.infer_with_heatmaps(frame)
            pf = PoseFrame(timestamp=ts, keypoints_2d=kps, scores=scores, frame_shape=frame.shape[:2])

            h36m = coco17_to_h36m17(kps, scores)
            buf3d.push(h36m)
            if frame_idx % 5 == 0 and buf3d.ready():
                try:
                    last_rig_3d = buf3d.lift(frame.shape[0], frame.shape[1])
                    pf.keypoints_3d = last_rig_3d
                except Exception as e:
                    print(f"3D lift error: {e}")

            state = fsm.update(kps, ts)
            if state == PhaseState.BOTTOM:
                last_bottom_frame = pf

            frame_drawn = renderer.draw_skeleton(frame, kps, scores)
            avg = (running_sum / rep_count) if rep_count else 0.0
            thai_text = worker.latest() or ""
            display = renderer.compose(
                frame_drawn,
                score=last_score,
                running_avg=avg,
                rep_count=rep_count,
                phase=state.value,
                thai_text=thai_text,
                rig_3d_kps=last_rig_3d,
                attention=aggregate_heatmaps(hms) if (show_attention and hms is not None) else None,
            )
            cv2.imshow("Workout AI", display)
            frame_idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                show_attention = not show_attention
    finally:
        worker.stop()
        cap.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Manual end-to-end test**

Run: `uv run python main.py`
Expected:
- Window opens after a ~15 s startup (LLM warmup).
- Skeleton tracks user.
- Phase HUD updates.
- After completing a full squat (down to bottom and back), within 1–3 s Thai feedback appears in the right panel.
- Pressing `a` toggles attention overlay.
- Pressing `q` quits cleanly.

- [ ] **Step 3: Commit**

```bash
git add src/workout_ai/app.py
git commit -m "feat(app): end-to-end loop with async Thai LLM feedback"
```

---

## Task 17: README + acceptance checklist

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write README**

Replace `README.md`:

```markdown
# Workout AI — Real-Time Squat Form Coach (Thai)

Real-time webcam-based squat form coach for macOS Apple Silicon.

## Setup

```bash
uv sync --all-extras
uv run python scripts/download_models.py     # ~6 GB, one-time
```

## Run

```bash
uv run python main.py
```

Keys:

- `q` — quit
- `a` — toggle model-attention overlay

## Stack

| Layer | Model | Framework |
|---|---|---|
| 2D pose | RTMPose-l | rtmlib (ONNX) |
| 3D lift | MotionBERT-Lite | PyTorch on MPS |
| Form analysis | Joint angles + phase FSM + attention | pure Python |
| Thai feedback | Qwen3.5-4B (mxfp4) | mlx-vlm |

## Acceptance criteria

- [ ] Skeleton overlay at ≥ 25 FPS on M2 or better.
- [ ] 3D rig updates at ≥ 5 Hz.
- [ ] Squat reps detected within ±1 over a 10-rep set.
- [ ] 0–100 form score shown per rep + running average.
- [ ] 2–3 sentence Thai feedback within 3 s of rep completion.
- [ ] Models live in `./models/` and load from disk on subsequent runs.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with setup, run, acceptance criteria"
```

---

## Self-review notes

- **Spec coverage:** All seven elements of the design (RTMPose-l, MotionBERT, angle rules, phase FSM, model attention, score 0–100, Thai LLM) have implementation tasks. Acceptance criteria are visible in README + spec.
- **Placeholder scan:** No "TBD", "implement later", or "similar to" patterns. Every code step has full source.
- **Type consistency:** `PoseFrame`, `PhaseState`, `RepAnalysis`, `RuleViolation` defined in Task 2 and used consistently in Tasks 4, 5, 11, 14, 15, 16.
- **Risks acknowledged:** Each ML wrapper has a one-line fallback (manual download for MotionBERT, 8-bit fallback for Qwen, attention overlay gracefully degrades if simcc not accessible).
- **The "GradCAM" claim:** the spec says three techniques including GradCAM. Implementation uses joint-heatmap aggregation, not true gradient CAM, and the design doc and code comments are explicit about that. True GradCAM is documented as a v1.1 swap.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-18-pose-form-coach.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration on a clean context per task.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

Which approach?

# Neck-Stretch Real-Time Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-hold `app.py` with a guided 2-minute neck-stretch demo — clickable Start, outline-based positioning, four 25s alternating-side holds with per-set countdowns, and spoken Thai coaching (Gemini TTS + macOS `say` fallback).

**Architecture:** A pure `RoutineFSM` (`src/routine.py`) drives the flow (SETUP → POSITIONING → COUNTDOWN → HOLD → TRANSITION → SUMMARY → DONE) advanced by `(now, pose_ready, in_target)` — no OpenCV/audio inside it. `app.py` owns the camera loop, mouse callback, per-frame pose inference, scoring accumulation, screen rendering, and audio. The 2D pose/measure/score code is reused unchanged; the 3D rig is dropped from this UI. Audio runs in a `TTSWorker` background thread.

**Tech Stack:** Python 3.12 (`uv`), OpenCV, rtmlib/ONNX (existing `Pose2D`), `mlx-vlm` (existing `ThaiCoachLLM`), `google-genai` (Gemini TTS — already a dependency), macOS `say`/`afplay` (system, no new deps), pytest.

Spec: `docs/superpowers/specs/2026-05-23-neck-stretch-realtime-demo-design.md`.

---

## File Structure

| File | Responsibility | Action |
| --- | --- | --- |
| `src/routine.py` | Pure routine state machine + config + events. No I/O. | Create |
| `src/feedback/tts.py` | `GeminiTTS` (synth + `say` fallback) and `TTSWorker` (cue queue + drop-stale feedback). | Create |
| `src/screens.py` | OpenCV draw functions for each screen + Start-button hit-test. | Create |
| `src/render.py` | Add module-level `put_thai_text()` helper (reused by screens). | Modify |
| `src/exercises/neck_stretch.py` | Add `NeckStretchRight`; DRY the shared `measure`; set holds to 25s. | Modify |
| `src/exercises/__init__.py` | Register `NeckStretchRight`. | Modify |
| `src/selector.py` | Add `choose_routine()` returning a routine key. | Modify |
| `src/app.py` | Rewritten orchestration + main loop. | Replace |
| `tests/office_syndrome/test_routine.py` | RoutineFSM transition tests. | Create |
| `tests/office_syndrome/test_tts.py` | WAV wrap + fallback + worker-state tests (mocked). | Create |
| `tests/office_syndrome/test_neck_stretch.py` | Add `NeckStretchRight` tests. | Modify |
| `tests/pipeline/test_render.py` | Add `put_thai_text` smoke test (create file if absent). | Modify/Create |
| `tests/office_syndrome/test_screens.py` | Button hit-test tests. | Create |
| `CLAUDE.md`, `README.md` | Document the new demo flow. | Modify |

**Conventions reused:** tests import from `src/` directly (pytest `pythonpath=["src"]`), so `from routine import ...`, `from feedback.tts import ...`. COCO indices live in `analysis/angles.py` (`NOSE=0, L_EAR=3, R_EAR=4, L_SHOULDER=5, R_SHOULDER=6, L_HIP=11, R_HIP=12`). Always run via `uv run`.

---

## Task 1: Add `NeckStretchRight` + DRY the shared measure

**Files:**
- Modify: `src/exercises/neck_stretch.py`
- Modify: `src/exercises/__init__.py`
- Test: `tests/office_syndrome/test_neck_stretch.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/office_syndrome/test_neck_stretch.py`:

```python
from exercises.neck_stretch import NeckStretchRight


def test_right_metadata():
    ex = NeckStretchRight()
    assert ex.name == "neck_stretch_right"
    assert ex.target.side == "right"
    assert ex.target.hold_seconds == 25.0
    assert ex.target.joints[0].name == "head_lateral_tilt"
    assert ex.target.joints[0].target_deg == 35.0


def test_right_valid_views_exclude_side():
    ex = NeckStretchRight()
    assert CameraView.SIDE not in ex.target.valid_views
    assert CameraView.FRONT in ex.target.valid_views


def test_right_measure_sign_for_right_tilt_is_positive():
    """Nose shifted to +x (toward R_shoulder) must yield a POSITIVE tilt to
    match the NeckStretchRight target of +35°."""
    ex = NeckStretchRight()
    kps, scores = _kps2d_with_head_shifted(+30.0)
    result = ex.measure(_make_frame(kps, scores))
    assert result["head_lateral_tilt"] > 0


def test_left_hold_is_now_25s():
    assert NeckStretchLeft().target.hold_seconds == 25.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/office_syndrome/test_neck_stretch.py -q`
Expected: FAIL — `ImportError: cannot import name 'NeckStretchRight'` (and `test_left_hold_is_now_25s` fails, hold is 20.0).

- [ ] **Step 3: Implement** — replace the body of `src/exercises/neck_stretch.py` from the `_NECK_TILT_TARGET_LEFT_DEG` constant onward. Keep the existing imports and the `_LIVE_TH` / `_SUMMARY_TH` templates above unchanged. Add a shared measure helper and both classes:

```python
_NECK_TILT_TARGET_LEFT_DEG = -35.0
_NECK_TILT_TARGET_RIGHT_DEG = 35.0
_NECK_TILT_TOLERANCE_DEG = 10.0
_HOLD_SECONDS = 25.0


def _measure_head_lateral_tilt(
    frame: PoseFrame, baseline: Optional[BaselinePose]
) -> dict[str, float]:
    """Shared measure for both neck-stretch sides: 2D head-lateral-tilt,
    baseline-subtracted when a BaselinePose is supplied (NaN-preserving)."""
    tilt = head_lateral_tilt_2d(frame.keypoints_2d, frame.scores)
    if (
        baseline is not None
        and not math.isnan(tilt)
        and not math.isnan(baseline.head_lateral_tilt_deg)
    ):
        tilt = tilt - baseline.head_lateral_tilt_deg
    return {"head_lateral_tilt": tilt}


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
        hold_seconds=_HOLD_SECONDS,
        side="left",
        valid_views=(CameraView.FRONT, CameraView.THREE_QUARTER),
    )
    prompt = PromptTemplate(live=_LIVE_TH, summary=_SUMMARY_TH)

    def measure(
        self, frame: PoseFrame, baseline: Optional[BaselinePose] = None
    ) -> dict[str, float]:
        return _measure_head_lateral_tilt(frame, baseline)


class NeckStretchRight:
    name = "neck_stretch_right"
    display_th = "ยืดคอด้านขวา"
    target = TargetPose(
        joints=(
            JointTarget(
                name="head_lateral_tilt",
                target_deg=_NECK_TILT_TARGET_RIGHT_DEG,
                tolerance_deg=_NECK_TILT_TOLERANCE_DEG,
                detail_th="เอียงศีรษะไปทางขวามากขึ้นอีกนิด",
            ),
        ),
        hold_seconds=_HOLD_SECONDS,
        side="right",
        valid_views=(CameraView.FRONT, CameraView.THREE_QUARTER),
    )
    prompt = PromptTemplate(live=_LIVE_TH, summary=_SUMMARY_TH)

    def measure(
        self, frame: PoseFrame, baseline: Optional[BaselinePose] = None
    ) -> dict[str, float]:
        return _measure_head_lateral_tilt(frame, baseline)
```

Then update `src/exercises/__init__.py`:

```python
"""Exercise registry. Each submodule registers its exercises here."""

from exercises.base import Exercise
from exercises.neck_stretch import NeckStretchLeft, NeckStretchRight


EXERCISES: dict[str, Exercise] = {
    "neck_stretch_left": NeckStretchLeft(),
    "neck_stretch_right": NeckStretchRight(),
}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/office_syndrome/test_neck_stretch.py tests/office_syndrome/test_exercises_registry.py -q`
Expected: PASS. (The existing `test_metadata` asserting `hold_seconds == 20.0` will now fail — update that assertion to `25.0` in the same file as part of this step.)

- [ ] **Step 5: Commit**

```bash
git add src/exercises/neck_stretch.py src/exercises/__init__.py tests/office_syndrome/test_neck_stretch.py
git commit -m "feat: add NeckStretchRight + 25s holds, DRY shared measure"
```

---

## Task 2: `routine.py` — types + SETUP → POSITIONING

**Files:**
- Create: `src/routine.py`
- Test: `tests/office_syndrome/test_routine.py`

- [ ] **Step 1: Write the failing test** — create `tests/office_syndrome/test_routine.py`:

```python
from routine import RoutineFSM, RoutineConfig, RoutinePhase


def kinds(events):
    return [e.kind for e in events]


def test_starts_in_setup():
    assert RoutineFSM().phase is RoutinePhase.SETUP


def test_config_defaults():
    c = RoutineConfig()
    assert c.order == ("left", "right", "left", "right")
    assert c.sets == 4
    assert c.hold_s == 25.0


def test_start_enters_positioning():
    fsm = RoutineFSM()
    fsm.start(0.0)
    assert fsm.phase is RoutinePhase.POSITIONING
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/office_syndrome/test_routine.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'routine'`.

- [ ] **Step 3: Implement** — create `src/routine.py`:

```python
"""Pure state machine for the guided neck-stretch routine.

No OpenCV, no audio, no pose code — advanced by timestamps + two booleans so it
is unit-testable and portable to the future streaming server. The caller
(`app.py`) maps the returned events to audio cues and screen rendering and owns
all I/O. Spec: docs/superpowers/specs/2026-05-23-neck-stretch-realtime-demo-design.md
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RoutinePhase(Enum):
    SETUP = "setup"
    POSITIONING = "positioning"
    COUNTDOWN = "countdown"
    HOLD = "hold"
    TRANSITION = "transition"
    SUMMARY = "summary"
    DONE = "done"


# Event kinds emitted by RoutineFSM.update(). The caller turns each into an
# audio cue and/or a one-shot UI change. `value` payloads noted inline.
EV_POSITION_OK = "position_ok"            # value: None
EV_COUNTDOWN = "countdown"                # value: int (3, 2, 1)
EV_SET_STARTED = "set_started"            # value: side str ("left"/"right")
EV_SET_COMPLETE = "set_complete"          # value: completed set index (int)
EV_SWITCH_SIDES = "switch_sides"          # value: next side str
EV_ROUTINE_COMPLETE = "routine_complete"  # value: None


@dataclass(frozen=True)
class RoutineEvent:
    kind: str
    value: object = None


@dataclass(frozen=True)
class RoutineConfig:
    hold_s: float = 25.0
    order: tuple[str, ...] = ("left", "right", "left", "right")
    position_hold_s: float = 3.0
    countdown_s: int = 3
    transition_s: float = 3.0
    summary_s: float = 10.0

    @property
    def sets(self) -> int:
        return len(self.order)


class RoutineFSM:
    """Advance with update(now, pose_ready, in_target) each frame; read .phase
    and the rendering-state attributes; map returned events to cues."""

    def __init__(self, config: Optional[RoutineConfig] = None):
        self.config = config or RoutineConfig()
        self.phase = RoutinePhase.SETUP
        self.set_index = 0
        # Rendering state, refreshed on each update().
        self.position_progress = 0.0
        self.countdown_value = self.config.countdown_s
        self.hold_elapsed_s = 0.0
        self.hold_remaining_s = self.config.hold_s
        self.transition_remaining_s = self.config.transition_s
        # Internal timers.
        self._phase_start: Optional[float] = None
        self._pos_ok_since: Optional[float] = None
        self._last_countdown_emitted: Optional[int] = None

    @property
    def current_side(self) -> Optional[str]:
        if self.phase in (RoutinePhase.COUNTDOWN, RoutinePhase.HOLD) and (
            0 <= self.set_index < self.config.sets
        ):
            return self.config.order[self.set_index]
        return None

    @property
    def next_side(self) -> Optional[str]:
        nxt = self.set_index + 1
        return self.config.order[nxt] if nxt < self.config.sets else None

    def start(self, now: float) -> None:
        if self.phase is RoutinePhase.SETUP:
            self.phase = RoutinePhase.POSITIONING
            self._pos_ok_since = None
            self.position_progress = 0.0

    def update(
        self, now: float, pose_ready: bool, in_target: bool
    ) -> list[RoutineEvent]:
        events: list[RoutineEvent] = []
        c = self.config

        if self.phase is RoutinePhase.POSITIONING:
            if pose_ready:
                if self._pos_ok_since is None:
                    self._pos_ok_since = now
                held = now - self._pos_ok_since
                self.position_progress = min(1.0, held / c.position_hold_s)
                if held >= c.position_hold_s:
                    events.append(RoutineEvent(EV_POSITION_OK))
                    self.set_index = 0
                    self._enter_countdown(now)
            else:
                self._pos_ok_since = None
                self.position_progress = 0.0

        elif self.phase is RoutinePhase.COUNTDOWN:
            elapsed = now - self._phase_start
            remaining = c.countdown_s - elapsed
            n = max(1, c.countdown_s - int(elapsed))
            self.countdown_value = n
            if self._last_countdown_emitted != n and remaining > 0:
                events.append(RoutineEvent(EV_COUNTDOWN, n))
                self._last_countdown_emitted = n
            if elapsed >= c.countdown_s:
                self._enter_hold(now, events)

        elif self.phase is RoutinePhase.HOLD:
            self.hold_elapsed_s = now - self._phase_start
            self.hold_remaining_s = max(0.0, c.hold_s - self.hold_elapsed_s)
            if self.hold_elapsed_s >= c.hold_s:
                events.append(RoutineEvent(EV_SET_COMPLETE, self.set_index))
                if self.set_index >= c.sets - 1:
                    self._enter_summary(now, events)
                else:
                    self._enter_transition(now, events)

        elif self.phase is RoutinePhase.TRANSITION:
            elapsed = now - self._phase_start
            self.transition_remaining_s = max(0.0, c.transition_s - elapsed)
            if elapsed >= c.transition_s:
                self.set_index += 1
                self._enter_countdown(now)

        elif self.phase is RoutinePhase.SUMMARY:
            if now - self._phase_start >= c.summary_s:
                self.phase = RoutinePhase.DONE

        return events

    def _enter_countdown(self, now: float) -> None:
        self.phase = RoutinePhase.COUNTDOWN
        self._phase_start = now
        self._last_countdown_emitted = None
        self.countdown_value = self.config.countdown_s

    def _enter_hold(self, now: float, events: list[RoutineEvent]) -> None:
        self.phase = RoutinePhase.HOLD
        self._phase_start = now
        self.hold_elapsed_s = 0.0
        self.hold_remaining_s = self.config.hold_s
        events.append(RoutineEvent(EV_SET_STARTED, self.current_side))

    def _enter_transition(self, now: float, events: list[RoutineEvent]) -> None:
        self.phase = RoutinePhase.TRANSITION
        self._phase_start = now
        self.transition_remaining_s = self.config.transition_s
        events.append(RoutineEvent(EV_SWITCH_SIDES, self.next_side))

    def _enter_summary(self, now: float, events: list[RoutineEvent]) -> None:
        self.phase = RoutinePhase.SUMMARY
        self._phase_start = now
        events.append(RoutineEvent(EV_ROUTINE_COMPLETE))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/office_syndrome/test_routine.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/routine.py tests/office_syndrome/test_routine.py
git commit -m "feat: add pure RoutineFSM scaffolding (setup/positioning)"
```

---

## Task 3: RoutineFSM POSITIONING gate (3s + reset)

**Files:**
- Test: `tests/office_syndrome/test_routine.py`
- (Implementation already written in Task 2 — this task verifies it with tests.)

- [ ] **Step 1: Write the failing test** — append:

```python
def test_positioning_progress_resets_when_pose_lost():
    cfg = RoutineConfig(position_hold_s=1.0)
    fsm = RoutineFSM(cfg)
    fsm.start(0.0)
    fsm.update(0.0, True, False)
    fsm.update(0.5, True, False)
    assert 0.4 < fsm.position_progress < 0.6
    fsm.update(0.6, False, False)  # lost the pose
    assert fsm.position_progress == 0.0
    assert fsm.phase is RoutinePhase.POSITIONING


def test_positioning_completes_after_full_hold():
    cfg = RoutineConfig(position_hold_s=1.0)
    fsm = RoutineFSM(cfg)
    fsm.start(0.0)
    fsm.update(0.0, True, False)
    evs = fsm.update(1.01, True, False)
    assert "position_ok" in kinds(evs)
    assert fsm.phase is RoutinePhase.COUNTDOWN
    assert fsm.set_index == 0
    assert fsm.current_side == "left"
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/office_syndrome/test_routine.py -q`
Expected: PASS (already implemented in Task 2). If a test fails, fix `routine.py` until green.

- [ ] **Step 3: Commit**

```bash
git add tests/office_syndrome/test_routine.py
git commit -m "test: cover RoutineFSM positioning gate + reset"
```

---

## Task 4: RoutineFSM COUNTDOWN → HOLD (first set)

**Files:**
- Test: `tests/office_syndrome/test_routine.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def _to_countdown(cfg):
    fsm = RoutineFSM(cfg)
    fsm.start(0.0)
    fsm.update(0.0, True, False)
    fsm.update(cfg.position_hold_s + 0.01, True, False)  # POSITIONING -> COUNTDOWN
    return fsm, cfg.position_hold_s + 0.01


def test_countdown_emits_3_2_1_then_set_started():
    cfg = RoutineConfig(position_hold_s=1.0, countdown_s=3)
    fsm, t0 = _to_countdown(cfg)
    emitted = []
    for t in (t0, t0 + 1.0, t0 + 2.0):
        e = fsm.update(t, True, False)
        emitted += [ev.value for ev in e if ev.kind == "countdown"]
    assert emitted == [3, 2, 1]
    e = fsm.update(t0 + 3.01, True, False)
    assert "set_started" in kinds(e)
    assert fsm.phase is RoutinePhase.HOLD
    assert fsm.current_side == "left"
    started_side = [ev.value for ev in e if ev.kind == "set_started"][0]
    assert started_side == "left"
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/office_syndrome/test_routine.py::test_countdown_emits_3_2_1_then_set_started -q`
Expected: PASS (Task 2 implementation covers this). Fix `routine.py` if red.

- [ ] **Step 3: Commit**

```bash
git add tests/office_syndrome/test_routine.py
git commit -m "test: cover RoutineFSM countdown -> hold"
```

---

## Task 5: RoutineFSM HOLD → TRANSITION → next set, full sequence + SUMMARY

**Files:**
- Test: `tests/office_syndrome/test_routine.py`

- [ ] **Step 1: Write the failing tests** — append:

```python
def _to_first_hold(cfg):
    fsm, t0 = _to_countdown(cfg)
    th = t0 + cfg.countdown_s + 0.01
    fsm.update(th, True, False)  # COUNTDOWN -> HOLD
    return fsm, th


def test_hold_completes_then_transition_to_next_side():
    cfg = RoutineConfig(position_hold_s=1.0, countdown_s=3, hold_s=2.0, transition_s=1.0)
    fsm, th = _to_first_hold(cfg)
    assert fsm.phase is RoutinePhase.HOLD
    e = fsm.update(th + 2.01, True, True)
    assert "set_complete" in kinds(e)
    assert "switch_sides" in kinds(e)
    assert fsm.phase is RoutinePhase.TRANSITION
    nxt = [ev.value for ev in e if ev.kind == "switch_sides"][0]
    assert nxt == "right"


def test_transition_advances_to_next_set():
    cfg = RoutineConfig(position_hold_s=1.0, countdown_s=3, hold_s=2.0, transition_s=1.0)
    fsm, th = _to_first_hold(cfg)
    t1 = th + 2.01
    fsm.update(t1, True, True)            # -> TRANSITION
    fsm.update(t1 + 1.01, True, False)    # transition done -> COUNTDOWN set 1
    assert fsm.phase is RoutinePhase.COUNTDOWN
    assert fsm.set_index == 1
    assert fsm.current_side == "right"


def test_full_routine_sequence_and_done():
    cfg = RoutineConfig(
        position_hold_s=0.5, countdown_s=1, hold_s=1.0, transition_s=0.5, summary_s=0.5
    )
    fsm = RoutineFSM(cfg)
    t = 0.0
    fsm.start(t)
    fsm.update(t, True, False)
    t += cfg.position_hold_s + 0.01
    fsm.update(t, True, False)
    started, completes, done = [], 0, False
    for _ in range(5000):
        t += 0.05
        for ev in fsm.update(t, True, True):
            if ev.kind == "set_started":
                started.append(ev.value)
            elif ev.kind == "set_complete":
                completes += 1
            elif ev.kind == "routine_complete":
                done = True
        if fsm.phase is RoutinePhase.DONE:
            break
    assert started == ["left", "right", "left", "right"]
    assert completes == 4
    assert done
    assert fsm.phase is RoutinePhase.DONE
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/office_syndrome/test_routine.py -q`
Expected: PASS (all routine tests). Fix `routine.py` if any are red.

- [ ] **Step 3: Commit**

```bash
git add tests/office_syndrome/test_routine.py
git commit -m "test: cover full RoutineFSM set sequence + summary/done"
```

---

## Task 6: `feedback/tts.py` — WAV wrapping + `GeminiTTS` with `say` fallback

**Files:**
- Create: `src/feedback/tts.py`
- Test: `tests/office_syndrome/test_tts.py`

- [ ] **Step 1: Write the failing tests** — create `tests/office_syndrome/test_tts.py`:

```python
import io
import wave


def test_pcm_to_wav_header():
    from feedback.tts import _pcm_to_wav

    pcm = b"\x00\x01" * 240  # 240 16-bit samples
    wav = _pcm_to_wav(pcm, sample_rate=24000)
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 24000
        assert wf.getnframes() == 240


def test_synthesize_falls_back_to_say_without_client(monkeypatch):
    from feedback.tts import GeminiTTS

    tts = GeminiTTS(api_key=None)  # no key -> no client
    assert tts._client is None
    monkeypatch.setattr(tts, "_synthesize_say", lambda text: b"AIFFDATA")
    audio, suffix = tts.synthesize("สวัสดี")
    assert audio == b"AIFFDATA"
    assert suffix == ".aiff"


def test_synthesize_falls_back_when_gemini_raises(monkeypatch):
    from feedback.tts import GeminiTTS

    tts = GeminiTTS(api_key=None)
    tts._client = object()  # pretend a client exists

    def boom(text):
        raise RuntimeError("network down")

    monkeypatch.setattr(tts, "_synthesize_gemini", boom)
    monkeypatch.setattr(tts, "_synthesize_say", lambda text: b"AIFFDATA")
    audio, suffix = tts.synthesize("สวัสดี")
    assert audio == b"AIFFDATA"
    assert suffix == ".aiff"


def test_synthesize_uses_gemini_when_available(monkeypatch):
    from feedback.tts import GeminiTTS

    tts = GeminiTTS(api_key=None)
    tts._client = object()
    monkeypatch.setattr(tts, "_synthesize_gemini", lambda text: b"WAVDATA")
    audio, suffix = tts.synthesize("hi")
    assert audio == b"WAVDATA"
    assert suffix == ".wav"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/office_syndrome/test_tts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'feedback.tts'`.

- [ ] **Step 3: Implement** — create `src/feedback/tts.py`:

```python
"""Thai text-to-speech for the neck-stretch demo.

GeminiTTS synthesizes via Google AI Studio (gemini-2.5-flash-preview-tts) and
falls back to the offline macOS `say -v Kanya` voice on any error, so the demo
never goes mute. TTSWorker plays audio on a background thread (afplay), with a
priority cue channel (pre-cached fixed phrases) and a drop-stale feedback
channel for live LLM coaching.
"""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import Optional

_GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
_GEMINI_VOICE = "Kore"  # multilingual prebuilt voice; handles Thai
_SAMPLE_RATE = 24000  # Gemini TTS returns 24kHz 16-bit mono PCM
_SAY_VOICE = "Kanya"  # macOS th_TH voice


def _pcm_to_wav(pcm: bytes, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """Wrap raw signed-16-bit mono PCM in a WAV container so afplay can play it."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


class GeminiTTS:
    """Synthesize Thai speech. Returns (audio_bytes, file_suffix)."""

    def __init__(self, api_key: Optional[str] = None, voice: str = _GEMINI_VOICE):
        self.voice = voice
        self._api_key = api_key or os.getenv("google_ai_studio_api_key")
        self._client = None
        if self._api_key:
            try:
                from google import genai

                self._client = genai.Client(api_key=self._api_key)
            except Exception as e:  # pragma: no cover - env dependent
                print(f"[tts] Gemini client init failed; using macOS say: {e}")
                self._client = None

    def synthesize(self, text: str) -> tuple[bytes, str]:
        if self._client is not None:
            try:
                return self._synthesize_gemini(text), ".wav"
            except Exception as e:
                print(f"[tts] Gemini synth failed ({e}); falling back to say")
        return self._synthesize_say(text), ".aiff"

    def _synthesize_gemini(self, text: str) -> bytes:
        from google.genai import types

        resp = self._client.models.generate_content(
            model=_GEMINI_TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.voice
                        )
                    )
                ),
            ),
        )
        pcm = resp.candidates[0].content.parts[0].inline_data.data
        return _pcm_to_wav(pcm)

    def _synthesize_say(self, text: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
            path = f.name
        try:
            subprocess.run(
                ["say", "-v", _SAY_VOICE, "-o", path, text],
                check=True,
                capture_output=True,
            )
            return Path(path).read_bytes()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/office_syndrome/test_tts.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/feedback/tts.py tests/office_syndrome/test_tts.py
git commit -m "feat: add GeminiTTS with macOS say fallback"
```

---

## Task 7: `feedback/tts.py` — `TTSWorker` (cue queue + drop-stale feedback)

**Files:**
- Modify: `src/feedback/tts.py` (append `TTSWorker`)
- Test: `tests/office_syndrome/test_tts.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/office_syndrome/test_tts.py`:

```python
class _FakeTTS:
    def __init__(self):
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        return (b"AUDIO", ".wav")


def test_precache_synthesizes_each_phrase():
    from feedback.tts import TTSWorker

    fake = _FakeTTS()
    w = TTSWorker(fake)
    w.precache({"count_3": "สาม", "done": "เยี่ยม"})
    assert sorted(fake.calls) == ["สาม", "เยี่ยม"]
    assert "count_3" in w._cue_cache and "done" in w._cue_cache


def test_play_cue_enqueues_cached_only():
    from feedback.tts import TTSWorker

    w = TTSWorker(_FakeTTS())
    w.precache({"count_3": "สาม"})
    w.play_cue("count_3")
    assert len(w._cue_queue) == 1
    w.play_cue("missing")  # unknown cue -> no-op
    assert len(w._cue_queue) == 1


def test_submit_feedback_is_drop_stale():
    from feedback.tts import TTSWorker

    w = TTSWorker(_FakeTTS())
    w.submit_feedback("a")
    w.submit_feedback("b")
    assert w._pending_feedback_text == "b"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/office_syndrome/test_tts.py -k "precache or play_cue or drop_stale" -q`
Expected: FAIL — `AttributeError`/`ImportError` (`TTSWorker` not defined).

- [ ] **Step 3: Implement** — append `TTSWorker` to `src/feedback/tts.py`:

```python
class TTSWorker:
    """Background audio player.

    - Cue channel: FIFO queue of pre-synthesized (bytes, suffix) clips. Never
      dropped; checked before feedback so cues win at scheduling boundaries.
    - Feedback channel: a single pending text (drop-stale, like LLMWorker).
      Synthesized in the worker thread and played only when no cue is queued.
    One clip plays at a time (afplay is blocking in the worker), so audio never
    overlaps.
    """

    def __init__(self, tts: GeminiTTS):
        self._tts = tts
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._cue_cache: dict[str, tuple[bytes, str]] = {}
        self._cue_queue: list[tuple[bytes, str]] = []
        self._pending_feedback_text: Optional[str] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def precache(self, phrases: dict[str, str]) -> None:
        """Synthesize fixed cue phrases once (blocking). Call at startup."""
        for key, text in phrases.items():
            self._cue_cache[key] = self._tts.synthesize(text)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def play_cue(self, key: str) -> None:
        with self._cv:
            clip = self._cue_cache.get(key)
            if clip is not None:
                self._cue_queue.append(clip)
                self._cv.notify()

    def submit_feedback(self, text: str) -> None:
        if not text:
            return
        with self._cv:
            self._pending_feedback_text = text  # newer overwrites older
            self._cv.notify()

    def stop(self) -> None:
        with self._cv:
            self._running = False
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while self._running:
            kind = None
            clip = None
            text = None
            with self._cv:
                while self._running and not self._cue_queue and (
                    self._pending_feedback_text is None
                ):
                    self._cv.wait(timeout=0.1)
                if not self._running:
                    return
                if self._cue_queue:
                    kind, clip = "cue", self._cue_queue.pop(0)
                elif self._pending_feedback_text is not None:
                    kind, text = "feedback", self._pending_feedback_text
                    self._pending_feedback_text = None
            if kind == "cue":
                self._play(clip[0], clip[1])
            elif kind == "feedback":
                try:
                    audio, suffix = self._tts.synthesize(text)
                except Exception as e:
                    print(f"[tts] feedback synth failed: {e}")
                    continue
                # A cue may have arrived during synthesis — it wins; drop this.
                with self._lock:
                    cue_waiting = bool(self._cue_queue)
                if not cue_waiting:
                    self._play(audio, suffix)

    def _play(self, audio: bytes, suffix: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio)
            path = f.name
        try:
            subprocess.run(["afplay", path], check=False)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/office_syndrome/test_tts.py -q`
Expected: PASS (7 tests). The worker thread is not started in tests, so no real audio plays.

- [ ] **Step 5: Commit**

```bash
git add src/feedback/tts.py tests/office_syndrome/test_tts.py
git commit -m "feat: add TTSWorker (cue queue + drop-stale feedback)"
```

---

## Task 8: `render.py` — `put_thai_text` module helper

**Files:**
- Modify: `src/render.py`
- Test: `tests/pipeline/test_render.py`

- [ ] **Step 1: Write the failing test** — create (or append to) `tests/pipeline/test_render.py`:

```python
import numpy as np

from render import put_thai_text


def test_put_thai_text_draws_pixels():
    img = np.zeros((120, 400, 3), dtype=np.uint8)
    put_thai_text(img, "ทดสอบ", (20, 40), font_size=28, color=(255, 255, 255))
    assert img.sum() > 0  # something was drawn


def test_put_thai_text_center_runs():
    img = np.zeros((120, 400, 3), dtype=np.uint8)
    put_thai_text(img, "กลาง", (200, 50), font_size=28, color=(0, 255, 0), center=True)
    assert img.sum() > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/pipeline/test_render.py -q`
Expected: FAIL — `ImportError: cannot import name 'put_thai_text'`.

- [ ] **Step 3: Implement** — add to `src/render.py` (module level, after the imports and `SKELETON`):

```python
_THAI_FONT_PATH = "/System/Library/Fonts/Supplemental/Ayuthaya.ttf"


def _load_thai_font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype(_THAI_FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def put_thai_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_size: int = 22,
    color: tuple[int, int, int] = (255, 255, 255),
    max_width: int | None = None,
    center: bool = False,
) -> None:
    """Draw Thai text onto a BGR ndarray in place.

    `color` is BGR (OpenCV convention). `max_width` (px) word-wraps; otherwise
    splits on '\\n'. `center=True` horizontally centers each line on org[0].
    """
    from PIL import Image, ImageDraw

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _load_thai_font(font_size)
    rgb = (color[2], color[1], color[0])
    x, y = org

    if max_width:
        lines, line = [], ""
        for word in text.split():
            test = (line + " " + word).strip()
            if draw.textbbox((0, 0), test, font=font)[2] > max_width and line:
                lines.append(line)
                line = word
            else:
                line = test
        if line:
            lines.append(line)
    else:
        lines = text.split("\n")

    line_h = int(font_size * 1.3)
    cy = y
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        w = bbox[2] - bbox[0]
        lx = x - w // 2 if center else x
        draw.text((lx, cy), ln, fill=rgb, font=font)
        cy += line_h

    np.copyto(img, cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/pipeline/test_render.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/render.py tests/pipeline/test_render.py
git commit -m "feat: add reusable put_thai_text helper to render"
```

---

## Task 9: `screens.py` — Start-button hit-test + screen draws

**Files:**
- Create: `src/screens.py`
- Test: `tests/office_syndrome/test_screens.py`

- [ ] **Step 1: Write the failing test** — create `tests/office_syndrome/test_screens.py`:

```python
from screens import SETUP_BUTTON_RECT, point_in_rect


def test_point_in_rect_inside():
    x, y, w, h = SETUP_BUTTON_RECT
    assert point_in_rect(x + 5, y + 5, SETUP_BUTTON_RECT)
    assert point_in_rect(x + w // 2, y + h // 2, SETUP_BUTTON_RECT)


def test_point_in_rect_outside():
    x, y, w, h = SETUP_BUTTON_RECT
    assert not point_in_rect(x - 1, y - 1, SETUP_BUTTON_RECT)
    assert not point_in_rect(x + w + 1, y + h + 1, SETUP_BUTTON_RECT)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/office_syndrome/test_screens.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'screens'`.

- [ ] **Step 3: Implement** — create `src/screens.py`. These are pure draw functions (return a fresh BGR canvas) plus the hit-test. The demo renders at the camera resolution `1280×720`.

```python
"""OpenCV screen rendering for the neck-stretch demo.

Each draw_* function returns a BGR canvas to imshow. Kept separate from the
pure RoutineFSM and from app's control flow so the visuals can be iterated on
in isolation. Thai text uses render.put_thai_text.
"""
from __future__ import annotations

import cv2
import numpy as np

from render import put_thai_text

W, H = 1280, 720
SETUP_BUTTON_RECT = (490, 410, 300, 90)  # x, y, w, h

_BG = (24, 24, 28)
_GREEN = (90, 220, 90)
_AMBER = (90, 200, 220)
_GREY = (160, 160, 160)
_WHITE = (245, 245, 245)

_SIDE_TH = {"left": "ซ้าย", "right": "ขวา"}


def point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def _blank() -> np.ndarray:
    canvas = np.empty((H, W, 3), dtype=np.uint8)
    canvas[:] = _BG
    return canvas


def _mirror(frame: np.ndarray) -> np.ndarray:
    """Selfie-view mirror, resized to the demo canvas."""
    f = cv2.resize(frame, (W, H))
    return cv2.flip(f, 1)


def draw_setup() -> np.ndarray:
    canvas = _blank()
    put_thai_text(canvas, "ยืดคอ คลายออฟฟิศซินโดรม", (W // 2, 150),
                  font_size=46, color=_WHITE, center=True)
    put_thai_text(canvas, "ใช้เวลา 2 นาที · ยืดคอสลับซ้าย-ขวา 4 เซ็ต",
                  (W // 2, 230), font_size=26, color=_GREY, center=True)
    x, y, w, h = SETUP_BUTTON_RECT
    cv2.rectangle(canvas, (x, y), (x + w, y + h), _GREEN, -1, cv2.LINE_AA)
    put_thai_text(canvas, "เริ่ม", (x + w // 2, y + 22), font_size=40,
                  color=(20, 20, 20), center=True)
    put_thai_text(canvas, "(คลิกปุ่มเพื่อเริ่ม · กด q เพื่อออก)", (W // 2, 560),
                  font_size=22, color=_GREY, center=True)
    return canvas


def _draw_outline(img: np.ndarray, ok: bool) -> None:
    """Semi-transparent humanoid guide centered on the canvas."""
    color = _GREEN if ok else _GREY
    overlay = img.copy()
    cx = W // 2
    cv2.circle(overlay, (cx, 180), 60, color, 3, cv2.LINE_AA)          # head
    cv2.line(overlay, (cx - 120, 300), (cx + 120, 300), color, 3, cv2.LINE_AA)  # shoulders
    cv2.line(overlay, (cx - 90, 520), (cx + 90, 520), color, 3, cv2.LINE_AA)    # hips
    cv2.line(overlay, (cx - 120, 300), (cx - 90, 520), color, 3, cv2.LINE_AA)   # torso L
    cv2.line(overlay, (cx + 120, 300), (cx + 90, 520), color, 3, cv2.LINE_AA)   # torso R
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0.0, dst=img)


def draw_positioning(frame: np.ndarray, progress: float, ok: bool) -> np.ndarray:
    canvas = _mirror(frame)
    _draw_outline(canvas, ok)
    msg = "ค้างไว้..." if ok else "ยืนให้กล้องเห็นหัว ไหล่ และสะโพก"
    put_thai_text(canvas, msg, (W // 2, 40), font_size=30,
                  color=_GREEN if ok else _AMBER, center=True)
    # progress bar
    bw, bx, by = 600, (W - 600) // 2, H - 60
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + 24), _GREY, 2, cv2.LINE_AA)
    fill = int(bw * max(0.0, min(1.0, progress)))
    cv2.rectangle(canvas, (bx, by), (bx + fill, by + 24), _GREEN, -1, cv2.LINE_AA)
    return canvas


def draw_countdown(frame: np.ndarray, number: int, side: str | None) -> np.ndarray:
    canvas = _mirror(frame)
    if side:
        put_thai_text(canvas, f"เตรียมยืดคอด้าน{_SIDE_TH.get(side, side)}",
                      (W // 2, 120), font_size=36, color=_WHITE, center=True)
    cv2.putText(canvas, str(number), (W // 2 - 40, H // 2 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 6.0, _WHITE, 12, cv2.LINE_AA)
    return canvas


def draw_hold(
    frame: np.ndarray,
    side: str | None,
    remaining_s: float,
    in_target: bool,
    view_ok: bool,
    thai_text: str,
) -> np.ndarray:
    canvas = _mirror(frame)
    border = _GREEN if in_target else _AMBER
    cv2.rectangle(canvas, (4, 4), (W - 4, H - 4), border, 8)
    put_thai_text(canvas, f"ยืดคอด้าน{_SIDE_TH.get(side, side or '')}", (30, 30),
                  font_size=34, color=_WHITE)
    cv2.putText(canvas, f"{int(np.ceil(remaining_s)):02d}", (W - 150, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, _WHITE, 5, cv2.LINE_AA)
    if not view_ok:
        put_thai_text(canvas, "หันหน้าเข้าหากล้อง", (W // 2, H // 2),
                      font_size=34, color=_AMBER, center=True)
    elif thai_text:
        put_thai_text(canvas, thai_text, (40, H - 120), font_size=28,
                      color=_WHITE, max_width=W - 80)
    return canvas


def draw_transition(frame: np.ndarray, next_side: str | None) -> np.ndarray:
    canvas = _mirror(frame)
    put_thai_text(canvas, "พักสักครู่", (W // 2, H // 2 - 40), font_size=40,
                  color=_WHITE, center=True)
    if next_side:
        put_thai_text(canvas, f"ต่อไปยืดด้าน{_SIDE_TH.get(next_side, next_side)}",
                      (W // 2, H // 2 + 30), font_size=32, color=_GREEN, center=True)
    return canvas


def draw_summary(set_scores: list[int], overall: int, thai_text: str) -> np.ndarray:
    canvas = _blank()
    put_thai_text(canvas, "จบการฝึก", (W // 2, 90), font_size=46,
                  color=_WHITE, center=True)
    put_thai_text(canvas, f"คะแนนรวม {overall}/100", (W // 2, 170), font_size=34,
                  color=_GREEN, center=True)
    for i, sc in enumerate(set_scores):
        put_thai_text(canvas, f"เซ็ต {i + 1}: {sc}/100", (W // 2, 240 + i * 44),
                      font_size=26, color=_GREY, center=True)
    if thai_text:
        put_thai_text(canvas, thai_text, (W // 2 - 400, 470), font_size=26,
                      color=_WHITE, max_width=800)
    put_thai_text(canvas, "(กด q เพื่อออก)", (W // 2, H - 50), font_size=22,
                  color=_GREY, center=True)
    return canvas
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/office_syndrome/test_screens.py -q`
Expected: PASS (2 tests). The draw_* functions are verified manually in Task 11.

- [ ] **Step 5: Commit**

```bash
git add src/screens.py tests/office_syndrome/test_screens.py
git commit -m "feat: add demo screen renderers + Start-button hit-test"
```

---

## Task 10: `app.py` — orchestration + main loop

**Files:**
- Replace: `src/app.py`
- Modify: `src/selector.py`

This task is GUI/audio integration; it is verified by a headless import smoke check plus a manual run (Task 11). Write the full file.

- [ ] **Step 1: Add `choose_routine()` to `src/selector.py`** — append:

```python
def choose_routine() -> str:
    """One-item launcher for the neck-stretch demo. Returns "neck_stretch".
    Press SPACE/Enter or click anywhere to start; q/Esc quits."""
    width, height = 780, 220
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    clicked = {"go": False}

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["go"] = True

    window = "Workout AI — Select"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_mouse)
    try:
        while True:
            canvas[:] = _PANEL_BG
            cv2.putText(canvas, "Neck Stretch (2 min)", (40, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, _TEXT_FG, 2, cv2.LINE_AA)
            cv2.putText(canvas, "Click / SPACE to start, q to quit", (40, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _HINT_FG, 1, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyWindow(window)
                raise SystemExit("User canceled")
            if key in (ord(" "), 13) or clicked["go"]:
                cv2.destroyWindow(window)
                return "neck_stretch"
    finally:
        pass
```

- [ ] **Step 2: Replace `src/app.py` entirely** with:

```python
"""Real-time guided neck-stretch demo.

Flow: selector -> setup (Start button) -> positioning (outline + calibration)
-> [countdown -> 25s hold -> transition] x4 -> summary. Spoken Thai coaching via
Gemini TTS (with macOS say fallback). The routine logic lives in the pure
RoutineFSM (src/routine.py); this module owns the camera loop, pose inference,
scoring accumulation, rendering, and audio.
"""
from __future__ import annotations

import time

import cv2

import screens
from analysis.angles import L_HIP, L_SHOULDER, NOSE, R_HIP, R_SHOULDER
from analysis.camera_view import classify_view
from analysis.rules_hold import score_frame, score_hold
from analysis.types import HoldAnalysis, HoldState, LiveSnapshot, PoseFrame, RuleViolation
from calibration import BaselinePose, CalibrationError, calibrate_from_samples
from capture import WebcamCapture
from exercises.neck_stretch import NeckStretchLeft, NeckStretchRight
from feedback.llm import ThaiCoachLLM
from feedback.tts import GeminiTTS, TTSWorker
from feedback.worker import LLMWorker
from pose2d import Pose2D
from routine import (
    EV_COUNTDOWN,
    EV_POSITION_OK,
    EV_ROUTINE_COMPLETE,
    EV_SET_COMPLETE,
    EV_SET_STARTED,
    EV_SWITCH_SIDES,
    RoutineConfig,
    RoutineFSM,
    RoutinePhase,
)
from selector import choose_routine

_REQUIRED_KPS = (NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
_KP_FLOOR = 0.3
_LIVE_SUBMIT_INTERVAL_S = 2.5
_VALID_VIEWS = NeckStretchLeft().target.valid_views
_WINDOW = "Workout AI"

# Fixed Thai cue phrases, pre-synthesized at startup so they play instantly.
CUE_PHRASES = {
    "count_3": "สาม",
    "count_2": "สอง",
    "count_1": "หนึ่ง",
    "start_left": "เริ่มยืดคอไปทางซ้าย ค่อย ๆ เอียงศีรษะ",
    "start_right": "เริ่มยืดคอไปทางขวา ค่อย ๆ เอียงศีรษะ",
    "switch_left": "เปลี่ยนไปยืดด้านซ้าย",
    "switch_right": "เปลี่ยนไปยืดด้านขวา",
    "done": "เยี่ยมมาก ทำครบทุกเซ็ตแล้ว",
    "face_camera": "กรุณาหันหน้าเข้าหากล้อง",
}


def _pose_ready(scores, view) -> bool:
    return (
        all(float(scores[i]) >= _KP_FLOOR for i in _REQUIRED_KPS)
        and view in _VALID_VIEWS
    )


def run():
    cap = WebcamCapture(device=0, width=1280, height=720)
    pose = Pose2D()

    print("Loading Qwen3.5-4B (this takes ~10 seconds the first time)...")
    llm = ThaiCoachLLM()
    print("Warming up LLM...")
    llm.warmup()
    worker = LLMWorker(llm)
    worker.start()

    print("Initializing TTS + pre-caching cues...")
    tts_worker = TTSWorker(GeminiTTS())
    tts_worker.precache(CUE_PHRASES)
    tts_worker.start()
    print("Ready.")

    side_exercises = {"left": NeckStretchLeft(), "right": NeckStretchRight()}
    cap.start()
    try:
        while True:
            try:
                choose_routine()
            except SystemExit:
                break
            run_neck_stretch_routine(cap, pose, worker, tts_worker, side_exercises)
    finally:
        worker.stop()
        tts_worker.stop()
        cap.stop()
        cv2.destroyAllWindows()


def _aggregate(set_analyses: list[HoldAnalysis], exercise) -> HoldAnalysis:
    n = max(1, len(set_analyses))
    comp = {
        k: round(sum(a.components.get(k, 0) for a in set_analyses) / n)
        for k in ("duration", "precision", "stability")
    }
    worst: dict[str, RuleViolation] = {}
    for a in set_analyses:
        for v in a.violations:
            if v.name not in worst or v.severity > worst[v.name].severity:
                worst[v.name] = v
    return HoldAnalysis(
        exercise_name=exercise.name,
        score=sum(comp.values()),
        components=comp,
        violations=list(worst.values()),
        in_target_ms=sum(a.in_target_ms for a in set_analyses),
        drift_count=sum(a.drift_count for a in set_analyses),
    )


def run_neck_stretch_routine(cap, pose, worker, tts_worker, side_exercises, config=None):
    fsm = RoutineFSM(config or RoutineConfig())
    cv2.namedWindow(_WINDOW)
    click = {"start": False}

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and screens.point_in_rect(
            x, y, screens.SETUP_BUTTON_RECT
        ):
            click["start"] = True

    cv2.setMouseCallback(_WINDOW, _on_mouse)

    baseline: BaselinePose | None = None
    calib_samples: list = []
    set_analyses: list[HoldAnalysis] = []
    cur: dict | None = None
    last_frame_ts = time.monotonic()
    last_live_submit = 0.0
    last_spoken = ""
    summary_submit_ts = 0.0
    spoke_summary = False
    view_bad_since: float | None = None

    while True:
        frame = cap.read_latest(timeout=2.0)
        if frame is None:
            return
        now = time.monotonic()

        kps, scores = pose.infer(frame)
        view = classify_view(kps, scores)
        pose_ready = _pose_ready(scores, view)

        side = fsm.current_side
        in_target = False
        violations: list[RuleViolation] = []
        view_ok = view in _VALID_VIEWS
        if fsm.phase is RoutinePhase.HOLD and side is not None and view_ok:
            ex = side_exercises[side]
            pf = PoseFrame(now, kps, scores, frame.shape[:2])
            measured = ex.measure(pf, baseline=baseline)
            in_target, violations = score_frame(ex.target, measured)

        if fsm.phase is RoutinePhase.POSITIONING and pose_ready:
            calib_samples.append((kps, scores))

        if click["start"]:
            click["start"] = False
            fsm.start(now)

        for ev in fsm.update(now, pose_ready, in_target):
            if ev.kind == EV_POSITION_OK:
                try:
                    baseline = calibrate_from_samples(calib_samples, min_clean_frames=30)
                    print(f"[calibration] OK tilt={baseline.head_lateral_tilt_deg:+.1f}")
                except CalibrationError as e:
                    print(f"[calibration] FAILED ({e}); absolute-angle mode")
                    baseline = BaselinePose(0.0, 0.0, 0.0, 0, now)
            elif ev.kind == EV_COUNTDOWN:
                tts_worker.play_cue(f"count_{ev.value}")
            elif ev.kind == EV_SET_STARTED:
                tts_worker.play_cue(f"start_{ev.value}")
                cur = {
                    "in_target_ms": 0,
                    "drift_count": 0,
                    "max_sev": {j.name: 0.0 for j in side_exercises[ev.value].target.joints},
                    "was_in_target": False,
                }
                last_live_submit = 0.0
                last_spoken = ""
            elif ev.kind == EV_SET_COMPLETE and cur is not None:
                done_side = fsm.config.order[ev.value]
                ex = side_exercises[done_side]
                meta = {
                    "in_target_ms": cur["in_target_ms"],
                    "drift_count": cur["drift_count"],
                    "completed_ts": now,
                }
                set_analyses.append(score_hold(ex.name, meta, ex.target, cur["max_sev"]))
                cur = None
            elif ev.kind == EV_SWITCH_SIDES:
                tts_worker.play_cue(f"switch_{ev.value}")
            elif ev.kind == EV_ROUTINE_COMPLETE:
                tts_worker.play_cue("done")
                agg = _aggregate(set_analyses, side_exercises["left"])
                worker.submit(agg, exercise=side_exercises["left"])
                summary_submit_ts = now
                spoke_summary = False

        # HOLD accumulation + live feedback.
        if fsm.phase is RoutinePhase.HOLD and cur is not None:
            dt = now - last_frame_ts
            if in_target:
                cur["in_target_ms"] += int(dt * 1000)
            if cur["was_in_target"] and not in_target:
                cur["drift_count"] += 1
            cur["was_in_target"] = in_target
            for v in violations:
                cur["max_sev"][v.name] = max(cur["max_sev"].get(v.name, 0.0), v.severity)

            if view_ok and side is not None and now - last_live_submit >= _LIVE_SUBMIT_INTERVAL_S:
                ex = side_exercises[side]
                progress = min(1.0, fsm.hold_elapsed_s / fsm.config.hold_s)
                snap = LiveSnapshot(
                    exercise_name=ex.name,
                    state=HoldState.HOLDING if in_target else HoldState.DRIFTED,
                    progress_ratio=progress,
                    current_violations=violations,
                )
                worker.submit(snap, exercise=ex)
                last_live_submit = now
            txt = worker.latest()
            if txt and txt != last_spoken and not txt.startswith("[LLM error"):
                tts_worker.submit_feedback(txt)
                last_spoken = txt

            # Spoken view nudge if framing stays invalid > 3s.
            if not view_ok:
                if view_bad_since is None:
                    view_bad_since = now
                elif now - view_bad_since > 3.0:
                    tts_worker.play_cue("face_camera")
                    view_bad_since = now + 5.0  # cooldown
            else:
                view_bad_since = None

        last_frame_ts = now

        # Render the active screen for this phase.
        canvas, spoke_summary = _compose(
            fsm, frame, in_target, view_ok, worker, set_analyses,
            summary_submit_ts, spoke_summary, now, tts_worker,
        )
        cv2.imshow(_WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or fsm.phase is RoutinePhase.DONE:
            return


def _compose(fsm, frame, in_target, view_ok, worker, set_analyses,
             summary_submit_ts, spoke_summary, now, tts_worker):
    """Pick + draw the screen for the current phase. Returns (canvas, spoke_summary)."""
    phase = fsm.phase
    if phase in (RoutinePhase.SETUP,):
        return screens.draw_setup(), spoke_summary
    if phase is RoutinePhase.POSITIONING:
        return screens.draw_positioning(frame, fsm.position_progress,
                                        ok=fsm.position_progress > 0.0), spoke_summary
    if phase is RoutinePhase.COUNTDOWN:
        return screens.draw_countdown(frame, fsm.countdown_value, fsm.current_side), spoke_summary
    if phase is RoutinePhase.HOLD:
        thai = worker.latest() or ""
        return screens.draw_hold(frame, fsm.current_side, fsm.hold_remaining_s,
                                 in_target, view_ok, thai), spoke_summary
    if phase is RoutinePhase.TRANSITION:
        return screens.draw_transition(frame, fsm.next_side), spoke_summary
    # SUMMARY / DONE
    scores = [a.score for a in set_analyses]
    overall = round(sum(scores) / len(scores)) if scores else 0
    summary_txt = ""
    if not spoke_summary and now - summary_submit_ts > 1.0:
        t = worker.latest()
        if t and not t.startswith("[LLM error"):
            tts_worker.submit_feedback(t)
            spoke_summary = True
    summary_txt = worker.latest() or ""
    return screens.draw_summary(scores, overall, summary_txt), spoke_summary


if __name__ == "__main__":
    run()
```

Note: `last_spoken` is updated inside the HOLD branch (it tracks the last line sent to TTS so the same coaching isn't spoken twice); it is intentionally local to the loop and not passed into `_compose`.

- [ ] **Step 3: Headless import smoke check**

Run: `uv run python -c "import app; import screens; import routine; print('import ok')"`
Expected: `import ok` (imports must not construct heavy objects — all model loading is inside `run()`).

- [ ] **Step 4: Run the pure-logic suite to confirm no regressions**

Run: `uv run pytest -k "not smoke" -q`
Expected: PASS (all pure-logic tests, including the new routine/tts/screens tests).

- [ ] **Step 5: Commit**

```bash
git add src/app.py src/selector.py
git commit -m "feat: rewrite app.py as guided neck-stretch routine with TTS"
```

---

## Task 11: Manual integration verification + docs

**Files:**
- Modify: `CLAUDE.md`, `README.md`

- [ ] **Step 1: Confirm `google-genai` is importable**

Run: `uv run python -c "from google import genai; from google.genai import types; print('genai ok')"`
Expected: `genai ok`. If it fails, run `uv sync --all-extras`.

- [ ] **Step 2: Manual end-to-end run** (requires webcam + the downloaded models; needs network for Gemini TTS, else it falls back to `say`)

Run: `uv run python main.py`
Verify, in order:
1. Selector window appears; clicking / SPACE launches the routine.
2. Setup screen shows the green **เริ่ม** button; clicking it (only the button rect) advances to positioning.
3. Positioning shows the humanoid outline; standing so head+shoulders+hips are visible turns the outline green and fills the progress bar over ~3s; stepping out resets it.
4. A spoken + on-screen 3·2·1 countdown announces the left side first.
5. A 25s hold runs with a counting-down timer, green/amber border tracking whether you are tilting enough, and spoken Thai coaching every few seconds.
6. At 25s a "switch sides" cue plays, a short transition screen shows, then the next set (right) counts down.
7. After 4 sets a "done" cue plays and the summary screen shows per-set + overall scores with a spoken wrap-up.
8. `q` quits cleanly at any point.

Note any issues and fix in `app.py` / `screens.py` before committing. (Tune `screens.py` coordinates/font sizes here if text clips.)

- [ ] **Step 3: Update `CLAUDE.md`** — in the "Project" → "Two exercise modes" section, replace the timed-hold bullet's description of the entry point to reflect the routine. Add under "Architecture" a short subsection:

```markdown
### Neck-stretch demo flow (current entry point)

`app.run()` loads pose + LLM + TTS weights once, then loops `choose_routine()` →
`run_neck_stretch_routine()`. The routine is driven by the pure `RoutineFSM`
(`src/routine.py`): SETUP (clickable Start) → POSITIONING (humanoid outline; the
3s clean-frame window doubles as calibration) → COUNTDOWN (3·2·1 + side cue) →
HOLD (fixed 25s wall-clock; form measured per frame via the 2D-direct tilt +
`score_frame`) → TRANSITION → repeat for 4 alternating sides → SUMMARY. Audio is
spoken Thai via `feedback/tts.py` (`GeminiTTS` + macOS `say` fallback, played by
`TTSWorker`). Screens are drawn by `src/screens.py`. The 3D rig is not used here.
```

- [ ] **Step 4: Update `README.md`** — update the run instructions to describe the neck-stretch demo (clickable Start, 2-minute alternating routine, spoken Thai). Keep it brief and match existing tone.

- [ ] **Step 5: Run the full suite once more**

Run: `uv run pytest -q`
Expected: PASS or only the pre-existing `*_smoke` skips (when models/fixtures are absent).

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document neck-stretch demo flow"
```

---

## Self-Review (completed by plan author)

**Spec coverage:** 4×25s alternating routine → Tasks 2-5 (FSM) + Task 10 (config/wiring). Fixed-wall-clock set → Task 5 + HOLD branch. Clickable Start → Tasks 9-10. Outline positioning + folded calibration → Tasks 9-10 (`EV_POSITION_OK` builds baseline from positioning samples). Per-set countdown + side announce → Tasks 4, 9, 10. Gemini TTS + `say` fallback → Task 6. Cue + drop-stale feedback channels → Task 7. Spoken set cues + live feedback → Task 10. Summary → Tasks 9-10. `NeckStretchRight` → Task 1. Error handling (calibration fallback, invalid view, quit) → Task 10. All spec sections map to a task.

**Deviation from spec (noted):** the spec mentioned reusing `HoldFSM` to compute the live holding/drifted *display* state. The plan instead derives display state directly from the per-frame `in_target` bool and counts drift edges for the stability score. Rationale: with a fixed-wall-clock set, `HoldFSM`'s completion/stability-window lifecycle is unnecessary and would add a second timer racing the set timer. The visible behavior (green when in target, amber otherwise; drift_count feeding `score_hold`) is identical. If you'd rather keep `HoldFSM`, say so and the HOLD branch in Task 10 can wrap it.

**Placeholder scan:** no TODO/TBD/"handle edge cases" left. Task 10's `app.py` is clean as-written (no phantom helpers); every referenced symbol is defined either in the plan or the existing codebase.

**Type consistency:** event kind constants (`EV_*`), `RoutineEvent(kind, value)`, `RoutineConfig` fields, `RoutineFSM` attributes (`current_side`, `next_side`, `hold_elapsed_s`, `countdown_value`, `position_progress`), `GeminiTTS.synthesize -> (bytes, str)`, `TTSWorker.play_cue/submit_feedback/precache`, and `screens.*` signatures are consistent across Tasks 2-10. `score_hold`/`score_frame`/`HoldAnalysis`/`LiveSnapshot` match their definitions in the existing codebase.

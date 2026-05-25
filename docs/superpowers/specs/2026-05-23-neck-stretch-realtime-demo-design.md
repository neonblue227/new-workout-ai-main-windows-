# Neck-Stretch Real-Time Demo — Design

**Date:** 2026-05-23
**Status:** Approved (pending spec review)
**Supersedes the entry-point flow in:** `src/app.py` (the single-hold `run_session` is replaced by a guided multi-set routine)

## Goal

Turn the verified 2D posture pipeline (`src/test_2D_3D.py`) into a guided, real-time
**neck-stretch demo** with spoken Thai coaching. The user picks "Neck Stretch," clicks
**Start**, gets positioned via an on-screen outline, then runs a fixed **2-minute** routine
of four 25-second holds alternating sides (L → R → L → R), with both spoken set cues and
spoken live form feedback.

This is a desktop OpenCV demo. Per `CLAUDE.md`, the analysis pipeline stays GUI-free so the
flow can later move behind a streaming server; only `app.py` and the renderer touch OpenCV's
GUI, and the routine logic is a pure, testable state machine.

## Decisions (from brainstorming)

| Decision | Choice |
| --- | --- |
| Routine structure | 4 holds × 25s, alternating **L → R → L → R** (~100s hold + countdowns/transitions ≈ 2 min) |
| Set timing | **Fixed 25s wall-clock** per set — ends on the timer regardless of form; form drives feedback + score, not duration |
| TTS engine | **Google AI Studio (Gemini) TTS**, with macOS `say -v Kanya` as the offline fallback |
| Audio strategy | Spoken **set cues** (start / switch / done / countdown) **and** spoken **live form feedback** |
| Start interaction | **Clickable on-screen button** (`cv2.setMouseCallback` + rect hit-test) |
| Calibration | Folded into the 3s positioning step — neutral baseline is captured from the same clean frames; no separate calibration screen |
| Architecture | Approach A — a pure `RoutineFSM` orchestrator that reuses the untouched 2D pipeline |

## Architecture

```
src/app.py (rewritten)
  ├─ owns the OpenCV window, mouse callback, main loop, and screen rendering
  ├─ drives RoutineFSM with (now, pose_ready, in_target) each frame
  └─ wires Pose2D → exercise.measure → score_frame → RoutineFSM → Renderer + TTSWorker

src/routine.py (new, pure — no OpenCV, no audio)
  ├─ RoutineConfig         (sets, hold_s, order, position_hold_s, countdown_s)
  ├─ RoutinePhase enum     (SETUP, POSITIONING, READY, SET, TRANSITION, SUMMARY, DONE)
  └─ RoutineFSM            advanced by timestamps + booleans; emits RoutineEvents

src/feedback/tts.py (new)
  ├─ GeminiTTS.synthesize(text) -> wav_bytes   (google-genai; say fallback)
  └─ TTSWorker  (bg thread; afplay playback; cue channel + drop-stale feedback channel)

src/exercises/neck_stretch.py (edited)  +NeckStretchRight (mirror, +35°)
src/exercises/__init__.py     (edited)  register the routine entry
src/selector.py               (edited)  single "Neck Stretch (2 min)" entry
```

The pose / measurement / scoring code (`pose2d.py`, `analysis/angles.py`,
`analysis/rules_hold.py`, `calibration.py`, `analysis/camera_view.py`) is **unchanged** and
reused as-is. The 3D lift is not required for this flow (measurement is 2D-direct); the rig
panel is optional and may be omitted from the demo UI.

## Routine state machine (`src/routine.py`)

`RoutineFSM` is a pure object. Inputs per tick: `now: float` (monotonic), `pose_ready: bool`
(the 5 required keypoints are visible AND the view is valid), `in_target: bool` (from
`score_frame`). It returns the current phase and a list of `RoutineEvent`s the caller turns
into audio + rendering. It holds no frames and calls nothing external.

### Phases

1. **SETUP** — idle until the caller reports a Start click (`fsm.start()`).
2. **POSITIONING** — requires `pose_ready` continuously for `position_hold_s` (3s). The 3s
   timer **resets** whenever `pose_ready` goes false. `pose_ready` =
   `nose, L_shoulder, R_shoulder, L_hip, R_hip` all ≥ 0.3 confidence **and**
   `classify_view ∈ {FRONT, THREE_QUARTER}`. The caller accumulates baseline samples from
   these clean frames; on exit it builds a `BaselinePose` (falls back to zero baseline /
   absolute mode if `calibrate_from_samples` raises, as today).
3. **READY** — one global `countdown_s` (3·2·1) "get ready" countdown. Emits a countdown
   cue per second.
4. **SET[i]** (i = 0..3) — preceded by a per-set side announcement + short 3·2·1
   ("countdown per set, indicating which side"). Then a **fixed 25s wall-clock** hold.
   - Emits `SET_STARTED(side)` once.
   - Each frame the caller feeds `in_target`; the FSM exposes `set_elapsed_s` /
     `set_remaining_s` for the countdown ring.
   - The caller accumulates `in_target_ms` and worst per-joint severity for the set's score.
   - At `elapsed ≥ hold_s` → emit `SET_COMPLETE(i, side)` → TRANSITION (or SUMMARY after the
     last set).
5. **TRANSITION** — short rest (~3s) with a "switch sides" cue and the next side announced,
   then the next SET's countdown. Skipped after the final set.
6. **SUMMARY** — overall score (mean of the four per-set scores) + a spoken LLM wrap-up.
   Then `DONE` (return to selector) on key/timeout.

### `RoutineEvent` types
`POSITION_OK`, `COUNTDOWN(n)`, `SET_STARTED(side)`, `SET_FEEDBACK_DUE`, `SET_COMPLETE(i)`,
`SWITCH_SIDES(next_side)`, `ROUTINE_COMPLETE`. The caller maps each to a cue (audio) and/or a
render change. `SET_FEEDBACK_DUE` fires on the existing ≥2.5s throttle and triggers an
`LLMWorker.submit(LiveSnapshot, …)`.

### `RoutineConfig` defaults
`sets=4, hold_s=25.0, order=("left","right","left","right"), position_hold_s=3.0,
countdown_s=3, transition_s=3.0`.

## Per-set scoring (reuse)

Per frame, when the view is valid: `measured = exercise.measure(pf, baseline)` →
`in_target, violations = score_frame(target, measured)`. The caller maintains, per set:
- `in_target_ms` (accumulated real time the user was in target during the 25s window),
- `max_severity_seen[joint]`.

At `SET_COMPLETE`, build a meta dict and call the existing
`rules_hold.score_hold(name, meta, target, max_severity_seen)` to get a 0–100 set score
(duration / precision / stability). `HoldFSM` is reused **only** to compute the live
`holding/drifted/entering` display state within a set; the set boundary is the wall-clock
timer, not `HoldState.COMPLETE`.

## Audio subsystem (`src/feedback/tts.py`)

### `GeminiTTS`
- `synthesize(text: str) -> bytes` returns WAV/PCM audio for Thai text via `google-genai`
  (Gemini TTS model; exact model id and request shape confirmed against current
  `google-genai` docs at implementation time — likely `gemini-2.5-flash-preview-tts`).
- On any exception (no network, API error, missing key), falls back to macOS
  `say -v Kanya -o <tmp.aiff>` then returns those bytes — **verified available** on this
  machine. If `say` also fails, raises; the worker then logs and stays silent (on-screen
  Thai text remains).

### `TTSWorker` (background thread, `afplay` playback)
Two logical channels, one playback at a time (no overlap):
- **Cue channel (priority, never dropped):** set start / switch / done / per-second
  countdown. The fixed cue phrases are **pre-synthesized once at startup** and cached as
  bytes, so they play with no synthesis latency. Cues queue and play in order; a cue may
  preempt an in-flight *feedback* clip.
- **Feedback channel (drop-stale, mirrors `LLMWorker`):** live LLM corrections. A new
  feedback request replaces any pending one. Feedback plays only when no cue is speaking and
  no other feedback is mid-playback — stale nudges are dropped rather than queued, since late
  coaching is worse than none.

Playback: write bytes to a temp WAV and `afplay` it (built into macOS, no new dependency);
track "is something playing" via the subprocess handle so the channels can coordinate.

## UI / rendering

The renderer gains screen-specific draw helpers (kept separate from the pure FSM):
- **Setup screen:** title, instructions, a drawn **Start** button; `cv2.setMouseCallback`
  hit-tests the button rect and sets a flag the loop reads.
- **Positioning:** live (mirrored) camera + a semi-transparent **humanoid outline** drawn
  programmatically (head circle, shoulder line, torso trapezoid, hip line) centered in frame.
  The outline turns **green** as the 5 required keypoints lock in; a 3s "hold position"
  progress ring fills while `pose_ready` holds.
- **Countdown:** large centered digits (global 3·2·1 and per-set 3·2·1), with voice.
- **During a set:** side badge (ซ้าย/ขวา), a 25s countdown ring, current tilt vs. target,
  and the live Thai feedback line (mirrors what is spoken).
- **Summary:** overall score + four per-set bars + spoken wrap-up.

Mirroring follows `test_2D_3D.py`: inference runs on the un-mirrored frame; only the drawn
overlay is flipped for a selfie view. `head_lateral_tilt_2d` is mirror-invariant, so the
sign convention is unaffected.

## Exercises

`NeckStretchRight` is added as a mirror of `NeckStretchLeft` (`target_deg = +35°`, same
tolerance, `side="right"`, same `valid_views`). The routine alternates the two instances by
the configured `order`. The selector shows one entry, "Neck Stretch (2 นาที)," which launches
the routine.

## Error handling

| Condition | Behavior |
| --- | --- |
| Gemini TTS network/API failure | Fall back to macOS `say`; if that also fails, log + stay silent (text remains on screen) |
| Calibration: too few clean frames | Zero baseline → absolute-angle mode (current behavior preserved) |
| Invalid camera view mid-set | On-screen + spoken "face the camera" nudge; in_target forced false so the set score reflects it (the 25s timer keeps running) |
| User presses `q` | Quit the session immediately from any phase |
| Webcam read timeout | Skip the frame; loop continues |

## Testing

- `tests/office_syndrome/test_routine.py` — pure `RoutineFSM` transitions: positioning
  reset on lost pose, 3s gate, READY countdown, set ordering L/R/L/R, fixed-25s set
  boundary, transition between sets, summary after the last set, event emission.
- `tests/office_syndrome/test_tts.py` — `GeminiTTS` fallback to `say` when `google-genai`
  raises (mocked), `TTSWorker` drop-stale feedback + cue priority/no-overlap (mocked
  playback, no real audio).
- `tests/office_syndrome/test_neck_stretch.py` — extend with `NeckStretchRight` measure /
  target sign.
- Existing pure-logic suites (`test_rules_hold`, `test_calibration`, `test_camera_view`,
  `test_hold_fsm`) continue to cover the reused pieces unchanged.

Pure-logic tests need no models or audio hardware. Live audio + Gemini calls are exercised
only by hand during the demo (and a `*_smoke` test gated on the API key, optional).

## Out of scope (YAGNI)

- The remaining 8 office-syndrome stretches (deferred content additions).
- The parked squat flow (left intact, still untouched by the entry point).
- A generic multi-exercise routine framework — this demo hardcodes the neck-stretch routine;
  generalization waits until a second routine actually exists.
- The streaming-server port — the design keeps the FSM GUI-free to enable it later, but does
  not build it now.

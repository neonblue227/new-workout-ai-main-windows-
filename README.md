# Workout AI — Real-Time Form Coach (Thai)

Real-time webcam-based form coach for macOS Apple Silicon. The current entry point is a guided neck-stretch routine: four alternating 25-second holds (left, right, left, right) with live Thai coaching and spoken cues. Joint angles are measured **directly from 2D keypoints** (nose / ears / shoulders) — robust to the seated, desk-camera framing where the 3D lift fails (see `CLAUDE.md` → "2D-direct measurement"); the 3D rig on screen is visualization-only. **Squat coaching code** is preserved in the codebase but is not currently wired to the launcher; see `CLAUDE.md` for how to re-enable it.

## Setup

```bash
uv sync --all-extras
uv run python scripts/download_models.py     # ~6 GB, one-time
```

## Run

```bash
uv run python main.py
```

Usage:

- **Start screen**: click the on-screen Start button to begin the routine.
- **Positioning**: stand so the camera clearly sees your head, shoulders, and hips. A humanoid outline guides placement. Hold still for 3 seconds — this window also captures your neutral posture for per-user calibration.
- **Countdown**: a 3·2·1 countdown plays with a spoken Thai side cue (e.g. "เอียงซ้าย").
- **Hold (25 s)**: tilt your neck to the indicated side and hold. Live Thai coaching plays during the hold. The 25-second timer counts down in real time regardless of your form — your tilt accuracy drives the live coaching and the set score, not the duration.
- **Routine**: four sets alternate left → right → left → right. A summary screen shows at the end.
- **Quit**: press `q` or Esc at any time to exit.

**TTS**: spoken cues use Google AI Studio (`gemini-2.5-flash-preview-tts`). Set `google_ai_studio_api_key` in a `.env` file at the repo root. If the key is absent or the request fails, the app falls back to the offline macOS `say -v Kanya` voice automatically.

### Use a phone as the camera (IP Webcam over LAN)

Install the **IP Webcam** Android app, start its server, and note the URL it shows (e.g. `http://192.168.1.104:8080`). Then, from a machine on the same network:

```bash
uv run python -m camera --url http://192.168.1.104:8080/video                 # preview the stream first (q/Esc to quit)
uv run python main.py --source ip --url http://192.168.1.104:8080/video       # run the routine on the phone feed
```

Frames are letterboxed to the app's working size, so pose angles stay correct regardless of the phone's resolution. With no `--source` flag, both entry points default to the local webcam.

Diagnostic / debugging:

```bash
uv run python src/test_2D_3D.py                                               # live 3-panel view: camera / 2D pose + metrics / 3D rig, with profiling
uv run python src/test_2D_3D.py --source ip --url http://192.168.1.104:8080/video   # same diagnostic, fed from the phone
uv run python scripts/profile_pipeline.py                                     # headless per-stage timing (CPU%, ms/stage)
```

## Development

```bash
uv run pytest                       # full test suite
uv run pytest -k "not smoke"        # skip tests that load heavy model weights
uv run pytest tests/office_syndrome # one section (pipeline / squat / office_syndrome)
uv run ruff check                   # lint
uv run ruff format                  # format
```

## Stack

| Layer | Model | Framework |
|---|---|---|
| 2D pose | YOLOX-m + RTMPose-m (balanced; default) | rtmlib (ONNX) on **CoreML / Neural Engine** |
| 3D lift (visualization only) | MotionBERT-Lite | PyTorch on MPS |
| Form analysis | **2D-direct** joint angles + per-user calibration + camera-view gating + phase FSM | pure Python |
| Thai feedback | Qwen3.5-4B (mxfp4) | mlx-vlm |

The 2D model defaults to the balanced tier on the Neural Engine (54 fps; far more accurate keypoints than the old lightweight+CPU path). CPU stays cool — ONNX threads are pinned to 2 and inference is throttled to ~15 Hz. See `docs/perf/2026-05-23-coreml-experiment.md`.

## Acceptance criteria

Shared pipeline (both modes):

- [x] Skeleton overlay ≥ 25 FPS on M-series (balanced + CoreML ≈ 54 fps; full per-frame infer+lift ≈ 23 ms).
- [x] 3D rig updates at ≥ 5 Hz (6 Hz in `app`, up to 30 Hz in the diagnostic).
- [x] Models live in `./models/` and load from disk on subsequent runs.

### Neck-stretch routine (current launcher mode)

- [x] Guided four-set routine (left, right, left, right) driven by `RoutineFSM` (SETUP → POSITIONING → COUNTDOWN → HOLD → TRANSITION → SUMMARY).
- [x] 3-second positioning + calibration window captures the user's neutral posture; targets scored as a delta from neutral.
- [x] Spoken Thai cues at each transition (`GeminiTTS` with macOS `say -v Kanya` offline fallback).
- [x] Hold FSM tracks `IDLE / ENTERING / HOLDING / DRIFTED / COMPLETE` with pause-on-drift timing and a stability penalty on drift count.
- [x] 100-pt score per set (50 duration / 30 precision / 20 stability) + Thai summary from the on-device VLM.
- [x] Live Thai nudges during the hold (throttled to ~1 every 3–4 s by Qwen latency).
- [x] **2D-direct measurement** (nose+shoulders), robust to seated/desk-camera framing where the 3D lift is structurally invalid.
- [x] **Camera-view gating** — exercise declares `valid_views`; FSM stays IDLE with a "rotate" coaching message in an unsupported view.

### Squat (preserved code, not currently exposed)

The original `SquatFSM` + `rules_squat.score_rep` + `RepAnalysis` flow is intact in the codebase. The Thai LLM still handles `RepAnalysis` payloads via the dispatch in `ThaiCoachLLM.generate`. The acceptance bar from the original spec — ±1 rep over a 10-rep set, 2–3 sentence Thai feedback within 3 s of rep completion — is unchanged on the code path, but no entry point currently routes to it.

Design specs:

- [Office syndrome stretches](docs/superpowers/specs/2026-05-22-office-syndrome-stretches-design.md) — current launcher mode.
- [3D scoring upgrade](docs/superpowers/specs/2026-05-19-3d-scoring-design.md).
- [Original squat coach](docs/superpowers/specs/2026-05-18-pose-form-coach-design.md).

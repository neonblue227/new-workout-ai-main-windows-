# Real-Time Squat Form Coach — Design (v1)

**Date:** 2026-05-18
**Status:** Approved — proceeding to implementation plan
**Target platform:** macOS, Apple Silicon (M-series)

## Goal

Build a real-time webcam application that:

1. Estimates the user's 2D pose, lifts it to 3D, and renders a live 3D rig overlay.
2. Analyzes squat form against a rule set, phase-based finite state machine, and a model-attention signal.
3. Displays a numerical form score (0–100) per rep, updated live.
4. Generates Thai-language coaching feedback via an on-device LLM, triggered at rep boundaries.

## Stack

| Layer | Tool | Role |
|---|---|---|
| Camera | OpenCV `cv2.VideoCapture` | Webcam capture, frame loop |
| 2D pose | **RTMPose-l** via `rtmlib` | 2D keypoints (COCO-17 or Halpe-26), ONNX runtime on CoreML/CPU |
| 3D lift | **MotionBERT** (official checkpoint) | 2D→3D lifting; PyTorch on MPS backend |
| Form analysis | Three techniques run together | (a) Joint-angle rules, (b) Phase-based FSM, (c) GradCAM on RTMPose heatmap |
| Score | Weighted aggregate 0–100 | Per-rep + running average overlay |
| Renderer | OpenCV draw calls | 2D skeleton overlay + 3D rig in a side panel (orthographic projection) |
| LLM feedback | **Qwen3.5-4B** mxfp4 via `mlx-vlm` | Async; triggered at rep boundaries, never per-frame |
| Output | Thai text in a sidebar + console | Generated from structured analysis (image input optional, second pass) |

### Verified facts

- `Qwen/Qwen3.5-4B` (Feb 2026 release) is a vision-language model with a vision encoder, image-text-to-text. `mlx-vlm` is the correct framework.
- Closest Q4-equivalent MLX-VLM build: `RepublicOfKorokke/Qwen3.5-4B-mlx-vlm-mxfp4`. Fallbacks: `mlx-community/Qwen3.5-4B-MLX-8bit`, or self-quantize via `mlx_vlm.convert`.
- Weights download to `models/` (gitignored). Path is project-local per the user's request.

## Architecture

```
src/
├── capture.py          # webcam thread, frame queue
├── pose2d.py           # rtmlib RTMPose-l wrapper -> 17 keypoints + heatmaps
├── pose3d.py           # MotionBERT wrapper, sliding window of 2D frames -> 3D
├── analysis/
│   ├── angles.py       # knee/hip/ankle angle computation
│   ├── phases.py       # squat FSM: standing -> descent -> bottom -> ascent -> standing
│   ├── gradcam.py      # GradCAM on RTMPose backbone, attention overlay
│   └── rules_squat.py  # squat-specific rule set + per-rep scoring
├── feedback/
│   ├── llm.py          # mlx-vlm Qwen3.5-4B wrapper, async worker
│   └── prompt_th.py    # Thai prompt template
├── render.py           # OpenCV overlay: skeleton, 3D rig panel, score, attention heatmap, Thai text
└── app.py              # main loop, ties it all together
models/                 # downloaded weights (gitignored)
docs/superpowers/specs/  # this document
```

## Real-Time Strategy

The three perception layers run at different cadences so the video loop never stalls.

- **Per-frame (~30 FPS target):** 2D pose, joint-angle computation, phase FSM update, score update, render. RTMPose-l on M-series CoreML should hit 30 FPS at 256×192 input.
- **Sliding window (every 5 frames):** MotionBERT 3D lift over a 27- or 81-frame buffer of 2D keypoints. 3D rig in the side panel updates at ~6 Hz, which is enough for visual feedback.
- **Per-rep (async):** at the bottom-of-rep frame, capture a GradCAM heatmap, build a structured summary (angles, rule violations, phase timings), and enqueue it for the LLM. LLM runs on a background thread; Thai feedback appears 1–3 s after rep completion. The video loop never blocks on LLM inference.

## Scoring

A single 0–100 form score per rep, weighted:

| Component | Weight | Rule |
|---|---|---|
| Depth | 30 | Hip Y ≥ knee Y at bottom (hip below knee), tolerance ±5% of frame height |
| Knee valgus | 25 | Knee X within ±15% of ankle X (knees track over toes) |
| Torso angle | 20 | Shoulder-hip vector forms 20°–55° with vertical |
| Symmetry | 15 | Abs(L knee angle − R knee angle) < 10° at bottom |
| Tempo | 10 | Descent duration ≥ ascent duration |

A running average is shown top-left; the per-rep score flashes at the bottom-of-rep moment.

## Squat-Specific Rules (v1)

- Knee angle at bottom ≤ ~90° (with tolerance)
- Hip-to-knee Y delta ≥ 0 at bottom (hip below knee)
- Knee X within ±15% of ankle X (valgus)
- Torso lean 20°–55° from vertical
- Reps detected by phase FSM crossing standing → bottom → standing

Rules live in `analysis/rules_squat.py`. Adding push-up or plank later is a new sibling file plus a registration entry; no refactor required.

## LLM Integration

- **Trigger:** rep completion (FSM enters standing after passing bottom).
- **Input on first pass:** structured analysis text only — phase timings, per-component scores, list of rule violations. Cheaper, faster, predictable.
- **Input on second pass (optional):** include the bottom-of-rep frame as image input to use the vision encoder. Defer until v1.1.
- **Prompt template:** Thai system prompt instructing the model to give a short critique (2–3 sentences) covering what was correct and what to fix.
- **Threading:** `concurrent.futures.ThreadPoolExecutor(max_workers=1)`. Newer reps cancel pending ones in the queue.

## Defaults Confirmed With User

1. Display: single OpenCV window with the 3D rig in a side panel.
2. Camera: built-in webcam, side view (essential for depth assessment).
3. Exercise scope: squat only for v1, modular for later.
4. LLM quant: start with `RepublicOfKorokke/Qwen3.5-4B-mlx-vlm-mxfp4`, fall back to 8-bit if it misbehaves.
5. LLM input mode: structured text on first pass.
6. Apple Silicon assumed.

## Out of Scope (v1)

- Multiple users in frame
- Multi-camera fusion
- Rep history / session log persistence
- Mobile or non-macOS targets
- Voice output (Thai TTS) — text only for v1
- Exercises other than squat — push-up, plank, deadlift come later
- Cloud LLM fallback

## Risks

- **MotionBERT inference time on MPS:** if it can't keep up at 6 Hz, fall back to running it only at the bottom-of-rep frame for the LLM context, and use 2D skeleton for the live rig (drop "live 3D" claim).
- **rtmlib + CoreML on Apple Silicon:** if RTMPose-l ONNX doesn't load on CoreML EP, fall back to CPU EP at reduced resolution, or drop to RTMPose-m.
- **Qwen3.5-4B mxfp4 quality for Thai:** if Thai output quality is poor, switch to 8-bit at the cost of ~2× memory.
- **mlx-vlm cold-start latency:** first inference can be 5–10 s. Warm the model at app start with a dummy prompt.
- **GradCAM on RTMPose's heatmap head:** the standard GradCAM formulation targets classification heads; applying it to a regression/heatmap head requires using the heatmap activations as the target. May need a custom implementation rather than off-the-shelf `pytorch-grad-cam`.

## Acceptance Criteria (v1)

- App launches, opens webcam, draws a 2D skeleton overlay at ≥ 25 FPS on M2 or better.
- 3D rig panel updates at ≥ 5 Hz with visibly correct hip/knee/ankle articulation.
- Squat reps are detected within ±1 rep over a 10-rep set.
- A 0–100 form score is shown per rep and as a running average.
- Within 3 s of rep completion, a 2–3 sentence Thai critique appears in the sidebar.
- Models download on first run with progress visible; subsequent runs load from disk.

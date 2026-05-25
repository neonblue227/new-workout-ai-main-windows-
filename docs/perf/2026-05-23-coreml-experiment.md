# CoreML Execution Provider experiment — 2026-05-23

> **TL;DR (corrected).** The first pass concluded "negative — do not adopt." That
> was **incomplete**: it tested only the lightweight model, with detection on CPU,
> using the default `NeuralNetwork` CoreML format. A second pass with `MLProgram`
> format + `RequireStaticInputShapes=1` (which also fixes the YOLOX zero-detection
> crash) + both sessions on CoreML shows CoreML/ANE **wins across the board, and
> the win scales hard with model size**. `accelerator="coreml"` is now a supported
> `Pose2D` option. The lightweight default stays on CPU; CoreML is the right choice
> the moment you move to `balanced` / `performance` for accuracy.

## Environment
- macOS 26.5 / ARM64 / M-series (18 logical cores)
- `onnxruntime` **1.26.0**, `rtmlib` 0.0.15

---

## Pass 1 (initial, incomplete) — "negative"

Tested: lightweight only, detection on CPU, RTMPose-only on CoreML, default
`NeuralNetwork` format.

- Full `body()` pipeline pure CPU (threads=2): **14.97 ms/frame**
- Full pipeline det-CPU + pose-CoreML: **15.21 ms/frame** → 2% *slower*
- `device="mps"` on the whole Body crashed YOLOX on zero detections:
  `Input (...) has a dynamic shape ({-1}) but the runtime shape ({0}) has zero elements`.

Concluded "do not adopt." **This conclusion did not generalize** — see Pass 2.

---

## Pass 2 (corrected) — the real picture

### The crash fix
`RequireStaticInputShapes="1"` keeps the dynamic-shape YOLOX NMS subgraph on the
CPU EP (via partitioning) while the conv backbone runs on CoreML. Verified: both
YOLOX-tiny and YOLOX-m run on CoreML EP and survive **zero-detection (blank image)
inputs** — the exact case that crashed in Pass 1. Pair with `ModelFormat=MLProgram`
(modern CoreML format, proper dynamic-shape handling).

### Isolated RTMPose inference (CPU threads=2 vs CoreML MLProgram)

| Model | Input | CPU | CoreML | Speedup |
|---|---|---:|---:|---:|
| RTMPose-s | 256×192 | 3.42 ms | 2.25 ms | 1.52× |
| RTMPose-m | 256×192 | 9.06 ms | 3.38 ms | 2.68× |
| RTMPose-x | 384×288 | 65.94 ms | 7.67 ms | 8.60× |

CoreML's per-inference overhead is fixed; compute scales — so the bigger the
model, the more ANE/GPU wins. Exactly the expected crossover behavior.

### Full pipeline (`Pose2D.infer()` = detection + pose + decode, real image)

| Mode | Accel | Median ms | FPS | CoreML speedup |
|---|---|---:|---:|---:|
| lightweight (YOLOX-tiny + RTMPose-s) | CPU | 14.71 | 68 | — |
| lightweight | CoreML | 11.80 | 85 | **1.25×** |
| balanced (YOLOX-m + RTMPose-m) | CPU | 114.43 | 8.7 | — |
| **balanced** | **CoreML** | **18.57** | **54** | **6.16×** |
| performance (YOLOX-x + RTMPose-x) | CPU | 415.57 | 2.4 | — |
| performance | CoreML | 89.20 | 11.2 | 4.66× |

Key reads:
- **Even lightweight is 1.25× faster** on CoreML in the full pipeline (Pass 1's
  "2% slower" was an artifact of the NeuralNetwork format + det-on-CPU).
- **Balanced is unusable on CPU (8.7 fps) but real-time on CoreML (54 fps)** — CoreML
  is the *enabling condition* for the higher-accuracy model, not just an optimization.
- Performance mode's lower 4.66× is because RTMPose-x has a dynamic input dim that
  emits `unbounded dimension` MLProgram compile warnings and partially falls back to
  CPU. Static-shaping it (`python -m onnxruntime.tools.make_dynamic_shape_fixed`)
  before loading would recover more. RTMPose-s/m compile clean.

## Decision

- **Keep `accelerator="cpu"` as the default** for the lightweight model. The CoreML
  win there (1.25×) is real but small, and CPU avoids the CoreML compile/warmup cost
  at startup and any fp16-on-ANE precision difference.
- **Use `accelerator="coreml"` whenever moving to `balanced` / `performance`.** That's
  the regime where ANE turns an unusable model into a real-time one, which matters if
  the 2D-direct posture metrics (CVA, forward-head, shoulder asymmetry — see plan B2)
  need more precise ear/shoulder localization than RTMPose-s provides.
- Caveats to validate before flipping any default: (1) CoreML model compile adds
  first-inference latency; (2) `MLComputeUnits=ALL` may use fp16 on ANE — confirm
  keypoint precision is adequate for the measurement tolerances; (3) static-shape
  RTMPose-x for a clean performance-mode path.

`Pose2D(mode=..., accelerator="coreml")` is wired and tested (no zero-detection
crash, simcc heatmap capture preserved). The CPU-side knobs from the A-workstream
(`threads=2`, 15 Hz inference cadence) still apply as the fallback / CPU-op budget.

"""Headless per-stage timing harness for the realtime pose pipeline.

Drives Pose2D + Pose3D + the diagnostic 3-panel canvas composition against a
static image fixture in a tight loop. Reports mean / p95 wall-clock per stage
plus aggregate CPU%, so subsequent optimization work can compare against this
baseline objectively.

Usage:

    uv run python scripts/profile_pipeline.py
    uv run python scripts/profile_pipeline.py --frames 1200 --warmup 50
    uv run python scripts/profile_pipeline.py --output docs/perf/2026-05-23-baseline.md

Notes:
- Uses the same fixture as `tests/office_syndrome/test_exercise_pipeline_smoke.py`
  so results are comparable across runs.
- 3D lift runs every Nth frame (matches `app.run_session` / `test_2D_3D.py`'s
  `LIFT_EVERY_N_FRAMES = 5`).
- `cv2.imshow` is intentionally NOT exercised here — that requires a window
  server and adds host-dependent variance. The non-display pipeline cost is
  what we want to optimize first; imshow can be measured separately later if
  the perf budget is unclear after A2–A5.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FIXTURE = PROJECT_ROOT / "data" / "neck_stretch" / "neck_stretch_01.jpg"


def _md_row(name: str, samples: list[float], call_count: int) -> str:
    if not samples:
        return f"| {name:<30s} | {call_count:>8d} | — | — |"
    mean_ms = statistics.mean(samples) * 1000
    sorted_v = sorted(samples)
    p95_ms = sorted_v[int(0.95 * (len(sorted_v) - 1))] * 1000
    return f"| {name:<30s} | {call_count:>8d} | {mean_ms:7.2f} | {p95_ms:7.2f} |"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--frames",
        type=int,
        default=600,
        help="number of frames to profile (after warmup)",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="frames to run before starting measurement",
    )
    ap.add_argument(
        "--lift-every",
        type=int,
        default=5,
        help="how often to run the 3D lift (matches LIFT_EVERY_N_FRAMES)",
    )
    ap.add_argument(
        "--pose-stride",
        type=int,
        default=1,
        help="run pose2d.infer() every Nth iteration (matches A3 throttling). "
        "stride=1: every frame (default); stride=2: half-rate (15 Hz at 30 fps cam).",
    )
    ap.add_argument(
        "--onnx-threads",
        type=int,
        default=None,
        help="optional override for ONNX intra_op thread count. "
        "None = library default (typically num_cpu_cores).",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="if set, append a markdown summary to this file",
    )
    ap.add_argument(
        "--label",
        type=str,
        default="baseline",
        help="label this run in the markdown output (e.g. 'baseline', "
        "'onnx_threads=4')",
    )
    args = ap.parse_args()

    assert FIXTURE.exists(), f"missing fixture: {FIXTURE}"

    from analysis.angles import head_lateral_tilt_2d
    from pose2d import Pose2D
    from pose3d import Pose3D, Pose3DBuffer, coco17_to_h36m17

    # Optional: thread tuning hook for A2. Not used in baseline run.
    if args.onnx_threads is not None:
        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = args.onnx_threads
        sess_opts.inter_op_num_threads = 1
        # The cleanest way to apply this to rtmlib is to monkey-patch
        # InferenceSession construction at the ort layer so the next sessions
        # created (by Pose2D below) pick it up.
        _orig_session_init = ort.InferenceSession.__init__

        def _patched_session_init(self, *a, **kw):  # type: ignore[no-untyped-def]
            kw.setdefault("sess_options", sess_opts)
            return _orig_session_init(self, *a, **kw)

        ort.InferenceSession.__init__ = _patched_session_init  # type: ignore[method-assign]

    print(
        f"[profile] platform: {platform.platform()}  python: {platform.python_version()}"
    )
    print(f"[profile] fixture: {FIXTURE.relative_to(PROJECT_ROOT)}")
    print(
        f"[profile] frames: {args.frames}  warmup: {args.warmup}  lift_every: {args.lift_every}"
    )
    if args.onnx_threads is not None:
        print(f"[profile] ONNX intra_op_num_threads override: {args.onnx_threads}")

    print("[profile] loading models...")
    t0 = time.time()
    pose2d = Pose2D(device="cpu", mode="lightweight")
    pose3d = Pose3D()
    buf = Pose3DBuffer(pose3d)
    print(f"[profile] models loaded in {time.time() - t0:.1f}s")

    img = cv2.imread(str(FIXTURE))
    assert img is not None
    H, W = img.shape[:2]

    # Warmup: prime ORT/CoreML and fill the 3D buffer.
    print(f"[profile] warmup ({args.warmup} frames)...")
    for _ in range(args.warmup):
        kps, scores = pose2d.infer(img)
        h36m = coco17_to_h36m17(kps, scores)
        buf.push(h36m)
        if buf.ready():
            _ = buf.lift(frame_h=H, frame_w=W)

    # Per-stage sample buckets.
    samples: dict[str, list[float]] = {
        "pose2d.infer (2D detection)": [],
        "coco17_to_h36m17": [],
        "Pose3DBuffer.push": [],
        "Pose3DBuffer.lift (3D, every Nth)": [],
        "head_lateral_tilt_2d": [],
        "canvas_compose (3-panel 1920x560)": [],
        "frame_total (sum of stages)": [],
    }
    counts = {k: 0 for k in samples}
    last_rig_3d: np.ndarray | None = None

    # CPU% sampling: start a baseline reading, then read again at the end.
    proc = psutil.Process()
    proc.cpu_percent(interval=None)  # prime
    t_wall_start = time.time()

    print(f"[profile] measuring ({args.frames} frames)...")
    # Pre-prime so the throttled branch always has a usable last-result.
    last_kps, last_scores = pose2d.infer(img)
    for i in range(args.frames):
        frame_t0 = time.perf_counter()

        if i % args.pose_stride == 0:
            s = time.perf_counter()
            last_kps, last_scores = pose2d.infer(img)
            samples["pose2d.infer (2D detection)"].append(time.perf_counter() - s)
            counts["pose2d.infer (2D detection)"] += 1

            s = time.perf_counter()
            h36m = coco17_to_h36m17(last_kps, last_scores)
            samples["coco17_to_h36m17"].append(time.perf_counter() - s)
            counts["coco17_to_h36m17"] += 1

            s = time.perf_counter()
            buf.push(h36m)
            samples["Pose3DBuffer.push"].append(time.perf_counter() - s)
            counts["Pose3DBuffer.push"] += 1
        kps, scores = last_kps, last_scores

        if buf.ready() and i % args.lift_every == 0:
            s = time.perf_counter()
            last_rig_3d = buf.lift(frame_h=H, frame_w=W)
            samples["Pose3DBuffer.lift (3D, every Nth)"].append(time.perf_counter() - s)
            counts["Pose3DBuffer.lift (3D, every Nth)"] += 1

        s = time.perf_counter()
        _ = head_lateral_tilt_2d(kps, scores)
        samples["head_lateral_tilt_2d"].append(time.perf_counter() - s)
        counts["head_lateral_tilt_2d"] += 1

        s = time.perf_counter()
        # Mimic the 3-panel canvas composition used in test_2D_3D.py.
        cam_panel = np.full((H + 80, W, 3), 22, dtype=np.uint8)
        cam_panel[40 : 40 + H, :W] = img
        two_d_panel = cam_panel.copy()
        cv2.line(two_d_panel, (100, 100), (200, 200), (255, 200, 0), 2)
        three_d_panel = np.full((H + 80, W, 3), 22, dtype=np.uint8)
        if last_rig_3d is not None:
            cv2.rectangle(three_d_panel, (20, 40), (W - 20, H + 20), (60, 60, 60), 1)
        _ = np.hstack([cam_panel, two_d_panel, three_d_panel])
        samples["canvas_compose (3-panel 1920x560)"].append(time.perf_counter() - s)
        counts["canvas_compose (3-panel 1920x560)"] += 1

        samples["frame_total (sum of stages)"].append(time.perf_counter() - frame_t0)
        counts["frame_total (sum of stages)"] += 1

    t_wall_end = time.time()
    wall_s = t_wall_end - t_wall_start
    cpu_pct = proc.cpu_percent(interval=None)
    n_cores = psutil.cpu_count(logical=True) or 1

    # Console summary.
    print()
    print(f"=== Per-stage timings (label: {args.label}) ===")
    print(f"{'stage':<35s} {'calls':>6s}   {'mean ms':>8s}   {'p95 ms':>8s}")
    print("-" * 70)
    for name, vals in samples.items():
        n = counts[name]
        if not vals:
            print(f"{name:<35s} {n:>6d}      --         --")
        else:
            mean_ms = statistics.mean(vals) * 1000
            sorted_v = sorted(vals)
            p95_ms = sorted_v[int(0.95 * (len(sorted_v) - 1))] * 1000
            print(f"{name:<35s} {n:>6d}   {mean_ms:7.2f}   {p95_ms:7.2f}")
    print()
    print(
        f"wall-clock total: {wall_s:.2f} s   throughput: {args.frames / wall_s:.1f} fps"
    )
    print(
        f"process CPU%:    {cpu_pct:7.1f}%   (of one core; {cpu_pct / n_cores:.1f}% of {n_cores}-core total)"
    )

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        header_needed = not out_path.exists()
        with out_path.open("a", encoding="utf-8") as f:
            if header_needed:
                f.write("# Pipeline performance baseline\n\n")
                f.write(
                    f"Platform: `{platform.platform()}` · Python {platform.python_version()}\n\n"
                )
            f.write(f"## {args.label}\n\n")
            f.write(f"- Frames measured: **{args.frames}** (warmup {args.warmup})\n")
            f.write(f"- 3D lift cadence: every **{args.lift_every}** frames\n")
            if args.onnx_threads is not None:
                f.write(f"- ONNX intra_op_num_threads: **{args.onnx_threads}**\n")
            if args.pose_stride > 1:
                f.write(
                    f"- Pose inference stride: every **{args.pose_stride}** frames\n"
                )
            f.write(
                f"- Wall-clock total: **{wall_s:.2f} s** · throughput **{args.frames / wall_s:.1f} fps**\n"
            )
            f.write(
                f"- Process CPU%: **{cpu_pct:.1f}%** of one core "
                f"(≈ **{cpu_pct / n_cores:.1f}%** of {n_cores}-core total)\n\n"
            )
            f.write("| stage | calls | mean ms | p95 ms |\n")
            f.write("|---|---:|---:|---:|\n")
            for name, vals in samples.items():
                f.write(_md_row(name, vals, counts[name]) + "\n")
            f.write("\n")
        print(f"[profile] appended to {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

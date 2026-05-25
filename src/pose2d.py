from pathlib import Path
from typing import Literal, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# rtmlib hardcodes its cache to ~/.cache/rtmlib via TORCH_HOME/XDG_CACHE_HOME and
# ignores any RTMLIB_CACHE env var. Redirect it to the in-repo models/ folder by
# patching the resolver before the first Body() instantiation triggers a download.
import rtmlib.tools.file as _rtmlib_file  # noqa: E402

_RTMLIB_HUB = PROJECT_ROOT / "models" / "rtmlib_cache" / "hub"
_rtmlib_file._get_rtmhub_dir = lambda: str(_RTMLIB_HUB)


# Default ONNX intra-op thread count for both detection and pose sessions.
# ORT's default is `num_logical_cpus`, which thrashes on an M-series and is
# objectively slower than 2-4 threads for our tiny models. Sweep on 2026-05-23
# (M-series, 18 logical cores, onnxruntime 1.26.0) — see
# docs/perf/2026-05-23-baseline.md:
#   threads=default (18) → 53 fps, 1247% process CPU
#   threads=2            → 59 fps, 239% process CPU  (+11% throughput, -81% CPU)
#   threads=4            → 61 fps, 527% process CPU
# Two threads is the sweet spot for our real-time target.
_ONNX_THREADS_DEFAULT = 2

# CoreML EP provider config that (a) routes the conv backbone onto the Apple
# Neural Engine / GPU and (b) keeps the dynamic-shape YOLOX NMS subgraph on CPU
# via `RequireStaticInputShapes=1`, which sidesteps the zero-detection crash
# (CoreML EP can't handle a {-1} tensor that becomes {0} at runtime). MLProgram
# is the modern CoreML format with proper dynamic-shape handling. Benchmark on
# 2026-05-23 (onnxruntime 1.26.0) shows ANE scales hard with model size:
# RTMPose-s 1.5x / RTMPose-m 2.7x / RTMPose-x 8.6x faster than CPU+threads=2.
# See docs/perf/2026-05-23-coreml-experiment.md.
#
# `ModelCacheDirectory` persists the compiled .mlmodelc so CoreML doesn't
# recompile both models on every process start (the dominant chunk of first-
# inference latency / startup CPU). ORT keys the cache on the model file-path
# hash, so the detector and pose sessions don't collide. NOTE: ORT never
# invalidates this cache — if you swap a model file or change EP options, clear
# the directory. `SpecializationStrategy=FastPrediction` optimizes the compiled
# model for prediction latency over load time (paid once, then cached).
_COREML_CACHE_DIR = PROJECT_ROOT / "models" / "coreml_cache"
_COREML_PROVIDER = (
    "CoreMLExecutionProvider",
    {
        "RequireStaticInputShapes": "1",
        "ModelFormat": "MLProgram",
        "MLComputeUnits": "ALL",
        "ModelCacheDirectory": str(_COREML_CACHE_DIR),
        "SpecializationStrategy": "FastPrediction",
    },
)

Accelerator = Literal["cpu", "coreml"]


def _rebuild_session(sub_model, providers: list, intra_op_threads: int) -> None:
    """Replace `sub_model.session` with an equivalent InferenceSession using the
    given execution-provider list + a constrained CPU thread pool. Works against
    rtmlib's YOLOX / RTMPose wrappers where `sub_model.session` and
    `sub_model.onnx_model` are public attributes."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_op_threads
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3  # silence CoreML compile chatter
    sub_model.session = ort.InferenceSession(
        sub_model.onnx_model,
        sess_options=opts,
        providers=providers,
    )


class Pose2D:
    """Wraps rtmlib's RTMPose. Single-person inference: returns the highest-score person.
    Optionally returns simcc-decoded heatmaps via `infer_with_heatmaps`.

    Model sizes (full-pipeline median ms on M-series, onnxruntime 1.26.0 —
    see docs/perf/2026-05-23-coreml-experiment.md):
      - `mode="lightweight"` (YOLOX-tiny + RTMPose-s): 14.7 ms CPU / 11.8 ms CoreML
      - `mode="balanced"` (YOLOX-m + RTMPose-m): 114 ms CPU / 18.6 ms CoreML  ← DEFAULT
      - `mode="performance"` (YOLOX-x + RTMPose-x): 416 ms CPU / 89 ms CoreML

    Default is `mode="balanced"` + `accelerator="coreml"`: at 54 fps it's well above
    the live 15 Hz inference target with far better keypoint accuracy than lightweight,
    and the conv compute runs on the Neural Engine instead of the CPU. balanced is
    unusable on CPU (8.7 fps), so the default only makes sense paired with CoreML.

    `accelerator` selects the execution provider for both sessions:
      - `"cpu"` (default): CPUExecutionProvider with `onnx_threads` intra-op threads.
        Best for the lightweight model — the conv backbone is small enough that
        CoreML's per-inference overhead isn't worth it.
      - `"coreml"`: routes the conv backbone onto the Apple Neural Engine / GPU
        while keeping the dynamic-shape YOLOX NMS subgraph on CPU
        (`RequireStaticInputShapes=1`) to dodge the zero-detection crash. Wins
        grow with model size — pick this when running `balanced` / `performance`.
        See `_COREML_PROVIDER` and `docs/perf/2026-05-23-coreml-experiment.md`.

    `onnx_threads` constrains the ONNX intra-op thread pool of both sessions.
    Defaults to `_ONNX_THREADS_DEFAULT = 2`, which is faster AND uses ~80% less
    CPU than ORT's default of "all cores" on M-series for our tiny models. With
    `accelerator="coreml"` this still bounds the CPU-side ops (NMS, fallbacks).

    `device` is retained for signature compatibility but is no longer used for
    EP selection — sessions are always rebuilt below per `accelerator`.
    """

    def __init__(
        self,
        device: str = "cpu",
        mode: Literal["lightweight", "balanced", "performance"] = "balanced",
        onnx_threads: int = _ONNX_THREADS_DEFAULT,
        accelerator: Accelerator = "coreml",
    ):
        from rtmlib import Body

        # Always construct on CPU; sessions are rebuilt below per `accelerator`.
        # (Constructing rtmlib's Body with device="mps" can crash YOLOX during
        # inference; routing CoreML via session rebuild gives us full control of
        # the provider options needed to avoid that.)
        self._body = Body(
            mode=mode, to_openpose=False, backend="onnxruntime", device="cpu"
        )

        if accelerator == "coreml":
            _COREML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            providers: list = [_COREML_PROVIDER, "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        threads = onnx_threads if onnx_threads > 0 else _ONNX_THREADS_DEFAULT
        _rebuild_session(self._body.det_model, providers, threads)
        _rebuild_session(self._body.pose_model, providers, threads)

        # rtmlib 0.0.15 exposes the pose estimator as `pose_model`
        self._pose = getattr(self._body, "pose_model", None)

        if self._pose is not None:
            # Monkey-patch inference to capture raw simcc outputs for attention overlay.
            pose_model = self._pose
            _orig_inference = pose_model.inference

            def _capturing_inference(image):
                result = _orig_inference(image)
                try:
                    # inference returns [simcc_x, simcc_y] — each shape (1, N_kpts, simcc_bins)
                    if isinstance(result, (list, tuple)) and len(result) == 2:
                        pose_model._last_simcc = result
                except Exception:
                    pass
                return result

            pose_model.inference = _capturing_inference

    def infer(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (keypoints (17,2) float32, scores (17,) float32) for the most prominent person."""
        keypoints, scores = self._body(image_bgr)
        if len(keypoints) == 0:
            return (
                np.zeros((17, 2), dtype=np.float32),
                np.zeros((17,), dtype=np.float32),
            )
        idx = int(np.argmax(scores.sum(axis=1)))
        return keypoints[idx].astype(np.float32), scores[idx].astype(np.float32)

    def infer_with_heatmaps(
        self, image_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Returns (keypoints, scores, heatmaps) where heatmaps is (17, H, W) reconstructed from simcc.
        If rtmlib version does not expose simcc outputs, heatmaps will be None."""
        kps, scores = self.infer(image_bgr)
        heatmaps = None
        if self._pose is not None:
            try:
                simcc_pair = getattr(self._pose, "_last_simcc", None)
                if simcc_pair is not None:
                    simcc_x, simcc_y = simcc_pair
                    # simcc_x: (1, N_kpts, W_bins), simcc_y: (1, N_kpts, H_bins)
                    sx = simcc_x[0]  # (N_kpts, W_bins)
                    sy = simcc_y[0]  # (N_kpts, H_bins)
                    hms = []
                    for k in range(sx.shape[0]):
                        # outer product: rows=H_bins, cols=W_bins
                        hm = np.outer(sy[k], sx[k])
                        hms.append(hm)
                    heatmaps = np.stack(hms, axis=0).astype(np.float32)
            except Exception:
                heatmaps = None
        return kps, scores, heatmaps

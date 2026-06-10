from pathlib import Path
from typing import Literal, Optional
import os
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Put torch's CUDA runtime DLLs on PATH so onnxruntime-gpu can find cublas/cudnn.
# PyTorch ships cublasLt64_12.dll, cudnn64_9.dll, etc. in torch/lib/.
# onnxruntime-gpu finds them via PATH but they aren't there by default on Windows.
_TORCH_LIB = PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
if _TORCH_LIB.exists():
    os.environ.setdefault("PATH", "")
    os.environ["PATH"] = str(_TORCH_LIB) + os.pathsep + os.environ["PATH"]

# rtmlib hardcodes its cache to ~/.cache/rtmlib via TORCH_HOME/XDG_CACHE_HOME and
# ignores any RTMLIB_CACHE env var. Redirect it to the in-repo models/ folder by
# patching the resolver before the first Body() instantiation triggers a download.
import rtmlib.tools.file as _rtmlib_file  # noqa: E402

_RTMLIB_HUB = PROJECT_ROOT / "models" / "rtmlib_cache" / "hub"
_rtmlib_file._get_rtmhub_dir = lambda: str(_RTMLIB_HUB)


# Default ONNX intra-op thread count — only used in the CPU fallback path when
# onnx2torch is unavailable. ORT's default is `num_logical_cpus`, which is
# objectively slower than 2-4 threads for our tiny models.  Sweep on 2026-05-23
# (M-series, 18 logical cores): threads=2 → 59 fps, 239% CPU (+11% vs default).
_ONNX_THREADS_DEFAULT = 2

# CoreML EP provider config (macOS only — Apple Neural Engine).
# `RequireStaticInputShapes=1` keeps the dynamic-shape YOLOX NMS subgraph on CPU
# to dodge the zero-detection crash. Cache persists compiled .mlmodelc across
# launches (`ModelCacheDirectory`). Clear `models/coreml_cache/` if you swap
# a model file or change EP options.
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

# CUDA EP provider config (Windows/Linux NVIDIA GPU) — used as fallback when
# onnx2torch is unavailable or model conversion fails.
_CUDA_PROVIDER = (
    "CUDAExecutionProvider",
    {"device_id": 0},
)

# DirectML provider name (Windows DirectML, no NVIDIA required)
_DML_PROVIDER = "DmlExecutionProvider"

Accelerator = Literal["cpu", "coreml", "cuda", "dml"]


def _get_default_accelerator() -> Accelerator:
    """Auto-detect platform and return the appropriate accelerator.

    - macOS: CoreML (Apple Neural Engine) for best performance
    - Windows/Linux: CUDA (NVIDIA GPU) if available, else CPU
    """
    if sys.platform == "darwin":
        return "coreml"
    # Windows and Linux: prefer CUDA via PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    # ORT DirectML fallback (Windows only, no NVIDIA required)
    if sys.platform == "win32":
        try:
            import onnxruntime as ort
            if "DmlExecutionProvider" in ort.get_available_providers():
                return "dml"
        except Exception:
            pass
    return "cpu"


# ---------------------------------------------------------------------------
# Session builder: dispatch to the fastest backend per-platform
# ---------------------------------------------------------------------------

def _try_ort_session(onnx_path: str, providers: list) -> "onnxruntime.InferenceSession":
    """Create an ORT session, falling back to CPU if the requested EP fails."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = _ONNX_THREADS_DEFAULT
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    try:
        return ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
    except Exception:
        return ort.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )


def _try_onnx2torch_session(onnx_path: str):
    """Convert ONNX → PyTorch nn.Module via onnx2torch.

    Returns (pt_model, input_names) or (None, None) if conversion fails.
    Only used for the CPU path since ORT CUDA is faster than PyTorch eager.
    """
    try:
        import onnx2torch
        import onnx
        import torch

        onnx_model = onnx.load(onnx_path)
        pt_model = onnx2torch.convert(onnx_model)
        pt_model.eval().to("cpu")
        input_names = [i.name for i in onnx_model.graph.input]
        return pt_model, input_names
    except Exception as exc:
        if onnx_path:
            print(f"[pose2d] onnx2torch skipped for {Path(onnx_path).name}: {exc}")
        return None, None


def _build_session(sub_model, device: str) -> None:
    """Replace `sub_model.session` with the best available backend.

    Dispatch:
      - cuda  → ORT CUDAExecutionProvider  (fastest ONNX path on NVIDIA)
      - cpu   → onnx2torch PyTorch, or ORT CPU if conversion fails
      - coreml → ORT CoreMLExecutionProvider
      - dml    → ORT DmlExecutionProvider
    """
    onnx_path = sub_model.onnx_model

    # ── CUDA: ORT CUDAExecutionProvider (fastest path for ONNX on NVIDIA) ──
    if device == "cuda":
        session = _try_ort_session(onnx_path, [_CUDA_PROVIDER, "CPUExecutionProvider"])
        backend = "ort+cuda"
        print(f"[pose2d] {Path(onnx_path).name}: backend={backend}")
        sub_model.session = session
        return

    # ── CPU: onnx2torch PyTorch (primary), ORT CPU fallback ───────────────
    if device == "cpu":
        pt_model, input_names = _try_onnx2torch_session(onnx_path)
        if pt_model is not None:
            from functools import partial as _partial
            import torch
            import numpy as np

            def _run_pt(feed, model=pt_model, names=input_names):
                tensors = [torch.from_numpy(feed[n]).cpu() for n in names]
                with torch.no_grad():
                    out = model(*tensors)
                if isinstance(out, torch.Tensor):
                    out = [out]
                elif isinstance(out, (list, tuple)):
                    pass
                else:
                    out = list(out)
                return [o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in out]

            session = _PytorchSession(pt_model, input_names, _run_pt)
            backend = "pytorch+cpu"
        else:
            session = _try_ort_session(onnx_path, ["CPUExecutionProvider"])
            backend = "ort+cpu"
        print(f"[pose2d] {Path(onnx_path).name}: backend={backend}")
        sub_model.session = session
        return

    # ── macOS CoreML / Windows DML ────────────────────────────────────────
    if device == "coreml":
        _COREML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        providers = [_COREML_PROVIDER, "CPUExecutionProvider"]
    elif device == "dml":
        providers = [_DML_PROVIDER, "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = _try_ort_session(onnx_path, providers)
    backend = device if device in ("coreml", "dml") else "cpu"
    print(f"[pose2d] {Path(onnx_path).name}: backend=ort+{backend}")
    sub_model.session = session


class _PytorchSession:
    """Thin wrapper that makes onnx2torch models look like an ORT InferenceSession.

    rtmlib expects `session.run(None, {input_name: array}) → list[np.ndarray]`.
    """

    def __init__(self, pt_model, input_names, run_fn):
        self._run_fn = run_fn
        self._pt_model = pt_model
        self._input_names = input_names

    def run(self, output_names, input_feed: dict):
        return self._run_fn(input_feed)

    def get_inputs(self):
        class _I:
            def __init__(self, name):
                self.name = name
        return [_I(n) for n in self._input_names]

    def get_outputs(self):
        return []


# ---------------------------------------------------------------------------
# Public Pose2D class
# ---------------------------------------------------------------------------

class Pose2D:
    """Wraps rtmlib's RTMPose.  Single-person inference: returns the highest-
    score person.  Optionally returns simcc-decoded heatmaps via
    `infer_with_heatmaps`.

    Model sizes (full-pipeline median ms on M-series — CoreML):
      - `mode="lightweight"` (YOLOX-tiny + RTMPose-s): 14.7 ms CPU / 11.8 ms CoreML
      - `mode="balanced"` (YOLOX-m + RTMPose-m): 114 ms CPU / 18.6 ms CoreML  ← DEFAULT
      - `mode="performance"` (YOLOX-x + RTMPose-x): 416 ms CPU / 89 ms CoreML

    Accelerator selection:
      - macOS: CoreML (Apple Neural Engine) — ORT session path
      - Windows/Linux with NVIDIA: CUDA via **PyTorch + onnx2torch** (primary),
        ORT CUDAExecutionProvider (fallback if onnx2torch conversion fails)
      - Windows without NVIDIA: DirectML via ORT session path
      - Everything else: CPU via PyTorch + onnx2torch (primary), ORT (fallback)

    `onnx_threads` applies only to the ORT fallback path.
    """

    def __init__(
        self,
        device: str = "cpu",
        mode: Literal["lightweight", "balanced", "performance"] | None = None,
        onnx_threads: int = _ONNX_THREADS_DEFAULT,
        accelerator: Accelerator | None = None,
    ):
        from rtmlib import Body

        # Auto-detect accelerator and pick a mode that runs well on it.
        if accelerator is None:
            accelerator = _get_default_accelerator()

        # CUDA on laptop GPUs (RTX 3050) can't sustain balanced at 25+ FPS.
        # Default to lightweight mode which runs at 36 FPS on CUDA.
        if mode is None:
            mode = "lightweight" if accelerator == "cuda" else "balanced"

        # Always construct rtmlib on CPU; we replace the sessions below.
        self._body = Body(
            mode=mode, to_openpose=False, backend="onnxruntime", device="cpu"
        )

        # Map accelerator → torch device string (used by _PytorchSession)
        _torch_device = {"cuda": "cuda", "cpu": "cpu"}.get(accelerator, accelerator)

        print(f"[pose2d] accelerator={accelerator}")
        _build_session(self._body.det_model, accelerator)
        _build_session(self._body.pose_model, accelerator)

        # rtmlib 0.0.15 exposes the pose estimator as `pose_model`
        self._pose = getattr(self._body, "pose_model", None)

        if self._pose is not None:
            # Monkey-patch inference to capture raw simcc outputs for the
            # attention overlay (works with both the PyTorch shim and ORT).
            pose_model = self._pose
            _orig_inference = pose_model.inference

            def _capturing_inference(image):
                result = _orig_inference(image)
                try:
                    # inference returns [simcc_x, simcc_y] — each (1, N_kpts, bins)
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

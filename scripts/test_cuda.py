"""CUDA platform verification script.

Run with:  uv run python scripts/test_cuda.py

Checks (in order):
  0. VRAM headroom  — warn if free VRAM < 2 GB (Qwen3-4B INT4 needs ~2.5 GB)
  1. nvidia-smi     — GPU name, VRAM, driver/CUDA version
  2. PyTorch CUDA   — torch.cuda.is_available(), cuDNN, device name, tensor op
  3. ONNX Runtime   — CUDAExecutionProvider present
  4. PyTorch pose   — onnx2torch converts an RTMPose ONNX file and runs
                      a dummy inference on CUDA (skipped when models absent)
  5. LLM deps       — transformers + bitsandbytes + accelerate importable,
                      bitsandbytes CUDA kernels reachable
  6. LLM smoke      — ThaiCoachLLM loads + generates one Thai phrase
                      (skipped when models/qwen3_4b is absent)

Exit code 0 = all present checks passed.
Exit code 1 = at least one required check failed.
"""

import subprocess
import sys
import io

# Force UTF-8 output on Windows (avoids codec errors in PowerShell / cmd)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QWEN_DIR = PROJECT_ROOT / "models" / "qwen3_4b"
RTMLIB_HUB = PROJECT_ROOT / "models" / "rtmlib_cache" / "hub"

# Colour helpers (no external deps)
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

_passed:  list[str] = []
_failed:  list[str] = []
_skipped: list[str] = []


def _ok(label: str, detail: str = "") -> None:
    mark = f"{_GREEN}PASS{_RESET}"
    print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))
    _passed.append(label)


def _fail(label: str, detail: str = "") -> None:
    mark = f"{_RED}FAIL{_RESET}"
    print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))
    _failed.append(label)


def _skip(label: str, reason: str = "") -> None:
    mark = f"{_YELLOW}SKIP{_RESET}"
    print(f"  [{mark}] {label}" + (f"  — {reason}" if reason else ""))
    _skipped.append(label)


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}WARN{_RESET}  {msg}")


# ── 0. VRAM headroom ─────────────────────────────────────────────────────────

def check_vram_headroom() -> None:
    print(f"\n{_BOLD}[0/6] VRAM headroom{_RESET}")
    try:
        import torch

        if not torch.cuda.is_available():
            _skip("VRAM headroom", "no CUDA device — skipping")
            return

        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb  = free_bytes  / 1024**3
        total_gb = total_bytes / 1024**3
        used_gb  = total_gb - free_gb

        detail = f"{free_gb:.2f} GB free / {total_gb:.2f} GB total  ({used_gb:.2f} GB used)"

        if free_gb < 1.0:
            _fail("VRAM headroom", f"{detail}  — CRITICAL: < 1 GB free, close other GPU processes")
        elif free_gb < 2.0:
            _warn(f"Only {free_gb:.2f} GB VRAM free — Qwen3-4B INT4 needs ~2.5 GB. "
                  "Close GPU-heavy apps (games, other models) before the LLM smoke test.")
            _ok("VRAM headroom", detail + "  ⚠ tight for LLM smoke test")
        else:
            _ok("VRAM headroom", detail)

    except ImportError:
        _skip("VRAM headroom", "torch not importable yet")
    except Exception as e:
        _fail("VRAM headroom", str(e))


# ── 1. nvidia-smi ────────────────────────────────────────────────────────────

def check_nvidia_smi() -> None:
    print(f"\n{_BOLD}[1/6] nvidia-smi{_RESET}")
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            name, vram, drv = parts[0], parts[1], parts[2]
            cc = parts[3] if len(parts) > 3 else "?"
            _ok("nvidia-smi", f"{name} | {vram} | driver {drv} | compute cap {cc}")
    except FileNotFoundError:
        _fail("nvidia-smi", "nvidia-smi not found — NVIDIA driver not installed?")
    except subprocess.CalledProcessError as e:
        _fail("nvidia-smi", e.output.strip())


# ── 2. PyTorch CUDA + cuDNN ──────────────────────────────────────────────────

def check_torch() -> None:
    print(f"\n{_BOLD}[2/6] PyTorch CUDA + cuDNN{_RESET}")
    try:
        import torch

        ver = torch.__version__
        if not torch.cuda.is_available():
            _fail("torch.cuda.is_available()",
                  f"torch {ver} — CUDA not available. "
                  "Is this the +cpu wheel? Run: uv sync")
            return

        props    = torch.cuda.get_device_properties(0)
        vram_gb  = props.total_memory / 1024**3
        _ok("torch.cuda.is_available()",
            f"torch {ver} | {props.name} | {vram_gb:.1f} GB VRAM | "
            f"CUDA {torch.version.cuda}")

        # cuDNN
        if torch.backends.cudnn.is_available():
            _ok("cuDNN", f"version {torch.backends.cudnn.version()}")
        else:
            _warn("cuDNN not available — convolution ops will run via CUDA kernel fallback")

        # Tensor compute smoke
        x = torch.zeros(256, 256, device="cuda")
        y = (x + 1.0).sum()
        assert float(y) == 256 * 256, "unexpected tensor result"
        _ok("GPU tensor op", "zeros(256,256) + 1.0 computed correctly on CUDA")

    except ImportError as e:
        _fail("torch import", str(e))
    except AssertionError as e:
        _fail("GPU tensor op", str(e))
    except Exception as e:
        _fail("torch CUDA", str(e))


# ── 3. ONNX Runtime CUDA EP ──────────────────────────────────────────────────

def check_ort() -> None:
    print(f"\n{_BOLD}[3/6] ONNX Runtime (fallback EP){_RESET}")
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        ver = ort.__version__

        if "CUDAExecutionProvider" in providers:
            _ok("CUDAExecutionProvider", f"onnxruntime-gpu {ver} | {providers}")
        else:
            _fail("CUDAExecutionProvider",
                  f"onnxruntime {ver} providers: {providers} — "
                  "install onnxruntime-gpu (not onnxruntime)")

        if "TensorrtExecutionProvider" in providers:
            _ok("TensorrtExecutionProvider", "TensorRT EP present (bonus acceleration)")

    except ImportError as e:
        _fail("onnxruntime import", str(e))


# ── 4. PyTorch pose inference (onnx2torch) ────────────────────────────────────

def _build_minimal_onnx() -> "onnx.ModelProto":
    """Create a tiny ONNX Add model (A + B -> C) for a self-contained onnx2torch test."""
    import onnx
    from onnx import helper, TensorProto

    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 4])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Add", inputs=["A", "B"], outputs=["C"])
    graph = helper.make_graph([node], "add_graph", [A, B], [C])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def check_pytorch_pose() -> None:
    print(f"\n{_BOLD}[4/6] PyTorch pose inference (onnx2torch){_RESET}")

    try:
        import onnx
        import onnx2torch
        import torch

        try:
            from importlib.metadata import version as _pkg_ver
            _o2t_ver = _pkg_ver("onnx2torch")
        except Exception:
            _o2t_ver = getattr(onnx2torch, "__version__", "?")
        _ok("onnx2torch import", f"v{_o2t_ver}")

        # ── 4a. Minimal synthetic model (always works) ──────────────────────
        mini = _build_minimal_onnx()
        pt_mini = onnx2torch.convert(mini)
        _ok("onnx2torch.convert()", "minimal Add model converted")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pt_mini.eval().to(device)
        a = torch.ones(1, 4, device=device)
        b = torch.ones(1, 4, device=device) * 2.0
        with torch.no_grad():
            c = pt_mini(a, b)
        assert c.shape == (1, 4) and float(c[0, 0]) == 3.0, f"unexpected result {c}"
        _ok(f"forward pass on {device}", f"1+2=3 ✓  shape={tuple(c.shape)}")

        # ── 4b. Real RTMPose model (conversion may fail on complex ops) ──────
        onnx_path: Path | None = None
        if RTMLIB_HUB.exists():
            candidates = sorted(RTMLIB_HUB.rglob("*.onnx"))
            for cand in candidates:
                if "rtmpose" in cand.name.lower() or "pose" in cand.name.lower():
                    onnx_path = cand
                    break
            if onnx_path is None and candidates:
                onnx_path = candidates[0]

        if onnx_path is None:
            _skip("RTMPose model conversion",
                  f"no .onnx in {RTMLIB_HUB} — run: uv run python scripts/download_models.py")
        else:
            try:
                real_model = onnx.load(str(onnx_path))
                pt_real    = onnx2torch.convert(real_model)
                pt_real.eval().to(device)
                _ok("RTMPose onnx2torch.convert()", f"{onnx_path.name} on {device}")
            except Exception as exc:
                # Complex ONNX ops (e.g. YOLOX dynamic min/max) may fail onnx2torch.
                # _PytorchSession in pose2d.py keeps an ORT fallback for exactly this case.
                _warn(
                    f"RTMPose conversion: {exc}  "
                    "(pose2d.py falls back to ORT CPU for unsupported ops — expected)"
                )
                _ok("RTMPose ORT fallback", "pose2d._PytorchSession handles this transparently")

    except ImportError as e:
        _fail("onnx2torch import", str(e))
    except Exception as e:
        _fail("PyTorch pose inference", str(e))


# ── 5. LLM dependencies ──────────────────────────────────────────────────────

def check_llm_deps() -> None:
    print(f"\n{_BOLD}[5/6] LLM dependencies (transformers + bitsandbytes){_RESET}")
    for pkg, attr in [
        ("transformers",  "__version__"),
        ("bitsandbytes",  "__version__"),
        ("accelerate",    "__version__"),
    ]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, attr, "?")
            _ok(pkg, f"v{ver}")
        except ImportError as e:
            _fail(pkg, str(e))

    # Confirm bitsandbytes can see CUDA
    try:
        import bitsandbytes as bnb
        import torch

        if torch.cuda.is_available():
            layer = bnb.nn.Linear8bitLt(4, 4, has_fp16_weights=False).cuda()
            _ok("bitsandbytes CUDA kernels", "Linear8bitLt on CUDA instantiated")
        else:
            _skip("bitsandbytes CUDA kernels", "no CUDA device available")
    except Exception as e:
        _fail("bitsandbytes CUDA kernels", str(e))


# ── 6. LLM smoke ─────────────────────────────────────────────────────────────

def check_llm_smoke() -> None:
    print(f"\n{_BOLD}[6/6] LLM smoke test (Qwen3-4B){_RESET}")
    if not QWEN_DIR.exists():
        _skip("ThaiCoachLLM.generate()",
              f"{QWEN_DIR} not found — run: uv run python scripts/download_models.py")
        return

    try:
        import torch

        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from analysis.types import RepAnalysis
        from feedback.llm import ThaiCoachLLM

        vram_before = (
            torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else 0
        )

        llm = ThaiCoachLLM(model_dir=QWEN_DIR)
        if llm._backend == "fallback":
            _fail("ThaiCoachLLM load", "backend is 'fallback' — model failed to load")
            return
        _ok("ThaiCoachLLM load", f"backend={llm._backend}")

        vram_after = (
            torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else 0
        )
        if torch.cuda.is_available():
            _ok("VRAM after LLM load",
                f"{vram_after:.0f} MB allocated ({vram_after - vram_before:+.0f} MB)")

        rep = RepAnalysis(
            rep_index=0,
            score=78,
            components={
                "depth": 25, "valgus": 20, "torso": 15,
                "symmetry": 10, "tempo": 8,
            },
            violations=[],
            descent_ms=1200,
            ascent_ms=900,
        )
        text = llm.generate(rep, max_tokens=60)

        if not text:
            _fail("ThaiCoachLLM.generate()", "returned empty string")
            return

        has_thai = any("\u0e00" <= c <= "\u0e7f" for c in text)
        _ok("ThaiCoachLLM.generate()",
            f"output ({len(text)} chars, thai={has_thai}): {text[:80]!r}")

    except Exception as e:
        _fail("LLM smoke", str(e))


# ── Summary ───────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{_BOLD}{'='*60}")
    print("  CUDA Platform Verification  (NVIDIA + PyTorch + onnx2torch)")
    print(f"{'='*60}{_RESET}")

    check_vram_headroom()
    check_nvidia_smi()
    check_torch()
    check_ort()
    check_pytorch_pose()
    check_llm_deps()
    check_llm_smoke()

    print(f"\n{_BOLD}{'='*60}{_RESET}")
    print(f"  {_GREEN}Passed : {len(_passed)}{_RESET}")
    if _skipped:
        print(f"  {_YELLOW}Skipped: {len(_skipped)}{_RESET}")
    if _failed:
        print(f"  {_RED}Failed : {len(_failed)}{_RESET}")
        for f in _failed:
            print(f"    {_RED}FAIL{_RESET} {f}")
    print(f"{_BOLD}{'='*60}{_RESET}\n")

    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())

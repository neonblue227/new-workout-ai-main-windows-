"""Download all model artifacts into ./models/ at the project root.

Usage
-----
Download everything:
    uv run python scripts/download_models.py

Download individual steps:
    uv run python scripts/download_models.py --rtmpose
    uv run python scripts/download_models.py --motionbert
    uv run python scripts/download_models.py --qwen

Skip specific steps (useful on slow/unstable connections):
    uv run python scripts/download_models.py --skip-qwen
    uv run python scripts/download_models.py --skip-rtmpose --skip-motionbert
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def download_rtmpose():
    """Pre-warm rtmlib's RTMPose-l download by instantiating it once."""
    print("[rtmpose] Downloading RTMPose-l via rtmlib...")
    import rtmlib.tools.file as _rtmlib_file

    hub = MODELS_DIR / "rtmlib_cache" / "hub"
    _rtmlib_file._get_rtmhub_dir = lambda: str(hub)
    from rtmlib import Body

    _ = Body(mode="performance", to_openpose=False, backend="onnxruntime", device="cpu")
    print("[rtmpose] RTMPose-l ready.")


def download_motionbert():
    print("[motionbert] Downloading MotionBERT checkpoint...")
    from huggingface_hub import hf_hub_download

    target_dir = MODELS_DIR / "motionbert"
    target_dir.mkdir(parents=True, exist_ok=True)
    ckpt = hf_hub_download(
        repo_id="walterzhu/MotionBERT",
        filename="checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
        local_dir=str(target_dir),
    )
    print(f"[motionbert] MotionBERT at {ckpt}")


def download_qwen():
    """Download Qwen3-4B (HuggingFace Transformers format) for CUDA inference.

    The model is stored in full bf16 precision (~8 GB on disk).
    bitsandbytes quantises it to NF4 INT4 at load time, using ~2.5 GB GPU VRAM.

    Run on a stable connection:
        uv run python scripts/download_models.py --qwen
    """
    print("[qwen] Downloading Qwen/Qwen3-4B (HuggingFace Transformers for CUDA)...")
    print("[qwen] Note: ~8 GB download. Use a stable connection.")
    from huggingface_hub import snapshot_download

    target_dir = MODELS_DIR / "qwen3_4b"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id="Qwen/Qwen3-4B",
        local_dir=str(target_dir),
        ignore_patterns=["*.gguf"],  # skip GGUF blobs — we use Transformers format
    )
    print(f"[qwen] Qwen3-4B at {path}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Explicit single-step flags
    g = p.add_argument_group("individual steps (run only this step)")
    g.add_argument("--rtmpose",    action="store_true", help="download RTMPose only")
    g.add_argument("--motionbert", action="store_true", help="download MotionBERT only")
    g.add_argument("--qwen",       action="store_true", help="download Qwen3-4B only")

    # Skip flags (run all except...)
    s = p.add_argument_group("skip flags (run all except these)")
    s.add_argument("--skip-rtmpose",    action="store_true")
    s.add_argument("--skip-motionbert", action="store_true")
    s.add_argument("--skip-qwen",       action="store_true", help="defer Qwen download (e.g. unstable WiFi)")

    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    MODELS_DIR.mkdir(exist_ok=True)

    # If any explicit step flag is set, run only those
    explicit = any([args.rtmpose, args.motionbert, args.qwen])
    if explicit:
        if args.rtmpose:
            download_rtmpose()
        if args.motionbert:
            download_motionbert()
        if args.qwen:
            download_qwen()
        print("\nDone.")
        return

    # Otherwise run all, respecting --skip-* flags
    if not args.skip_rtmpose:
        download_rtmpose()
    else:
        print("[rtmpose] skipped.")

    if not args.skip_motionbert:
        download_motionbert()
    else:
        print("[motionbert] skipped.")

    if not args.skip_qwen:
        download_qwen()
    else:
        print("[qwen] skipped — run later with:  uv run python scripts/download_models.py --qwen")

    print("\nAll requested models downloaded.")


if __name__ == "__main__":
    sys.exit(main())

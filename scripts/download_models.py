"""Download all model artifacts into ./models/ at the project root."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def download_rtmpose():
    """Pre-warm rtmlib's RTMPose-l download by instantiating it once."""
    print("[1/3] Downloading RTMPose-l via rtmlib...")
    import rtmlib.tools.file as _rtmlib_file

    hub = MODELS_DIR / "rtmlib_cache" / "hub"
    _rtmlib_file._get_rtmhub_dir = lambda: str(hub)
    from rtmlib import Body

    _ = Body(mode="performance", to_openpose=False, backend="onnxruntime", device="cpu")
    print("       RTMPose-l ready.")


def download_motionbert():
    print("[2/3] Downloading MotionBERT checkpoint...")
    from huggingface_hub import hf_hub_download

    target_dir = MODELS_DIR / "motionbert"
    target_dir.mkdir(parents=True, exist_ok=True)
    ckpt = hf_hub_download(
        repo_id="walterzhu/MotionBERT",
        filename="checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
        local_dir=str(target_dir),
    )
    print(f"       MotionBERT at {ckpt}")


def download_qwen():
    print("[3/3] Downloading Qwen3.5-4B (mxfp4 mlx-vlm)...")
    from huggingface_hub import snapshot_download

    target_dir = MODELS_DIR / "qwen3_5_4b_mxfp4"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id="RepublicOfKorokke/Qwen3.5-4B-mlx-vlm-mxfp4",
        local_dir=str(target_dir),
    )
    print(f"       Qwen at {path}")


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    download_rtmpose()
    download_motionbert()
    download_qwen()
    print("\nAll models downloaded.")


if __name__ == "__main__":
    sys.exit(main())

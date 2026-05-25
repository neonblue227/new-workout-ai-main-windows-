import numpy as np
import pytest
from pathlib import Path

MOTIONBERT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "vendor" / "motionbert"
)
CKPT = (
    Path(__file__).resolve().parent.parent.parent
    / "models"
    / "motionbert"
    / "checkpoint"
    / "pose3d"
    / "FT_MB_lite_MB_ft_h36m_global_lite"
    / "best_epoch.bin"
)


@pytest.mark.skipif(
    not (MOTIONBERT_PATH.exists() and CKPT.exists()),
    reason="MotionBERT vendor code or checkpoint missing",
)
def test_pose3d_returns_17x3():
    from pose3d import Pose3D

    lifter = Pose3D()
    window = np.zeros((27, 17, 3), dtype=np.float32)
    window[..., 2] = 1.0
    out = lifter.infer(window)
    assert out.shape == (17, 3)

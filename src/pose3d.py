import sys
from pathlib import Path
from collections import deque
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOTIONBERT_DIR = PROJECT_ROOT / "vendor" / "motionbert"
CKPT_PATH = (
    PROJECT_ROOT
    / "models"
    / "motionbert"
    / "checkpoint"
    / "pose3d"
    / "FT_MB_lite_MB_ft_h36m_global_lite"
    / "best_epoch.bin"
)
CONFIG_PATH = MOTIONBERT_DIR / "configs" / "pose3d" / "MB_ft_h36m_global_lite.yaml"

# COCO-17 -> Human3.6M-17 reordering.
COCO_TO_H36M = {
    0: -1,  # pelvis: synthesized as (L_hip + R_hip) / 2
    1: 12,  # R hip
    2: 14,  # R knee
    3: 16,  # R ankle
    4: 11,  # L hip
    5: 13,  # L knee
    6: 15,  # L ankle
    7: -2,  # spine: midpoint of pelvis and thorax
    8: -3,  # thorax: midpoint of shoulders
    9: 0,  # neck/nose -> nose
    10: 0,  # head -> nose (coarse)
    11: 5,  # L shoulder
    12: 7,  # L elbow
    13: 9,  # L wrist
    14: 6,  # R shoulder
    15: 8,  # R elbow
    16: 10,  # R wrist
}


def coco17_to_h36m17(kps_coco: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Convert (17, 2) COCO + (17,) scores -> (17, 3) H36M with (x, y, score)."""
    out = np.zeros((17, 3), dtype=np.float32)
    l_hip, r_hip = kps_coco[11], kps_coco[12]
    pelvis = (l_hip + r_hip) / 2.0
    l_sh, r_sh = kps_coco[5], kps_coco[6]
    thorax = (l_sh + r_sh) / 2.0
    spine = (pelvis + thorax) / 2.0

    for h_idx, c_idx in COCO_TO_H36M.items():
        if c_idx == -1:
            out[h_idx, :2] = pelvis
            out[h_idx, 2] = min(scores[11], scores[12])
        elif c_idx == -2:
            out[h_idx, :2] = spine
            out[h_idx, 2] = min(scores[5], scores[6], scores[11], scores[12])
        elif c_idx == -3:
            out[h_idx, :2] = thorax
            out[h_idx, 2] = min(scores[5], scores[6])
        else:
            out[h_idx, :2] = kps_coco[c_idx]
            out[h_idx, 2] = scores[c_idx]
    return out


def _normalize_2d(kps: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
    out = kps.copy()
    out[..., 0] = out[..., 0] / frame_w * 2 - 1
    out[..., 1] = out[..., 1] / frame_w * 2 - frame_h / frame_w
    return out


class Pose3D:
    """MotionBERT-Lite wrapper for 2D->3D lifting on a sliding 27-frame window."""

    def __init__(self, window_size: int = 27, device: str | None = None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        sys.path.insert(0, str(MOTIONBERT_DIR))
        from lib.utils.tools import get_config
        from lib.model.DSTformer import DSTformer

        cfg = get_config(str(CONFIG_PATH))
        self._model = DSTformer(
            dim_in=3,
            dim_out=3,
            dim_feat=cfg.dim_feat,
            dim_rep=cfg.dim_rep,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            num_joints=cfg.num_joints,
            maxlen=cfg.maxlen,
        )
        ckpt = torch.load(CKPT_PATH, map_location=self.device, weights_only=False)
        state = ckpt.get("model_pos", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self._model.load_state_dict(state, strict=False)
        self._model.eval().to(self.device)
        self.window_size = window_size

    @torch.no_grad()
    def infer(
        self, window: np.ndarray, frame_h: int = 720, frame_w: int = 1280
    ) -> np.ndarray:
        norm = _normalize_2d(window, frame_h, frame_w)
        x = torch.from_numpy(norm).float().unsqueeze(0).to(self.device)
        out = self._model(x)
        out = out.squeeze(0).cpu().numpy()
        centre = out[self.window_size // 2]
        return centre


class Pose3DBuffer:
    """Rolling 2D buffer that yields a 3D pose on demand."""

    def __init__(self, lifter: Pose3D):
        self._buf: deque[np.ndarray] = deque(maxlen=lifter.window_size)
        self._lifter = lifter

    def push(self, h36m_kps: np.ndarray):
        self._buf.append(h36m_kps)

    def ready(self) -> bool:
        return len(self._buf) == self._buf.maxlen

    def lift(self, frame_h: int, frame_w: int) -> np.ndarray:
        if not self.ready():
            raise RuntimeError("buffer not full")
        window = np.stack(self._buf, axis=0)
        return self._lifter.infer(window, frame_h, frame_w)

import numpy as np


LOWER_BODY_JOINTS = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles


def aggregate_heatmaps(
    heatmaps: np.ndarray, joints: list[int] | None = None
) -> np.ndarray:
    """heatmaps: (N_kpts, H, W). Returns a single (H, W) attention map in [0, 1].

    Squat-specific: weight lower-body joints heavier.
    """
    if joints is None:
        joints = LOWER_BODY_JOINTS
    weights = np.ones(heatmaps.shape[0], dtype=np.float32)
    for j in joints:
        weights[j] = 3.0
    agg = (heatmaps * weights[:, None, None]).sum(axis=0)
    agg = agg - agg.min()
    rng = agg.max() - agg.min()
    if rng < 1e-9:
        return np.zeros_like(agg)
    return (agg / rng).astype(np.float32)

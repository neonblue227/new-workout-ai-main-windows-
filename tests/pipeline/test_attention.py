import numpy as np
from analysis.attention import aggregate_heatmaps


def test_aggregate_heatmaps_normalised_to_0_1():
    hm = np.random.rand(17, 64, 48).astype(np.float32) * 5.0
    out = aggregate_heatmaps(hm)
    assert out.shape == (64, 48)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_aggregate_focuses_on_joint_peaks():
    hm = np.zeros((17, 64, 48), dtype=np.float32)
    hm[13, 30, 24] = 1.0
    out = aggregate_heatmaps(hm)
    assert out[30, 24] == out.max()

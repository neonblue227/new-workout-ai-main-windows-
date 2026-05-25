import numpy as np

from screens import (
    PANEL_W,
    SETUP_BUTTON_RECT,
    build_debug_panel,
    draw_skeleton_overlay,
    h36m_joint_confidences,
    health_color,
    point_in_rect,
)


def test_point_in_rect_inside():
    x, y, w, h = SETUP_BUTTON_RECT
    assert point_in_rect(x + 5, y + 5, SETUP_BUTTON_RECT)
    assert point_in_rect(x + w // 2, y + h // 2, SETUP_BUTTON_RECT)


def test_point_in_rect_outside():
    x, y, w, h = SETUP_BUTTON_RECT
    assert not point_in_rect(x - 1, y - 1, SETUP_BUTTON_RECT)
    assert not point_in_rect(x + w + 1, y + h + 1, SETUP_BUTTON_RECT)


def test_h36m_joint_confidences_shape_and_pelvis_min():
    scores = np.full(17, 0.9, dtype=np.float32)
    scores[11] = 0.2  # L_HIP low
    jc = h36m_joint_confidences(scores)
    assert jc.shape == (17,)
    assert jc[0] == 0.2  # pelvis = min(L_HIP, R_HIP)


def test_health_color_nan_is_grey():
    grey = health_color(float("nan"), True)
    green = health_color(50.0, True)
    amber = health_color(10.0, False)
    assert grey != green and green != amber


def _synthetic_pose():
    kps = np.full((17, 2), 360.0, dtype=np.float32)
    kps[:, 0] = 640.0
    kps[0] = (640.0, 150.0)  # nose
    kps[5] = (560.0, 300.0)  # L_shoulder
    kps[6] = (720.0, 300.0)  # R_shoulder
    scores = np.full(17, 0.9, dtype=np.float32)
    return kps, scores


def _metrics():
    return {
        "tilt": -12.3,
        "cva": 55.0,
        "fwd": 0.2,
        "neck_flex": 5.0,
        "sh_asym": 0.01,
        "view": "front",
        "view_ok": True,
        "conf": {"nose": 0.9, "lsh": 0.9, "rsh": 0.9, "lhip": 0.8, "rhip": 0.8},
        "fps": 28,
        "infer_ms": 18.0,
        "lift_ms": 12.0,
    }


def test_draw_skeleton_overlay_runs_in_place():
    kps, scores = _synthetic_pose()
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_skeleton_overlay(canvas, kps, scores, (720, 1280))
    assert canvas.sum() > 0  # something was drawn


def test_build_debug_panel_shape_with_and_without_rig():
    jc = h36m_joint_confidences(_synthetic_pose()[1])
    for rig in (None, np.random.randn(17, 3).astype(np.float32)):
        panel = build_debug_panel(720, rig, jc, _metrics())
        assert panel.shape == (720, PANEL_W, 3)


def test_build_debug_panel_tolerates_nan_metrics():
    jc = h36m_joint_confidences(_synthetic_pose()[1])
    m = dict(_metrics(), tilt=float("nan"), cva=float("nan"), fwd=float("nan"))
    panel = build_debug_panel(720, None, jc, m)
    assert panel.shape == (720, PANEL_W, 3)

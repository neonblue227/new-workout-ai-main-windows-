import numpy as np
from render import Renderer


def test_draw_skeleton_does_not_crash():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    kps = np.array([[100 + i * 5, 100 + i * 5] for i in range(17)], dtype=np.float32)
    scores = np.ones((17,), dtype=np.float32) * 0.9
    r = Renderer(panel_width=320)
    out = r.draw_skeleton(frame, kps, scores)
    assert out.shape == frame.shape


def test_compose_with_panel_returns_wider_image():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    r = Renderer(panel_width=320)
    out = r.compose(
        frame, score=85, running_avg=72.5, rep_count=4, phase="bottom", thai_text=""
    )
    assert out.shape == (480, 640 + 320, 3)


def test_compose_accepts_hold_kwargs_without_error():
    r = Renderer(panel_width=320)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = r.compose(
        frame,
        score=None,
        running_avg=0.0,
        rep_count=0,
        phase="holding",
        thai_text="",
        hold_state="holding",
        hold_progress=0.5,
    )
    assert out.shape == (480, 640 + 320, 3)


def test_load_thai_font_is_cached():
    from render import _load_thai_font

    a = _load_thai_font(28)
    b = _load_thai_font(28)
    assert a is b  # cached per size, not reloaded on every put_thai_text call


def test_thai_sprite_is_cached():
    from render import _thai_sprite

    a = _thai_sprite("ทดสอบ", (20, 40), 28, (255, 255, 255), None, False, 120, 400)
    b = _thai_sprite("ทดสอบ", (20, 40), 28, (255, 255, 255), None, False, 120, 400)
    assert a is b  # repeated text reuses the rendered sprite instead of re-rasterizing


def test_thai_sprite_none_for_empty_text():
    from render import _thai_sprite

    assert (
        _thai_sprite("", (20, 40), 28, (255, 255, 255), None, False, 120, 400) is None
    )


def test_put_thai_text_empty_is_noop():
    from render import put_thai_text

    img = np.zeros((120, 400, 3), dtype=np.uint8)
    put_thai_text(img, "", (20, 40), font_size=28)
    assert img.sum() == 0  # nothing drawn for empty text


def test_put_thai_text_draws_pixels():
    from render import put_thai_text

    img = np.zeros((120, 400, 3), dtype=np.uint8)
    put_thai_text(img, "ทดสอบ", (20, 40), font_size=28, color=(255, 255, 255))
    assert img.sum() > 0  # something was drawn


def test_put_thai_text_center_runs():
    from render import put_thai_text

    img = np.zeros((120, 400, 3), dtype=np.uint8)
    put_thai_text(img, "กลาง", (200, 50), font_size=28, color=(0, 255, 0), center=True)
    assert img.sum() > 0

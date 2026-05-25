"""OpenCV screen rendering for the neck-stretch demo.

Each draw_* function returns a BGR canvas to imshow. Kept separate from the
pure RoutineFSM and from app's control flow so the visuals can be iterated on
in isolation. Thai text uses render.put_thai_text.
"""

from __future__ import annotations

import cv2
import numpy as np

from analysis.angles import (
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
)
from render import SKELETON, put_thai_text

W, H = 1280, 720
PANEL_W = 360  # width of the right-hand debug panel (window total = W + PANEL_W)
SETUP_BUTTON_RECT = (490, 410, 300, 90)  # x, y, w, h

_BG = (24, 24, 28)
_GREEN = (90, 220, 90)
_AMBER = (90, 200, 220)
_GREY = (160, 160, 160)
_WHITE = (245, 245, 245)

_SIDE_TH = {"left": "ซ้าย", "right": "ขวา"}


def point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def _blank() -> np.ndarray:
    canvas = np.empty((H, W, 3), dtype=np.uint8)
    canvas[:] = _BG
    return canvas


def _mirror(frame: np.ndarray) -> np.ndarray:
    """Selfie-view mirror, resized to the demo canvas."""
    f = cv2.resize(frame, (W, H))
    return cv2.flip(f, 1)


def draw_setup() -> np.ndarray:
    canvas = _blank()
    put_thai_text(
        canvas,
        "ยืดคอ คลายออฟฟิศซินโดรม",
        (W // 2, 150),
        font_size=46,
        color=_WHITE,
        center=True,
    )
    put_thai_text(
        canvas,
        "ใช้เวลา 2 นาที · ยืดคอสลับซ้าย-ขวา 4 เซ็ต",
        (W // 2, 230),
        font_size=26,
        color=_GREY,
        center=True,
    )
    x, y, w, h = SETUP_BUTTON_RECT
    cv2.rectangle(canvas, (x, y), (x + w, y + h), _GREEN, -1, cv2.LINE_AA)
    put_thai_text(
        canvas,
        "เริ่ม",
        (x + w // 2, y + 22),
        font_size=40,
        color=(20, 20, 20),
        center=True,
    )
    put_thai_text(
        canvas,
        "(คลิกปุ่มเพื่อเริ่ม · กด q เพื่อออก)",
        (W // 2, 560),
        font_size=22,
        color=_GREY,
        center=True,
    )
    return canvas


def _draw_outline(img: np.ndarray, ok: bool) -> None:
    """Semi-transparent humanoid guide centered on the canvas."""
    color = _GREEN if ok else _GREY
    overlay = img.copy()
    cx = W // 2
    cv2.circle(overlay, (cx, 180), 60, color, 3, cv2.LINE_AA)  # head
    cv2.line(
        overlay, (cx - 120, 300), (cx + 120, 300), color, 3, cv2.LINE_AA
    )  # shoulders
    cv2.line(overlay, (cx - 90, 520), (cx + 90, 520), color, 3, cv2.LINE_AA)  # hips
    cv2.line(overlay, (cx - 120, 300), (cx - 90, 520), color, 3, cv2.LINE_AA)  # torso L
    cv2.line(overlay, (cx + 120, 300), (cx + 90, 520), color, 3, cv2.LINE_AA)  # torso R
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0.0, dst=img)


def draw_positioning(frame: np.ndarray, progress: float, ok: bool) -> np.ndarray:
    canvas = _mirror(frame)
    _draw_outline(canvas, ok)
    msg = "ค้างไว้..." if ok else "ยืนให้กล้องเห็นหัว ไหล่ และสะโพก"
    put_thai_text(
        canvas,
        msg,
        (W // 2, 40),
        font_size=30,
        color=_GREEN if ok else _AMBER,
        center=True,
    )
    bw, bx, by = 600, (W - 600) // 2, H - 60
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + 24), _GREY, 2, cv2.LINE_AA)
    fill = int(bw * max(0.0, min(1.0, progress)))
    cv2.rectangle(canvas, (bx, by), (bx + fill, by + 24), _GREEN, -1, cv2.LINE_AA)
    return canvas


def draw_countdown(frame: np.ndarray, number: int, side: str | None) -> np.ndarray:
    canvas = _mirror(frame)
    if side:
        put_thai_text(
            canvas,
            f"เตรียมยืดคอด้าน{_SIDE_TH.get(side, side)}",
            (W // 2, 120),
            font_size=36,
            color=_WHITE,
            center=True,
        )
    cv2.putText(
        canvas,
        str(number),
        (W // 2 - 40, H // 2 + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        6.0,
        _WHITE,
        12,
        cv2.LINE_AA,
    )
    return canvas


def draw_hold(
    frame: np.ndarray,
    side: str | None,
    remaining_s: float,
    in_target: bool,
    view_ok: bool,
    thai_text: str,
) -> np.ndarray:
    canvas = _mirror(frame)
    border = _GREEN if in_target else _AMBER
    cv2.rectangle(canvas, (4, 4), (W - 4, H - 4), border, 8)
    put_thai_text(
        canvas,
        f"ยืดคอด้าน{_SIDE_TH.get(side or '', '')}",
        (30, 30),
        font_size=34,
        color=_WHITE,
    )
    cv2.putText(
        canvas,
        f"{int(np.ceil(remaining_s)):02d}",
        (W - 150, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.2,
        _WHITE,
        5,
        cv2.LINE_AA,
    )
    if not view_ok:
        put_thai_text(
            canvas,
            "หันหน้าเข้าหากล้อง",
            (W // 2, H // 2),
            font_size=34,
            color=_AMBER,
            center=True,
        )
    elif thai_text:
        put_thai_text(
            canvas,
            thai_text,
            (40, H - 120),
            font_size=28,
            color=_WHITE,
            max_width=W - 80,
        )
    return canvas


def draw_transition(frame: np.ndarray, next_side: str | None) -> np.ndarray:
    canvas = _mirror(frame)
    put_thai_text(
        canvas, "พักสักครู่", (W // 2, H // 2 - 40), font_size=40, color=_WHITE, center=True
    )
    if next_side:
        put_thai_text(
            canvas,
            f"ต่อไปยืดด้าน{_SIDE_TH.get(next_side, next_side)}",
            (W // 2, H // 2 + 30),
            font_size=32,
            color=_GREEN,
            center=True,
        )
    return canvas


def draw_summary(set_scores: list[int], overall: int, thai_text: str) -> np.ndarray:
    canvas = _blank()
    put_thai_text(
        canvas, "จบการฝึก", (W // 2, 90), font_size=46, color=_WHITE, center=True
    )
    put_thai_text(
        canvas,
        f"คะแนนรวม {overall}/100",
        (W // 2, 170),
        font_size=34,
        color=_GREEN,
        center=True,
    )
    for i, sc in enumerate(set_scores):
        put_thai_text(
            canvas,
            f"เซ็ต {i + 1}: {sc}/100",
            (W // 2, 240 + i * 44),
            font_size=26,
            color=_GREY,
            center=True,
        )
    if thai_text:
        put_thai_text(
            canvas,
            thai_text,
            (W // 2 - 400, 470),
            font_size=26,
            color=_WHITE,
            max_width=800,
        )
    put_thai_text(
        canvas,
        "(กด q เพื่อออก)",
        (W // 2, H - 50),
        font_size=22,
        color=_GREY,
        center=True,
    )
    return canvas


# --- Debug overlays + 3D rig panel (adapted from src/test_2D_3D.py) ---

JOINT_VISIBILITY_THRESHOLD = 0.3

# H36M-17 skeleton bones; per-joint visibility decides which subset draws.
_H36M_BONES = [
    (0, 1),
    (1, 2),
    (2, 3),  # right leg
    (0, 4),
    (4, 5),
    (5, 6),  # left leg
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),  # pelvis -> spine -> thorax -> neck -> head
    (8, 11),
    (11, 12),
    (12, 13),  # left arm
    (8, 14),
    (14, 15),
    (15, 16),  # right arm
]


def _draw_text(img, text, org, color=(240, 240, 240), scale=0.5, thick=1):
    """Outlined ASCII text (black halo) so debug readouts stay legible."""
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thick + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA
    )


def health_color(value, healthy: bool) -> tuple[int, int, int]:
    """Grey if value is NaN/None, green if within the healthy band, amber otherwise."""
    try:
        if value is None or np.isnan(value):
            return _GREY
    except TypeError:
        pass
    return _GREEN if healthy else _AMBER


def draw_skeleton_overlay(canvas, kps, scores, frame_shape, threshold=0.3) -> None:
    """Draw the COCO-17 skeleton + head-tilt reference lines onto a mirrored,
    resized camera canvas (W x H), in place. `kps` are original frame-pixel
    coords; they are scaled to the canvas and mirrored to match `_mirror`."""
    fh, fw = frame_shape[:2]
    sx, sy = W / fw, H / fh
    disp = np.empty((kps.shape[0], 2), dtype=np.float32)
    disp[:, 0] = (W - 1) - kps[:, 0] * sx
    disp[:, 1] = kps[:, 1] * sy
    for a, b in SKELETON:
        if scores[a] < threshold or scores[b] < threshold:
            continue
        cv2.line(
            canvas,
            (int(disp[a, 0]), int(disp[a, 1])),
            (int(disp[b, 0]), int(disp[b, 1])),
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )
    for i in range(disp.shape[0]):
        if scores[i] < threshold:
            continue
        cv2.circle(
            canvas, (int(disp[i, 0]), int(disp[i, 1])), 4, (0, 255, 0), -1, cv2.LINE_AA
        )
    if (
        scores[NOSE] >= threshold
        and scores[L_SHOULDER] >= threshold
        and scores[R_SHOULDER] >= threshold
    ):
        ls, rs, n = disp[L_SHOULDER], disp[R_SHOULDER], disp[NOSE]
        mid = (ls + rs) / 2.0
        cv2.line(
            canvas,
            (int(ls[0]), int(ls[1])),
            (int(rs[0]), int(rs[1])),
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (int(mid[0]), int(mid[1])),
            (int(n[0]), int(n[1])),
            (60, 60, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            canvas, (int(mid[0]), int(mid[1])), 5, (0, 180, 255), -1, cv2.LINE_AA
        )


def h36m_joint_confidences(scores) -> np.ndarray:
    """Map COCO-17 confidences (17,) to per-H36M-17-joint confidences (17,),
    mirroring the synthesis rules in pose3d.coco17_to_h36m17."""
    out = np.zeros(17, dtype=np.float32)
    out[0] = float(min(scores[L_HIP], scores[R_HIP]))  # pelvis
    out[1] = float(scores[R_HIP])
    out[2] = float(scores[R_KNEE])
    out[3] = float(scores[R_ANKLE])
    out[4] = float(scores[L_HIP])
    out[5] = float(scores[L_KNEE])
    out[6] = float(scores[L_ANKLE])
    out[7] = float(
        min(scores[L_SHOULDER], scores[R_SHOULDER], scores[L_HIP], scores[R_HIP])
    )  # spine
    out[8] = float(min(scores[L_SHOULDER], scores[R_SHOULDER]))  # thorax
    out[9] = float(scores[NOSE])  # neck
    out[10] = float(scores[NOSE])  # head
    out[11] = float(scores[L_SHOULDER])
    out[12] = float(scores[L_ELBOW])
    out[13] = float(scores[L_WRIST])
    out[14] = float(scores[R_SHOULDER])
    out[15] = float(scores[R_ELBOW])
    out[16] = float(scores[R_WRIST])
    return out


def _render_rig_3d(
    panel, rig, box, joint_conf, threshold=JOINT_VISIBILITY_THRESHOLD, mirror=True
) -> None:
    """Render an H36M-17 rig inside box (x0, y0, w, h). Per-joint visibility from
    joint_conf; figure auto-scales to the visible joints."""
    x0, y0, w, h = box
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (60, 60, 60), 1)
    if rig is None:
        _draw_text(
            panel,
            "3D warming up (need 27 frames)",
            (x0 + 8, y0 + h // 2),
            (160, 160, 160),
            0.45,
        )
        return
    visible = joint_conf >= threshold
    idxs = np.where(visible)[0]
    if idxs.size == 0:
        _draw_text(
            panel, "no joints detected", (x0 + 8, y0 + h // 2), (160, 160, 160), 0.45
        )
        return
    pts = rig[:, :2].astype(np.float32)
    bb = pts[idxs]
    bmin, bmax = bb.min(axis=0), bb.max(axis=0)
    span = max(float((bmax - bmin).max()), 1e-3)
    margin = 16
    scale = (min(w, h) - 2 * margin) / span
    centre = (bmin + bmax) / 2.0
    pcentre = np.array([x0 + w / 2.0, y0 + h / 2.0], dtype=np.float32)
    screen = (pts - centre) * scale + pcentre
    if mirror:
        screen[:, 0] = (2 * x0 + w) - screen[:, 0]
    rect = (x0, y0, w, h)
    for a, b in _H36M_BONES:
        if not (visible[a] and visible[b]):
            continue
        ok, p1, p2 = cv2.clipLine(
            rect,
            (int(screen[a, 0]), int(screen[a, 1])),
            (int(screen[b, 0]), int(screen[b, 1])),
        )
        if ok:
            cv2.line(panel, p1, p2, (0, 200, 255), 2, cv2.LINE_AA)
    for i in range(17):
        if not visible[i]:
            continue
        x, y = int(screen[i, 0]), int(screen[i, 1])
        if x0 <= x <= x0 + w and y0 <= y <= y0 + h:
            cv2.circle(panel, (x, y), 3, (0, 200, 255), -1, cv2.LINE_AA)


def _draw_metric_block(panel, x, y, lines, line_h=22, box_w=340) -> None:
    """Draw (text, color) lines over a translucent dark box for legibility."""
    if not lines:
        return
    top = max(0, y - 17)
    bottom = min(panel.shape[0], y - 17 + line_h * len(lines) + 8)
    left = max(0, x - 6)
    right = min(panel.shape[1], x - 6 + box_w)
    sub = panel[top:bottom, left:right]
    if sub.size:
        cv2.addWeighted(sub, 0.35, np.zeros_like(sub), 0.65, 0.0, dst=sub)
    for i, (text, color) in enumerate(lines):
        _draw_text(panel, text, (x, y + i * line_h), color=color, scale=0.5)


def _debug_lines(m: dict) -> list[tuple[str, tuple[int, int, int]]]:
    """Build the (text, color) readout lines for the debug panel from a metrics dict."""

    def num(v, fmt):
        try:
            return fmt.format(v) if not np.isnan(v) else "  --"
        except (TypeError, ValueError):
            return "  --"

    tilt, cva, fwd = m["tilt"], m["cva"], m["fwd"]
    nf, sa, conf = m["neck_flex"], m["sh_asym"], m["conf"]
    return [
        (
            f"head tilt : {num(tilt, '{:+5.1f}')}",
            _WHITE if not np.isnan(tilt) else _GREY,
        ),
        (
            f"CVA       : {num(cva, '{:5.1f}')}",
            health_color(cva, not np.isnan(cva) and cva >= 50.0),
        ),
        (
            f"fwd head  : {num(fwd, '{:5.2f}')}",
            health_color(fwd, not np.isnan(fwd) and fwd < 0.30),
        ),
        (
            f"neck flex : {num(nf, '{:+5.1f}')}",
            health_color(nf, not np.isnan(nf) and abs(nf) < 25.0),
        ),
        (
            f"sh asym   : {num(sa, '{:+5.2f}')}",
            health_color(sa, not np.isnan(sa) and abs(sa) < 0.05),
        ),
        (f"view      : {m['view']}", _GREEN if m["view_ok"] else _AMBER),
        (f"nose {conf['nose']:.2f} Lsh {conf['lsh']:.2f} Rsh {conf['rsh']:.2f}", _GREY),
        (f"Lhip {conf['lhip']:.2f} Rhip {conf['rhip']:.2f}", _GREY),
        (f"fps       : {m['fps']:2d}", _GREEN if m["fps"] >= 15 else _AMBER),
        (f"infer {m['infer_ms']:4.0f}ms  lift {m['lift_ms']:4.0f}ms", _GREY),
    ]


def build_debug_panel(height: int, rig, joint_conf, metrics: dict) -> np.ndarray:
    """Right-hand debug panel: 3D rig on top, 2D cookbook + diagnostic readouts below.

    `metrics` keys: tilt, cva, fwd, neck_flex, sh_asym (floats; NaN ok),
    view (str), view_ok (bool), conf (dict nose/lsh/rsh/lhip/rhip), fps (int),
    infer_ms, lift_ms (floats).
    """
    panel = np.full((height, PANEL_W, 3), 22, dtype=np.uint8)
    _draw_text(panel, "DEBUG", (10, 24), (255, 255, 255), 0.7, 2)
    _render_rig_3d(panel, rig, (10, 36, PANEL_W - 20, 240), joint_conf, mirror=True)
    _draw_metric_block(panel, 12, 308, _debug_lines(metrics), box_w=PANEL_W - 20)
    return panel

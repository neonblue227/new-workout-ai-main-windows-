"""Live 3-panel diagnostic app for the 2D → 3D pose pipeline.

Three panels side-by-side:

  [ Camera (raw) ]  [ 2D pose overlay ]  [ 3D rig (lifted)  ]

Each panel carries a measurement readout so you can see — in real time, on your
own desk-camera framing — whether the 2D nose+shoulders neck-tilt (the new
canonical measurement, per recommendation 2 of the seated-pipeline review) and
the 3D pelvis-rooted neck-tilt (the old measurement, still computed for
comparison) agree or diverge.

Run:

    uv run python src/test_2D_3D.py

Keys:
    q / Esc — quit

Why this exists
---------------
MotionBERT-Lite was trained on Human3.6M with full-body visible. For a desk
camera that frames only the head/shoulders/torso, the lifted hips and legs
are out-of-distribution and the body axes derived from them (pelvis → thorax,
L_hip → R_hip) are unreliable. The 2D head-lateral-tilt uses only COCO-17
nose + shoulders and does not depend on the lifted lower body at all. Watch
the two numbers in the panels: if 2D is stable while 3D jitters or is biased,
that's the seated-OOD effect in action.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analysis.angles import (  # noqa: E402
    NOSE,
    L_ANKLE,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    R_ANKLE,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    craniovertebral_angle_2d,
    forward_head_offset_normalized_2d,
    head_lateral_tilt_2d,
    neck_flexion_2d,
    shoulder_elevation_asymmetry_2d,
)
from analysis.angles_3d import head_lateral_tilt_3d  # noqa: E402
from analysis.camera_view import classify_view  # noqa: E402
from camera import build_capture  # noqa: E402
from exercises.neck_stretch import NeckStretchLeft  # noqa: E402
from pose2d import Pose2D  # noqa: E402
from render import SKELETON  # noqa: E402

CAM_WIDTH = 640
CAM_HEIGHT = 480
PANEL_PAD = 80  # extra vertical room below each panel for readouts

# Diagnostic runs both heavy stages at the camera's native 30 fps. Profiled
# 2026-05-23 (balanced + CoreML): 2D infer + 3D lift combined = ~22.8 ms median
# (p95 24.8 ms), which fits the 33.3 ms/frame budget with ~10 ms headroom. Running
# at 30 Hz (vs the earlier 15 Hz) does two things: the rig redraws as smoothly as
# the camera, AND the 27-frame window fills twice as fast, halving the window-centre
# lag (867 ms → 433 ms). The 15 Hz figure made sense in the lightweight+CPU era
# (halved CPU load); with the conv work now on the Neural Engine it's cheap enough
# to run full-rate here. NOTE: `app.run_session` deliberately stays at 15 Hz —
# held-pose scoring doesn't need 30 Hz and the lower rate saves power/thermals over
# a multi-minute coaching session. This higher rate is a diagnostic-only choice.
#
# 2D inference is gated on a new camera-frame timestamp (see read_latest_with_ts,
# matching app.py) rather than a wall-clock throttle, so it runs once per unique
# frame; the 30 fps loop cap below bounds that rate. _POSE_INFERENCE_HZ remains
# the assumed rate used for the rig-lag estimate.
_POSE_INFERENCE_HZ = 30
_LIFT_HZ = 30
_LIFT_INTERVAL_S = 1.0 / _LIFT_HZ

# Per-H36M-joint visibility threshold. A joint draws (and any bone touching
# it draws) only when the joint's source 2D confidence meets this floor.
# 0.3 matches `head_lateral_tilt_2d`'s gate and the per-frame floor used
# elsewhere in the pipeline.
JOINT_VISIBILITY_THRESHOLD = 0.3

# H36M-17 skeleton bones. Per-frame visibility decides which subset draws.
H36M_BONES = [
    (0, 1),
    (1, 2),
    (2, 3),  # right leg (pelvis → R_hip → R_knee → R_ankle)
    (0, 4),
    (4, 5),
    (5, 6),  # left leg
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),  # pelvis → spine → thorax → neck → head
    (8, 11),
    (11, 12),
    (12, 13),  # left arm
    (8, 14),
    (14, 15),
    (15, 16),  # right arm
]


_GREEN = (90, 220, 90)
_AMBER = (90, 200, 220)
_GREY = (160, 160, 160)


def _draw_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int] = (240, 240, 240),
    scale: float = 0.55,
    thick: int = 1,
) -> None:
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


def _health_color(value: float, healthy: bool) -> tuple[int, int, int]:
    """Grey if the metric is NaN (not measurable), green if within the healthy
    band, amber otherwise."""
    if np.isnan(value):
        return _GREY
    return _GREEN if healthy else _AMBER


def _draw_metric_block(
    panel: np.ndarray,
    x: int,
    y: int,
    lines: list[tuple[str, tuple[int, int, int]]],
    line_h: int = 22,
    box_w: int = 250,
) -> None:
    """Draw a list of (text, color) over a translucent dark box so the metric
    readout stays legible on top of a busy camera image."""
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


def _make_panel(w: int, h: int, title: str) -> np.ndarray:
    panel = np.full((h + PANEL_PAD, w, 3), 22, dtype=np.uint8)
    _draw_text(panel, title, (10, 22), color=(255, 255, 255), scale=0.7, thick=2)
    return panel


def _draw_skeleton_2d(
    img: np.ndarray, kps: np.ndarray, scores: np.ndarray, threshold: float = 0.3
) -> None:
    for a, b in SKELETON:
        if scores[a] < threshold or scores[b] < threshold:
            continue
        pa = (int(kps[a, 0]), int(kps[a, 1]))
        pb = (int(kps[b, 0]), int(kps[b, 1]))
        cv2.line(img, pa, pb, (255, 200, 0), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(kps):
        if scores[i] < threshold:
            continue
        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1, cv2.LINE_AA)


def _draw_head_tilt_2d_overlay(
    img: np.ndarray, kps: np.ndarray, scores: np.ndarray, threshold: float = 0.3
) -> None:
    """Visualize the inputs to head_lateral_tilt_2d: mid-shoulders → nose line
    and the body-lateral reference (L_shoulder → R_shoulder). Makes it obvious
    which keypoints the measurement is actually reading from."""
    if (
        scores[NOSE] < threshold
        or scores[L_SHOULDER] < threshold
        or scores[R_SHOULDER] < threshold
    ):
        return
    l = kps[L_SHOULDER]
    r = kps[R_SHOULDER]
    n = kps[NOSE]
    mid = (l + r) / 2.0
    cv2.line(
        img,
        (int(l[0]), int(l[1])),
        (int(r[0]), int(r[1])),
        (0, 180, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        img,
        (int(mid[0]), int(mid[1])),
        (int(n[0]), int(n[1])),
        (60, 60, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(img, (int(mid[0]), int(mid[1])), 5, (0, 180, 255), -1, cv2.LINE_AA)


def _h36m_joint_confidences(scores: np.ndarray) -> np.ndarray:
    """Map COCO-17 confidences (17,) to per-H36M-17-joint confidences (17,).

    Mirrors the synthesis rules in `pose3d.coco17_to_h36m17`: pelvis / spine /
    thorax are synthesized from multiple COCO joints, so their confidence is
    the minimum across their sources. Drives per-joint visibility in the 3D
    panel, so the rig fills in one joint at a time as more of the body enters
    the frame rather than appearing all-at-once at a single global threshold.
    """
    out = np.zeros(17, dtype=np.float32)
    out[0] = float(min(scores[L_HIP], scores[R_HIP]))  # pelvis = midpoint of hips
    out[1] = float(scores[R_HIP])  # R_hip
    out[2] = float(scores[R_KNEE])  # R_knee
    out[3] = float(scores[R_ANKLE])  # R_ankle
    out[4] = float(scores[L_HIP])  # L_hip
    out[5] = float(scores[L_KNEE])  # L_knee
    out[6] = float(scores[L_ANKLE])  # L_ankle
    out[7] = float(  # spine: shoulders + hips
        min(scores[L_SHOULDER], scores[R_SHOULDER], scores[L_HIP], scores[R_HIP])
    )
    out[8] = float(
        min(scores[L_SHOULDER], scores[R_SHOULDER])
    )  # thorax = mid-shoulders
    out[9] = float(scores[NOSE])  # neck   (synth from nose)
    out[10] = float(scores[NOSE])  # head   (synth from nose)
    out[11] = float(scores[L_SHOULDER])
    out[12] = float(scores[7])  # COCO L_elbow
    out[13] = float(scores[9])  # COCO L_wrist
    out[14] = float(scores[R_SHOULDER])
    out[15] = float(scores[8])  # COCO R_elbow
    out[16] = float(scores[10])  # COCO R_wrist
    return out


def _render_rig_3d(
    panel: np.ndarray,
    rig: Optional[np.ndarray],
    box: tuple[int, int, int, int],
    joint_conf: np.ndarray,
    threshold: float = JOINT_VISIBILITY_THRESHOLD,
    mirror: bool = False,
) -> None:
    """Render an H36M-17 rig inside the bounding box (x0, y0, w, h).

    Design notes:

    1. **No y-negation.** MotionBERT outputs image-y-down (head at y ≈ -0.5,
       feet at y ≈ +0.5); OpenCV canvas y also grows down. So `rig[:, 1]` maps
       to canvas-y directly. The matplotlib notebooks flip it because mpl's 3D
       Z-axis is up; OpenCV doesn't need that.
    2. **Per-joint visibility.** Each H36M joint draws iff its 2D-source
       confidence ≥ `threshold`. Bones draw iff both endpoints draw. This
       produces graceful per-joint reveal as more of the body enters the
       frame — hips appear before knees, knees before ankles — instead of
       an all-or-nothing global gate that hides the rig until feet are seen.
    3. **Scaling by visible joints.** The bbox used for scaling includes only
       visible joints, so the figure always fills the panel regardless of how
       much body is in frame.
    """
    x0, y0, w, h = box
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (60, 60, 60), 1)
    if rig is None:
        _draw_text(
            panel,
            "warming up (need 27 frames)",
            (x0 + 12, y0 + h // 2),
            color=(160, 160, 160),
            scale=0.5,
        )
        return

    visible = joint_conf >= threshold
    visible_idxs = np.where(visible)[0]
    if visible_idxs.size == 0:
        _draw_text(
            panel,
            "no joints detected",
            (x0 + 12, y0 + h // 2),
            color=(160, 160, 160),
            scale=0.5,
        )
        return

    pts = rig[:, :2].astype(np.float32)
    bbox_pts = pts[visible_idxs]
    bbox_min = bbox_pts.min(axis=0)
    bbox_max = bbox_pts.max(axis=0)
    span = max(float((bbox_max - bbox_min).max()), 1e-3)

    margin = 16
    avail = float(min(w, h) - 2 * margin)
    scale = avail / span
    centre = (bbox_min + bbox_max) / 2.0
    panel_centre = np.array([x0 + w / 2.0, y0 + h / 2.0], dtype=np.float32)
    screen = (pts - centre) * scale + panel_centre

    # Mirror horizontally within the box so the rig's left/right matches the
    # mirrored camera panel (display-only; rig coords themselves are untouched).
    if mirror:
        screen[:, 0] = (2 * x0 + w) - screen[:, 0]

    rect = (x0, y0, w, h)

    for a, b in H36M_BONES:
        if not (visible[a] and visible[b]):
            continue
        p1 = (int(screen[a, 0]), int(screen[a, 1]))
        p2 = (int(screen[b, 0]), int(screen[b, 1]))
        ok, p1c, p2c = cv2.clipLine(rect, p1, p2)
        if ok:
            cv2.line(panel, p1c, p2c, (0, 200, 255), 2, cv2.LINE_AA)

    for i in range(17):
        if not visible[i]:
            continue
        x, y = int(screen[i, 0]), int(screen[i, 1])
        if x0 <= x <= x0 + w and y0 <= y <= y0 + h:
            cv2.circle(panel, (x, y), 3, (0, 200, 255), -1, cv2.LINE_AA)


def _try_load_pose3d():
    """Optional — Pose3D needs MotionBERT weights and vendor code. Don't crash
    the diagnostic app if either is missing; just disable the 3D panel."""
    try:
        from pose3d import Pose3D, Pose3DBuffer, coco17_to_h36m17
    except Exception as exc:  # pragma: no cover - environment-dependent
        print(f"[test_2D_3D] 3D lifter unavailable: {exc}")
        return None, None, None

    try:
        lifter = Pose3D()
    except Exception as exc:  # pragma: no cover - weights missing etc.
        print(f"[test_2D_3D] Pose3D() construction failed: {exc}")
        return None, None, None

    return lifter, Pose3DBuffer(lifter), coco17_to_h36m17


def _tilt_status(
    tilt: float, target: float, tol: float
) -> tuple[str, tuple[int, int, int]]:
    if np.isnan(tilt):
        return "no measurement", (160, 160, 160)
    if abs(tilt - target) <= tol:
        return "IN TARGET", (90, 220, 90)
    return "out of target", (90, 200, 220)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="test_2D_3D", description="2D/3D pose pipeline diagnostic"
    )
    p.add_argument("--source", choices=["webcam", "ip"], default="webcam")
    p.add_argument(
        "--url",
        default=None,
        help="IP Webcam MJPEG URL (required for --source ip)",
    )
    args = p.parse_args(argv)
    if args.source == "ip" and not args.url:
        p.error("--source ip requires --url (e.g. http://192.168.1.42:8080/video)")
    return args


def run(argv=None) -> None:
    args = _parse_args(argv)
    exercise = NeckStretchLeft()
    target = exercise.target.joints[0].target_deg
    tol = exercise.target.joints[0].tolerance_deg

    print("[test_2D_3D] Loading 2D pose (default: balanced + CoreML)...")
    pose2d = Pose2D()  # balanced + coreml default

    print("[test_2D_3D] Loading 3D lifter (MotionBERT-Lite)...")
    lifter, buf, coco_to_h36m = _try_load_pose3d()
    have_3d = lifter is not None

    print(f"[test_2D_3D] Opening {args.source} source...")
    cam = build_capture(args.source, args.url, width=CAM_WIDTH, height=CAM_HEIGHT)
    cam.start()

    last_rig_3d: Optional[np.ndarray] = None
    last_kps: Optional[np.ndarray] = None
    last_scores: Optional[np.ndarray] = None
    last_processed_ts = -1.0
    last_lift_ts = 0.0

    # Live profiling state. `prof` holds the most recent per-stage ms; `lift_stamps`
    # is a rolling window of lift timestamps used to compute the actual lift fps.
    prof = {"infer_ms": 0.0, "lift_ms": 0.0}
    lift_stamps: list[float] = []
    # MotionBERT returns the centre of its 27-frame window, so the rig trails
    # live by ~half a window at the push rate. Surfaced in the readout.
    lift_lag_ms = (
        (lifter.window_size // 2) / _POSE_INFERENCE_HZ * 1000 if have_3d else 0.0
    )

    # Smooth FPS over a small window.
    frame_times: list[float] = []

    window_name = "test_2D_3D"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            t0 = time.time()
            read = cam.read_latest_with_ts(timeout=0.5)
            if read is None:
                continue
            frame_bgr, frame_ts = read

            # Run 2D inference only on a new camera frame (timestamp gate, matching
            # app.py); reuse the last result on duplicate frames so the panels still
            # render at the loop rate without paying ORT cost.
            if frame_ts != last_processed_ts:
                last_processed_ts = frame_ts
                _ti = time.perf_counter()
                last_kps, last_scores = pose2d.infer(frame_bgr)
                prof["infer_ms"] = (time.perf_counter() - _ti) * 1000
                if have_3d and coco_to_h36m is not None and buf is not None:
                    buf.push(coco_to_h36m(last_kps, last_scores))
            kps, scores = last_kps, last_scores
            assert kps is not None and scores is not None

            # Full 2D cookbook + camera-view classification (everything the
            # real app now computes).
            tilt_2d = head_lateral_tilt_2d(kps, scores)
            cva = craniovertebral_angle_2d(kps, scores)
            fwd_head = forward_head_offset_normalized_2d(kps, scores)
            neck_flex = neck_flexion_2d(kps, scores)
            sh_asym = shoulder_elevation_asymmetry_2d(kps, scores)
            view = classify_view(kps, scores)

            # --- sign-debug for head_lateral_tilt_2d ---
            # Surfaces the two quantities that determine the SIGN: the nose's
            # lateral offset from mid-shoulder (flips with tilt direction) and
            # the up_component (must stay > 0 — if it goes <= 0 the atan2 angle
            # wraps past 90° and the sign stops behaving monotonically).
            _mid = (kps[L_SHOULDER] + kps[R_SHOULDER]) / 2.0
            _sh_dx = float(kps[R_SHOULDER, 0] - kps[L_SHOULDER, 0])
            dbg_nose_dx = float(kps[NOSE, 0] - _mid[0]) * (1.0 if _sh_dx > 0 else -1.0)
            dbg_up = -float(kps[NOSE, 1] - _mid[1])  # nose height above shoulders

            tilt_3d = float("nan")
            joint_conf = _h36m_joint_confidences(scores)
            if have_3d and buf is not None and lifter is not None:
                # 3D lift gated on wall-clock so it stays ≈ _LIFT_HZ.
                if buf.ready() and (t0 - last_lift_ts) >= _LIFT_INTERVAL_S:
                    _tl = time.perf_counter()
                    last_rig_3d = buf.lift(
                        frame_h=frame_bgr.shape[0], frame_w=frame_bgr.shape[1]
                    )
                    prof["lift_ms"] = (time.perf_counter() - _tl) * 1000
                    lift_stamps.append(time.time())
                    last_lift_ts = t0
                if last_rig_3d is not None:
                    tilt_3d = head_lateral_tilt_3d(last_rig_3d)

            # ----- display mirroring (selfie view) -----
            # Inference + all measurements above ran on the un-mirrored frame, so
            # they are unchanged. Here we mirror only what's DRAWN, so on-screen
            # motion matches a mirror (intuitive). The drawing keypoints get their
            # x flipped to line up with the flipped image; `kps`/`scores` used for
            # measurement are untouched.
            W = frame_bgr.shape[1]
            disp_frame = cv2.flip(frame_bgr, 1)
            kps_disp = kps.copy()
            kps_disp[:, 0] = (W - 1) - kps[:, 0]

            # ----- compose panels -----
            cam_panel = _make_panel(CAM_WIDTH, CAM_HEIGHT, "1. Camera (mirror)")
            cam_panel[30 : 30 + CAM_HEIGHT, :CAM_WIDTH] = disp_frame
            # Camera-view classifier (B3): green when the framing is valid for the
            # active exercise, amber otherwise — mirrors the app's view gate.
            view_ok = view in exercise.target.valid_views
            _draw_metric_block(
                cam_panel,
                12,
                52,
                [(f"view: {view.value}", _GREEN if view_ok else _AMBER)],
                box_w=210,
            )

            two_d_panel = _make_panel(CAM_WIDTH, CAM_HEIGHT, "2. 2D pose + metrics")
            overlay = disp_frame.copy()
            _draw_skeleton_2d(overlay, kps_disp, scores)
            _draw_head_tilt_2d_overlay(overlay, kps_disp, scores)
            two_d_panel[30 : 30 + CAM_HEIGHT, :CAM_WIDTH] = overlay
            # B2 cookbook metrics, colored by their published healthy bands.
            _draw_metric_block(
                two_d_panel,
                12,
                52,
                [
                    (
                        f"head tilt : {tilt_2d:+5.1f}deg"
                        if not np.isnan(tilt_2d)
                        else "head tilt :   --",
                        _health_color(tilt_2d, abs(tilt_2d - target) <= tol),
                    ),
                    (
                        f"CVA       : {cva:5.1f}deg"
                        if not np.isnan(cva)
                        else "CVA       :   --",
                        _health_color(cva, cva >= 50.0),
                    ),
                    (
                        f"fwd head  : {fwd_head:5.2f}"
                        if not np.isnan(fwd_head)
                        else "fwd head  :   --",
                        _health_color(fwd_head, fwd_head < 0.30),
                    ),
                    (
                        f"neck flex : {neck_flex:+5.1f}deg"
                        if not np.isnan(neck_flex)
                        else "neck flex :   --",
                        _health_color(neck_flex, abs(neck_flex) < 25.0),
                    ),
                    (
                        f"sh asym   : {sh_asym:+5.2f}"
                        if not np.isnan(sh_asym)
                        else "sh asym   :   --",
                        _health_color(sh_asym, abs(sh_asym) < 0.05),
                    ),
                    # sign-debug: lat should flip +/- as you tilt; up must stay > 0.
                    (
                        f"  lat={dbg_nose_dx:+5.0f} up={dbg_up:+5.0f}",
                        _GREY if dbg_up > 0 else (90, 90, 220),
                    ),
                ],
            )

            three_d_panel = _make_panel(
                CAM_WIDTH, CAM_HEIGHT, "3. 3D rig (MotionBERT-Lite)"
            )
            _render_rig_3d(
                three_d_panel,
                last_rig_3d,
                box=(20, 40, CAM_WIDTH - 40, CAM_HEIGHT - 20),
                joint_conf=joint_conf,
                mirror=True,  # match the mirrored camera + 2D panels
            )

            # ----- readouts -----
            # Camera panel: per-keypoint confidences for the ones the 2D measurement uses,
            # plus the hips that the 3D measurement secretly leans on.
            base_y = 30 + CAM_HEIGHT + 22
            for i, (name, idx) in enumerate(
                [
                    ("nose", NOSE),
                    ("L_sh", L_SHOULDER),
                    ("R_sh", R_SHOULDER),
                    ("L_hip", L_HIP),
                    ("R_hip", R_HIP),
                ]
            ):
                s = float(scores[idx])
                color = (
                    (90, 220, 90)
                    if s >= 0.5
                    else (90, 200, 220)
                    if s >= 0.3
                    else (90, 90, 220)
                )
                _draw_text(
                    cam_panel, f"{name}: {s:.2f}", (12 + i * 120, base_y), color=color
                )

            now = time.time()
            frame_times.append(now)
            frame_times = [t for t in frame_times if now - t < 1.0]
            fps = len(frame_times)
            _draw_text(
                cam_panel,
                f"fps={fps}  press q to quit",
                (12, 30 + CAM_HEIGHT + 50),
                color=(180, 180, 180),
                scale=0.5,
            )

            # 2D panel: the canonical measurement.
            label_2d, color_2d = _tilt_status(tilt_2d, target, tol)
            tilt_str_2d = f"{tilt_2d:+.1f}°" if not np.isnan(tilt_2d) else "--"
            _draw_text(
                two_d_panel,
                f"head_lateral_tilt_2d = {tilt_str_2d}    target {target:+.0f}° ± {tol:.0f}°    [{label_2d}]",
                (12, 30 + CAM_HEIGHT + 22),
                color=color_2d,
                scale=0.55,
                thick=1,
            )
            _draw_text(
                two_d_panel,
                "uses: nose, L_shoulder, R_shoulder (COCO-17). desk-camera safe.",
                (12, 30 + CAM_HEIGHT + 50),
                color=(170, 170, 170),
                scale=0.45,
            )

            # 3D panel: legacy comparison + which lifted joints feed into it.
            if not have_3d:
                _draw_text(
                    three_d_panel,
                    "3D lifter unavailable (see console).",
                    (12, 30 + CAM_HEIGHT + 22),
                    color=(170, 170, 170),
                )
            else:
                tilt_str_3d = f"{tilt_3d:+.1f}°" if not np.isnan(tilt_3d) else "--"
                label_3d, color_3d = _tilt_status(tilt_3d, target, tol)
                _draw_text(
                    three_d_panel,
                    f"head_lateral_tilt_3d = {tilt_str_3d}    [{label_3d}]",
                    (12, 30 + CAM_HEIGHT + 22),
                    color=color_3d,
                )
                visible_mask = joint_conf >= JOINT_VISIBILITY_THRESHOLD
                n_visible = int(visible_mask.sum())
                n_lower_visible = int(visible_mask[[1, 2, 3, 4, 5, 6]].sum())
                _draw_text(
                    three_d_panel,
                    f"{n_visible}/17 joints visible   (lower body {n_lower_visible}/6)",
                    (12, 30 + CAM_HEIGHT + 50),
                    color=(170, 170, 170),
                    scale=0.45,
                )
                # Live profiling: actual lift rate + cost + the temporal lag the
                # window-centre output imposes. Overlaid in the rig area so the
                # "why is 3D laggy" question is answerable on-screen.
                now = time.time()
                lift_stamps[:] = [s for s in lift_stamps if now - s < 1.0]
                lift_fps = len(lift_stamps)
                _draw_metric_block(
                    three_d_panel,
                    12,
                    62,
                    [
                        (f"lift cost : {prof['lift_ms']:4.1f}ms", _GREY),
                        (
                            f"lift rate : {lift_fps:2d} fps",
                            _GREEN if lift_fps >= 10 else _AMBER,
                        ),
                        (f"2D infer  : {prof['infer_ms']:4.1f}ms", _GREY),
                        (f"rig lag   : ~{lift_lag_ms:.0f}ms (window centre)", _AMBER),
                    ],
                    box_w=240,
                )

            canvas = np.hstack([cam_panel, two_d_panel, three_d_panel])
            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            # Cap UI loop at 30 fps (matches a typical webcam's native rate). With
            # inference throttled to 15 Hz separately, the camera panel still updates
            # every loop iteration but inference results refresh every other frame —
            # plenty of perceived smoothness without spinning the canvas at 60 fps.
            elapsed = time.time() - t0
            if elapsed < 1 / 30:
                time.sleep(1 / 30 - elapsed)
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()

import numpy as np

NOSE = 0
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def angle_between_3_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees, B is the vertex."""
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def knee_angles(kps: np.ndarray) -> tuple[float, float]:
    left = angle_between_3_points(kps[L_HIP], kps[L_KNEE], kps[L_ANKLE])
    right = angle_between_3_points(kps[R_HIP], kps[R_KNEE], kps[R_ANKLE])
    return left, right


def torso_lean_deg(kps: np.ndarray) -> float:
    """Angle between the vector (mid-hip -> mid-shoulder) and vertical (up)."""
    mid_shoulder = (kps[L_SHOULDER] + kps[R_SHOULDER]) / 2.0
    mid_hip = (kps[L_HIP] + kps[R_HIP]) / 2.0
    v = mid_shoulder - mid_hip
    vertical = np.array([0.0, -1.0])
    cos = np.dot(v, vertical) / (np.linalg.norm(v) + 1e-9)
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def hip_below_knee(kps: np.ndarray) -> bool:
    """In image coords (y down), hip below knee means hip_y > knee_y."""
    mid_hip_y = (kps[L_HIP, 1] + kps[R_HIP, 1]) / 2.0
    mid_knee_y = (kps[L_KNEE, 1] + kps[R_KNEE, 1]) / 2.0
    return bool(mid_hip_y > mid_knee_y)


def knee_valgus_ratio(kps: np.ndarray) -> tuple[float, float]:
    """Signed normalized distance: (knee_x - ankle_x) / hip_width.
    Positive = knee inward of ankle on that side. Returns (left, right).
    Used as a valgus signal; abs value > 0.15 considered valgus."""
    hip_width = abs(kps[R_HIP, 0] - kps[L_HIP, 0]) + 1e-6
    l = (kps[L_KNEE, 0] - kps[L_ANKLE, 0]) / hip_width
    r = (kps[R_ANKLE, 0] - kps[R_KNEE, 0]) / hip_width
    return float(l), float(r)


def head_lateral_tilt_2d(
    kps_2d: np.ndarray,
    scores: np.ndarray | None = None,
    score_threshold: float = 0.3,
) -> float:
    """Signed head-lateral-tilt angle in degrees, computed in image space from
    nose + shoulders only — robust to lower-body occlusion (desk-camera, seated).

    Sign convention matches `angles_3d.head_lateral_tilt_3d`:
    positive = head tilted toward the body's R_shoulder side (body_lateral = R - L);
    negative = tilted toward the L_shoulder side. Independent of camera mirroring
    because the lateral reference is the body's own shoulder vector.

    Up reference is image-vertical (the spec accepts camera-roll dependence
    for the phone-on-tripod target). Returns NaN if nose / either shoulder has
    confidence below `score_threshold`, or shoulders are coincident.
    """
    if scores is not None:
        if (
            scores[NOSE] < score_threshold
            or scores[L_SHOULDER] < score_threshold
            or scores[R_SHOULDER] < score_threshold
        ):
            return float("nan")

    nose = kps_2d[NOSE]
    l_sh = kps_2d[L_SHOULDER]
    r_sh = kps_2d[R_SHOULDER]
    mid_sh = (l_sh + r_sh) / 2.0

    shoulder_dx = float(r_sh[0] - l_sh[0])
    if abs(shoulder_dx) < 1e-6:
        return float("nan")

    head_dx = float(nose[0] - mid_sh[0])
    head_dy = float(nose[1] - mid_sh[1])  # image y grows down

    # Project head vector onto body axes. Lateral direction is sign(shoulder_dx)
    # along image-x (mirror-independent). Up direction is image-up = (0, -1).
    lat_component = head_dx * (1.0 if shoulder_dx > 0 else -1.0)
    up_component = -head_dy

    if lat_component == 0.0 and up_component == 0.0:
        return float("nan")

    return float(np.degrees(np.arctan2(lat_component, up_component)))


# ---- Office-syndrome posture cookbook (2D, image-plane) ----
# Each function follows the same shape as `head_lateral_tilt_2d`: pure 2D,
# NaN-gated on the inputs it reads, signed where the sign carries meaning.
# Source: posture-coach research synthesis 2026-05-23 (clinical CVA validated
# at Pearson r > 0.98 vs goniometer; other formulas are common-practice
# heuristics widely used by open-source posture monitors).


def craniovertebral_angle_2d(
    kps_2d: np.ndarray,
    scores: np.ndarray | None = None,
    score_threshold: float = 0.3,
    side: str = "auto",
) -> float:
    """Craniovertebral angle (CVA) in degrees — forward-head-posture proxy.

    Definition: angle between (a) the horizontal line through the shoulder and
    (b) the line from shoulder→ear. Computed in image-pixel coords with y-down.
    Healthy ≥ 50°; < 48° indicates forward-head posture per published thresholds.

    `side` ∈ {"left", "right", "auto"}. `"auto"` picks the side whose ear AND
    shoulder both have higher confidence — useful for three-quarter and frontal
    views where one ear may be occluded. NaN if neither side has clean inputs.
    """
    sides: list[tuple[int, int]] = []
    if side in ("left", "auto"):
        sides.append((L_EAR, L_SHOULDER))
    if side in ("right", "auto"):
        sides.append((R_EAR, R_SHOULDER))

    best_angle = float("nan")
    best_score = -1.0
    for ear_i, sh_i in sides:
        if scores is not None and (
            scores[ear_i] < score_threshold or scores[sh_i] < score_threshold
        ):
            continue
        ear = kps_2d[ear_i]
        sh = kps_2d[sh_i]
        # vector shoulder → ear. atan2 of (shoulder_y - ear_y, ear_x - shoulder_x)
        # gives the angle above horizontal (positive when ear is above shoulder,
        # which it is in a normal upright head).
        dy = float(sh[1] - ear[1])  # image-y grows down: ear above shoulder → dy > 0
        dx = float(ear[0] - sh[0])
        if dx == 0.0 and dy == 0.0:
            continue
        angle = float(np.degrees(np.arctan2(dy, abs(dx) + 1e-6)))
        s = float(min(scores[ear_i], scores[sh_i])) if scores is not None else 1.0
        if s > best_score:
            best_score = s
            best_angle = angle
    return best_angle


def forward_head_offset_normalized_2d(
    kps_2d: np.ndarray,
    scores: np.ndarray | None = None,
    score_threshold: float = 0.3,
) -> float:
    """Front-view forward-head fallback: horizontal ear-shoulder offset
    normalized by shoulder width. ~0.30 is the commonly-cited threshold.

    Returns the MAX of left and right offsets (so the worse side wins) — front-
    facing forward-head shows on both sides roughly equally. Skips a side if its
    ear or shoulder confidence is below threshold. NaN if no usable side or if
    shoulder width is collapsed.
    """
    if scores is not None:
        if scores[L_SHOULDER] < score_threshold or scores[R_SHOULDER] < score_threshold:
            return float("nan")
    shoulder_width = abs(float(kps_2d[R_SHOULDER, 0] - kps_2d[L_SHOULDER, 0]))
    if shoulder_width < 1e-6:
        return float("nan")

    offsets: list[float] = []
    for ear_i, sh_i in ((L_EAR, L_SHOULDER), (R_EAR, R_SHOULDER)):
        if scores is not None and scores[ear_i] < score_threshold:
            continue
        offsets.append(abs(float(kps_2d[ear_i, 0] - kps_2d[sh_i, 0])) / shoulder_width)
    if not offsets:
        return float("nan")
    return max(offsets)


def neck_flexion_2d(
    kps_2d: np.ndarray,
    scores: np.ndarray | None = None,
    score_threshold: float = 0.3,
    side: str = "auto",
) -> float:
    """Neck flexion/extension angle in degrees: angle between shoulder→ear and
    image vertical `(0, -1)`. Side-view metric; healthy < 25° flexion.

    Positive = forward flexion (ear forward of shoulder along image-x);
    negative = extension (ear behind shoulder). Magnitude reflects how far
    the head deviates from upright. `side="auto"` picks whichever side has
    the higher-confidence ear+shoulder pair.
    """
    sides: list[tuple[int, int, float]] = []
    pairs: list[tuple[int, int]] = []
    if side in ("left", "auto"):
        pairs.append((L_EAR, L_SHOULDER))
    if side in ("right", "auto"):
        pairs.append((R_EAR, R_SHOULDER))
    for ear_i, sh_i in pairs:
        if scores is not None and (
            scores[ear_i] < score_threshold or scores[sh_i] < score_threshold
        ):
            continue
        s = float(min(scores[ear_i], scores[sh_i])) if scores is not None else 1.0
        sides.append((ear_i, sh_i, s))
    if not sides:
        return float("nan")
    ear_i, sh_i, _ = max(sides, key=lambda t: t[2])
    dx = float(kps_2d[ear_i, 0] - kps_2d[sh_i, 0])
    dy = float(kps_2d[ear_i, 1] - kps_2d[sh_i, 1])
    # Image-up = (0, -1). Angle from image-up to (dx, dy) is atan2(dx, -dy).
    up_y = -dy  # how far ear is above shoulder
    if dx == 0.0 and up_y == 0.0:
        return float("nan")
    return float(np.degrees(np.arctan2(dx, up_y)))


def shoulder_elevation_asymmetry_2d(
    kps_2d: np.ndarray,
    scores: np.ndarray | None = None,
    score_threshold: float = 0.3,
) -> float:
    """Signed shoulder-y delta normalized by shoulder width.

    Positive = L_shoulder higher (smaller image-y) than R_shoulder;
    negative = R higher. Healthy magnitude < ~0.05 (≈3°). NaN if either
    shoulder is below threshold or shoulders are coincident.
    """
    if scores is not None and (
        scores[L_SHOULDER] < score_threshold or scores[R_SHOULDER] < score_threshold
    ):
        return float("nan")
    shoulder_width = abs(float(kps_2d[R_SHOULDER, 0] - kps_2d[L_SHOULDER, 0]))
    if shoulder_width < 1e-6:
        return float("nan")
    # Positive when L_shoulder.y < R_shoulder.y, i.e. L is higher (smaller image y).
    delta = float(kps_2d[R_SHOULDER, 1] - kps_2d[L_SHOULDER, 1])
    return delta / shoulder_width


def shoulder_protraction_ratio_2d(
    kps_2d: np.ndarray,
    scores: np.ndarray | None = None,
    baseline_width_px: float = 0.0,
    score_threshold: float = 0.3,
) -> float:
    """Current shoulder width divided by user-baseline shoulder width.

    Pulling shoulders forward (protraction / rounded shoulders) reduces the
    *apparent* shoulder width from a frontal camera, so this ratio drops below
    1.0. Ratio above 1.0 indicates retraction (chest opening). NaN if either
    shoulder is below threshold or no baseline width was provided.
    """
    if baseline_width_px <= 0.0:
        return float("nan")
    if scores is not None and (
        scores[L_SHOULDER] < score_threshold or scores[R_SHOULDER] < score_threshold
    ):
        return float("nan")
    width = abs(float(kps_2d[R_SHOULDER, 0] - kps_2d[L_SHOULDER, 0]))
    if width < 1e-6:
        return float("nan")
    return width / baseline_width_px


def wrist_extension_2d(
    kps_2d: np.ndarray,
    scores: np.ndarray | None = None,
    side: str = "auto",
    score_threshold: float = 0.3,
) -> float:
    """Vertical wrist-to-elbow offset normalized by forearm length.

    Positive = wrist BELOW elbow (relaxed, neutral); negative = wrist ABOVE
    elbow (extended / dorsiflexed — the typing-strain posture). Returns the
    side with higher-confidence inputs when `side="auto"`. NaN if no usable
    side.
    """
    pairs: list[tuple[int, int, float]] = []
    candidates: list[tuple[int, int]] = []
    if side in ("left", "auto"):
        candidates.append((L_WRIST, L_ELBOW))
    if side in ("right", "auto"):
        candidates.append((R_WRIST, R_ELBOW))
    for wrist_i, elbow_i in candidates:
        if scores is not None and (
            scores[wrist_i] < score_threshold or scores[elbow_i] < score_threshold
        ):
            continue
        s = float(min(scores[wrist_i], scores[elbow_i])) if scores is not None else 1.0
        pairs.append((wrist_i, elbow_i, s))
    if not pairs:
        return float("nan")
    wrist_i, elbow_i, _ = max(pairs, key=lambda t: t[2])
    forearm_len = float(np.linalg.norm(kps_2d[wrist_i] - kps_2d[elbow_i]))
    if forearm_len < 1e-6:
        return float("nan")
    # Image y grows down; wrist BELOW elbow means wrist.y > elbow.y → positive.
    return float(kps_2d[wrist_i, 1] - kps_2d[elbow_i, 1]) / forearm_len

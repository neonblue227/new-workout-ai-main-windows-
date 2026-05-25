import numpy as np

# H36M-17 indices used in this module.
PELVIS = 0
R_HIP = 1
R_KNEE = 2
R_ANKLE = 3
L_HIP = 4
L_KNEE = 5
L_ANKLE = 6
THORAX = 8
HEAD = 10


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def body_frame_axes(
    kps_3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (body_up, body_lateral, body_forward) as unit vectors in MotionBERT's frame.

    - body_up: pelvis → thorax (head-ward along the torso).
    - body_lateral: L_hip → R_hip projected orthogonal to body_up.
    - body_forward: body_up × body_lateral (sagittal axis, forward through the chest).
    """
    up = _norm(kps_3d[THORAX] - kps_3d[PELVIS])
    hip = kps_3d[R_HIP] - kps_3d[L_HIP]
    lateral_raw = hip - float(np.dot(hip, up)) * up
    lateral = _norm(lateral_raw)
    forward = np.cross(up, lateral)
    return up, lateral, forward


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two vectors in degrees.

    Returns NaN if either vector has near-zero length.
    """
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos = float(np.dot(v1, v2) / (n1 * n2))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def knee_flexion_3d(kps_3d: np.ndarray) -> tuple[float, float]:
    """Per-knee flexion in degrees, computed in full 3D. Returns (left, right).

    180° = straight leg, smaller values = more bent. NaN if any bone has zero length.
    """
    left = _angle_deg(
        kps_3d[L_HIP] - kps_3d[L_KNEE],
        kps_3d[L_ANKLE] - kps_3d[L_KNEE],
    )
    right = _angle_deg(
        kps_3d[R_HIP] - kps_3d[R_KNEE],
        kps_3d[R_ANKLE] - kps_3d[R_KNEE],
    )
    return left, right


# Image vertical in MotionBERT's normalized 3D frame (image-y down → up = -y).
# This equals gravity-up when the camera is upright; the spec accepts this
# camera-roll dependency for the current target (phone-on-tripod).
IMAGE_UP = np.array([0.0, -1.0, 0.0], dtype=np.float32)


def torso_lean_3d(kps_3d: np.ndarray) -> float:
    """Angle in degrees between body_up (pelvis → thorax) and image vertical."""
    torso = kps_3d[THORAX] - kps_3d[PELVIS]
    return _angle_deg(torso, IMAGE_UP)


def _project_frontal(p: np.ndarray, lateral: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Project a 3D point onto the (lateral, up) frontal plane.
    Returns a 2-vector (lat_coord, up_coord)."""
    return np.array([float(np.dot(p, lateral)), float(np.dot(p, up))], dtype=np.float32)


def valgus_offset_3d(kps_3d: np.ndarray) -> tuple[float, float]:
    """Signed perpendicular offset of each knee from the hip-ankle line in the
    body's frontal plane, normalized by 3D shin length.

    Sign convention: positive = medial (knee toward midline = valgus),
    negative = lateral (knee outside the hip-ankle line).
    Returns (left, right).
    """
    up, lateral, _ = body_frame_axes(kps_3d)

    def _one_side(hip_i: int, knee_i: int, ankle_i: int, medial_sign: float) -> float:
        h2 = _project_frontal(kps_3d[hip_i], lateral, up)
        k2 = _project_frontal(kps_3d[knee_i], lateral, up)
        a2 = _project_frontal(kps_3d[ankle_i], lateral, up)
        d = a2 - h2
        line_len = float(np.linalg.norm(d))
        if line_len < 1e-9:
            return 0.0
        # Signed 2D cross product gives signed perpendicular distance × |d|.
        cross = float(d[0] * (k2[1] - h2[1]) - d[1] * (k2[0] - h2[0]))
        perp = cross / line_len
        shin = float(np.linalg.norm(kps_3d[ankle_i] - kps_3d[knee_i]))
        if shin < 1e-9:
            return 0.0
        return medial_sign * perp / shin

    # See spec §Valgus: for R leg the cross-product convention puts medial on the
    # negative side, so we flip the sign. For L leg the natural sign is medial-positive.
    right = _one_side(R_HIP, R_KNEE, R_ANKLE, medial_sign=-1.0)
    left = _one_side(L_HIP, L_KNEE, L_ANKLE, medial_sign=+1.0)
    return left, right


def head_lateral_tilt_3d(kps_3d: np.ndarray) -> float:
    """Signed head-tilt angle (degrees) in the body's frontal plane.

    Positive = head tilted toward body_lateral (+) direction (L_hip → R_hip);
    negative = tilted the opposite way. Use the sign convention defined by
    body_frame_axes(). Returns NaN if the body frame is degenerate.
    """
    up, lateral, _ = body_frame_axes(kps_3d)
    # Reject degenerate frames (zero vectors come back from _norm unchanged).
    if float(np.linalg.norm(up)) < 1e-9 or float(np.linalg.norm(lateral)) < 1e-9:
        return float("nan")
    head_vec = kps_3d[HEAD] - kps_3d[THORAX]
    if float(np.linalg.norm(head_vec)) < 1e-9:
        return float("nan")
    lat_component = float(np.dot(head_vec, lateral))
    up_component = float(
        np.dot(head_vec, up)
    )  # up is pelvis→thorax; head is further along up
    return float(np.degrees(np.arctan2(lat_component, up_component)))

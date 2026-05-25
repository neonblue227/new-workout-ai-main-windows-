from analysis.angles import (
    knee_angles,
    torso_lean_deg,
    knee_valgus_ratio,
)
from analysis.angles_3d import (
    knee_flexion_3d,
    torso_lean_3d,
    valgus_offset_3d,
)
from analysis.types import PoseFrame, RepAnalysis, RuleViolation

VALGUS_THRESHOLD = 0.15
VALGUS_THRESHOLD_3D = 0.10
VALGUS_SAT_3D = 0.30
DEPTH_FULL_DEG = 90.0
DEPTH_ZERO_DEG = 130.0


def score_rep(
    bottom_frame: PoseFrame, descent_ms: int, ascent_ms: int, rep_index: int = 0
) -> RepAnalysis:
    """Score a completed rep. Dispatches to the 3D scorer when keypoints_3d
    is present; falls back to the 2D scorer otherwise."""
    if bottom_frame.keypoints_3d is not None:
        return _score_rep_3d(bottom_frame, descent_ms, ascent_ms, rep_index)
    return _score_rep_2d(bottom_frame, descent_ms, ascent_ms, rep_index)


def _score_rep_2d(
    bottom_frame: PoseFrame, descent_ms: int, ascent_ms: int, rep_index: int = 0
) -> RepAnalysis:
    kps = bottom_frame.keypoints_2d
    violations: list[RuleViolation] = []
    components: dict[str, int] = {}

    l_knee_ang, r_knee_ang = knee_angles(kps)
    mean_knee_ang = (l_knee_ang + r_knee_ang) / 2.0

    # --- Depth (30 pts) — linear ramp on mean knee flexion ---
    if mean_knee_ang <= DEPTH_FULL_DEG:
        components["depth"] = 30
    elif mean_knee_ang >= DEPTH_ZERO_DEG:
        components["depth"] = 0
        violations.append(
            RuleViolation(
                name="shallow_depth",
                severity=1.0,
                detail_th="ยังลงไม่ลึกพอ ลองงอเข่าให้สะโพกต่ำกว่าหัวเข่า",
            )
        )
    else:
        ratio = (DEPTH_ZERO_DEG - mean_knee_ang) / (DEPTH_ZERO_DEG - DEPTH_FULL_DEG)
        components["depth"] = int(30 * ratio)
        violations.append(
            RuleViolation(
                name="shallow_depth",
                severity=1.0 - ratio,
                detail_th="ยังลงไม่ลึกพอ ลองงอเข่าให้สะโพกต่ำกว่าหัวเข่า",
            )
        )

    # --- Valgus (25 pts) ---
    l_v, r_v = knee_valgus_ratio(kps)
    worst_valgus = max(abs(l_v), abs(r_v))
    if worst_valgus < VALGUS_THRESHOLD:
        components["valgus"] = 25
    else:
        severity = min(1.0, (worst_valgus - VALGUS_THRESHOLD) / 0.3)
        components["valgus"] = int(25 * (1.0 - severity))
        violations.append(
            RuleViolation(
                name="knee_valgus",
                severity=severity,
                detail_th="หัวเข่าเข้าด้านใน ควรกางหัวเข่าออกตามแนวปลายเท้า",
            )
        )

    # --- Torso (20 pts) ---
    lean = torso_lean_deg(kps)
    if 20.0 <= lean <= 55.0:
        components["torso"] = 20
    elif lean > 55.0:
        components["torso"] = max(0, int(20 * (1 - (lean - 55) / 30)))
        violations.append(
            RuleViolation(
                name="excessive_forward_lean",
                severity=min(1.0, (lean - 55) / 30),
                detail_th="ลำตัวโน้มไปข้างหน้ามากเกินไป",
            )
        )
    else:
        components["torso"] = max(0, int(20 * (1 - (20 - lean) / 20)))
        violations.append(
            RuleViolation(
                name="too_upright",
                severity=min(1.0, (20 - lean) / 20),
                detail_th="ลำตัวตั้งตรงเกินไป ควรโน้มไปข้างหน้าเล็กน้อย",
            )
        )

    # --- Symmetry (15 pts) ---
    delta = abs(l_knee_ang - r_knee_ang)
    if delta < 10.0:
        components["symmetry"] = 15
    else:
        severity = min(1.0, (delta - 10.0) / 20.0)
        components["symmetry"] = int(15 * (1.0 - severity))
        violations.append(
            RuleViolation(
                name="asymmetric",
                severity=severity,
                detail_th="ซ้ายและขวาไม่สมมาตร",
            )
        )

    # --- Tempo (10 pts) ---
    if descent_ms >= ascent_ms:
        components["tempo"] = 10
    else:
        ratio = descent_ms / max(1, ascent_ms)
        components["tempo"] = int(10 * ratio)
        violations.append(
            RuleViolation(
                name="fast_descent",
                severity=1.0 - ratio,
                detail_th="ลงเร็วกว่าขึ้น ควรลงช้าๆ ควบคุมการเคลื่อนไหว",
            )
        )

    total = sum(components.values())
    return RepAnalysis(
        rep_index=rep_index,
        score=total,
        components=components,
        violations=violations,
        descent_ms=descent_ms,
        ascent_ms=ascent_ms,
        bottom_frame_keypoints_2d=kps.copy(),
        bottom_frame_keypoints_3d=bottom_frame.keypoints_3d.copy()
        if bottom_frame.keypoints_3d is not None
        else None,
        metric_source="2d",
    )


def _score_rep_3d(
    bottom_frame: PoseFrame, descent_ms: int, ascent_ms: int, rep_index: int = 0
) -> RepAnalysis:
    """3D scoring path. Uses keypoints_3d for depth, valgus, torso, symmetry.
    Tempo is computed identically to the 2D path."""
    kps_3d = bottom_frame.keypoints_3d
    assert kps_3d is not None, "_score_rep_3d called without keypoints_3d"

    violations: list[RuleViolation] = []
    components: dict[str, int] = {}

    l_flex, r_flex = knee_flexion_3d(kps_3d)
    mean_flex = (l_flex + r_flex) / 2.0

    # --- Depth (30 pts) — linear ramp on mean 3D knee flexion ---
    if mean_flex <= DEPTH_FULL_DEG:
        components["depth"] = 30
    elif mean_flex >= DEPTH_ZERO_DEG:
        components["depth"] = 0
        violations.append(
            RuleViolation(
                name="shallow_depth",
                severity=1.0,
                detail_th="ยังลงไม่ลึกพอ ลองงอเข่าให้สะโพกต่ำกว่าหัวเข่า",
            )
        )
    else:
        ratio = (DEPTH_ZERO_DEG - mean_flex) / (DEPTH_ZERO_DEG - DEPTH_FULL_DEG)
        components["depth"] = int(30 * ratio)
        violations.append(
            RuleViolation(
                name="shallow_depth",
                severity=1.0 - ratio,
                detail_th="ยังลงไม่ลึกพอ ลองงอเข่าให้สะโพกต่ำกว่าหัวเข่า",
            )
        )

    # --- Valgus (25 pts) — signed max in frontal plane, medial-only penalty ---
    l_v, r_v = valgus_offset_3d(kps_3d)
    worst = max(l_v, r_v)  # signed: only medial (positive) counts
    if worst < VALGUS_THRESHOLD_3D:
        components["valgus"] = 25
    else:
        severity = min(
            1.0,
            (worst - VALGUS_THRESHOLD_3D) / (VALGUS_SAT_3D - VALGUS_THRESHOLD_3D),
        )
        components["valgus"] = int(25 * (1.0 - severity))
        violations.append(
            RuleViolation(
                name="knee_valgus",
                severity=severity,
                detail_th="หัวเข่าเข้าด้านใน ควรกางหัวเข่าออกตามแนวปลายเท้า",
            )
        )

    # --- Torso (20 pts) — angle from image up ---
    lean = torso_lean_3d(kps_3d)
    if 20.0 <= lean <= 55.0:
        components["torso"] = 20
    elif lean > 55.0:
        components["torso"] = max(0, int(20 * (1 - (lean - 55) / 30)))
        violations.append(
            RuleViolation(
                name="excessive_forward_lean",
                severity=min(1.0, (lean - 55) / 30),
                detail_th="ลำตัวโน้มไปข้างหน้ามากเกินไป",
            )
        )
    else:
        components["torso"] = max(0, int(20 * (1 - (20 - lean) / 20)))
        violations.append(
            RuleViolation(
                name="too_upright",
                severity=min(1.0, (20 - lean) / 20),
                detail_th="ลำตัวตั้งตรงเกินไป ควรโน้มไปข้างหน้าเล็กน้อย",
            )
        )

    # --- Symmetry (15 pts) — L vs R 3D knee flexion ---
    delta = abs(l_flex - r_flex)
    if delta < 10.0:
        components["symmetry"] = 15
    else:
        severity = min(1.0, (delta - 10.0) / 20.0)
        components["symmetry"] = int(15 * (1.0 - severity))
        violations.append(
            RuleViolation(
                name="asymmetric",
                severity=severity,
                detail_th="ซ้ายและขวาไม่สมมาตร",
            )
        )

    # --- Tempo (10 pts) — unchanged ---
    if descent_ms >= ascent_ms:
        components["tempo"] = 10
    else:
        ratio = descent_ms / max(1, ascent_ms)
        components["tempo"] = int(10 * ratio)
        violations.append(
            RuleViolation(
                name="fast_descent",
                severity=1.0 - ratio,
                detail_th="ลงเร็วกว่าขึ้น ควรลงช้าๆ ควบคุมการเคลื่อนไหว",
            )
        )

    total = sum(components.values())
    return RepAnalysis(
        rep_index=rep_index,
        score=total,
        components=components,
        violations=violations,
        descent_ms=descent_ms,
        ascent_ms=ascent_ms,
        bottom_frame_keypoints_2d=bottom_frame.keypoints_2d.copy(),
        bottom_frame_keypoints_3d=kps_3d.copy(),
        metric_source="3d",
    )

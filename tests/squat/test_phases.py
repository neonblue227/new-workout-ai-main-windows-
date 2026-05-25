import numpy as np
from analysis.phases import SquatFSM
from analysis.types import PhaseState


def make_kps(knee_angle_deg: float) -> np.ndarray:
    """Build a synthetic keypoint array with a target knee angle."""
    kps = np.zeros((17, 2), dtype=np.float32)
    import math

    theta = math.radians(180 - knee_angle_deg)
    kps[11] = (0.0, 0.0)
    kps[13] = (0.0, 100.0)
    kps[15] = (100.0 * math.sin(theta), 100.0 + 100.0 * math.cos(theta))
    kps[12] = kps[11] + (50, 0)
    kps[14] = kps[13] + (50, 0)
    kps[16] = kps[15] + (50, 0)
    kps[5] = kps[11] + (0, -50)
    kps[6] = kps[12] + (0, -50)
    return kps


def test_starts_standing():
    fsm = SquatFSM()
    assert fsm.state == PhaseState.STANDING


def test_full_rep_cycle():
    fsm = SquatFSM()
    rep_completed = []

    def on_rep(meta):
        rep_completed.append(meta)

    fsm.on_rep_complete = on_rep
    fsm.update(make_kps(175.0), timestamp=0.0)
    assert fsm.state == PhaseState.STANDING
    fsm.update(make_kps(140.0), timestamp=0.5)
    assert fsm.state == PhaseState.DESCENT
    fsm.update(make_kps(85.0), timestamp=1.0)
    assert fsm.state == PhaseState.BOTTOM
    fsm.update(make_kps(110.0), timestamp=1.5)
    assert fsm.state == PhaseState.ASCENT
    fsm.update(make_kps(175.0), timestamp=2.0)
    assert fsm.state == PhaseState.STANDING
    assert len(rep_completed) == 1
    meta = rep_completed[0]
    assert meta["descent_ms"] == 1000
    assert meta["ascent_ms"] == 1000


def test_descent_then_back_up_no_rep():
    fsm = SquatFSM()
    rep_completed = []
    fsm.on_rep_complete = lambda m: rep_completed.append(m)
    fsm.update(make_kps(175.0), timestamp=0.0)
    fsm.update(make_kps(140.0), timestamp=0.5)
    fsm.update(make_kps(175.0), timestamp=1.0)
    assert fsm.state == PhaseState.STANDING
    assert len(rep_completed) == 0


def test_rep_completes_when_starting_already_mid_squat():
    """Regression: if the first observed frame is below STAND_THRESHOLD, descent_start_ts
    must fall back to the transition timestamp instead of remaining None."""
    fsm = SquatFSM()
    rep_completed = []
    fsm.on_rep_complete = lambda m: rep_completed.append(m)

    # First frame: user already descending (knee=140 < 160). FSM goes STANDING -> DESCENT.
    fsm.update(make_kps(140.0), timestamp=0.5)
    assert fsm.state == PhaseState.DESCENT
    fsm.update(make_kps(85.0), timestamp=1.0)
    assert fsm.state == PhaseState.BOTTOM
    fsm.update(make_kps(110.0), timestamp=1.5)
    assert fsm.state == PhaseState.ASCENT
    # Completing the rep used to throw `TypeError: float - NoneType`.
    fsm.update(make_kps(175.0), timestamp=2.0)
    assert fsm.state == PhaseState.STANDING
    assert len(rep_completed) == 1
    meta = rep_completed[0]
    # descent_start_ts fell back to t=0.5 (the transition frame). descent_ms = (1.0 - 0.5) * 1000 = 500.
    assert meta["descent_ms"] == 500
    assert meta["ascent_ms"] == 1000

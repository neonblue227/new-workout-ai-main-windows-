from analysis.phases import HoldFSM
from analysis.types import HoldState


def test_starts_idle():
    fsm = HoldFSM(target_seconds=20.0)
    assert fsm.state is HoldState.IDLE


def test_enters_after_first_in_target_frame():
    fsm = HoldFSM(target_seconds=20.0)
    assert fsm.update(in_target=True, timestamp=0.0) is HoldState.ENTERING


def test_promotes_to_holding_after_stability_window():
    fsm = HoldFSM(target_seconds=20.0, stability_window_s=0.5)
    fsm.update(True, 0.0)  # ENTERING
    fsm.update(True, 0.3)  # still ENTERING (< 0.5)
    state = fsm.update(True, 0.6)  # crosses 0.5 → HOLDING
    assert state is HoldState.HOLDING


def test_fly_through_does_not_start_hold():
    fsm = HoldFSM(target_seconds=20.0, stability_window_s=0.5)
    fsm.update(True, 0.0)
    fsm.update(True, 0.3)
    state = fsm.update(False, 0.4)  # in-target window < 0.5 → reset
    assert state is HoldState.IDLE


def test_clean_hold_reaches_complete():
    fsm = HoldFSM(target_seconds=2.0, stability_window_s=0.5)
    completes = []
    fsm.on_hold_complete = lambda meta: completes.append(meta)

    t = 0.0
    state = None
    while state is not HoldState.COMPLETE and t < 10.0:
        state = fsm.update(True, t)
        t += 0.1
    assert t < 10.0, "FSM never reached COMPLETE"
    assert state is HoldState.COMPLETE
    assert completes, "callback must fire on COMPLETE"
    meta = completes[0]
    assert meta["in_target_ms"] >= 2000
    assert meta["drift_count"] == 0


def test_drift_within_grace_resumes_holding():
    fsm = HoldFSM(target_seconds=5.0, stability_window_s=0.5, drift_grace_s=0.3)
    fsm.update(True, 0.0)
    fsm.update(True, 0.6)  # HOLDING
    fsm.update(False, 1.0)  # DRIFTED
    state = fsm.update(True, 1.2)  # within grace → HOLDING
    assert state is HoldState.HOLDING


def test_drift_beyond_grace_transitions_to_entering_and_counts_drift():
    fsm = HoldFSM(target_seconds=5.0, stability_window_s=0.5, drift_grace_s=0.3)
    fsm.update(True, 0.0)
    fsm.update(True, 0.6)  # HOLDING
    fsm.update(False, 1.0)  # DRIFTED
    state = fsm.update(False, 1.5)  # past grace
    assert state is HoldState.ENTERING
    assert fsm.drift_count == 1


def test_timer_pauses_during_drift():
    """In-target ms accumulates only while HOLDING — not while DRIFTED."""
    fsm = HoldFSM(target_seconds=10.0, stability_window_s=0.5, drift_grace_s=0.3)
    fsm.update(True, 0.0)
    fsm.update(True, 0.6)  # HOLDING start
    fsm.update(True, 1.6)  # +1.0s in-target
    fsm.update(False, 1.7)  # DRIFTED — timer pauses
    fsm.update(False, 1.9)  # still in grace, still paused
    fsm.update(True, 2.0)  # resume HOLDING
    fsm.update(True, 2.5)  # +0.5s more in-target
    assert 1400 <= fsm.in_target_ms <= 1600


def test_drift_beyond_grace_preserves_in_target_ms():
    """Per spec: drift past grace records a stability penalty but DOES NOT
    throw away accumulated in-target time. User can re-prove the 0.5s window
    and resume the hold from where they left off."""
    fsm = HoldFSM(target_seconds=2.0, stability_window_s=0.1, drift_grace_s=0.3)
    # Accumulate ~0.9s of in-target time then drift out past grace.
    fsm.update(True, 0.0)  # ENTERING
    fsm.update(True, 0.2)  # HOLDING
    fsm.update(True, 1.1)  # +0.9s in-target
    accumulated = fsm.in_target_ms
    fsm.update(False, 1.2)  # DRIFTED
    fsm.update(False, 1.6)  # past 0.3s grace → ENTERING, drift_count=1
    assert fsm.state is HoldState.ENTERING
    assert fsm.drift_count == 1
    # in_target_ms preserved across drift-expire — NOT reset.
    assert fsm.in_target_ms == accumulated


def test_hold_completes_via_drift_recovery():
    """After drift-expire, re-entering and accumulating more in-target time
    reaches COMPLETE without restarting from zero."""
    fsm = HoldFSM(target_seconds=1.0, stability_window_s=0.1, drift_grace_s=0.2)
    completes = []
    fsm.on_hold_complete = lambda meta: completes.append(meta)

    # First leg: ~0.6s in-target.
    fsm.update(True, 0.0)
    fsm.update(True, 0.15)  # HOLDING
    fsm.update(True, 0.75)  # +0.6s
    assert fsm.in_target_ms >= 500

    # Drift past grace → ENTERING.
    fsm.update(False, 0.8)  # DRIFTED
    fsm.update(False, 1.1)  # past grace → ENTERING

    # Second leg: re-prove stability + accumulate more — should complete.
    fsm.update(True, 1.2)  # ENTERING fresh window
    fsm.update(True, 1.35)  # HOLDING
    fsm.update(True, 1.85)  # +0.5s more, total > 1s → COMPLETE
    assert fsm.state is HoldState.COMPLETE
    assert completes

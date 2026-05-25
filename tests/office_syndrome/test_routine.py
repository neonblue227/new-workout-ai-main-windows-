from routine import RoutineFSM, RoutineConfig, RoutinePhase


def kinds(events):
    return [e.kind for e in events]


def test_starts_in_setup():
    assert RoutineFSM().phase is RoutinePhase.SETUP


def test_config_defaults():
    c = RoutineConfig()
    assert c.order == ("left", "right", "left", "right")
    assert c.sets == 4
    assert c.hold_s == 25.0


def test_start_enters_positioning():
    fsm = RoutineFSM()
    fsm.start(0.0)
    assert fsm.phase is RoutinePhase.POSITIONING


def test_positioning_progress_resets_when_pose_lost():
    cfg = RoutineConfig(position_hold_s=1.0)
    fsm = RoutineFSM(cfg)
    fsm.start(0.0)
    fsm.update(0.0, True, False)
    fsm.update(0.5, True, False)
    assert 0.4 < fsm.position_progress < 0.6
    fsm.update(0.6, False, False)  # lost the pose
    assert fsm.position_progress == 0.0
    assert fsm.phase is RoutinePhase.POSITIONING


def test_positioning_completes_after_full_hold():
    cfg = RoutineConfig(position_hold_s=1.0)
    fsm = RoutineFSM(cfg)
    fsm.start(0.0)
    fsm.update(0.0, True, False)
    evs = fsm.update(1.01, True, False)
    assert "position_ok" in kinds(evs)
    assert fsm.phase is RoutinePhase.COUNTDOWN
    assert fsm.set_index == 0
    assert fsm.current_side == "left"


def _to_countdown(cfg):
    fsm = RoutineFSM(cfg)
    fsm.start(0.0)
    fsm.update(0.0, True, False)
    fsm.update(cfg.position_hold_s + 0.01, True, False)  # POSITIONING -> COUNTDOWN
    return fsm, cfg.position_hold_s + 0.01


def test_countdown_emits_3_2_1_then_set_started():
    cfg = RoutineConfig(position_hold_s=1.0, countdown_s=3)
    fsm, t0 = _to_countdown(cfg)
    emitted = []
    for t in (t0, t0 + 1.0, t0 + 2.0):
        e = fsm.update(t, True, False)
        emitted += [ev.value for ev in e if ev.kind == "countdown"]
    assert emitted == [3, 2, 1]
    e = fsm.update(t0 + 3.01, True, False)
    assert "set_started" in kinds(e)
    assert fsm.phase is RoutinePhase.HOLD
    assert fsm.current_side == "left"
    started_side = [ev.value for ev in e if ev.kind == "set_started"][0]
    assert started_side == "left"


def _to_first_hold(cfg):
    fsm, t0 = _to_countdown(cfg)
    th = t0 + cfg.countdown_s + 0.01
    fsm.update(th, True, False)  # COUNTDOWN -> HOLD
    return fsm, th


def test_hold_completes_then_transition_to_next_side():
    cfg = RoutineConfig(
        position_hold_s=1.0, countdown_s=3, hold_s=2.0, transition_s=1.0
    )
    fsm, th = _to_first_hold(cfg)
    assert fsm.phase is RoutinePhase.HOLD
    e = fsm.update(th + 2.01, True, True)
    assert "set_complete" in kinds(e)
    assert "switch_sides" in kinds(e)
    assert fsm.phase is RoutinePhase.TRANSITION
    nxt = [ev.value for ev in e if ev.kind == "switch_sides"][0]
    assert nxt == "right"


def test_transition_advances_to_next_set():
    cfg = RoutineConfig(
        position_hold_s=1.0, countdown_s=3, hold_s=2.0, transition_s=1.0
    )
    fsm, th = _to_first_hold(cfg)
    t1 = th + 2.01
    fsm.update(t1, True, True)  # -> TRANSITION
    fsm.update(t1 + 1.01, True, False)  # transition done -> COUNTDOWN set 1
    assert fsm.phase is RoutinePhase.COUNTDOWN
    assert fsm.set_index == 1
    assert fsm.current_side == "right"


def test_full_routine_sequence_and_done():
    cfg = RoutineConfig(
        position_hold_s=0.5, countdown_s=1, hold_s=1.0, transition_s=0.5, summary_s=0.5
    )
    fsm = RoutineFSM(cfg)
    t = 0.0
    fsm.start(t)
    fsm.update(t, True, False)
    t += cfg.position_hold_s + 0.01
    fsm.update(t, True, False)
    started, completes, done = [], 0, False
    for _ in range(5000):
        t += 0.05
        for ev in fsm.update(t, True, True):
            if ev.kind == "set_started":
                started.append(ev.value)
            elif ev.kind == "set_complete":
                completes += 1
            elif ev.kind == "routine_complete":
                done = True
        if fsm.phase is RoutinePhase.DONE:
            break
    assert started == ["left", "right", "left", "right"]
    assert completes == 4
    assert done
    assert fsm.phase is RoutinePhase.DONE

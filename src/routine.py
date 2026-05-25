"""Pure state machine for the guided neck-stretch routine.

No OpenCV, no audio, no pose code — advanced by timestamps + two booleans so it
is unit-testable and portable to the future streaming server. The caller
(`app.py`) maps the returned events to audio cues and screen rendering and owns
all I/O. Spec: docs/superpowers/specs/2026-05-23-neck-stretch-realtime-demo-design.md
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RoutinePhase(Enum):
    SETUP = "setup"
    POSITIONING = "positioning"
    COUNTDOWN = "countdown"
    HOLD = "hold"
    TRANSITION = "transition"
    SUMMARY = "summary"
    DONE = "done"


# Event kinds emitted by RoutineFSM.update(). The caller turns each into an
# audio cue and/or a one-shot UI change. `value` payloads noted inline.
EV_POSITION_OK = "position_ok"  # value: None
EV_COUNTDOWN = "countdown"  # value: int (3, 2, 1)
EV_SET_STARTED = "set_started"  # value: side str ("left"/"right")
EV_SET_COMPLETE = "set_complete"  # value: completed set index (int)
EV_SWITCH_SIDES = "switch_sides"  # value: next side str
EV_ROUTINE_COMPLETE = "routine_complete"  # value: None


@dataclass(frozen=True)
class RoutineEvent:
    kind: str
    value: object = None


@dataclass(frozen=True)
class RoutineConfig:
    hold_s: float = 25.0
    order: tuple[str, ...] = ("left", "right", "left", "right")
    position_hold_s: float = 3.0
    countdown_s: int = 3
    transition_s: float = 3.0
    summary_s: float = 10.0

    @property
    def sets(self) -> int:
        return len(self.order)


class RoutineFSM:
    """Advance with update(now, pose_ready, in_target) each frame; read .phase
    and the rendering-state attributes; map returned events to cues."""

    def __init__(self, config: Optional[RoutineConfig] = None):
        self.config = config or RoutineConfig()
        self.phase = RoutinePhase.SETUP
        self.set_index = 0
        # Rendering state, refreshed on each update().
        self.position_progress = 0.0
        self.countdown_value = self.config.countdown_s
        self.hold_elapsed_s = 0.0
        self.hold_remaining_s = self.config.hold_s
        self.transition_remaining_s = self.config.transition_s
        # Internal timers.
        self._phase_start: float = 0.0  # always set before read by _enter_* helpers
        self._pos_ok_since: Optional[float] = None
        self._last_countdown_emitted: Optional[int] = None

    @property
    def current_side(self) -> Optional[str]:
        """Active side ("left"/"right") during COUNTDOWN or HOLD; None otherwise
        (i.e., during POSITIONING, TRANSITION, SUMMARY, or DONE)."""
        if self.phase in (RoutinePhase.COUNTDOWN, RoutinePhase.HOLD) and (
            0 <= self.set_index < self.config.sets
        ):
            return self.config.order[self.set_index]
        return None

    @property
    def next_side(self) -> Optional[str]:
        nxt = self.set_index + 1
        return self.config.order[nxt] if nxt < self.config.sets else None

    def start(self, now: float) -> None:
        # Idempotent: only acts in SETUP, no-op if called again in any other phase.
        if self.phase is RoutinePhase.SETUP:
            self.phase = RoutinePhase.POSITIONING
            self._pos_ok_since = None
            self.position_progress = 0.0

    def update(
        self, now: float, pose_ready: bool, in_target: bool
    ) -> list[RoutineEvent]:
        # `in_target` is accepted but currently unused; reserved for a future
        # drift-detection extension that will pause hold progress on drift.
        events: list[RoutineEvent] = []
        c = self.config

        if self.phase is RoutinePhase.POSITIONING:
            if pose_ready:
                if self._pos_ok_since is None:
                    self._pos_ok_since = now
                held = now - self._pos_ok_since
                self.position_progress = min(1.0, held / c.position_hold_s)
                if held >= c.position_hold_s:
                    events.append(RoutineEvent(EV_POSITION_OK))
                    self.set_index = 0
                    self._enter_countdown(now)
            else:
                self._pos_ok_since = None
                self.position_progress = 0.0

        elif self.phase is RoutinePhase.COUNTDOWN:
            elapsed = now - self._phase_start
            remaining = c.countdown_s - elapsed
            # Shift slightly upward before truncation so a timestamp landing at e.g.
            # 0.9999999997 (float rounding) truncates to 1, not 0. 1e-9 is far below any
            # real countdown step (steps are >= 1 s apart), so it never fires a step early.
            n = max(1, c.countdown_s - int(elapsed + 1e-9))
            self.countdown_value = n
            if self._last_countdown_emitted != n and remaining > 0:
                events.append(RoutineEvent(EV_COUNTDOWN, n))
                self._last_countdown_emitted = n
            if elapsed >= c.countdown_s:
                self._enter_hold(now, events)

        elif self.phase is RoutinePhase.HOLD:
            self.hold_elapsed_s = now - self._phase_start
            self.hold_remaining_s = max(0.0, c.hold_s - self.hold_elapsed_s)
            if self.hold_elapsed_s >= c.hold_s:
                events.append(RoutineEvent(EV_SET_COMPLETE, self.set_index))
                if self.set_index >= c.sets - 1:
                    self._enter_summary(now, events)
                else:
                    self._enter_transition(now, events)

        elif self.phase is RoutinePhase.TRANSITION:
            elapsed = now - self._phase_start
            self.transition_remaining_s = max(0.0, c.transition_s - elapsed)
            if elapsed >= c.transition_s:
                self.set_index += 1
                self._enter_countdown(now)

        elif self.phase is RoutinePhase.SUMMARY:
            if now - self._phase_start >= c.summary_s:
                self.phase = RoutinePhase.DONE

        return events

    def _enter_countdown(self, now: float) -> None:
        self.phase = RoutinePhase.COUNTDOWN
        self._phase_start = now
        self._last_countdown_emitted = None
        self.countdown_value = self.config.countdown_s

    def _enter_hold(self, now: float, events: list[RoutineEvent]) -> None:
        self.phase = RoutinePhase.HOLD
        self._phase_start = now
        self.hold_elapsed_s = 0.0
        self.hold_remaining_s = self.config.hold_s
        events.append(RoutineEvent(EV_SET_STARTED, self.current_side))

    def _enter_transition(self, now: float, events: list[RoutineEvent]) -> None:
        self.phase = RoutinePhase.TRANSITION
        self._phase_start = now
        self.transition_remaining_s = self.config.transition_s
        events.append(RoutineEvent(EV_SWITCH_SIDES, self.next_side))

    def _enter_summary(self, now: float, events: list[RoutineEvent]) -> None:
        self.phase = RoutinePhase.SUMMARY
        self._phase_start = now
        events.append(RoutineEvent(EV_ROUTINE_COMPLETE))

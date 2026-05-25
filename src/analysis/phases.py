from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

from analysis.angles import knee_angles
from analysis.types import HoldState, PhaseState


STAND_THRESHOLD = 160.0
BOTTOM_THRESHOLD = 100.0


@dataclass
class SquatFSM:
    state: PhaseState = PhaseState.STANDING
    descent_start_ts: Optional[float] = None
    bottom_ts: Optional[float] = None
    on_rep_complete: Optional[Callable[[dict], None]] = None

    def update(self, kps: np.ndarray, timestamp: float) -> PhaseState:
        l, r = knee_angles(kps)
        angle = (l + r) / 2.0

        if self.state == PhaseState.STANDING:
            if angle < STAND_THRESHOLD:
                self.state = PhaseState.DESCENT
                if self.descent_start_ts is None:
                    self.descent_start_ts = timestamp
            else:
                self.descent_start_ts = timestamp  # track last confirmed standing frame
        elif self.state == PhaseState.DESCENT:
            if angle < BOTTOM_THRESHOLD:
                self.state = PhaseState.BOTTOM
                self.bottom_ts = timestamp
            elif angle >= STAND_THRESHOLD:
                self.state = PhaseState.STANDING
                self.descent_start_ts = None
        elif self.state == PhaseState.BOTTOM:
            if angle >= BOTTOM_THRESHOLD:
                self.state = PhaseState.ASCENT
        elif self.state == PhaseState.ASCENT:
            if angle >= STAND_THRESHOLD:
                meta = {
                    "descent_ms": int((self.bottom_ts - self.descent_start_ts) * 1000),
                    "ascent_ms": int((timestamp - self.bottom_ts) * 1000),
                    "bottom_ts": self.bottom_ts,
                    "completed_ts": timestamp,
                }
                self.state = PhaseState.STANDING
                self.descent_start_ts = None
                self.bottom_ts = None
                if self.on_rep_complete:
                    self.on_rep_complete(meta)
        return self.state


@dataclass
class HoldFSM:
    target_seconds: float
    stability_window_s: float = 0.5
    drift_grace_s: float = 0.3
    on_hold_complete: Optional[Callable[[dict], None]] = None

    state: HoldState = HoldState.IDLE
    entry_start_ts: Optional[float] = None  # when current ENTERING window began
    last_hold_ts: Optional[float] = None  # last frame in HOLDING (for accumulation)
    drift_start_ts: Optional[float] = None  # when current drift began
    in_target_ms: int = 0
    drift_count: int = 0

    def update(self, in_target: bool, timestamp: float) -> HoldState:
        if self.state is HoldState.IDLE:
            if in_target:
                self.state = HoldState.ENTERING
                self.entry_start_ts = timestamp

        elif self.state is HoldState.ENTERING:
            if not in_target:
                # Aborted entry — reset.
                self.state = HoldState.IDLE
                self.entry_start_ts = None
            elif (
                self.entry_start_ts is not None
                and timestamp - self.entry_start_ts >= self.stability_window_s
            ):
                self.state = HoldState.HOLDING
                self.last_hold_ts = timestamp

        elif self.state is HoldState.HOLDING:
            if in_target:
                if self.last_hold_ts is not None:
                    delta = max(0.0, timestamp - self.last_hold_ts)
                    self.in_target_ms += int(delta * 1000)
                self.last_hold_ts = timestamp
                if self.in_target_ms >= int(self.target_seconds * 1000):
                    self._fire_complete(timestamp)
            else:
                self.state = HoldState.DRIFTED
                self.drift_start_ts = timestamp

        elif self.state is HoldState.DRIFTED:
            if in_target:
                self.state = HoldState.HOLDING
                self.drift_start_ts = None
                self.last_hold_ts = timestamp
            elif (
                self.drift_start_ts is not None
                and timestamp - self.drift_start_ts >= self.drift_grace_s
            ):
                # Spec: hold continues, quality drops. Preserve in_target_ms;
                # drift_count carries the penalty into the stability score.
                self.drift_count += 1
                self.state = HoldState.ENTERING
                self.entry_start_ts = timestamp
                self.drift_start_ts = None
                self.last_hold_ts = None
                # do NOT reset in_target_ms — preserve accumulated hold time

        return self.state

    def _fire_complete(self, ts: float) -> None:
        self.state = HoldState.COMPLETE
        if self.on_hold_complete:
            self.on_hold_complete(
                {
                    "in_target_ms": self.in_target_ms,
                    "drift_count": self.drift_count,
                    "completed_ts": ts,
                }
            )

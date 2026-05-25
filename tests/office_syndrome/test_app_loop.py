"""Integration test for the app camera-loop's inference gating.

The display loop runs faster than the camera, so `run_neck_stretch_routine`
must run pose inference only once per *unique* camera frame (keyed on the
capture timestamp from `WebcamCapture.read_latest_with_ts`), reusing the last
result on duplicate frames. This keeps the visualization identical while not
re-paying the dominant per-frame cost on a frame it already processed.
"""

from __future__ import annotations

import numpy as np

import app as app_mod
from exercises.neck_stretch import NeckStretchLeft, NeckStretchRight

# A plausible standing pose so the 2D angle cookbook gets sane (non-degenerate)
# geometry instead of NaN-everywhere.
_KPS = np.array(
    [
        [640, 300],  # nose
        [650, 290],
        [630, 290],  # eyes
        [665, 295],
        [615, 295],  # ears
        [720, 380],
        [560, 380],  # shoulders
        [740, 460],
        [540, 460],  # elbows
        [750, 540],
        [530, 540],  # wrists
        [700, 560],
        [580, 560],  # hips
        [705, 680],
        [575, 680],  # knees
        [710, 710],
        [570, 710],  # ankles
    ],
    dtype=np.float32,
)
_SCORES = np.ones(17, dtype=np.float32) * 0.9


class _FakeCap:
    def __init__(self, ts_sequence):
        self._ts = ts_sequence
        self._i = 0

    def read_latest_with_ts(self, timeout=2.0):
        ts = self._ts[min(self._i, len(self._ts) - 1)]
        self._i += 1
        return np.zeros((720, 1280, 3), dtype=np.uint8), ts


class _FakePose:
    def __init__(self):
        self.calls = 0

    def infer(self, frame):
        self.calls += 1
        return _KPS.copy(), _SCORES.copy()


class _FakeWorker:
    def latest(self):
        return ""

    def submit(self, *args, **kwargs):
        pass


class _FakeTTS:
    def play_cue(self, *args, **kwargs):
        pass

    def submit_feedback(self, *args, **kwargs):
        pass


def test_inference_runs_once_per_unique_frame(monkeypatch):
    # 5 reads, 3 distinct capture timestamps (1.0 seen 3x, then 2.0, then 3.0).
    ts_sequence = [1.0, 1.0, 1.0, 2.0, 3.0]
    cap = _FakeCap(ts_sequence)
    pose = _FakePose()

    wait = {"n": 0}

    def fake_waitkey(_delay):
        wait["n"] += 1
        return ord("q") if wait["n"] >= len(ts_sequence) else 255

    monkeypatch.setattr(app_mod.cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(app_mod.cv2, "setMouseCallback", lambda *a, **k: None)
    monkeypatch.setattr(app_mod.cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(app_mod.cv2, "waitKey", fake_waitkey)
    monkeypatch.setattr(
        app_mod.time, "sleep", lambda _s: None
    )  # don't pace in the test

    app_mod.run_neck_stretch_routine(
        cap,
        pose,
        _FakeWorker(),
        _FakeTTS(),
        {"left": NeckStretchLeft(), "right": NeckStretchRight()},
    )

    # 5 loop iterations but only 3 unique frames -> 3 inferences, not 5.
    assert pose.calls == 3

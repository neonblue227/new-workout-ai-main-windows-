import time

import cv2
import numpy as np
from capture import WebcamCapture


def test_read_latest_with_ts_reports_capture_timestamp(monkeypatch):
    class FakeVC:
        def __init__(self, idx):
            self.opened = True

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self):
            self.opened = False

        def set(self, *args, **kwargs):
            return True

    monkeypatch.setattr(cv2, "VideoCapture", FakeVC)
    cap = WebcamCapture(device=0)
    cap.start()
    try:
        result = cap.read_latest_with_ts(timeout=1.0)
        assert result is not None
        frame, ts = result
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        assert ts > 0.0
    finally:
        cap.stop()


def test_read_latest_with_ts_stable_when_no_new_frame(monkeypatch):
    """The capture timestamp only advances when a fresh frame is decoded, so the
    app loop can detect a duplicate frame (loop faster than camera) and skip
    re-running pose inference on it."""
    state = {"produced": False}

    class FakeVC:
        def __init__(self, idx):
            pass

        def isOpened(self):
            return True

        def read(self):
            if not state["produced"]:
                state["produced"] = True
                return True, np.zeros((480, 640, 3), dtype=np.uint8)
            time.sleep(0.005)
            return False, None  # camera produced no new frame after the first

        def release(self):
            pass

        def set(self, *args, **kwargs):
            return True

    monkeypatch.setattr(cv2, "VideoCapture", FakeVC)
    cap = WebcamCapture(device=0)
    cap.start()
    try:
        r1 = cap.read_latest_with_ts(timeout=1.0)
        assert r1 is not None
        time.sleep(0.05)
        r2 = cap.read_latest_with_ts(timeout=1.0)
        assert r2 is not None
        assert r1[1] == r2[1]  # same capture ts -> duplicate frame -> skip re-inference
    finally:
        cap.stop()


def test_capture_thread_yields_frames(monkeypatch):
    class FakeVC:
        def __init__(self, idx):
            self.opened = True

        def isOpened(self):
            return self.opened

        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self):
            self.opened = False

        def set(self, *args, **kwargs):
            return True

    monkeypatch.setattr(cv2, "VideoCapture", FakeVC)
    cap = WebcamCapture(device=0)
    cap.start()
    try:
        frame = cap.read_latest(timeout=1.0)
        assert frame is not None
        assert frame.shape == (480, 640, 3)
    finally:
        cap.stop()

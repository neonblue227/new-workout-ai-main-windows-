import time

import cv2
import numpy as np
import pytest
import requests

from camera import build_capture
from camera.__main__ import _parse_args
from camera.ip_webcam import (
    IPWebcamCapture,
    _letterbox,
    _normalize_url,
    iter_jpeg_frames,
)
from capture import WebcamCapture


def _jpeg(value: int) -> bytes:
    """A real, decodable JPEG (starts FFD8, ends FFD9) tagged by fill value."""
    img = np.full((8, 8, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_single_complete_frame():
    a = _jpeg(10)
    frames, remainder = iter_jpeg_frames(a)
    assert frames == [a]
    assert remainder == b""


def test_two_frames_in_one_buffer():
    a, b = _jpeg(10), _jpeg(200)
    frames, remainder = iter_jpeg_frames(a + b)
    assert frames == [a, b]
    assert remainder == b""


def test_frame_split_across_chunk_boundary():
    a = _jpeg(10)
    cut = len(a) // 2
    frames1, rem1 = iter_jpeg_frames(a[:cut])
    assert frames1 == []  # no EOI yet -> buffered
    assert rem1 == a[:cut]
    frames2, rem2 = iter_jpeg_frames(rem1 + a[cut:])
    assert frames2 == [a]
    assert rem2 == b""


def test_trailing_partial_frame_returned_as_remainder():
    a, b = _jpeg(10), _jpeg(200)
    partial = b[: len(b) // 2]
    frames, remainder = iter_jpeg_frames(a + partial)
    assert frames == [a]
    assert remainder == partial


def test_multipart_headers_between_frames_are_ignored():
    a, b = _jpeg(10), _jpeg(200)
    headers = b"\r\n--myboundary\r\nContent-Type: image/jpeg\r\n\r\n"
    frames, remainder = iter_jpeg_frames(a + headers + b)
    assert frames == [a, b]
    assert remainder == b""


def test_no_complete_frame_yields_nothing():
    frames, remainder = iter_jpeg_frames(b"--boundary\r\nContent-Type: image/jpeg")
    assert frames == []
    assert remainder == b""


def test_trailing_lone_ff_byte_is_kept_as_possible_split_soi():
    a = _jpeg(10)
    frames, remainder = iter_jpeg_frames(a + b"\xff")
    assert frames == [a]
    assert remainder == b"\xff"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("http://192.168.1.42:8080/video", "http://192.168.1.42:8080/video"),
        ("192.168.1.42:8080", "http://192.168.1.42:8080/video"),
        ("http://192.168.1.42:8080", "http://192.168.1.42:8080/video"),
        ("http://192.168.1.42:8080/", "http://192.168.1.42:8080/video"),
        ("  192.168.1.42:8080  ", "http://192.168.1.42:8080/video"),
        ("http://192.168.1.42:8080/videofeed", "http://192.168.1.42:8080/videofeed"),
    ],
)
def test_normalize_url(raw, expected):
    assert _normalize_url(raw) == expected


class _FakeResp:
    """Fake streaming response. `chunks` is the byte sequence iter_content yields;
    pass empty (b"") chunks to simulate an open stream with no new frame."""

    def __init__(self, chunks, pause_s: float = 0.0):
        self._chunks = chunks
        self._pause_s = pause_s

    def iter_content(self, chunk_size=1):
        for c in self._chunks:
            if self._pause_s:
                time.sleep(self._pause_s)
            yield c

    def raise_for_status(self):
        pass

    def close(self):
        pass


def _patch_get(monkeypatch, resp_factory):
    monkeypatch.setattr(requests, "get", lambda url, **kw: resp_factory(), raising=True)


def test_start_decodes_latest_frame(monkeypatch):
    a = _jpeg(123)
    # one JPEG, then a long tail of empty chunks so the stream stays "open"
    # (no reconnect) and the frame ts stays stable.
    _patch_get(
        monkeypatch,
        lambda: _FakeResp([a] + [b""] * 200, pause_s=0.005),
    )
    cap = IPWebcamCapture("192.168.1.42:8080")
    cap.start()
    try:
        result = cap.read_latest_with_ts(timeout=1.0)
        assert result is not None
        frame, ts = result
        assert frame.shape == (8, 8, 3)
        assert ts > 0.0
    finally:
        cap.stop()


def test_ts_stable_when_no_new_frame(monkeypatch):
    """ts only advances when a new JPEG is decoded, so the app's duplicate-frame
    dedup gate (frame_ts != last_processed_ts) keeps working."""
    a = _jpeg(50)
    _patch_get(
        monkeypatch,
        lambda: _FakeResp([a] + [b""] * 200, pause_s=0.005),
    )
    cap = IPWebcamCapture("192.168.1.42:8080")
    cap.start()
    try:
        r1 = cap.read_latest_with_ts(timeout=1.0)
        assert r1 is not None
        time.sleep(0.05)
        r2 = cap.read_latest_with_ts(timeout=1.0)
        assert r2 is not None
        assert r1[1] == r2[1]  # no new frame -> same ts
    finally:
        cap.stop()


def test_start_raises_on_connection_failure(monkeypatch):
    def _boom(url, **kw):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", _boom, raising=True)
    cap = IPWebcamCapture("192.168.1.42:8080")
    with pytest.raises(RuntimeError, match="Could not open IP webcam stream"):
        cap.start()


def test_read_latest_returns_none_before_any_frame(monkeypatch):
    # stream opens but yields only empty chunks -> never a frame
    _patch_get(monkeypatch, lambda: _FakeResp([b""] * 50, pause_s=0.005))
    cap = IPWebcamCapture("192.168.1.42:8080")
    cap.start()
    try:
        assert cap.read_latest(timeout=0.2) is None
    finally:
        cap.stop()


def test_letterbox_conforms_size_and_preserves_aspect():
    # 16:9 source into a 4:3 target -> uniform scale to 640x360, black bars
    # top/bottom (geometry preserved, so pose angles are unaffected).
    src = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    out = _letterbox(src, 640, 480)
    assert out.shape == (480, 640, 3)
    assert out[0, 320].tolist() == [0, 0, 0]  # top bar is black
    assert out[240, 320].tolist() == [200, 200, 200]  # centre is content
    assert out[479, 320].tolist() == [0, 0, 0]  # bottom bar is black


def test_letterbox_noop_when_size_matches():
    src = np.full((480, 640, 3), 100, dtype=np.uint8)
    out = _letterbox(src, 640, 480)
    assert out is src  # already the target size -> returned unchanged


def test_ipwebcam_conforms_frame_to_requested_size(monkeypatch):
    a = _jpeg(77)  # native 8x8 JPEG
    _patch_get(monkeypatch, lambda: _FakeResp([a] + [b""] * 200, pause_s=0.005))
    cap = IPWebcamCapture("192.168.1.42:8080", width=320, height=240)
    cap.start()
    try:
        frame = cap.read_latest(timeout=1.0)
        assert frame is not None
        assert frame.shape == (240, 320, 3)
    finally:
        cap.stop()


def test_build_capture_webcam_returns_webcamcapture():
    cap = build_capture("webcam", None, width=1280, height=720)
    assert isinstance(cap, WebcamCapture)


def test_build_capture_ip_returns_ipwebcamcapture():
    cap = build_capture("ip", "192.168.1.42:8080", width=1280, height=720)
    assert isinstance(cap, IPWebcamCapture)


def test_build_capture_ip_without_url_raises():
    with pytest.raises(ValueError, match="--url"):
        build_capture("ip", None, width=1280, height=720)


def test_preview_parse_args_requires_url():
    with pytest.raises(SystemExit):
        _parse_args([])  # no --url


def test_preview_parse_args_accepts_url():
    args = _parse_args(["--url", "192.168.1.42:8080"])
    assert args.url == "192.168.1.42:8080"

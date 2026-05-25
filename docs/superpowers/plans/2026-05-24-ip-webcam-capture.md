# IP Webcam Capture Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `IPWebcamCapture` source that pulls an MJPEG stream from an Android IP Webcam app over the LAN, as a drop-in replacement for `WebcamCapture`, and wire it into `app.py` and `test_2D_3D.py` behind a `--source` CLI flag.

**Architecture:** A new `src/camera/` package. A pure `iter_jpeg_frames` parser extracts complete JPEGs from a byte buffer (SOI/EOI marker scan); `IPWebcamCapture` runs a background thread that streams `/video`, decodes the latest JPEG, and keeps only the newest frame — mirroring `WebcamCapture`'s `start()/read_latest()/read_latest_with_ts()/stop()` interface so nothing downstream changes. A `build_capture(...)` selector picks the source from CLI args; both entry points gain `--source {webcam,ip}` / `--url`.

**Tech Stack:** Python 3.12, `requests` (HTTP streaming), `opencv-python` (`cv2.imdecode`), `numpy`, `pytest`. Run everything via `uv run`.

---

## File Structure

```
src/camera/
  __init__.py      # exports iter_jpeg_frames, IPWebcamCapture, build_capture
  ip_webcam.py     # iter_jpeg_frames, _normalize_url, IPWebcamCapture
  select.py        # build_capture(source, url, *, width, height)
  __main__.py      # standalone preview: open stream, cv2.imshow, q/Esc
tests/pipeline/
  test_ip_webcam.py   # pure-logic + fake-stream tests (no phone, no network)
```

Modified:
- `pyproject.toml` — add `requests` to `[project].dependencies`
- `src/app.py` — argparse + `build_capture`
- `src/test_2D_3D.py` — argparse + `build_capture`
- `CLAUDE.md` — document the new source + CLI flags

---

## Task 1: JPEG frame parser + package skeleton

**Files:**
- Modify: `pyproject.toml` (add `requests`)
- Create: `src/camera/__init__.py`
- Create: `src/camera/ip_webcam.py`
- Test: `tests/pipeline/test_ip_webcam.py`

- [ ] **Step 1: Add `requests` to dependencies**

In `pyproject.toml`, find the `[project]` `dependencies` array containing `"opencv-python>=4.10.0"` and `"numpy>=1.26"`, and add a line so it reads (keep the existing entries, just add the `requests` line):

```toml
    "opencv-python>=4.10.0",
    "numpy>=1.26",
    "requests>=2.31",
```

Then run:

```bash
uv sync --all-extras
```

Expected: resolves and installs without error (`requests` was already present transitively, so this just declares it).

- [ ] **Step 2: Write the failing parser tests**

Create `tests/pipeline/test_ip_webcam.py`:

```python
import cv2
import numpy as np

from camera.ip_webcam import iter_jpeg_frames


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
    assert frames1 == []          # no EOI yet -> buffered
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_ip_webcam.py -v`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'camera.ip_webcam'` (or `cannot import name 'iter_jpeg_frames'`).

- [ ] **Step 4: Create the package `__init__.py`**

Create `src/camera/__init__.py`. (It exports only the `ip_webcam` names for now; Task 4 adds `build_capture` once `select.py` exists — importing it here before then would break collection.)

```python
"""Frame capture sources for the pose pipeline.

`IPWebcamCapture` pulls an MJPEG stream from an Android IP Webcam app over the
LAN; it mirrors the interface of `capture.WebcamCapture` so it is a drop-in
replacement (see CLAUDE.md "streaming server for mobile client").
"""

from camera.ip_webcam import IPWebcamCapture, iter_jpeg_frames

__all__ = ["IPWebcamCapture", "iter_jpeg_frames"]
```

- [ ] **Step 5: Implement `iter_jpeg_frames`**

Create `src/camera/ip_webcam.py`:

```python
"""MJPEG capture source for the Android IP Webcam app.

The app serves a `multipart/x-mixed-replace` MJPEG stream at `/video`. We read
the stream in a background thread, extract each complete JPEG with the pure
`iter_jpeg_frames` parser, decode it to a BGR ndarray, and keep only the latest
frame — exactly like `capture.WebcamCapture`, so it is a drop-in source.
"""

from __future__ import annotations

_SOI = b"\xff\xd8"  # JPEG Start Of Image
_EOI = b"\xff\xd9"  # JPEG End Of Image


def iter_jpeg_frames(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Extract every complete JPEG from `buffer`.

    Scans for SOI (0xFFD8) ... EOI (0xFFD9) marker pairs rather than parsing the
    multipart boundary string (robust to per-part header variation across IP
    Webcam app versions). Returns (complete_jpeg_blobs, remainder); `remainder`
    is the trailing partial frame (from the last unmatched SOI) or a lone 0xFF
    that may be the first half of a split SOI, to prepend to the next chunk.
    Bytes before the first SOI (e.g. multipart headers) are discarded.
    """
    frames: list[bytes] = []
    while True:
        soi = buffer.find(_SOI)
        if soi == -1:
            # No frame start. Keep a trailing 0xFF (possible split SOI), else drop.
            return frames, b"\xff" if buffer.endswith(b"\xff") else b""
        eoi = buffer.find(_EOI, soi + 2)
        if eoi == -1:
            return frames, buffer[soi:]  # incomplete frame; buffer it
        frames.append(buffer[soi : eoi + 2])
        buffer = buffer[eoi + 2 :]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_ip_webcam.py -v`
Expected: PASS (all 7 tests).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/camera/__init__.py src/camera/ip_webcam.py tests/pipeline/test_ip_webcam.py
git commit -m "feat: add iter_jpeg_frames MJPEG parser + camera package

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: URL normalization

**Files:**
- Modify: `src/camera/ip_webcam.py`
- Test: `tests/pipeline/test_ip_webcam.py`

- [ ] **Step 1: Add imports, then write the failing tests**

First, keep imports at the top (Ruff's default rules flag mid-file imports as E402). Edit the import block at the top of `tests/pipeline/test_ip_webcam.py`:

Change:

```python
import cv2
import numpy as np

from camera.ip_webcam import iter_jpeg_frames
```

to:

```python
import cv2
import numpy as np
import pytest

from camera.ip_webcam import _normalize_url, iter_jpeg_frames
```

Then append the test function to the bottom of the file:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_ip_webcam.py::test_normalize_url -v`
Expected: FAIL — `cannot import name '_normalize_url'`.

- [ ] **Step 3: Implement `_normalize_url`**

In `src/camera/ip_webcam.py`, add the import near the top (after the module docstring, before `_SOI`):

```python
from urllib.parse import urlsplit, urlunsplit
```

Then add this function below `iter_jpeg_frames`:

```python
def _normalize_url(url: str) -> str:
    """Accept `<ip>:<port>`, with or without scheme/path, and return a full
    MJPEG URL. Adds `http://` if no scheme, and `/video` if the path is empty."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    parts = urlsplit(url)
    path = parts.path if parts.path not in ("", "/") else "/video"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_ip_webcam.py::test_normalize_url -v`
Expected: PASS (all 6 params).

- [ ] **Step 5: Commit**

```bash
git add src/camera/ip_webcam.py tests/pipeline/test_ip_webcam.py
git commit -m "feat: normalize IP webcam URLs (scheme + /video path)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: IPWebcamCapture class

**Files:**
- Modify: `src/camera/ip_webcam.py`
- Test: `tests/pipeline/test_ip_webcam.py`

- [ ] **Step 1: Add imports, then write the failing tests**

Edit the import block at the top of `tests/pipeline/test_ip_webcam.py`.

Change:

```python
import cv2
import numpy as np
import pytest

from camera.ip_webcam import _normalize_url, iter_jpeg_frames
```

to:

```python
import time

import cv2
import numpy as np
import pytest
import requests

from camera.ip_webcam import IPWebcamCapture, _normalize_url, iter_jpeg_frames
```

Then append to the bottom of the file:

```python
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
    monkeypatch.setattr(
        requests, "get", lambda url, **kw: resp_factory(), raising=True
    )


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_ip_webcam.py -k "IPWebcam or start_ or ts_stable or read_latest_returns" -v`
Expected: FAIL — `cannot import name 'IPWebcamCapture'`.

- [ ] **Step 3: Implement `IPWebcamCapture`**

In `src/camera/ip_webcam.py`, extend the imports at the top so they read:

```python
from __future__ import annotations

import threading
import time
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import requests
```

Then append the class to the end of the file:

```python
class IPWebcamCapture:
    """Background thread that streams an MJPEG feed from an IP Webcam app and
    keeps the latest decoded frame available.

    Drop-in for `capture.WebcamCapture`: same start()/read_latest()/
    read_latest_with_ts()/stop() semantics. Takes a URL instead of a device int
    (resolution is configured in the phone app, so there is no width/height).
    """

    def __init__(
        self,
        url: str,
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 5.0,
        chunk_size: int = 4096,
        reconnect_backoff_s: float = 0.5,
    ):
        self._url = _normalize_url(url)
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._chunk_size = chunk_size
        self._reconnect_backoff_s = reconnect_backoff_s
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._running = False
        self._response = None

    def _open_stream(self):
        resp = requests.get(
            self._url,
            stream=True,
            timeout=(self._connect_timeout, self._read_timeout),
        )
        resp.raise_for_status()
        return resp

    def start(self):
        try:
            self._response = self._open_stream()
        except Exception as exc:
            raise RuntimeError(
                f"Could not open IP webcam stream at {self._url}: {exc}"
            ) from exc
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        buffer = b""
        resp = self._response
        self._response = None
        while self._running:
            if resp is None:
                try:
                    resp = self._open_stream()
                except Exception:
                    time.sleep(self._reconnect_backoff_s)
                    continue
                buffer = b""
            try:
                for chunk in resp.iter_content(chunk_size=self._chunk_size):
                    if not self._running:
                        break
                    if not chunk:
                        continue
                    buffer += chunk
                    frames, buffer = iter_jpeg_frames(buffer)
                    if frames:
                        img = cv2.imdecode(
                            np.frombuffer(frames[-1], np.uint8), cv2.IMREAD_COLOR
                        )
                        if img is not None:
                            with self._lock:
                                self._latest = img
                                self._latest_ts = time.monotonic()
            except Exception:
                pass  # drop + reconnect below
            try:
                resp.close()
            except Exception:
                pass
            resp = None
            if self._running:
                time.sleep(self._reconnect_backoff_s)

    def read_latest(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        result = self.read_latest_with_ts(timeout)
        return None if result is None else result[0]

    def read_latest_with_ts(
        self, timeout: float = 1.0
    ) -> Optional[tuple[np.ndarray, float]]:
        """Return (frame_copy, capture_ts) for the most recent frame, or None on
        timeout. `capture_ts` advances only when a new JPEG is decoded, so a
        consumer looping faster than the stream can skip duplicate frames."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None:
                    return self._latest.copy(), self._latest_ts
            time.sleep(0.005)
        return None

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._response is not None:
            try:
                self._response.close()
            except Exception:
                pass
            self._response = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_ip_webcam.py -v`
Expected: PASS (all tests, including the 4 new class tests).

- [ ] **Step 5: Commit**

```bash
git add src/camera/ip_webcam.py tests/pipeline/test_ip_webcam.py
git commit -m "feat: IPWebcamCapture background MJPEG reader (WebcamCapture drop-in)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: build_capture source selector

**Files:**
- Create: `src/camera/select.py`
- Modify: `src/camera/__init__.py`
- Test: `tests/pipeline/test_ip_webcam.py`

- [ ] **Step 1: Add imports, then write the failing tests**

Edit the import block at the top of `tests/pipeline/test_ip_webcam.py`.

Change:

```python
from camera.ip_webcam import IPWebcamCapture, _normalize_url, iter_jpeg_frames
```

to:

```python
from camera import build_capture
from camera.ip_webcam import IPWebcamCapture, _normalize_url, iter_jpeg_frames
from capture import WebcamCapture
```

Then append to the bottom of the file:

```python
def test_build_capture_webcam_returns_webcamcapture():
    cap = build_capture("webcam", None, width=1280, height=720)
    assert isinstance(cap, WebcamCapture)


def test_build_capture_ip_returns_ipwebcamcapture():
    cap = build_capture("ip", "192.168.1.42:8080", width=1280, height=720)
    assert isinstance(cap, IPWebcamCapture)


def test_build_capture_ip_without_url_raises():
    with pytest.raises(ValueError, match="--url"):
        build_capture("ip", None, width=1280, height=720)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_ip_webcam.py -k build_capture -v`
Expected: FAIL — `cannot import name 'build_capture' from 'camera'`.

- [ ] **Step 3: Implement `build_capture`**

Create `src/camera/select.py`:

```python
"""Pick a frame-capture source from CLI args. Kept separate from the entry
points so the selection logic is testable without importing the heavy pipeline
modules (Pose2D, MotionBERT, Qwen)."""

from __future__ import annotations

from camera.ip_webcam import IPWebcamCapture
from capture import WebcamCapture


def build_capture(source: str, url: str | None, *, width: int, height: int):
    """Return a capture source. `source` is "webcam" (local device 0) or "ip"
    (Android IP Webcam MJPEG stream — `url` required). Raises ValueError if
    source=="ip" without a url."""
    if source == "ip":
        if not url:
            raise ValueError(
                "source 'ip' requires --url (e.g. http://192.168.1.42:8080/video)"
            )
        return IPWebcamCapture(url)
    return WebcamCapture(device=0, width=width, height=height)
```

- [ ] **Step 4: Add `build_capture` to the package exports**

Replace the contents of `src/camera/__init__.py` with:

```python
"""Frame capture sources for the pose pipeline.

`IPWebcamCapture` pulls an MJPEG stream from an Android IP Webcam app over the
LAN; it mirrors the interface of `capture.WebcamCapture` so it is a drop-in
replacement (see CLAUDE.md "streaming server for mobile client").
"""

from camera.ip_webcam import IPWebcamCapture, iter_jpeg_frames
from camera.select import build_capture

__all__ = ["IPWebcamCapture", "iter_jpeg_frames", "build_capture"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_ip_webcam.py -v`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add src/camera/select.py src/camera/__init__.py tests/pipeline/test_ip_webcam.py
git commit -m "feat: build_capture source selector (webcam | ip)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Standalone preview (`python -m camera`)

**Files:**
- Create: `src/camera/__main__.py`
- Test: `tests/pipeline/test_ip_webcam.py`

This is the manual smoke check you run against a real phone before wiring into
the pipeline. The cv2 display loop can't run headless, so the automated test
only covers the argument parser; the loop is verified manually in Step 5.

- [ ] **Step 1: Add import, then write the failing arg-parser test**

Edit the import block at the top of `tests/pipeline/test_ip_webcam.py`.

Change:

```python
from camera import build_capture
from camera.ip_webcam import IPWebcamCapture, _normalize_url, iter_jpeg_frames
from capture import WebcamCapture
```

to:

```python
from camera import build_capture
from camera.__main__ import _parse_args
from camera.ip_webcam import IPWebcamCapture, _normalize_url, iter_jpeg_frames
from capture import WebcamCapture
```

Then append to the bottom of the file:

```python
def test_preview_parse_args_requires_url():
    with pytest.raises(SystemExit):
        _parse_args([])  # no --url


def test_preview_parse_args_accepts_url():
    args = _parse_args(["--url", "192.168.1.42:8080"])
    assert args.url == "192.168.1.42:8080"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_ip_webcam.py -k preview_parse_args -v`
Expected: FAIL — `No module named 'camera.__main__'`.

- [ ] **Step 3: Implement the preview**

Create `src/camera/__main__.py`:

```python
"""Standalone IP Webcam preview — confirm the phone stream works before wiring
it into the pose pipeline.

    uv run python -m camera --url http://192.168.1.42:8080/video

Keys: q / Esc to quit.
"""

from __future__ import annotations

import argparse
import sys

import cv2

from camera.ip_webcam import IPWebcamCapture


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m camera", description="Preview an Android IP Webcam MJPEG stream"
    )
    p.add_argument(
        "--url",
        required=True,
        help="IP Webcam URL, e.g. http://192.168.1.42:8080/video or 192.168.1.42:8080",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    cap = IPWebcamCapture(args.url)
    print(f"[camera] connecting to {cap._url} ...")
    try:
        cap.start()
    except RuntimeError as exc:
        print(f"[camera] {exc}", file=sys.stderr)
        return 1
    print("[camera] connected. Press q or Esc to quit.")
    win = "IP Webcam preview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = cap.read_latest(timeout=2.0)
            if frame is None:
                print("[camera] waiting for frames ...")
                continue
            cv2.imshow(win, frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        cap.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_ip_webcam.py -k preview_parse_args -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Manual verification (with a phone, optional now)**

Start the IP Webcam app on the phone, note its URL, then run:

```bash
uv run python -m camera --url http://<phone-ip>:8080/video
```

Expected: a window showing the live phone camera feed; `q`/`Esc` quits. (Skip if no phone is available right now — the automated tests cover the wiring.)

- [ ] **Step 6: Commit**

```bash
git add src/camera/__main__.py tests/pipeline/test_ip_webcam.py
git commit -m "feat: 'python -m camera' standalone IP webcam preview

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Wire into `app.py`

**Files:**
- Modify: `src/app.py`

`run()` is currently `def run():` and its first line builds the capture
(`src/app.py:110-111`). We add `argparse`, default to the local webcam, and use
`build_capture`. `main.py` calls `run()` with no args, so `run()` reads
`sys.argv` itself.

- [ ] **Step 1: Replace the `WebcamCapture` import with `build_capture` + argparse**

In `src/app.py`, find this import line (around line 40):

```python
from capture import WebcamCapture
```

Replace it with:

```python
from camera import build_capture
```

And add `import argparse` to the top-of-file imports (next to `import time`):

```python
import argparse
import time
```

- [ ] **Step 2: Add the arg parser and update `run()`**

In `src/app.py`, find:

```python
def run():
    cap = WebcamCapture(device=0, width=1280, height=720)
    pose = Pose2D()
```

Replace with:

```python
def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="workout-ai", description="Guided neck-stretch demo"
    )
    p.add_argument(
        "--source",
        choices=["webcam", "ip"],
        default="webcam",
        help="Frame source (default: webcam)",
    )
    p.add_argument(
        "--url",
        default=None,
        help="IP Webcam MJPEG URL (required for --source ip), "
        "e.g. http://192.168.1.42:8080/video",
    )
    args = p.parse_args(argv)
    if args.source == "ip" and not args.url:
        p.error("--source ip requires --url (e.g. http://192.168.1.42:8080/video)")
    return args


def run(argv=None):
    args = _parse_args(argv)
    cap = build_capture(args.source, args.url, width=1280, height=720)
    pose = Pose2D()
```

- [ ] **Step 3: Verify the CLI wiring via `--help`**

Run: `uv run python main.py --help`
Expected: argparse help text listing `--source {webcam,ip}` and `--url`, then exit 0 (no camera/model load).

- [ ] **Step 4: Verify the existing test suite still imports/passes app**

Run: `uv run pytest tests/office_syndrome -k "not smoke" -q`
Expected: PASS (no regressions; `import app` still works with the new imports).

- [ ] **Step 5: Commit**

```bash
git add src/app.py
git commit -m "feat: --source webcam|ip + --url flags on app.py

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Wire into `test_2D_3D.py`

**Files:**
- Modify: `src/test_2D_3D.py`

`run()` is currently `def run() -> None:` and builds
`cam = WebcamCapture(device=0, width=CAM_WIDTH, height=CAM_HEIGHT)` (around
`src/test_2D_3D.py:393`). `__main__` calls `run()`.

- [ ] **Step 1: Replace the `WebcamCapture` import**

In `src/test_2D_3D.py`, find:

```python
from capture import WebcamCapture  # noqa: E402
```

Replace with:

```python
from camera import build_capture  # noqa: E402
```

And add `import argparse` to the top imports (next to `import sys`):

```python
import argparse
import sys
```

- [ ] **Step 2: Add the arg parser and update `run()`**

In `src/test_2D_3D.py`, find:

```python
def run() -> None:
    exercise = NeckStretchLeft()
```

Replace the signature line with (keep the body that follows `exercise = ...`):

```python
def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="test_2D_3D", description="2D/3D pose pipeline diagnostic"
    )
    p.add_argument("--source", choices=["webcam", "ip"], default="webcam")
    p.add_argument(
        "--url",
        default=None,
        help="IP Webcam MJPEG URL (required for --source ip)",
    )
    args = p.parse_args(argv)
    if args.source == "ip" and not args.url:
        p.error("--source ip requires --url (e.g. http://192.168.1.42:8080/video)")
    return args


def run(argv=None) -> None:
    args = _parse_args(argv)
    exercise = NeckStretchLeft()
```

- [ ] **Step 3: Replace the capture construction**

In `src/test_2D_3D.py`, find:

```python
    print(f"[test_2D_3D] Opening webcam at {CAM_WIDTH}x{CAM_HEIGHT}...")
    cam = WebcamCapture(device=0, width=CAM_WIDTH, height=CAM_HEIGHT)
    cam.start()
```

Replace with:

```python
    print(f"[test_2D_3D] Opening {args.source} source...")
    cam = build_capture(args.source, args.url, width=CAM_WIDTH, height=CAM_HEIGHT)
    cam.start()
```

- [ ] **Step 4: Verify the CLI wiring via `--help`**

Run: `uv run python src/test_2D_3D.py --help`
Expected: argparse help listing `--source` / `--url`, then exit 0 (no camera/model load).

- [ ] **Step 5: Commit**

```bash
git add src/test_2D_3D.py
git commit -m "feat: --source webcam|ip + --url flags on test_2D_3D.py

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Full suite, lint, and docs

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run the full pure-logic suite**

Run: `uv run pytest -k "not smoke" -q`
Expected: PASS — the previous 169 tests plus the new `test_ip_webcam.py` tests, no failures.

- [ ] **Step 2: Lint and format**

Run: `uv run ruff check`

The repo has **4 pre-existing errors that are NOT in scope** (do not fix them — unrelated refactoring):
- `notebooks/neck_stretch_test_pipeline/neck_stretch_video_to_3d.ipynb` cell 15 — E402
- `src/analysis/angles.py:51` — E741 (`l`)
- `src/analysis/phases.py:21` — E741 (`l`)
- `src/test_2D_3D.py:209` — E741 (`l`, inside the pre-existing `_draw_head_tilt_2d_overlay`)

Expected: exactly those 4 errors and **no new ones** introduced by this branch. Verify the new files are clean by scoping the check:

```bash
uv run ruff check src/camera tests/pipeline/test_ip_webcam.py
```

Expected: `All checks passed!`

Then format only what this branch touched and re-stage if anything changes:

```bash
uv run ruff format src/camera tests/pipeline/test_ip_webcam.py src/app.py src/test_2D_3D.py
```

- [ ] **Step 3: Document the new source in `CLAUDE.md`**

In `CLAUDE.md`, in the `## Commands` section, find the line:

```bash
uv run python main.py                         # guided neck-stretch routine: Start screen → positioning/calibration → 4×25s alternating holds with spoken Thai cues → summary; q/Esc quits
```

Add these lines immediately after it:

```bash
uv run python main.py --source ip --url http://<phone-ip>:8080/video   # same routine, frames from an Android IP Webcam app over the LAN
uv run python -m camera --url http://<phone-ip>:8080/video             # standalone IP webcam preview (verify the phone stream first)
```

Then, in the `### Threading model` section, find the bullet:

```
1. **WebcamCapture._loop** — pulls frames, keeps only the latest. `read_latest()` returns a copy with timeout.
```

Replace it with:

```
1. **WebcamCapture._loop / IPWebcamCapture._loop** — pulls frames, keeps only the latest. `read_latest()` returns a copy with timeout. `IPWebcamCapture` (`src/camera/`) is a drop-in source that streams an Android IP Webcam app's MJPEG feed over the LAN (`iter_jpeg_frames` parses the stream; reconnects on drop); pick it with `--source ip --url ...` on `app.py` / `test_2D_3D.py` via `camera.build_capture`. It is the first step toward the streaming-server direction below.
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document IP webcam source + CLI flags in CLAUDE.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Verification Checklist (end-to-end)

After all tasks:

- [ ] `uv run pytest -k "not smoke" -q` — green, includes `test_ip_webcam.py`.
- [ ] `uv run ruff check` — clean.
- [ ] `uv run python main.py --help` — shows `--source` / `--url`.
- [ ] `uv run python src/test_2D_3D.py --help` — shows `--source` / `--url`.
- [ ] `uv run python main.py` (no args) — still runs the local-webcam routine unchanged.
- [ ] (with phone) `uv run python -m camera --url http://<phone-ip>:8080/video` — live preview window.
- [ ] (with phone) `uv run python main.py --source ip --url http://<phone-ip>:8080/video` — full routine driven by the phone feed.
```


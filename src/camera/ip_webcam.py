"""MJPEG capture source for the Android IP Webcam app.

The app serves a `multipart/x-mixed-replace` MJPEG stream at `/video`. We read
the stream in a background thread, extract each complete JPEG with the pure
`iter_jpeg_frames` parser, decode it to a BGR ndarray, and keep only the latest
frame — exactly like `capture.WebcamCapture`, so it is a drop-in source.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import requests

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


def _normalize_url(url: str) -> str:
    """Accept `<ip>:<port>`, with or without scheme/path, and return a full
    MJPEG URL. Adds `http://` if no scheme, and `/video` if the path is empty."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    parts = urlsplit(url)
    path = parts.path if parts.path not in ("", "/") else "/video"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def _letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize `frame` to exactly (height, width) preserving aspect ratio, padding
    the remainder with black bars.

    Uses a single uniform scale for both axes, so pose geometry is preserved —
    `head_lateral_tilt_2d` and the other atan2/ratio measurements are unchanged
    (a direct, non-uniform resize would distort the angles). Returns the input
    untouched when it is already the target size.
    """
    h, w = frame.shape[:2]
    if (w, h) == (width, height):
        return frame
    scale = min(width / w, height / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=frame.dtype)
    x0, y0 = (width - new_w) // 2, (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


class IPWebcamCapture:
    """Background thread that streams an MJPEG feed from an IP Webcam app and
    keeps the latest decoded frame available.

    Drop-in for `capture.WebcamCapture`: same start()/read_latest()/
    read_latest_with_ts()/stop() semantics. Takes a URL instead of a device int.
    The phone app controls the *captured* resolution, but `width`/`height` (when
    given) conform each delivered frame to that size via `_letterbox` (uniform
    scale + black bars), so this is a true drop-in for consumers that assume a
    fixed frame size. Leave them None to pass frames through at native size.
    """

    def __init__(
        self,
        url: str,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 5.0,
        chunk_size: int = 4096,
        reconnect_backoff_s: float = 0.5,
    ):
        self._url = _normalize_url(url)
        self._width = width
        self._height = height
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
                            if self._width and self._height:
                                img = _letterbox(img, self._width, self._height)
                            with self._lock:
                                self._latest = img
                                self._latest_ts = time.monotonic()
            except Exception:
                logging.debug(
                    "IPWebcamCapture stream error; reconnecting", exc_info=True
                )
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
        # Only set in the brief window before _loop claims it; once the thread
        # is running, _loop owns the response and closes it on exit.
        if self._response is not None:
            try:
                self._response.close()
            except Exception:
                pass
            self._response = None

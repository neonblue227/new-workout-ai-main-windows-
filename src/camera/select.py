"""Pick a frame-capture source from CLI args. Kept separate from the entry
points so the selection logic is testable without importing the heavy pipeline
modules (Pose2D, MotionBERT, Qwen)."""

from __future__ import annotations

from camera.ip_webcam import IPWebcamCapture
from capture import WebcamCapture


def build_capture(source: str, url: str | None, *, width: int, height: int):
    """Return a capture source. `source` is "webcam" (local device 0) or "ip"
    (Android IP Webcam MJPEG stream — `url` required). `width`/`height` set the
    frame size both sources deliver (the IP source letterboxes the phone's native
    resolution to fit), so downstream code sees a consistent size regardless of
    source. Raises ValueError if source=="ip" without a url."""
    if source == "ip":
        if not url:
            raise ValueError(
                "source 'ip' requires --url (e.g. http://192.168.1.42:8080/video)"
            )
        return IPWebcamCapture(url, width=width, height=height)
    return WebcamCapture(device=0, width=width, height=height)

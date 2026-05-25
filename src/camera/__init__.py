"""Frame capture sources for the pose pipeline.

`IPWebcamCapture` pulls an MJPEG stream from an Android IP Webcam app over the
LAN; it mirrors the interface of `capture.WebcamCapture` so it is a drop-in
replacement (see CLAUDE.md "streaming server for mobile client").
"""

from camera.ip_webcam import IPWebcamCapture, iter_jpeg_frames
from camera.select import build_capture

__all__ = ["IPWebcamCapture", "iter_jpeg_frames", "build_capture"]

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

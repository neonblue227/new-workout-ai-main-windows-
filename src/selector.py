"""Pre-loop OpenCV menu to choose an exercise.

Returns the selected Exercise. Press 1..9 to pick, q/Esc to quit.
The window stays modal until the user picks or cancels.
"""

import cv2
import numpy as np

from exercises import EXERCISES
from exercises.base import Exercise


_PANEL_BG = (24, 24, 28)
_TEXT_FG = (240, 240, 240)
_HINT_FG = (160, 160, 180)


def choose_exercise(default: str | None = None) -> Exercise:
    items = list(EXERCISES.items())
    if not items:
        raise RuntimeError("No exercises registered")

    width, line_h = 780, 36
    height = 80 + line_h * (len(items) + 2)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    while True:
        canvas[:] = _PANEL_BG
        cv2.putText(
            canvas,
            "Choose an exercise (number to pick, q to quit):",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            _TEXT_FG,
            1,
            cv2.LINE_AA,
        )
        for i, (key, ex) in enumerate(items):
            row = 80 + i * line_h
            # 1..9 then 0 for the 10th (limit: ≤ 10 entries reachable via keys).
            shortcut = str(i + 1) if i < 9 else "0"
            label = f"{shortcut}. [{key}] {ex.display_th}"
            cv2.putText(
                canvas,
                label,
                (24, row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                _TEXT_FG,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            "(English keys; Thai is for the panel display only)",
            (16, height - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _HINT_FG,
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("Workout AI — Select Exercise", canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow("Workout AI — Select Exercise")
            raise SystemExit("User canceled exercise selection")
        idx = None
        if ord("1") <= key <= ord("9"):
            idx = key - ord("1")  # '1' → 0 … '9' → 8
        elif key == ord("0"):
            idx = 9  # '0' → 10th item
        if idx is not None and 0 <= idx < len(items):
            cv2.destroyWindow("Workout AI — Select Exercise")
            return items[idx][1]


def choose_routine() -> str:
    """One-item launcher for the neck-stretch demo. Returns "neck_stretch".
    Press SPACE/Enter or click anywhere to start; q/Esc quits."""
    width, height = 780, 220
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    clicked = {"go": False}

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["go"] = True

    window = "Workout AI — Select"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_mouse)
    while True:
        canvas[:] = _PANEL_BG
        cv2.putText(
            canvas,
            "Neck Stretch (2 min)",
            (40, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            _TEXT_FG,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Click / SPACE to start, q to quit",
            (40, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            _HINT_FG,
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(window)
            raise SystemExit("User canceled")
        if key in (ord(" "), 13) or clicked["go"]:
            cv2.destroyWindow(window)
            return "neck_stretch"

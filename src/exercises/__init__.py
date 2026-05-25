"""Exercise registry. Each submodule registers its exercises here."""

from exercises.base import Exercise
from exercises.neck_stretch import NeckStretchLeft, NeckStretchRight


EXERCISES: dict[str, Exercise] = {
    "neck_stretch_left": NeckStretchLeft(),
    "neck_stretch_right": NeckStretchRight(),
}

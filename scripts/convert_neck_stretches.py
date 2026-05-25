"""Convert/copy the three neck-stretch images to JPG in data/neck_stretch/."""

from pathlib import Path

from PIL import Image

SRC_DIR = Path(__file__).resolve().parent.parent / "data"
DST_DIR = SRC_DIR / "neck_stretch"

SOURCES = [
    "innovation_blog_7-Best-Stretching-Exercises-for-Neck-Pain-Relief3.webp",
    "neck-stretches1.jpg",
    "Seated-Neck-Release.jpg",
]


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(SOURCES, start=1):
        src = SRC_DIR / name
        dst = DST_DIR / f"neck_stretch_{i:02d}.jpg"
        with Image.open(src) as img:
            img.convert("RGB").save(dst, "JPEG", quality=92)
        print(f"{src.name} -> {dst.relative_to(SRC_DIR.parent)}")


if __name__ == "__main__":
    main()

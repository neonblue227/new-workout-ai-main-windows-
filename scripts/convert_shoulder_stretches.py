"""Convert/copy the three shoulder-stretch images to JPG in data/shoulder_stretch/."""

from pathlib import Path

from PIL import Image

SRC_DIR = Path(__file__).resolve().parent.parent / "data"
DST_DIR = SRC_DIR / "shoulder_stretch"

SOURCES = [
    "tight-shoulder-stretch.jpg",
    "cross+body+shoulder+stretch+.webp",
    "AdobeStock_314857823-1024x683.jpeg",
]


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(SOURCES, start=1):
        src = SRC_DIR / name
        dst = DST_DIR / f"shoulder_stretch_{i:02d}.jpg"
        with Image.open(src) as img:
            img.convert("RGB").save(dst, "JPEG", quality=92)
        print(f"{src.name} -> {dst.relative_to(SRC_DIR.parent)}")


if __name__ == "__main__":
    main()

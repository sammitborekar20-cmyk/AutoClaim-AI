
from pathlib import Path
import cv2
import random

IMAGE_DIR = Path("data/raw/train/images")
LABEL_DIR = Path("data/raw/train/labels")
OUTPUT_DIR = Path("reports/boot_label_audit")

CLASS_ID = 1
NUM_IMAGES = 20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

matching = []

for label_file in LABEL_DIR.glob("*.txt"):

    lines = label_file.read_text().splitlines()

    boot_lines = [
        line for line in lines
        if line.strip() and line.strip().split()[0] == str(CLASS_ID)
    ]

    if not boot_lines:
        continue

    image_path = next(
        (
            IMAGE_DIR / (label_file.stem + ext)
            for ext in extensions
            if (IMAGE_DIR / (label_file.stem + ext)).exists()
        ),
        None
    )

    if image_path:
        matching.append((image_path, label_file, boot_lines))

random.seed(42)
selected = random.sample(matching, min(NUM_IMAGES, len(matching)))

for index, (image_path, label_file, boot_lines) in enumerate(selected, 1):

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    height, width = image.shape[:2]

    for line in boot_lines:

        parts = line.split()
        _, xc, yc, bw, bh = map(float, parts)

        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            image,
            "BOOT-DENT",
            (max(0, x1), max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    output_path = OUTPUT_DIR / f"boot_audit_{index:02d}.jpg"
    cv2.imwrite(str(output_path), image)

    print(f"Saved: {output_path}")

print(f"\nTotal boot-dent images found: {len(matching)}")
print(f"Images exported for inspection: {len(selected)}")
print(f"Output folder: {OUTPUT_DIR.resolve()}")


from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

# AutoClaim AI dataset
IMAGE_DIR = Path("data/raw/train/images")
LABEL_DIR = Path("data/raw/train/labels")

# Class ID 1 = boot-dent
TARGET_CLASS = 1

# Number of examples to display
NUM_IMAGES = 12

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():

    matching_images = []

    for label_file in LABEL_DIR.glob("*.txt"):

        lines = label_file.read_text().splitlines()

        has_boot_damage = any(
            line.strip().split()[0] == str(TARGET_CLASS)
            for line in lines
            if line.strip()
        )

        if has_boot_damage:
            image_path = None

            for extension in IMAGE_EXTENSIONS:
                candidate = IMAGE_DIR / (label_file.stem + extension)

                if candidate.exists():
                    image_path = candidate
                    break

            if image_path:
                matching_images.append((image_path, label_file))

    print("Boot-dent images found:", len(matching_images))

    if not matching_images:
        print("No boot-dent images found.")
        return

    random.seed(42)
    selected = random.sample(
        matching_images,
        min(NUM_IMAGES, len(matching_images))
    )

    columns = 3
    rows = (len(selected) + columns - 1) // columns

    fig, axes = plt.subplots(
        rows, columns,
        figsize=(15, 5 * rows)
    )

    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (image_path, label_path) in zip(axes, selected):

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]

        for line in label_path.read_text().splitlines():

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            if class_id != TARGET_CLASS:
                continue

            x_center, y_center, box_width, box_height = map(
                float, parts[1:]
            )

            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

        ax.imshow(image)
        ax.set_title(image_path.name)
        ax.axis("off")

    for ax in axes[len(selected):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("boot_damage_inspection.png", dpi=150)
    plt.show()

    print("Saved: boot_damage_inspection.png")


if __name__ == "__main__":
    main()

from pathlib import Path
from collections import Counter
import yaml

# --------------------------------------------------
# AUTOCLAIM AI - DATASET LABEL AND CLASS CHECK
# --------------------------------------------------

DATA_YAML = Path("data/raw/data.yaml")
DATASET_ROOT = Path("data/raw")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if not DATA_YAML.exists():
        print(f"ERROR: Cannot find {DATA_YAML.resolve()}")
        return

    with open(DATA_YAML, "r") as file:
        config = yaml.safe_load(file)

    class_names = config.get("names", [])
    number_of_classes = config.get("nc", len(class_names))

    print("\n========== DATASET CONFIGURATION ==========")
    print(f"Number of classes in YAML: {number_of_classes}")
    print(f"Number of class names:     {len(class_names)}")

    print("\nClass ID mapping:")
    for class_id, class_name in enumerate(class_names):
        print(f"{class_id}: {class_name}")

    if number_of_classes != len(class_names):
        print("\nWARNING: nc does not match the number of class names!")

    print("\n========== LABEL VALIDATION ==========")

    for split in ["train", "valid", "test"]:
        label_dir = DATASET_ROOT / split / "labels"
        image_dir = DATASET_ROOT / split / "images"

        if not label_dir.exists():
            print(f"\nWARNING: Missing label folder: {label_dir}")
            continue

        label_files = list(label_dir.glob("*.txt"))
        image_files = [
            file for file in image_dir.iterdir()
            if file.suffix.lower() in IMAGE_EXTENSIONS
        ] if image_dir.exists() else []

        class_counts = Counter()
        invalid_lines = 0
        invalid_class_ids = 0
        invalid_coordinates = 0

        for label_file in label_files:
            for line_number, line in enumerate(
                label_file.read_text().splitlines(), start=1
            ):
                parts = line.strip().split()

                if not parts:
                    continue

                if len(parts) != 5:
                    invalid_lines += 1
                    continue

                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                except ValueError:
                    invalid_lines += 1
                    continue

                class_counts[class_id] += 1

                if class_id < 0 or class_id >= number_of_classes:
                    invalid_class_ids += 1

                coordinates = [x_center, y_center, width, height]

                if (
                    any(value < 0 or value > 1 for value in coordinates)
                    or width <= 0
                    or height <= 0
                ):
                    invalid_coordinates += 1

        print(f"\n{split.upper()} split")
        print(f"Images:                 {len(image_files)}")
        print(f"Label files:            {len(label_files)}")
        print(f"Invalid label lines:    {invalid_lines}")
        print(f"Invalid class IDs:      {invalid_class_ids}")
        print(f"Invalid box values:     {invalid_coordinates}")

        print("Object instances by class:")
        for class_id in range(number_of_classes):
            print(
                f"  {class_id} - {class_names[class_id]}: "
                f"{class_counts[class_id]}"
            )

    print("\nCheck completed.")


if __name__ == "__main__":
    main()
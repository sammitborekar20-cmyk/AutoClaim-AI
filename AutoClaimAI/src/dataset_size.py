from pathlib import Path

# --------------------------------------------------
# AUTOCLAIM AI - DATASET SIZE
# Counts images and YOLO label files in each split
# --------------------------------------------------

DATASET_PATH = Path("data/raw")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

SPLITS = {
    "Train": "train",
    "Validation": "valid",
    "Test": "test"
}


def count_images(folder):
    if not folder.exists():
        return 0

    return sum(
        1
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_labels(folder):
    if not folder.exists():
        return 0

    return sum(
        1
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() == ".txt"
    )


def main():
    total_images = 0
    total_labels = 0

    print("\n==========================================")
    print("       AUTOCLAIM AI DATASET SIZE")
    print("==========================================")

    for split_name, folder_name in SPLITS.items():

        image_folder = DATASET_PATH / folder_name / "images"
        label_folder = DATASET_PATH / folder_name / "labels"

        images = count_images(image_folder)
        labels = count_labels(label_folder)

        total_images += images
        total_labels += labels

        print(f"\n{split_name} Dataset")
        print("------------------------------------------")
        print(f"Images:       {images}")
        print(f"Label files:  {labels}")

    print("\n==========================================")
    print("             TOTAL DATASET")
    print("==========================================")
    print(f"Total images:      {total_images}")
    print(f"Total label files: {total_labels}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
from pathlib import Path
import random
import cv2
import os


IMAGE_DIR = Path("data/raw/train/images")
LABEL_DIR = Path("data/raw/train/labels")

OUTPUT_DIR = Path(
    "runs/detect/runs/damage/training_dataset_audit"
)

NUMBER_OF_IMAGES = 30

CLASS_NAMES = [
    "bonnet-dent",
    "boot-dent",
    "doorouter-dent",
    "fender-dent",
    "front-bumper-dent",
    "Front-windscreen-damage",
    "Headlight-damage",
    "quaterpanel-dent",
    "rear-bumper-dent",
    "Rear-windscreen-Damage",
    "roof-dent",
    "Runningboard-Damage",
    "Sidemirror-Damage",
    "Taillight-Damage"
]


def yolo_to_xyxy(
    x_center,
    y_center,
    width,
    height,
    image_width,
    image_height
):
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height

    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    return x1, y1, x2, y2


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = [
        file for file in IMAGE_DIR.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    random.seed(42)

    selected_images = random.sample(
        image_files,
        min(NUMBER_OF_IMAGES, len(image_files))
    )

    print()
    print("========================================")
    print("     TRAINING DATASET AUDIT")
    print("========================================")
    print()
    print(f"Training images available : {len(image_files)}")
    print(f"Images selected for audit : {len(selected_images)}")
    print()

    for index, image_path in enumerate(selected_images, start=1):

        label_path = (
            LABEL_DIR /
            f"{image_path.stem}.txt"
        )

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not load: {image_path.name}")
            continue

        height, width = image.shape[:2]

        ground_truths = []

        if label_path.exists():

            with open(label_path, "r") as file:
                lines = file.readlines()

            for line in lines:

                values = line.strip().split()

                if len(values) < 5:
                    continue

                class_id = int(values[0])

                x_center = float(values[1])
                y_center = float(values[2])
                box_width = float(values[3])
                box_height = float(values[4])

                x1, y1, x2, y2 = yolo_to_xyxy(
                    x_center,
                    y_center,
                    box_width,
                    box_height,
                    width,
                    height
                )

                class_name = CLASS_NAMES[class_id]

                ground_truths.append(class_name)

                # GREEN = ground truth
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    image,
                    f"GT: {class_name}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        output_path = OUTPUT_DIR / image_path.name

        cv2.imwrite(
            str(output_path),
            image
        )

        print(
            f"[{index}/{len(selected_images)}] "
            f"{image_path.name}"
        )

        print(
            f"  Labels: {ground_truths}"
        )

    print()
    print("========================================")
    print("       AUDIT IMAGES CREATED")
    print("========================================")
    print()
    print(f"Output directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
from ultralytics import YOLO
import cv2
import os
import random


MODEL_PATH = (
    "runs/detect/runs/damage/"
    "autoclaim_20epochs/weights/best.pt"
)

IMAGE_DIR = "data/raw/test/images"
LABEL_DIR = "data/raw/test/labels"

OUTPUT_DIR = (
    "runs/detect/runs/damage/"
    "random_test_inspection"
)

CONFIDENCE = 0.50
NUMBER_OF_IMAGES = 10

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


def yolo_to_xyxy(values, image_width, image_height):
    x_center = float(values[0]) * image_width
    y_center = float(values[1]) * image_height
    width = float(values[2]) * image_width
    height = float(values[3]) * image_height

    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    return x1, y1, x2, y2


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)

    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.seed(42)
    selected = random.sample(
        image_files,
        min(NUMBER_OF_IMAGES, len(image_files))
    )

    print()
    print("========================================")
    print(" RANDOM TEST SET INSPECTION")
    print("========================================")
    print(f"Images selected: {len(selected)}")
    print()

    for index, filename in enumerate(selected, start=1):

        image_path = os.path.join(IMAGE_DIR, filename)

        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(LABEL_DIR, label_filename)

        image = cv2.imread(image_path)

        if image is None:
            continue

        height, width = image.shape[:2]

        # -----------------------------
        # Ground truth - GREEN
        # -----------------------------

        ground_truths = []

        if os.path.exists(label_path):

            with open(label_path, "r") as file:
                lines = file.readlines()

            for line in lines:

                values = line.strip().split()

                if len(values) < 5:
                    continue

                class_id = int(values[0])

                box = yolo_to_xyxy(
                    values[1:5],
                    width,
                    height
                )

                ground_truths.append(
                    CLASS_NAMES[class_id]
                )

                x1, y1, x2, y2 = box

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    f"GT: {CLASS_NAMES[class_id]}",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        # -----------------------------
        # Predictions - RED
        # -----------------------------

        results = model.predict(
            source=image_path,
            conf=CONFIDENCE,
            device="mps",
            verbose=False
        )

        predictions = []

        for box in results[0].boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            predictions.append(
                CLASS_NAMES[class_id]
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                image,
                f"Pred: {CLASS_NAMES[class_id]} "
                f"{confidence:.2f}",
                (x1, min(y2 + 20, height - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        cv2.imwrite(output_path, image)

        print(f"[{index}/{len(selected)}] {filename}")
        print(f"  Ground truth: {ground_truths}")
        print(f"  Prediction   : {predictions}")
        print()

    print("========================================")
    print("INSPECTION COMPLETED")
    print("========================================")
    print()
    print("Output:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
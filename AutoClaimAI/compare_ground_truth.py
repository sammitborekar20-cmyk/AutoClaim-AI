from ultralytics import YOLO
import cv2
import os


MODEL_PATH = (
    "runs/detect/runs/damage/"
    "autoclaim_20epochs/weights/best.pt"
)

IMAGE_PATH = (
    "data/raw/test/images/"
    "0123_JPEG_jpg.rf.86bf174672a49ce3f5bf94dbd495f63b.jpg"
)

LABEL_PATH = (
    "data/raw/test/labels/"
    "0123_JPEG_jpg.rf.86bf174672a49ce3f5bf94dbd495f63b.txt"
)

OUTPUT_DIR = "data/processed"

OUTPUT_PATH = (
    "data/processed/"
    "0123_ground_truth_vs_prediction.jpg"
)


# YOLO class names from data.yaml
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading image...")
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("ERROR: Could not load image.")
        return

    image_height, image_width = image.shape[:2]

    # ------------------------------------------------
    # Draw ground-truth annotation
    # ------------------------------------------------

    print("Reading ground-truth label...")

    with open(LABEL_PATH, "r") as file:
        lines = file.readlines()

    for line in lines:

        values = line.strip().split()

        class_id = int(values[0])

        x_center = float(values[1])
        y_center = float(values[2])
        width = float(values[3])
        height = float(values[4])

        x1, y1, x2, y2 = yolo_to_xyxy(
            x_center,
            y_center,
            width,
            height,
            image_width,
            image_height
        )

        label = (
            "GT: "
            + CLASS_NAMES[class_id]
        )

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
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # ------------------------------------------------
    # Run YOLO prediction
    # ------------------------------------------------

    print("Running YOLO prediction...")

    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=IMAGE_PATH,
        conf=0.25,
        device="mps",
        verbose=False
    )

    result = results[0]

    # ------------------------------------------------
    # Draw predictions
    # ------------------------------------------------

    prediction_count = 0

    if result.boxes is not None:

        for box in result.boxes:

            prediction_count += 1

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            coordinates = box.xyxy[0].tolist()

            x1, y1, x2, y2 = map(
                int,
                coordinates
            )

            label = (
                f"Pred: {CLASS_NAMES[class_id]} "
                f"{confidence * 100:.1f}%"
            )

            # RED = prediction
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            cv2.putText(
                image,
                label,
                (x1, min(y2 + 25, image_height - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )

    # ------------------------------------------------
    # Save result
    # ------------------------------------------------

    cv2.imwrite(
        OUTPUT_PATH,
        image
    )

    print()
    print("========================================")
    print("       COMPARISON COMPLETED")
    print("========================================")
    print()
    print(f"Ground-truth boxes : {len(lines)}")
    print(f"Predicted boxes    : {prediction_count}")
    print()
    print(f"Output saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
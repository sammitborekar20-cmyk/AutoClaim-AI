from ultralytics import YOLO
import cv2
import os


# ==========================================
# 1. MODEL PATH
# ==========================================

model_path = (
    "runs/detect/runs/damage/"
    "autoclaim_detector_test/"
    "weights/best.pt"
)


# ==========================================
# 2. TEST IMAGE PATH
# ==========================================

image_path = (
    "data/raw/test/images/"
    "signlight-10-53-_jpg.rf."
    "8769dd83f6c826b27b7c164dca5d50c4.jpg"
)


# ==========================================
# 3. OUTPUT PATH
# ==========================================

output_path = "data/processed/annotated_damage.jpg"


# Create output directory if it doesn't exist
os.makedirs("data/processed", exist_ok=True)


# ==========================================
# 4. LOAD YOLO MODEL
# ==========================================

print("Loading YOLO model...")

model = YOLO(model_path)


# ==========================================
# 5. LOAD IMAGE
# ==========================================

print("Loading test image...")

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not load the image.")
    print(f"Image path: {image_path}")
    exit()


# ==========================================
# 6. RUN YOLO PREDICTION
# ==========================================

print("Running damage detection...")

results = model.predict(
    source=image_path,
    conf=0.25
)


# ==========================================
# 7. FIND STRONGEST DETECTION
# ==========================================

strongest_detection = None
highest_confidence = 0


for result in results:

    for box in result.boxes:

        confidence = float(box.conf[0])

        if confidence > highest_confidence:

            highest_confidence = confidence

            class_id = int(box.cls[0])

            damage_type = model.names[class_id]

            coordinates = box.xyxy[0].tolist()

            strongest_detection = {
                "damage_type": damage_type,
                "confidence": confidence,
                "box": coordinates
            }


# ==========================================
# 8. DRAW STRONGEST DETECTION
# ==========================================

if strongest_detection is not None:

    damage_type = strongest_detection["damage_type"]
    confidence = strongest_detection["confidence"]
    box = strongest_detection["box"]

    # Convert coordinates to integers
    x1, y1, x2, y2 = map(int, box)

    # Draw bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    # Create label
    label = (
        f"{damage_type} "
        f"{confidence * 100:.1f}%"
    )

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    # Get label size
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness
    )

    # Draw label background
    cv2.rectangle(
        image,
        (x1, y1 - text_height - 15),
        (x1 + text_width + 10, y1),
        (0, 255, 0),
        -1
    )

    # Draw label text
    cv2.putText(
        image,
        label,
        (x1 + 5, y1 - 7),
        font,
        font_scale,
        (0, 0, 0),
        thickness
    )

    print()
    print("================================")
    print("       DAMAGE DETECTION")
    print("================================")
    print(f"Damage Type : {damage_type}")
    print(f"Confidence  : {confidence * 100:.1f}%")
    print(
        f"Bounding Box: "
        f"{[round(x, 2) for x in box]}"
    )

else:

    print()
    print("No damage detected.")


# ==========================================
# 9. SAVE ANNOTATED IMAGE
# ==========================================

success = cv2.imwrite(
    output_path,
    image
)


if success:

    print()
    print("================================")
    print("      OUTPUT IMAGE SAVED")
    print("================================")
    print(f"Output path: {output_path}")

else:

    print("ERROR: Could not save the output image.")
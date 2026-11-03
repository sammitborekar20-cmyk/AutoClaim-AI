from ultralytics import YOLO
import os


# ==========================================
# 1. MODEL PATH
# ==========================================

model_path = (
    "runs/detect/runs/damage/"
    "autoclaim_mps_test/weights/best.pt"
)


# ==========================================
# 2. TEST IMAGE DIRECTORY
# ==========================================

test_image_dir = "data/raw/test/images"


# ==========================================
# 3. OUTPUT DIRECTORY
# ==========================================

output_dir = "runs/detect/test_predictions"

os.makedirs(output_dir, exist_ok=True)


# ==========================================
# 4. LOAD MODEL
# ==========================================

print("Loading trained model...")

model = YOLO(model_path)


# ==========================================
# 5. GET TEST IMAGES
# ==========================================

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
)

image_files = [
    os.path.join(test_image_dir, file)
    for file in os.listdir(test_image_dir)
    if file.endswith(image_extensions)
]


# ==========================================
# 6. SELECT SEVERAL IMAGES
# ==========================================

# Evaluate the first 10 test images
selected_images = image_files[:10]

print()
print(f"Total test images available: {len(image_files)}")
print(f"Images selected for evaluation: {len(selected_images)}")


# ==========================================
# 7. RUN PREDICTIONS
# ==========================================

print()
print("Running predictions...")
print()


for image_path in selected_images:

    print("----------------------------------------")
    print(f"Image: {os.path.basename(image_path)}")

    # Run prediction
    results = model.predict(
        source=image_path,
        conf=0.25,
        save=False,
        device="mps"
    )

    result = results[0]

    # --------------------------------------
    # Print detections
    # --------------------------------------

    if result.boxes is None or len(result.boxes) == 0:

        print("Prediction: No damage detected.")

    else:

        print(
            f"Predictions: {len(result.boxes)}"
        )

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            damage_type = model.names[class_id]

            print(
                f"  {damage_type} "
                f"({confidence * 100:.1f}%)"
            )

    # --------------------------------------
    # Save annotated image
    # --------------------------------------

    annotated_image = result.plot()

    filename = os.path.basename(image_path)

    output_path = os.path.join(
        output_dir,
        filename
    )

    # Ultralytics returns BGR image,
    # which OpenCV can save directly.
    import cv2

    cv2.imwrite(
        output_path,
        annotated_image
    )

    print(f"Saved: {output_path}")


# ==========================================
# 8. FINISHED
# ==========================================

print()
print("========================================")
print("       EVALUATION COMPLETED")
print("========================================")
print(
    f"Annotated images saved in:\n"
    f"{output_dir}"
)
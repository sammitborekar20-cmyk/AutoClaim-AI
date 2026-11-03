from ultralytics import YOLO
import os


MODEL_PATH = (
    "runs/detect/runs/damage/"
    "autoclaim_20epochs/weights/best.pt"
)

TEST_IMAGE_DIR = "data/raw/test/images"

OUTPUT_DIR = (
    "runs/detect/runs/damage/"
    "test_predictions_20epochs"
)


def main():

    print()
    print("========================================")
    print("   AUTOCLAIM AI - TEST PREDICTIONS")
    print("========================================")
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load trained model
    print("Loading 20-epoch YOLO11 model...")
    model = YOLO(MODEL_PATH)

    # Get test images
    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG"
    )

    image_files = [
        os.path.join(TEST_IMAGE_DIR, file)
        for file in os.listdir(TEST_IMAGE_DIR)
        if file.endswith(image_extensions)
    ]

    image_files.sort()

    print(f"Test images found: {len(image_files)}")
    print()

    print("Generating annotated predictions...")
    print()

    # Process every test image
    for index, image_path in enumerate(image_files, start=1):

        print(
            f"[{index}/{len(image_files)}] "
            f"{os.path.basename(image_path)}"
        )

        results = model.predict(
            source=image_path,
            conf=0.50,
            device="mps",
            save=False,
            verbose=False
        )

        result = results[0]

        # Generate annotated image
        annotated_image = result.plot()

        # Save using original filename
        filename = os.path.basename(image_path)

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        # Save annotated image
        import cv2

        success = cv2.imwrite(
            output_path,
            annotated_image
        )

        if not success:
            print(f"  ERROR saving: {output_path}")

    print()
    print("========================================")
    print("       PREDICTIONS COMPLETED")
    print("========================================")
    print()
    print(f"Images processed : {len(image_files)}")
    print()
    print("Annotated images saved in:")
    print(OUTPUT_DIR)
    print()


if __name__ == "__main__":
    main()
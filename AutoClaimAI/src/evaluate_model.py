from ultralytics import YOLO
from pathlib import Path

# --------------------------------------------------
# AUTOCLAIM AI - YOLO MODEL EVALUATION
# Measures Precision, Recall, mAP@0.50 and mAP@0.50:0.95
# --------------------------------------------------

# Trained model path
MODEL_PATH = Path(
    "runs/detect/runs/detect/runs/damage/"
    "autoclaim_improved_v1/weights/best.pt"
)

# Dataset configuration file
DATA_YAML = Path("data/raw/data.yaml")


def main():
    # Check whether model file exists
    if not MODEL_PATH.exists():
        print("ERROR: Trained model was not found.")
        print(f"Checked path: {MODEL_PATH.resolve()}")
        print("\nSearch for your best.pt file using:")
        print("find runs -name best.pt")
        return

    # Check whether dataset YAML exists
    if not DATA_YAML.exists():
        print("ERROR: Dataset configuration file was not found.")
        print(f"Checked path: {DATA_YAML.resolve()}")
        return

    print("\nLoading trained YOLO model...")
    model = YOLO(str(MODEL_PATH))

    print("Starting evaluation on the test dataset...\n")

    # Evaluate on the test split
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=8,
        device="mps",       # Apple Silicon Mac
        conf=0.001,
        iou=0.6,
        plots=True
    )

    # Display results
    print("\n==========================================")
    print("       AUTOCLAIM AI MODEL RESULTS")
    print("==========================================")

    print(f"Precision:       {metrics.box.mp:.4f}")
    print(f"Recall:          {metrics.box.mr:.4f}")
    print(f"mAP@0.50:        {metrics.box.map50:.4f}")
    print(f"mAP@0.50:0.95:   {metrics.box.map:.4f}")

    print("==========================================")
    print("Evaluation completed.")
    print("YOLO evaluation plots and results are saved")
    print("inside the runs/detect/val folder.")
    print("==========================================\n")


if __name__ == "__main__":
    main()
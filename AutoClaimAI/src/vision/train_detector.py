from ultralytics import YOLO


def main():

    print("Loading YOLO11 model...")

    model = YOLO("yolo11n.pt")

    print("Starting 20-epoch training on Apple M1 GPU (MPS)...")

    model.train(
        data="data/raw/data.yaml",

        # Training duration
        epochs=20,

        # Image size
        imgsz=640,

        # Batch size
        batch=8,

        # Apple Silicon GPU
        device="mps",

        # Training output
        project="runs/damage",
        name="autoclaim_20epochs",
        exist_ok=True
    )

    print()
    print("========================================")
    print("       TRAINING COMPLETED")
    print("========================================")
    print(
        "Model location:"
    )
    print(
        "runs/detect/runs/damage/"
        "autoclaim_20epochs/weights/best.pt"
    )


if __name__ == "__main__":
    main()